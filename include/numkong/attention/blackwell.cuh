/**
 *  @file include/numkong/attention/blackwell.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Ragged attention for the NVIDIA compute capability 10.x family.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/ampere.cuh
 *  @sa include/numkong/dots/blackwell.cuh
 *
 *  FlashAttention-4 on `tcgen05.mma`: persistent blocks take items of up to two 128-row Q tiles of
 *  one K and V head, whose products take turns on the tensor cores. One thread streams 128-position
 *  panels of K and V through a ring of stages with the Tensor Memory Accelerator, and one thread
 *  issues every product. S lands in tensor memory, where a warpgroup per tile reads one row per
 *  thread and writes P over it, and P · V reads P from there as its A operand into an F32 O that
 *  stays in tensor memory too. A correction warpgroup rescales a row of O only when its maximum
 *  moved, which BF16 lets it do only past 2⁸, and finally normalizes and stores it. BF16 multiplies
 *  as @c kind::f16, E4M3 as @c kind::f8f6f4 with weights e4m3(256 · p), and I8, which this family
 *  has no tensor-core kind for, as exact F16: its pack stores K and V widened and the softmax warps
 *  widen Q as they stage it, under the U8 weights of the serial contract. The pack keeps V
 *  position-major, which P · V reads MN-major, so one tensor map reaches every plane; depths above
 *  256 keep the @c cuda layout and kernel. Only the "100f" family code carries these instructions,
 *  so every other device pass traps.
 *
 *  The BF16 backward stages K, V, Q and dO once each in the same swizzled rows, which the products
 *  read K-major or MN-major as each needs. Blocks of 128 keys take Sᵀ and dPᵀ into tensor memory,
 *  turn them into Pᵀ and dSᵀ in place a lane per key, and accumulate dV and dK from there over the
 *  rows that see them; blocks of 128 rows then do the same for dQ. Past 128 dimensions the other
 *  side spans 64 rows or keys, so the four operands fit in shared memory, and dK and dV take two
 *  128-column slices of tensor memory, recomputing Sᵀ and dPᵀ for each. Heads past 256 run the
 *  @c cuda loops.
 *
 *  The single-pass backward computes S, dP and dS once per key block and row chunk, 5 products per
 *  pair where the two passes take 7: dK and dV stay in tensor memory, and the partial dS · K is
 *  written over the spent dP columns and added into the query gradient by TMA reduce-add of F32
 *  boxes. It takes BF16 heads of 64 or 128 dimensions whose query head group divides 128, with
 *  16-byte aligned rows and strides, and needs dO in BF16 and D once per row, which it allocates
 *  with @c cudaMallocAsync on the caller's stream and frees with @c cudaFreeAsync after the
 *  kernels. Any other call, or a failed allocation, takes the two passes.
 */
#ifndef NUMKONG_ATTENTION_BLACKWELL_CUH
#define NUMKONG_ATTENTION_BLACKWELL_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_CUDA_BLACKWELL_

#include "numkong/attention/ampere.cuh" // `nk_f32x2_to_bf16x2_ampere_`, `nk_attention_fallback_bf16_cuda_`
#include "numkong/dots/blackwell.cuh"   // `nk_mma_f16_blackwell_`, `nk_cross_map_blackwell_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_attention_tile_rows_blackwell_k = 128,
    nk_attention_panel_blackwell_k = 128,
    // Copies span half a panel, as the pack pads every plane only to a multiple of 64 positions.
    nk_attention_box_rows_blackwell_k = 64,
    nk_attention_block_bytes_blackwell_k = nk_attention_panel_blackwell_k * 128,
    nk_attention_stages_blackwell_k = 8,
    nk_attention_narrow_depth_blackwell_k = 128,
    nk_attention_wide_depth_blackwell_k = 256,
    nk_attention_tensor_columns_blackwell_k = 512,
    // Tensor-memory columns: tile t keeps S from 128 t, with P over its first columns, and O from
    // 256 + 128 t, which a wide head's only tile extends to the last column.
    nk_attention_scores_column_blackwell_k = 0,
    nk_attention_output_column_blackwell_k = 256,
    // Warps 0 to 7 run the softmax of tiles 0 and 1, 8 to 11 correct O, 12 multiplies, 13 loads.
    nk_attention_correct_warp_blackwell_k = 8,
    nk_attention_multiply_warp_blackwell_k = 12,
    nk_attention_load_warp_blackwell_k = 13,
    nk_attention_threads_blackwell_k = 14 * 32,
    nk_attention_shared_limit_blackwell_k = 232448,
};

/** Barriers and the warps' exchange, past the Q tiles and the ring of stages. Per stage, @c loaded
 *  completes with the copies and @c consumed once a later S shows the products done with it, which
 *  the softmax warps see for K and the correction warps for V; per work slot, @c scheduled with the
 *  load warp and @c taken with every consumer warp; per tile, @c queries_staged with its softmax
 *  warps and @c queries_read with its last S, @c scored with each S, @c weighed_half with the first
 *  half of each P and @c weighed with all of it, @c rescaling with each factor past an item's first
 *  panel, @c corrected with each rescaled or stored O, @c accumulated with an item's last P · V,
 *  @c summed with the item's row sums and @c sums_read with the correction warps' read of them.
 *  Single-tile items alternate S between both tiles' columns, the second with the barriers and
 *  factors of slot 2, which tile 1's warps never see, as they may already wait on slot 1 for a
 *  later item; such items signal @c multiplied with each P · V but the last. A shared item's
 *  correction warps complete @c exchanged in the other block with each row's log-sum-exp, and rank
 *  0's complete @c merged in rank 1 once O lands. */
typedef struct {
    nk_u64_t loaded[nk_attention_stages_blackwell_k];
    nk_u64_t consumed[nk_attention_stages_blackwell_k];
    nk_u64_t scheduled[2];
    nk_u64_t taken[2];
    nk_u64_t queries_staged[2];
    nk_u64_t queries_read[2];
    nk_u64_t scored[3];
    nk_u64_t weighed_half[3];
    nk_u64_t weighed[3];
    nk_u64_t rescaling[3];
    nk_u64_t corrected[2];
    nk_u64_t accumulated[2];
    nk_u64_t multiplied;
    nk_u64_t summed[2];
    nk_u64_t sums_read[2];
    nk_u64_t exchanged[2][2];
    nk_u64_t merged[2];
    nk_attention_work_t works[2];
    nk_f32_t factors[3][nk_attention_tile_rows_blackwell_k];
    nk_f32_t sums[2][nk_attention_tile_rows_blackwell_k];
    nk_f32_t log_sum_exps[2][nk_attention_tile_rows_blackwell_k];
    nk_f32_t pair_log_sum_exps[2][2][nk_attention_tile_rows_blackwell_k];
    nk_u32_t tensor_memory;
} nk_attention_control_blackwell_t;

/** Everything one launch shares: the attention arguments, the tensor map of the packed buffer as
 *  rows of a K or V plane, whose payload starts a whole number of rows in, the map of Q as tokens
 *  of heads of head rows, which reaches @c query_tokens tokens, or none when zero, and whether
 *  clusters of two blocks split each item's keys. A grid constant, so the maps keep an address. */
typedef struct {
    CUtensorMap map;
    CUtensorMap queries_map;
    nk_size_t query_tokens;
    int paired;
    nk_attention_arguments_t attention;
} nk_attention_tile_arguments_blackwell_t;

/** What one kernel computes, fixed at compile time: the bytes of its input element, and whether its
 *  Q widens to F16, as I8's does against the F16 K and V its pack holds; up to 128 depths in two Q
 *  tiles per item, or up to 256 in one; whether panels crossing the band's edges mask their scores,
 *  while panels past the segment's keys always do; and the operand format code of both products. */
typedef struct {
    unsigned element_bytes;
    int widens;
    nk_attention_width_t width;
    nk_attention_mask_t mask;
    nk_u32_t format;
} nk_attention_kernel_blackwell_t;

/** What a kernel and its launch make of the tiles, the ring and the products. */
typedef struct {

    /** Whether Q widens from I8 codes to the F16 the products read. */
    int widens;

    /** Q tiles an item takes at most, and its rows. */
    unsigned tiles, item_rows;

    /** Bytes of an input element, then of a packed K or V row, and the bytes and count of the
     *  copies along it. */
    unsigned element_bytes, row_bytes, box_bytes, boxes;

    /** Bytes of a staged row as the products read it, and the 32-byte steps of S along it. */
    unsigned staged_bytes, depth_steps;

    /** The columns of O, the steps of P · V per panel and the bytes each moves along V. */
    unsigned value_columns, value_steps, value_step_bytes;

    /** Bytes of a Q tile, of a stage, and the ring's stages. */
    unsigned queries_bytes, stage_bytes, ring_stages;

    nk_u32_t scores_instruction, values_instruction;

    /** The base-2 score multiplier, with the power of two the operands put on scores undone. */
    nk_f32_t scale2;

    /** How far past a row's maximum a panel's must reach before the row moves to it: 8 for BF16 and
     *  F16, whose weights hold the 2⁸ of a stale maximum, so O is rarely rescaled, and 0 for the
     *  dtypes whose weights need the true maximum. */
    nk_f32_t rescale_threshold;

    /** The row of the pack's tensor map that the payload starts at. */
    nk_size_t payload_row;

    /** The keys each query row sees, as @c nk_attention_kernel_band_simt_ gives them. */
    nk_diagonal_band_t band;

    /** Whether the load warp copies Q tiles starting on a whole token through the map of Q. */
    int queries_mapped;

    /** Whether a pair of blocks shares each item, rank 0 taking the first half of its panels and
     *  rank 1 the rest, and this block's rank in it. */
    int paired;
    nk_u32_t pair_rank;
} nk_attention_schedule_blackwell_t;

/** One work item as every role sees it: its panels of keys, and where its planes start. */
typedef struct {
    nk_attention_work_t work;

    /** Q tiles with rows. */
    unsigned tiles;

    /** The first 128-position panel some row sees, and the panels from it to the last one. */
    unsigned panel_first, panels;

    /** Keys of the segment, and their count rounded up to the pack's 64. */
    nk_size_t length, positions_padded;

    /** Map rows of the item's first K and first V position. */
    nk_size_t keys_row, values_row;

    /** Whether both blocks of a pair take a share of the item's panels and merge their rows. */
    int shared;
} nk_attention_item_blackwell_t;

/** Bytes of one operand element as the products read it: 1 for E4M3 codes, else 2, as 16-bit
 *  inputs are and 8-bit integers widen to. */
NUMKONG_CONSTEXPR unsigned nk_attention_operand_bytes_blackwell_(unsigned element_bytes, int widens) {
    return element_bytes == 1 && !widens ? 1u : 2u;
}

/** Bytes of a staged row at the deepest head @p width takes, as the products read it. */
NUMKONG_CONSTEXPR unsigned nk_attention_staged_bytes_blackwell_(unsigned element_bytes, int widens,
                                                                nk_attention_width_t width) {
    return (width == nk_attention_width_128_k ? nk_attention_narrow_depth_blackwell_k
                                              : nk_attention_wide_depth_blackwell_k) *
           nk_attention_operand_bytes_blackwell_(element_bytes, widens);
}

NUMKONG_CONSTEXPR unsigned nk_attention_tiles_blackwell_(nk_attention_width_t width) {
    return width == nk_attention_width_128_k ? 2u : 1u;
}

/** Bytes of one stage: a panel of the rows the products read. */
NUMKONG_CONSTEXPR unsigned nk_attention_stage_bytes_blackwell_(unsigned element_bytes, int widens,
                                                               nk_attention_width_t width) {
    return nk_attention_panel_blackwell_k * nk_attention_staged_bytes_blackwell_(element_bytes, widens, width);
}

/** Stages fitting beside the Q tiles and barriers, at most @c nk_attention_stages_blackwell_k. */
NUMKONG_CONSTEXPR unsigned nk_attention_ring_stages_blackwell_(unsigned element_bytes, int widens,
                                                               nk_attention_width_t width) {
    unsigned const queries_bytes = nk_attention_tiles_blackwell_(width) * nk_attention_tile_rows_blackwell_k *
                                   nk_attention_staged_bytes_blackwell_(element_bytes, widens, width);
    unsigned const stages = (nk_attention_shared_limit_blackwell_k - 1024 -
                             (unsigned)sizeof(nk_attention_control_blackwell_t) - queries_bytes) /
                            nk_attention_stage_bytes_blackwell_(element_bytes, widens, width);
    return stages < nk_attention_stages_blackwell_k ? stages : nk_attention_stages_blackwell_k;
}

/** Dynamic shared memory of a block: Q tiles, the ring and the barriers, plus the slack that aligns
 *  them to 1024 bytes. */
NUMKONG_CONSTEXPR unsigned nk_attention_shared_bytes_blackwell_(unsigned element_bytes, int widens,
                                                                nk_attention_width_t width) {
    return 1024 +
           nk_attention_tiles_blackwell_(width) * nk_attention_tile_rows_blackwell_k *
               nk_attention_staged_bytes_blackwell_(element_bytes, widens, width) +
           nk_attention_ring_stages_blackwell_(element_bytes, widens, width) *
               nk_attention_stage_bytes_blackwell_(element_bytes, widens, width) +
           (unsigned)sizeof(nk_attention_control_blackwell_t);
}

#pragma endregion Configuration

#pragma region Instructions

/** A 128-row step reading A from tensor memory at @p a, 16 depths per 8 columns, and B through
 *  its shared-memory descriptor. */
NUMKONG_DEVICE void nk_mma_f16_tmem_blackwell_(nk_u32_t accumulator, nk_u32_t a, nk_u64_t b, nk_u32_t instruction,
                                               nk_u32_t accumulate) {
    asm volatile("{\n.reg .pred accumulate;\n"      //
                 "setp.ne.b32 accumulate, %4, 0;\n" //
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, accumulate;\n}\n" ::"r"(accumulator),
                 "r"(a), "l"(b), "r"(instruction), "r"(accumulate)
                 : "memory");
}

/** A 128-row step reading A from tensor memory at @p a, 32 depths per 8 columns. */
NUMKONG_DEVICE void nk_mma_f8f6f4_tmem_blackwell_(nk_u32_t accumulator, nk_u32_t a, nk_u64_t b, nk_u32_t instruction,
                                                  nk_u32_t accumulate) {
    asm volatile("{\n.reg .pred accumulate;\n"      //
                 "setp.ne.b32 accumulate, %4, 0;\n" //
                 "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, accumulate;\n}\n" ::"r"(accumulator),
                 "r"(a), "l"(b), "r"(instruction), "r"(accumulate)
                 : "memory");
}

/* The largest of 32 consecutive F32 columns of this thread's lane, or their smallest when
 *  @p smallest, reduced once loaded. */
NUMKONG_DEVICE nk_f32_t nk_tmem_load_extreme_x32_blackwell_(nk_u32_t address, int smallest) {
    nk_u32_t values[32];
    nk_tmem_load_x32_blackwell_(address, values);
    nk_f32_t extreme = __uint_as_float(values[0]);
#pragma unroll
    for (unsigned offset = 1; offset < 32; ++offset)
        extreme = smallest ? fminf(extreme, __uint_as_float(values[offset]))
                           : fmaxf(extreme, __uint_as_float(values[offset]));
    return extreme;
}

/* Two F32 lanes of a 64-bit register at once: @p a times @p b plus @p c. */
NUMKONG_DEVICE nk_u64_t nk_f32x2_fma_blackwell_(nk_u64_t a, nk_u64_t b, nk_u64_t c) {
    nk_u64_t result;
    asm("fma.rn.f32x2 %0, %1, %2, %3;\n" : "=l"(result) : "l"(a), "l"(b), "l"(c));
    return result;
}

NUMKONG_DEVICE nk_u64_t nk_f32x2_mul_blackwell_(nk_u64_t a, nk_u64_t b) {
    nk_u64_t result;
    asm("mul.rn.f32x2 %0, %1, %2;\n" : "=l"(result) : "l"(a), "l"(b));
    return result;
}

NUMKONG_DEVICE nk_u64_t nk_f32x2_add_blackwell_(nk_u64_t a, nk_u64_t b) {
    nk_u64_t result;
    asm("add.rn.f32x2 %0, %1, %2;\n" : "=l"(result) : "l"(a), "l"(b));
    return result;
}

/* Adds two F32 lane pairs, rounding toward negative infinity. */
NUMKONG_DEVICE nk_u64_t nk_f32x2_add_down_blackwell_(nk_u64_t a, nk_u64_t b) {
    nk_u64_t result;
    asm("add.rm.f32x2 %0, %1, %2;\n" : "=l"(result) : "l"(a), "l"(b));
    return result;
}

/* Whether this lane is the one lane `elect.sync` picks from the converged warp. */
NUMKONG_DEVICE int nk_elect_one_blackwell_(void) {
    nk_u32_t elected;
    asm volatile("{\n.reg .pred elected;\n"            //
                 "elect.sync _|elected, 0xFFFFFFFF;\n" //
                 "selp.u32 %0, 1, 0, elected;\n}\n"
                 : "=r"(elected));
    return (int)elected;
}

/* Copies the box at byte @p column, head @p head and token @p token of the tensor @p map describes,
 *  zero-filling what lies outside it, and completes its bytes on @p barrier. */
NUMKONG_DEVICE void nk_load_box_3d_blackwell_(nk_u32_t destination, void const *map, nk_u32_t barrier, nk_i32_t column,
                                              nk_i32_t head, nk_i32_t token) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes " //
                 "[%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(destination),
                 "l"(map), "r"(column), "r"(head), "r"(token), "r"(barrier)
                 : "memory");
}

/* Writes 8 consecutive columns of this thread's lane, without waiting. */
NUMKONG_DEVICE void nk_tmem_store_x8_blackwell_(nk_u32_t address, nk_u32_t const values[8]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n" ::"r"(address),
                 "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]), "r"(values[4]), "r"(values[5]),
                 "r"(values[6]), "r"(values[7])
                 : "memory");
}

/* Writes 16 consecutive columns of this thread's lane, without waiting. */
NUMKONG_DEVICE void nk_tmem_store_x16_blackwell_(nk_u32_t address, nk_u32_t const values[16]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "                                              //
                 "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n" ::"r"(address), //
                 "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]), "r"(values[4]), "r"(values[5]),
                 "r"(values[6]), "r"(values[7]), "r"(values[8]), "r"(values[9]), "r"(values[10]), "r"(values[11]),
                 "r"(values[12]), "r"(values[13]), "r"(values[14]), "r"(values[15])
                 : "memory");
}

/* Writes 32 consecutive columns of this thread's lane, without waiting. */
NUMKONG_DEVICE void nk_tmem_store_x32_blackwell_(nk_u32_t address, nk_u32_t const values[32]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "                                             //
                 "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "                  //
                 "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};\n" ::"r"( //
                     address),
                 "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]), "r"(values[4]), "r"(values[5]),
                 "r"(values[6]), "r"(values[7]), "r"(values[8]), "r"(values[9]), "r"(values[10]), "r"(values[11]),
                 "r"(values[12]), "r"(values[13]), "r"(values[14]), "r"(values[15]), "r"(values[16]), "r"(values[17]),
                 "r"(values[18]), "r"(values[19]), "r"(values[20]), "r"(values[21]), "r"(values[22]), "r"(values[23]),
                 "r"(values[24]), "r"(values[25]), "r"(values[26]), "r"(values[27]), "r"(values[28]), "r"(values[29]),
                 "r"(values[30]), "r"(values[31])
                 : "memory");
}

/* Writes @p value to cluster address @p address, in any block of the cluster. */
NUMKONG_DEVICE void nk_store_shared_cluster_f32_blackwell_(nk_u32_t address, nk_f32_t value) {
    asm volatile("st.shared::cluster.f32 [%0], %1;\n" ::"r"(address), "f"(value) : "memory");
}

/* Arrives on the barrier at cluster address @p barrier, releasing this thread's writes to the
 *  cluster. */
NUMKONG_DEVICE void nk_mbarrier_arrive_release_cluster_blackwell_(nk_u32_t barrier) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" ::"r"(barrier) : "memory");
}

/* Spins until the phase of parity @p parity completes, acquiring what the cluster released. */
NUMKONG_DEVICE void nk_mbarrier_wait_acquire_cluster_blackwell_(nk_u32_t barrier, nk_u32_t parity) {
    nk_u32_t done;
    do {
        asm volatile("{\n.reg .pred ready;\n"                                                      //
                     "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [%1], %2;\n" //
                     "selp.u32 %0, 1, 0, ready;\n}\n"
                     : "=r"(done)
                     : "r"(barrier), "r"(parity)
                     : "memory");
    } while (!done);
}

#pragma endregion Instructions

#pragma region Fragments

/** Two F32 values as the lanes of one 64-bit register, @p low first. */
NUMKONG_DEVICE nk_u64_t nk_f32x2_blackwell_(nk_f32_t low, nk_f32_t high) {
    return ((nk_u64_t)__float_as_uint(high) << 32) | __float_as_uint(low);
}

NUMKONG_DEVICE nk_f32_t nk_f32x2_low_blackwell_(nk_u64_t pair) { return __uint_as_float((nk_u32_t)pair); }
NUMKONG_DEVICE nk_f32_t nk_f32x2_high_blackwell_(nk_u64_t pair) { return __uint_as_float((nk_u32_t)(pair >> 32)); }

/** BF16 weights, summed as rounded, as the Ampere tile sums them. */
NUMKONG_DEVICE void nk_attention_weights_bf16_blackwell_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                         nk_u64_t *sums) {
    packed[0] = nk_f32x2_to_bf16x2_ampere_(probabilities[0], probabilities[1]);
    packed[1] = nk_f32x2_to_bf16x2_ampere_(probabilities[2], probabilities[3]);
#pragma unroll
    for (unsigned word = 0; word < 2; ++word)
        *sums = nk_f32x2_add_blackwell_(*sums, ((nk_u64_t)(packed[word] & 0xFFFF0000u) << 32) | (packed[word] << 16));
}

/** F16 weights, summed as rounded. */
NUMKONG_DEVICE void nk_attention_weights_f16_blackwell_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                        nk_u64_t *sums) {
    packed[0] = nk_f32x2_to_f16x2_ampere_(probabilities[0], probabilities[1]);
    packed[1] = nk_f32x2_to_f16x2_ampere_(probabilities[2], probabilities[3]);
#pragma unroll
    for (unsigned word = 0; word < 2; ++word)
        *sums = nk_f32x2_add_blackwell_(
            *sums, nk_f32x2_blackwell_(__half2float(__ushort_as_half((unsigned short)(packed[word] & 0xFFFFu))),
                                       __half2float(__ushort_as_half((unsigned short)(packed[word] >> 16)))));
}

/** Four F32 values as E4M3 codes of one word, the first in the low byte, rounding to nearest and
 *  saturating, with no permute to join the pairs. */
NUMKONG_DEVICE nk_u32_t nk_f32x4_to_e4m3x4_blackwell_(nk_f32_t const values[4]) {
    nk_u32_t codes;
    asm("{\n.reg .b16 low, high;\n"                   //
        "cvt.rn.satfinite.e4m3x2.f32 low, %2, %1;\n"  //
        "cvt.rn.satfinite.e4m3x2.f32 high, %4, %3;\n" //
        "mov.b32 %0, {low, high};\n}\n"
        : "=r"(codes)
        : "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]));
    return codes;
}

/** Adds the two F16 halves of @p halves to the two F32 lanes of @p sums, widening as they add. */
NUMKONG_DEVICE nk_u64_t nk_f32x2_add_f16x2_blackwell_(nk_u64_t sums, nk_u32_t halves) {
    nk_f32_t low = nk_f32x2_low_blackwell_(sums), high = nk_f32x2_high_blackwell_(sums);
    asm("{\n.reg .b16 low, high;\n"     //
        "mov.b32 {low, high}, %2;\n"    //
        "add.rn.f32.f16 %0, low, %0;\n" //
        "add.rn.f32.f16 %1, high, %1;\n}\n"
        : "+f"(low), "+f"(high)
        : "r"(halves));
    return nk_f32x2_blackwell_(low, high);
}

/** E4M3 weights e4m3(256 · p) of probabilities the softmax already amplified by 256, summed as
 *  rounded, which the output's division by the sum undoes. */
NUMKONG_DEVICE void nk_attention_weights_e4m3_blackwell_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                         nk_u64_t *sums) {
    nk_u32_t const codes = nk_f32x4_to_e4m3x4_blackwell_(probabilities);
    packed[0] = codes, packed[1] = 0;
    *sums = nk_f32x2_add_f16x2_blackwell_(*sums, nk_e4m3x2_to_f16x2_ada_((unsigned short)(codes & 0xFFFFu)));
    *sums = nk_f32x2_add_f16x2_blackwell_(*sums, nk_e4m3x2_to_f16x2_ada_((unsigned short)(codes >> 16)));
}

/** U8 weights round(255 · p) as F16 pairs of integers, the max-scoring position landing on 255,
 *  which @c kind::f16 multiplies exactly against the widened V. Adding 2²³ rounding down
 *  truncates the scaled probabilities. */
NUMKONG_DEVICE void nk_attention_weights_u8_blackwell_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                       nk_u64_t *sums) {
    nk_u64_t const scale = nk_f32x2_blackwell_(255.0f, 255.0f), half = nk_f32x2_blackwell_(0.5f, 0.5f);
    nk_u64_t const shift = nk_f32x2_blackwell_(8388608.0f, 8388608.0f);
    nk_u64_t const unshift = nk_f32x2_blackwell_(-8388608.0f, -8388608.0f);
#pragma unroll
    for (unsigned word = 0; word < 2; ++word) {
        nk_u64_t const scaled = nk_f32x2_fma_blackwell_(
            nk_f32x2_blackwell_(probabilities[2 * word], probabilities[2 * word + 1]), scale, half);
        nk_u64_t const weights = nk_f32x2_add_blackwell_(nk_f32x2_add_down_blackwell_(scaled, shift), unshift);
        packed[word] = nk_f32x2_to_f16x2_ampere_(nk_f32x2_low_blackwell_(weights), nk_f32x2_high_blackwell_(weights));
        *sums = nk_f32x2_add_blackwell_(*sums, weights);
    }
}

/** Waits for the phase of @p barrier with parity @p parity. */
NUMKONG_DEVICE void nk_attention_wait_blackwell_(nk_u64_t *barrier, nk_u32_t parity) {
    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(barrier), parity);
}

/** Arrives on @p barrier once for the calling warp, after every lane gets here. */
NUMKONG_DEVICE void nk_attention_warp_arrive_blackwell_(nk_u64_t *barrier) {
    __syncwarp();
    if ((threadIdx.x & 31) == 0) nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(barrier));
}

/** Arrives on @p barrier once every product issued so far completes. */
NUMKONG_DEVICE void nk_attention_commit_blackwell_(nk_u64_t *barrier) {
    nk_mma_commit_blackwell_(nk_shared_address_ampere_(barrier));
}

NUMKONG_DEVICE void nk_attention_advance_blackwell_(nk_attention_schedule_blackwell_t const *schedule, nk_u32_t *stage,
                                                    nk_u32_t *phase) {
    if (++*stage == schedule->ring_stages) *stage = 0, *phase ^= 1;
}

#pragma endregion Fragments

#pragma region Schedule

/** Finds work item @p item of @p rows rows past the cursor of @p tasks, moving it to the item's
 *  segment, 32 segments at a time with every lane of the calling warp. Returns 0 once the window
 *  has no more, which items asked for in increasing order reach only at the end. */
NUMKONG_DEVICE int nk_attention_next_work_blackwell_(nk_attention_schedule_t *tasks, nk_size_t rows, nk_size_t item,
                                                     nk_attention_work_t *work) {
    unsigned const lane = threadIdx.x & 31;
    while (tasks->chunk_first < tasks->segment_end) {
        nk_u64_t const items = nk_attention_segment_items_simt_(tasks, tasks->chunk_first + lane, rows);
        nk_u64_t inclusive = items;
#pragma unroll
        for (unsigned offset = 1; offset < 32; offset <<= 1) {
            nk_u64_t const other = nk_shuffle_up_u64_cuda_(inclusive, offset);
            if (lane >= offset) inclusive += other;
        }
        nk_u64_t const total = __shfl_sync(0xFFFFFFFFu, inclusive, 31);
        if (item < tasks->items_before + total) {
            unsigned const found = __ffs(__ballot_sync(0xFFFFFFFFu, tasks->items_before + inclusive > item)) - 1;
            tasks->items_before += __shfl_sync(0xFFFFFFFFu, inclusive - items, found);
            tasks->chunk_first += found;
            nk_attention_segment_work_simt_(tasks, tasks->chunk_first, item - tasks->items_before, rows, work);
            return 1;
        }
        tasks->items_before += total, tasks->chunk_first += 32;
    }
    return 0;
}

/** Everything every role derives from @p work: its tiles, its panels and its planes. */
NUMKONG_DEVICE void nk_attention_item_blackwell_(nk_attention_arguments_t const *arguments,
                                                 nk_attention_schedule_blackwell_t const *schedule,
                                                 nk_attention_work_t const *work, nk_attention_item_blackwell_t *item) {
    nk_size_t const segments = ((nk_attention_packed_header_t const *)arguments->packed)->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_simt_(arguments->packed);
    nk_size_t const length = nk_attention_packed_key_lengths_simt_(arguments->packed, segments)[work->segment];
    item->work = *work;
    item->length = length;
    item->positions_padded = nk_size_round_up_to_multiple_(length, nk_attention_box_rows_blackwell_k);
    item->keys_row = schedule->payload_row + payload_offsets[work->segment] / schedule->row_bytes +
                     work->key_value_head * item->positions_padded;
    item->values_row = item->keys_row + arguments->key_value_head_count * item->positions_padded;
    nk_size_t const tiles = nk_size_divide_round_up_(work->row_count, nk_attention_tile_rows_blackwell_k);
    item->tiles = tiles < schedule->tiles ? (unsigned)tiles : schedule->tiles;
    nk_size_t begin, end;
    nk_attention_rows_keys_simt_(schedule->band, nk_attention_row_position_simt_(work, 0, length),
                                 nk_attention_row_position_simt_(work, work->row_count - 1, length), length, &begin,
                                 &end);
    item->panel_first = (unsigned)(begin / nk_attention_panel_blackwell_k);
    item->panels = begin < end
                       ? (unsigned)nk_size_divide_round_up_(end, nk_attention_panel_blackwell_k) - item->panel_first
                       : 0;
    item->shared = schedule->paired && item->panels >= 2;
    if (schedule->paired && !item->shared && schedule->pair_rank) item->panels = 0;
    if (!item->shared) return;
    unsigned const leading = nk_u32_divide_round_up_(item->panels, 2);
    item->panel_first += schedule->pair_rank * leading;
    item->panels = schedule->pair_rank ? item->panels - leading : leading;
}

#pragma endregion Schedule

#pragma region Tile

NUMKONG_DEVICE nk_attention_schedule_blackwell_t nk_attention_schedule_blackwell_(
    nk_attention_kernel_blackwell_t const *kernel, nk_attention_tile_arguments_blackwell_t const *tile_arguments) {
    nk_attention_arguments_t const *const arguments = &tile_arguments->attention;
    nk_attention_schedule_blackwell_t schedule;
    unsigned const operand_bytes = nk_attention_operand_bytes_blackwell_(kernel->element_bytes, kernel->widens);
    schedule.widens = kernel->widens;
    schedule.tiles = nk_attention_tiles_blackwell_(kernel->width);
    schedule.item_rows = schedule.tiles * nk_attention_tile_rows_blackwell_k;
    schedule.element_bytes = kernel->element_bytes;
    schedule.row_bytes = (unsigned)nk_size_round_up_to_multiple_(arguments->depth * operand_bytes,
                                                                 nk_attention_step_bytes_k);
    schedule.box_bytes = 128;
    schedule.boxes = nk_u32_divide_round_up_(schedule.row_bytes, schedule.box_bytes);
    schedule.staged_bytes = schedule.row_bytes;
    schedule.depth_steps = schedule.staged_bytes / 32;
    schedule.value_columns = schedule.staged_bytes / operand_bytes;
    schedule.value_steps = nk_attention_panel_blackwell_k * operand_bytes / 32;
    schedule.value_step_bytes = 32 / operand_bytes * 128;
    schedule.queries_bytes = nk_attention_tile_rows_blackwell_k *
                             nk_attention_staged_bytes_blackwell_(kernel->element_bytes, kernel->widens, kernel->width);
    schedule.stage_bytes = nk_attention_stage_bytes_blackwell_(kernel->element_bytes, kernel->widens, kernel->width);
    schedule.ring_stages = nk_attention_ring_stages_blackwell_(kernel->element_bytes, kernel->widens, kernel->width);
    schedule.scores_instruction = nk_mma_instruction_blackwell_(
        kernel->format, kernel->format, nk_attention_tile_rows_blackwell_k, nk_attention_panel_blackwell_k,
        nk_major_k_k, nk_major_k_k);
    schedule.values_instruction = nk_mma_instruction_blackwell_(kernel->format, kernel->format,
                                                                nk_attention_tile_rows_blackwell_k,
                                                                schedule.value_columns, nk_major_k_k, nk_major_mn_k);
    schedule.scale2 = arguments->scale2;
    schedule.rescale_threshold = schedule.element_bytes == 2 ? 8.0f : 0.0f;
    nk_size_t const segments = ((nk_attention_packed_header_t const *)arguments->packed)->segments;
    schedule.payload_row = nk_attention_payload_offset_simt_(segments, schedule.row_bytes) / schedule.row_bytes;
    schedule.band = nk_attention_kernel_band_simt_(kernel->mask, arguments);
    schedule.queries_mapped = tile_arguments->query_tokens != 0;
    schedule.paired = tile_arguments->paired;
    schedule.pair_rank = schedule.paired ? nk_cluster_rank_blackwell_() : 0;
    return schedule;
}

/** Initializes the barriers and allocates every tensor-memory column, returning their address.
 *  Every thread of the block calls it. */
NUMKONG_DEVICE nk_u32_t nk_attention_initialize_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                           nk_attention_control_blackwell_t *control) {
    if (threadIdx.x == 0) {
        // Every warp but the load warp takes each work item.
        unsigned const consumers = nk_attention_load_warp_blackwell_k;
        for (unsigned stage = 0; stage < schedule->ring_stages; ++stage) {
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->loaded[stage]), 1);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->consumed[stage]), 1);
        }
        for (unsigned index = 0; index < 2; ++index) {
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->scheduled[index]), 1);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->taken[index]), consumers);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->queries_staged[index]), 4);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->queries_read[index]), 1);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->corrected[index]), 4);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->accumulated[index]), 1);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->summed[index]), 4);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->sums_read[index]), 4);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->exchanged[0][index]),
                                        nk_attention_tile_rows_blackwell_k);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->exchanged[1][index]),
                                        nk_attention_tile_rows_blackwell_k);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->merged[index]),
                                        nk_attention_tile_rows_blackwell_k);
        }
        for (unsigned slot = 0; slot < 3; ++slot) {
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->scored[slot]), 1);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->weighed_half[slot]), 4);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->weighed[slot]), 4);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->rescaling[slot]), 4);
        }
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->multiplied), 1);
        nk_mbarrier_init_fence_blackwell_();
    }
    if (threadIdx.x >> 5 == nk_attention_multiply_warp_blackwell_k)
        nk_tmem_alloc_blackwell_(nk_shared_address_ampere_(&control->tensor_memory),
                                 nk_attention_tensor_columns_blackwell_k);
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    // The pair arrives on each other's barriers only once both are initialized.
    if (schedule->paired) nk_cluster_sync_blackwell_();
    nk_tmem_fence_after_blackwell_();
    return control->tensor_memory;
}

/** Frees the tensor memory once every thread is done with it. */
NUMKONG_DEVICE void nk_attention_release_blackwell_(nk_u32_t tensor_memory) {
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    if (threadIdx.x >> 5 != nk_attention_multiply_warp_blackwell_k) return;
    nk_tmem_fence_after_blackwell_();
    nk_tmem_dealloc_blackwell_(tensor_memory, nk_attention_tensor_columns_blackwell_k);
}

/** Waits for the work item of slot @p sequence mod 2 and copies it, freeing the slot for the
 *  calling warp. */
NUMKONG_DEVICE void nk_attention_take_work_blackwell_(nk_attention_control_blackwell_t *control, nk_u32_t sequence,
                                                      nk_attention_work_t *work) {
    nk_attention_wait_blackwell_(&control->scheduled[sequence & 1], (sequence >> 1) & 1);
    *work = control->works[sequence & 1];
    nk_attention_warp_arrive_blackwell_(&control->taken[sequence & 1]);
}

/** Copies one panel of the plane from map row @p plane_row into the next stage, half a panel per
 *  copy, the second half rereading the plane's last 64 positions when the plane ends before it. */
NUMKONG_DEVICE void nk_attention_load_panel_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                       nk_attention_control_blackwell_t *control,
                                                       nk_u32_t stages_address, void const *map,
                                                       nk_attention_item_blackwell_t const *item, nk_size_t plane_row,
                                                       nk_size_t position, nk_u32_t *stage, nk_u32_t *phase) {
    nk_u32_t const loaded = nk_shared_address_ampere_(&control->loaded[*stage]);
    nk_u32_t const destination = stages_address + *stage * schedule->stage_bytes;
    nk_attention_wait_blackwell_(&control->consumed[*stage], *phase ^ 1);
    nk_mbarrier_expect_bytes_blackwell_(loaded, schedule->boxes * nk_attention_panel_blackwell_k * schedule->box_bytes);
    for (unsigned box = 0; box < schedule->boxes; ++box)
        for (unsigned half = 0; half < 2; ++half) {
            nk_size_t const first = position + half * nk_attention_box_rows_blackwell_k;
            nk_size_t const row = first < item->positions_padded
                                      ? first
                                      : item->positions_padded - nk_attention_box_rows_blackwell_k;
            nk_load_box_blackwell_(
                destination + (box * nk_attention_panel_blackwell_k + half * nk_attention_box_rows_blackwell_k) *
                                  schedule->box_bytes,
                map, loaded, (nk_i32_t)(box * schedule->box_bytes), (nk_i32_t)(plane_row + row));
        }
    nk_attention_advance_blackwell_(schedule, stage, phase);
}

/** Whether an item of @p tiles tiles alternates S between both tiles' columns, so the next S runs
 *  beside the softmax, given a ring deep enough that the next K needn't wait on this stage. */
NUMKONG_DEVICE int nk_attention_double_scores_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                         unsigned tiles) {
    return tiles == 1 && schedule->ring_stages > 2;
}

/** Whether the load warp copies the Q tiles of @p work, whose rows then start on a whole token. */
NUMKONG_DEVICE int nk_attention_queries_mapped_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                          nk_attention_work_t const *work) {
    return schedule->queries_mapped && work->row_first % work->heads_selected == 0;
}

/** Copies the Q tiles of @p item, once the products are done with the previous ones, as boxes of
 *  whole tokens that land in the swizzle the softmax warps stage, arriving for those warps. Counts
 *  every item with panels in @p items, as they do. */
NUMKONG_DEVICE void nk_attention_load_queries_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                         nk_attention_control_blackwell_t *control,
                                                         nk_u32_t queries_address, void const *map,
                                                         nk_attention_item_blackwell_t const *item, nk_u32_t items[2]) {
    nk_attention_work_t const *work = &item->work;
    int const mapped = nk_attention_queries_mapped_blackwell_(schedule, work);
    for (unsigned tile = 0; tile < item->tiles; ++tile) {
        nk_u32_t const parity = (items[tile]++ & 1) ^ 1;
        if (!mapped) continue;
        nk_attention_wait_blackwell_(&control->queries_read[tile], parity);
        nk_u32_t const staged = nk_shared_address_ampere_(&control->queries_staged[tile]);
        nk_size_t const token = work->query_first +
                                (work->row_first + tile * nk_attention_tile_rows_blackwell_k) / work->heads_selected;
        nk_mbarrier_expect_bytes_blackwell_(staged, schedule->boxes * nk_attention_block_bytes_blackwell_k);
        for (unsigned box = 0; box < schedule->boxes; ++box)
            nk_load_box_3d_blackwell_(
                queries_address + tile * schedule->queries_bytes + box * nk_attention_block_bytes_blackwell_k, map,
                staged, (nk_i32_t)(box * 128), (nk_i32_t)work->head_first, (nk_i32_t)token);
        for (unsigned warp = 1; warp < 4; ++warp) nk_mbarrier_arrive_blackwell_(staged);
    }
}

/** Finds every item of the block, publishing each to the other roles through the two work slots and
 *  streaming its panels of K and V into the ring, K before V, from the load warp's first lane, and
 *  its Q tiles when mapped from the second. Ends with an item of no rows. */
NUMKONG_DEVICE void nk_attention_load_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                 nk_attention_control_blackwell_t *control, unsigned char *queries,
                                                 unsigned char *stages,
                                                 nk_attention_tile_arguments_blackwell_t const *arguments) {
    unsigned const lane = threadIdx.x & 31;
    void const *map = &arguments->map;
    nk_u32_t const queries_address = nk_shared_address_ampere_(queries);
    nk_u32_t const stages_address = nk_shared_address_ampere_(stages);
    nk_attention_schedule_t tasks;
    int const started = nk_attention_schedule_init_simt_(&arguments->attention, &tasks);
    if (lane == 0) nk_prefetch_map_blackwell_(map);
    if (lane == 1 && schedule->queries_mapped) nk_prefetch_map_blackwell_(&arguments->queries_map);
    nk_u32_t stage = 0, phase = 0, sequence = 0, query_items[2] = {0, 0};
    unsigned const item_blocks = schedule->paired ? 2 : 1;
    for (nk_size_t index = blockIdx.x / item_blocks;; index += gridDim.x / item_blocks, ++sequence) {
        nk_attention_work_t work;
        int const found = started && nk_attention_next_work_blackwell_(&tasks, schedule->item_rows, index, &work);
        if (!found) work.row_count = 0;
        if (lane == 0) {
            nk_attention_wait_blackwell_(&control->taken[sequence & 1], ((sequence >> 1) & 1) ^ 1);
            control->works[sequence & 1] = work;
            nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->scheduled[sequence & 1]));
        }
        if (!found) return;
        nk_attention_item_blackwell_t item;
        nk_attention_item_blackwell_(&arguments->attention, schedule, &work, &item);
        if (lane == 1 && item.panels)
            nk_attention_load_queries_blackwell_(schedule, control, queries_address, &arguments->queries_map, &item,
                                                 query_items);
        for (unsigned panel = 0; lane == 0 && panel < item.panels; ++panel) {
            nk_size_t const position = (nk_size_t)(item.panel_first + panel) * nk_attention_panel_blackwell_k;
            nk_attention_load_panel_blackwell_(schedule, control, stages_address, map, &item, item.keys_row, position,
                                               &stage, &phase);
            nk_attention_load_panel_blackwell_(schedule, control, stages_address, map, &item, item.values_row, position,
                                               &stage, &phase);
        }
        __syncwarp();
    }
}

/** Per-tile counts a role keeps across items, which pick the parities its barriers wait on. */
typedef struct {

    /** The ring position of the next stage. */
    nk_u32_t stage, phase;

    /** Items each tile took part in, panels it went through, and P each S slot held. */
    nk_u32_t items[2], panels[2], weighings[3];
} nk_attention_progress_blackwell_t;

/** Waits for the next stage to reach the products, returning its shared address. */
NUMKONG_DEVICE nk_u32_t nk_attention_acquire_stage_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                              nk_attention_control_blackwell_t *control,
                                                              nk_u32_t stages_address,
                                                              nk_attention_progress_blackwell_t *progress) {
    nk_attention_wait_blackwell_(&control->loaded[progress->stage], progress->phase);
    nk_tmem_fence_after_blackwell_();
    nk_u32_t const address = stages_address + progress->stage * schedule->stage_bytes;
    nk_attention_advance_blackwell_(schedule, &progress->stage, &progress->phase);
    return address;
}

/** Frees the stage the copy counted @p use since the launch landed in, from the one thread that saw
 *  the products done with it. */
NUMKONG_DEVICE void nk_attention_release_stage_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                          nk_attention_control_blackwell_t *control, nk_u32_t use) {
    nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->consumed[use % schedule->ring_stages]));
}

/** The 16 bytes of staged Q at chunk @p chunk of a row from @p row, null past the item: codes as
 *  they are, or I8 codes widened to F16 values, zero past the row's @p row_bytes. Whole 16-byte or
 *  8-byte loads when @p aligned. */
NUMKONG_DEVICE uint4 nk_attention_query_chunk_blackwell_(unsigned char const *row, unsigned chunk, nk_size_t row_bytes,
                                                         int widen, int aligned) {
    unsigned const width = widen ? 8 : 16, first = chunk * width;
    // Zero I8 codes widen to F16 zeros, so absent ones skip the widening.
    if (!row || first >= row_bytes) return make_uint4(0, 0, 0, 0);
    if (aligned && !widen) return *(uint4 const *)(row + first);
    nk_u32_t codes[4] = {0, 0, 0, 0};
    if (aligned) {
        uint2 const loaded = *(uint2 const *)(row + first);
        codes[0] = loaded.x, codes[1] = loaded.y;
    }
    else
        for (unsigned byte = 0; byte < width && first + byte < row_bytes; ++byte)
            codes[byte / 4] |= (nk_u32_t)row[first + byte] << (8 * (byte % 4));
    if (!widen) return make_uint4(codes[0], codes[1], codes[2], codes[3]);
    nk_u32_t const subtrahend = (0x6400u | 128u) * 0x00010001u;
    nk_u32_t halves[4];
    nk_u8x4_to_f16x4_blackwell_(codes[0] ^ 0x80808080u, subtrahend, &halves[0], &halves[1]);
    nk_u8x4_to_f16x4_blackwell_(codes[1] ^ 0x80808080u, subtrahend, &halves[2], &halves[3]);
    return make_uint4(halves[0], halves[1], halves[2], halves[3]);
}

/** Stages the 128 query rows of @p tile into @p queries, 128-byte column blocks in the 128-byte
 *  swizzle, rows past the item and depths past the head zeroed, from the tile's softmax warps. */
NUMKONG_DEVICE void nk_attention_stage_queries_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                          nk_attention_arguments_t const *arguments,
                                                          nk_attention_work_t const *work, unsigned char *queries,
                                                          unsigned tile) {
    int const widen = schedule->widens;
    nk_size_t const row_bytes = arguments->depth * schedule->element_bytes;
    int const aligned = ((((nk_size_t)arguments->queries) | arguments->query_stride | row_bytes) & 15) == 0;
    unsigned const chunks = schedule->staged_bytes / 16;
    for (unsigned index = threadIdx.x % nk_attention_tile_rows_blackwell_k;
         index < nk_attention_tile_rows_blackwell_k * chunks; index += nk_attention_tile_rows_blackwell_k) {
        unsigned const row = index / chunks, chunk = index % chunks;
        nk_size_t const local = tile * nk_attention_tile_rows_blackwell_k + row;
        unsigned char const *source = local < work->row_count ? nk_attention_query_row_simt_(arguments, work, local,
                                                                                             schedule->element_bytes)
                                                              : NUMKONG_NULL;
        uint4 *const block_row = (uint4 *)(queries + chunk / 8 * nk_attention_block_bytes_blackwell_k + row * 128);
        block_row[(chunk ^ row) & 7] = nk_attention_query_chunk_blackwell_(source, chunk, row_bytes, widen, aligned);
    }
    nk_fence_async_shared_blackwell_();
}

/** The largest of this thread's 128 scores of a masked panel at tensor-memory address @p scores,
 *  scaled to base 2, over the keys set in @p visible, a word per 32. */
NUMKONG_DEVICE nk_f32_t nk_attention_masked_maximum_blackwell_(nk_u32_t scores, nk_f32_t scale2,
                                                               nk_u32_t const visible[4]) {
    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    nk_f32_t maxima[4] = {negative_infinity, negative_infinity, negative_infinity, negative_infinity};
#pragma unroll 1
    for (unsigned chunk = 0; chunk < nk_attention_panel_blackwell_k; chunk += 32) {
        nk_u32_t bits[32];
        nk_tmem_load_x32_blackwell_(scores + chunk, bits);
#pragma unroll
        for (unsigned offset = 0; offset < 32; ++offset) {
            nk_f32_t const score = (visible[chunk / 32] >> offset) & 1 ? __uint_as_float(bits[offset]) * scale2
                                                                       : negative_infinity;
            maxima[offset & 3] = fmaxf(maxima[offset & 3], score);
        }
    }
    return fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
}

/** Multiplies this thread's row of O at tensor-memory address @p output by @p factor, unless the
 *  factor of every row of the warp is 1. */
NUMKONG_DEVICE void nk_attention_rescale_blackwell_(nk_attention_schedule_blackwell_t const *schedule, nk_u32_t output,
                                                    nk_f32_t factor) {
    if (!__any_sync(0xFFFFFFFFu, factor != 1.0f)) return;
    nk_u64_t const factors = nk_f32x2_blackwell_(factor, factor);
#pragma unroll 1
    for (unsigned column = 0; column < schedule->value_columns; column += 32) {
        nk_u32_t bits[32];
        nk_tmem_load_x32_blackwell_(output + column, bits);
#pragma unroll
        for (unsigned offset = 0; offset < 32; offset += 2) {
            nk_u64_t const pair = nk_f32x2_mul_blackwell_(((nk_u64_t)bits[offset + 1] << 32) | bits[offset], factors);
            bits[offset] = (nk_u32_t)pair, bits[offset + 1] = (nk_u32_t)(pair >> 32);
        }
        nk_tmem_store_x32_blackwell_(output + column, bits);
    }
    nk_tmem_wait_store_blackwell_();
}

/** Stores this thread's row of O at tensor-memory address @p output times @p multiplier, or zeros
 *  and a −∞ log-sum-exp for an item whose rows see no keys, skipping rows past the item. With
 *  @p adds, adds the row to the one stored instead, leaving the log-sum-exp. */
NUMKONG_DEVICE void nk_attention_store_output_blackwell_(nk_attention_arguments_t const *arguments,
                                                         nk_attention_work_t const *work, nk_u32_t output,
                                                         unsigned tile, int seen, nk_f32_t multiplier, int adds) {
    nk_size_t const local = tile * nk_attention_tile_rows_blackwell_k +
                            threadIdx.x % nk_attention_tile_rows_blackwell_k;
    nk_size_t const depth = arguments->depth;
    int const inside = local < work->row_count;
    nk_f32_t *const log_sum_exp_slot = !seen && inside && !adds
                                           ? nk_attention_log_sum_exp_slot_simt_(arguments, work, local)
                                           : NUMKONG_NULL;
    if (log_sum_exp_slot) *log_sum_exp_slot = nk_attention_negative_infinity_simt_();
    nk_f32_t *const destination = inside ? nk_attention_output_row_simt_(arguments, work, local) : arguments->output;
    int const aligned = ((nk_size_t)destination & 15) == 0;
#pragma unroll 1
    for (unsigned column = 0; column < depth; column += 32) {
        nk_u32_t bits[32];
        if (seen) nk_tmem_load_x32_blackwell_(output + column, bits);
        else
#pragma unroll
            for (unsigned offset = 0; offset < 32; ++offset) bits[offset] = 0;
        if (!inside) continue;
#pragma unroll
        for (unsigned offset = 0; offset < 32; ++offset)
            bits[offset] = __float_as_uint(__uint_as_float(bits[offset]) * multiplier);
        if (adds)
#pragma unroll
            for (unsigned offset = 0; offset < 32; ++offset)
                if (column + offset < depth)
                    bits[offset] = __float_as_uint(__uint_as_float(bits[offset]) + destination[column + offset]);
        if (aligned && column + 32 <= depth) {
            uint4 *destination_quartets = (uint4 *)(destination + column);
#pragma unroll
            for (unsigned quartet = 0; quartet < 8; ++quartet)
                destination_quartets[quartet] = make_uint4(bits[4 * quartet], bits[4 * quartet + 1],
                                                           bits[4 * quartet + 2], bits[4 * quartet + 3]);
            continue;
        }
#pragma unroll
        for (unsigned offset = 0; offset < 32; ++offset)
            if (column + offset < depth) destination[column + offset] = __uint_as_float(bits[offset]);
    }
}

/** Stores this thread's row of O at tensor-memory address @p output as its block's share of a
 *  paired item, of the keys whose weights sum to @p row_sum and log-sum-exp is @p log_sum_exp:
 *  swaps log-sum-exps with the pair, then rank 0 stores its weighted row and the joint
 *  log-sum-exp, and rank 1 adds its own once rank 0's lands, so rows sum in rank order. @p merge
 *  counts the tile's paired items before. */
NUMKONG_DEVICE void nk_attention_merge_output_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                         nk_attention_control_blackwell_t *control,
                                                         nk_attention_arguments_t const *arguments,
                                                         nk_attention_work_t const *work, nk_u32_t output,
                                                         unsigned tile, int seen, nk_f32_t row_sum,
                                                         nk_f32_t log_sum_exp, nk_u32_t merge) {
    unsigned const tile_row = threadIdx.x % nk_attention_tile_rows_blackwell_k;
    nk_f32_t *const pair_slot = &control->pair_log_sum_exps[merge & 1][tile][tile_row];
    // Alternate merges take alternate barriers, so a block one merge ahead never completes two.
    nk_u32_t const exchanged = nk_shared_address_ampere_(&control->exchanged[merge & 1][tile]);
    nk_u32_t const merged = nk_shared_address_ampere_(&control->merged[tile]);
    nk_u32_t const pair_rank = schedule->pair_rank ^ 1;
    nk_store_shared_cluster_f32_blackwell_(
        nk_cluster_address_blackwell_(nk_shared_address_ampere_(pair_slot), pair_rank), log_sum_exp);
    nk_mbarrier_arrive_release_cluster_blackwell_(nk_cluster_address_blackwell_(exchanged, pair_rank));
    nk_mbarrier_wait_acquire_cluster_blackwell_(exchanged, (merge >> 1) & 1);
    nk_f32_t const pair_log_sum_exp = *pair_slot;
    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    nk_f32_t const maximum = fmaxf(log_sum_exp, pair_log_sum_exp);
    nk_f32_t const joint = maximum == negative_infinity
                               ? maximum
                               : maximum + logf(expf(log_sum_exp - maximum) + expf(pair_log_sum_exp - maximum));
    nk_f32_t const share = joint == negative_infinity ? 0.0f : expf(log_sum_exp - joint);
    nk_f32_t const multiplier = row_sum > 0 ? share / row_sum : 0.0f;
    if (schedule->pair_rank) {
        nk_mbarrier_wait_acquire_cluster_blackwell_(merged, merge & 1);
        if (seen) nk_attention_store_output_blackwell_(arguments, work, output, tile, 1, multiplier, 1);
        return;
    }
    nk_attention_store_output_blackwell_(arguments, work, output, tile, seen, multiplier, 0);
    nk_size_t const local = tile * nk_attention_tile_rows_blackwell_k + tile_row;
    nk_f32_t *const log_sum_exp_slot = local < work->row_count
                                           ? nk_attention_log_sum_exp_slot_simt_(arguments, work, local)
                                           : NUMKONG_NULL;
    if (log_sum_exp_slot) *log_sum_exp_slot = joint;
    nk_mbarrier_arrive_release_cluster_blackwell_(nk_cluster_address_blackwell_(merged, 1));
}

/** Rescales every tile's O as its maximum moves, between its products, then normalizes and stores
 *  it, one row per thread, from the correction warpgroup. */
NUMKONG_DEVICE void nk_attention_correct_blackwell_(nk_attention_schedule_blackwell_t const *schedule,
                                                    nk_attention_control_blackwell_t *control, nk_u32_t tensor_memory,
                                                    nk_attention_tile_arguments_blackwell_t const *arguments) {
    unsigned const tile_row = threadIdx.x % nk_attention_tile_rows_blackwell_k;
    nk_u32_t const outputs = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32,
                                                       nk_attention_output_column_blackwell_k);
    // The last tile's S of a panel follows every product reading the previous panel's V.
    int const releaser = tile_row == 0 && schedule->ring_stages > 1;
    nk_u32_t rescalings[3] = {0, 0, 0}, panels[2] = {0, 0}, items[2] = {0, 0}, merges[2] = {0, 0}, uses = 0,
             multiplications = 0;
    for (nk_u32_t sequence = 0;; ++sequence) {
        nk_attention_work_t work;
        nk_attention_take_work_blackwell_(control, sequence, &work);
        if (!work.row_count) return;
        nk_attention_item_blackwell_t item;
        nk_attention_item_blackwell_(&arguments->attention, schedule, &work, &item);
        nk_u32_t const first_use = uses;
        uses += 2 * item.panels;
        int const doubled = nk_attention_double_scores_blackwell_(schedule, item.tiles);
        if (doubled) {
            for (unsigned panel = 1; panel < item.panels; ++panel) {
                unsigned const slot = panel % 2 * 2;
                nk_attention_wait_blackwell_(&control->rescaling[slot], rescalings[slot]++ & 1);
                nk_attention_wait_blackwell_(&control->multiplied, multiplications++ & 1);
                nk_tmem_fence_after_blackwell_();
                if (releaser) nk_attention_release_stage_blackwell_(schedule, control, first_use + 2 * panel - 1);
                nk_attention_rescale_blackwell_(schedule, outputs, control->factors[slot][tile_row]);
                nk_tmem_fence_before_blackwell_();
                nk_attention_warp_arrive_blackwell_(&control->corrected[0]);
            }
            panels[0] += item.panels - item.panels / 2;
        }
        else
            for (unsigned panel = 1; panel < item.panels; ++panel)
                for (unsigned tile = 0; tile < item.tiles; ++tile) {
                    nk_u32_t const output = outputs + tile * nk_attention_tile_rows_blackwell_k;
                    nk_attention_wait_blackwell_(&control->rescaling[tile], rescalings[tile]++ & 1);
                    nk_attention_wait_blackwell_(&control->scored[tile], (panels[tile] + panel) & 1);
                    nk_tmem_fence_after_blackwell_();
                    if (releaser && tile + 1 == item.tiles)
                        nk_attention_release_stage_blackwell_(schedule, control, first_use + 2 * panel - 1);
                    nk_attention_rescale_blackwell_(schedule, output, control->factors[tile][tile_row]);
                    nk_tmem_fence_before_blackwell_();
                    nk_attention_warp_arrive_blackwell_(&control->corrected[tile]);
                }
        if (!doubled)
            for (unsigned tile = 0; tile < item.tiles; ++tile) panels[tile] += item.panels;
        for (unsigned tile = 0; tile < item.tiles; ++tile) {
            nk_u32_t const output = outputs + tile * nk_attention_tile_rows_blackwell_k;
            if (!item.panels && item.shared)
                nk_attention_merge_output_blackwell_(schedule, control, &arguments->attention, &work, output, tile, 0,
                                                     0, nk_attention_negative_infinity_simt_(), merges[tile]++);
            else if (!item.panels && !schedule->pair_rank)
                nk_attention_store_output_blackwell_(&arguments->attention, &work, output, tile, 0, 0, 0);
            if (!item.panels) continue;
            nk_attention_wait_blackwell_(&control->accumulated[tile], items[tile] & 1);
            nk_attention_wait_blackwell_(&control->summed[tile], items[tile]++ & 1);
            nk_f32_t const row_sum = control->sums[tile][tile_row];
            nk_f32_t const log_sum_exp = control->log_sum_exps[tile][tile_row];
            nk_attention_warp_arrive_blackwell_(&control->sums_read[tile]);
            nk_tmem_fence_after_blackwell_();
            if (releaser && tile + 1 == item.tiles)
                nk_attention_release_stage_blackwell_(schedule, control, first_use + 2 * item.panels - 1);
            if (item.shared)
                nk_attention_merge_output_blackwell_(schedule, control, &arguments->attention, &work, output, tile, 1,
                                                     row_sum, log_sum_exp, merges[tile]++);
            else
                nk_attention_store_output_blackwell_(&arguments->attention, &work, output, tile, 1,
                                                     row_sum > 0 ? 1.0f / row_sum : 0.0f, 0);
            nk_tmem_fence_before_blackwell_();
            nk_attention_warp_arrive_blackwell_(&control->corrected[tile]);
        }
    }
}

#pragma endregion Tile

#pragma region Backward

enum {
    // Keys of a block in the key pass and rows of a chunk in the query pass, one per lane.
    nk_attention_backward_rows_blackwell_k = 128,
    nk_attention_backward_depth_blackwell_k = 256,
    // Tensor-memory columns: S with P over its first half, dP with dS over its first half, then the
    // accumulators: 128-column slices of dK and dV in the key pass, or all of dQ in the query pass.
    nk_attention_backward_scores_column_blackwell_k = 0,
    nk_attention_backward_gradients_column_blackwell_k = 128,
    nk_attention_backward_first_column_blackwell_k = 256,
    nk_attention_backward_second_column_blackwell_k = 384,
    nk_attention_backward_slice_blackwell_k = 128,
    // Four warps own the tensor-memory lanes and issue the products; eight more stage the operands
    // the products stream through a ring.
    nk_attention_backward_compute_threads_blackwell_k = 128,
    nk_attention_backward_threads_blackwell_k = 384,
};

/** A backward block's state past its operands: the rows of each ring slot, the products' barrier,
 *  the barriers marking each slot full and empty, and the tensor-memory address. */
typedef struct {
    nk_attention_backward_rows_ampere_t rows[2];
    nk_u64_t done, full[2], empty[2], staged;
    nk_u32_t tensor_memory;
} nk_attention_backward_control_blackwell_t;

/** Everything a backward launch shares: the arguments and, when @c mapped, the tensor maps of Q and
 *  of the prepared dO as boxes of 128 / group tokens of a head group, and of the packed rows as
 *  boxes of 64 positions. When @c single, also the map of dQ as F32 boxes the single pass reduces
 *  into, and its workspace: dO as BF16 rows of @c head_count × depth per token, then D per row.
 *  Passed as a grid constant, so the maps keep an address. */
typedef struct {
    CUtensorMap queries_map;
    CUtensorMap gradients_map;
    CUtensorMap planes_map;
    CUtensorMap sums_map;
    nk_attention_backward_arguments_t arguments;
    int mapped;
    int single;
    unsigned char *prepared_gradients;
    nk_f32_t *prepared_dots;
} nk_attention_backward_maps_blackwell_t;

/** Rows or keys of the other side that each product spans: 128, or 64 past 128 dimensions, so
 *  all four operands fit in shared memory. */
NUMKONG_CONSTEXPR unsigned nk_attention_backward_inner_blackwell_(nk_size_t depth) { return depth <= 128 ? 128 : 64; }

/** Bytes of a staged operand row of @p depth BF16 values, whole 128-byte column blocks. */
NUMKONG_CONSTEXPR unsigned nk_attention_backward_staged_bytes_blackwell_(nk_size_t depth) {
    return (unsigned)nk_size_round_up_to_multiple_(depth * 2, 128);
}

/** Ring slots of the streamed operands: two while they fit beside the resident ones, else one. */
NUMKONG_CONSTEXPR unsigned nk_attention_backward_slots_blackwell_(nk_size_t depth) {
    return nk_attention_backward_staged_bytes_blackwell_(depth) <= 384 ? 2 : 1;
}

/** Dynamic shared memory of a backward block: the two resident operands of 128 rows, each ring
 *  slot's two streamed ones of the inner side, the state, and the slack that aligns all of them to
 *  a 1024-byte boundary. */
NUMKONG_CONSTEXPR nk_size_t nk_attention_backward_shared_bytes_blackwell_(nk_size_t depth) {
    return 1024 +
           2 *
               (nk_attention_backward_rows_blackwell_k +
                nk_attention_backward_slots_blackwell_(depth) * nk_attention_backward_inner_blackwell_(depth)) *
               nk_attention_backward_staged_bytes_blackwell_(depth) +
           sizeof(nk_attention_backward_control_blackwell_t);
}

/** Initializes the barriers and allocates every tensor-memory column from warp 0, returning their
 *  address. Every thread of the block calls it. */
NUMKONG_DEVICE nk_u32_t
nk_attention_backward_initialize_blackwell_(nk_attention_backward_control_blackwell_t *control) {
    if (threadIdx.x == 0) {
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->done), 1);
        for (unsigned slot = 0; slot < 2; ++slot) {
            nk_mbarrier_init_blackwell_(
                nk_shared_address_ampere_(&control->full[slot]),
                nk_attention_backward_threads_blackwell_k - nk_attention_backward_compute_threads_blackwell_k);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->empty[slot]), 1);
        }
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->staged), 1);
        nk_mbarrier_init_fence_blackwell_();
    }
    if (threadIdx.x >> 5 == 0)
        nk_tmem_alloc_blackwell_(nk_shared_address_ampere_(&control->tensor_memory),
                                 nk_attention_tensor_columns_blackwell_k);
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    nk_tmem_fence_after_blackwell_();
    return control->tensor_memory;
}

/** Frees the tensor memory from warp 0 once every thread is done with it. */
NUMKONG_DEVICE void nk_attention_backward_release_blackwell_(nk_u32_t tensor_memory) {
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    if (threadIdx.x >> 5 != 0) return;
    nk_tmem_fence_after_blackwell_();
    nk_tmem_dealloc_blackwell_(tensor_memory, nk_attention_tensor_columns_blackwell_k);
}

/** Issues, from @p thread of @p threads, the asynchronous copies of @p rows positions of a
 *  position-major plane from @p first_position into @p operand, in @p chunks 16-byte chunks per row
 *  of 128-byte column blocks in the 128-byte swizzle, zero past the plane's @p positions_padded and
 *  its @p row_bytes. */
NUMKONG_DEVICE void nk_attention_backward_copy_plane_blackwell_(unsigned char *operand, unsigned char const *plane,
                                                                nk_size_t first_position, nk_size_t positions_padded,
                                                                unsigned rows, unsigned row_bytes, unsigned chunks,
                                                                unsigned thread, unsigned threads) {
    for (unsigned index = thread; index < rows * chunks; index += threads) {
        unsigned const row = index / chunks, chunk = index % chunks;
        nk_size_t const position = first_position + row;
        int const inside = position < positions_padded && chunk * 16 < row_bytes;
        nk_copy_b128_async_ampere_(
            nk_shared_address_ampere_(operand + nk_swizzled_panel_offset_simt_(row, chunk * 16, 128, rows * 128)),
            inside ? plane + position * row_bytes + chunk * 16 : plane, inside ? 16 : 0);
    }
}

/** Writes each folded row's dO as BF16 over the start of its dQ row and D = dO · O over that row's
 *  last float, a warp per row, so both passes copy them as they copy Q. */
NUMKONG_DEVICE void nk_attention_backward_prepare_blackwell_(nk_attention_backward_arguments_t const *arguments) {
    unsigned const lane = threadIdx.x & 31;
    nk_size_t const depth = arguments->depth, group = arguments->head_count / arguments->key_value_head_count;
    nk_size_t const warp = (blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x) >> 5;
    nk_size_t const warps = (gridDim.x * (nk_size_t)blockDim.x) >> 5;
    nk_size_t segment_first = arguments->tasks_begin / arguments->key_value_head_count, items_before = 0, task, folded;
    for (nk_size_t item = warp;
         nk_attention_backward_next_cuda_(arguments, 0, 1, &segment_first, &items_before, item, &task, &folded);
         item += warps) {
        nk_size_t const segment = task / arguments->key_value_head_count;
        nk_size_t const key_value_head = task % arguments->key_value_head_count;
        nk_size_t const token = arguments->query_offsets[segment] + folded / group;
        nk_size_t const head = key_value_head * group + folded % group, offset = token * arguments->output_stride;
        nk_f32_t const *output_row = (nk_f32_t const *)((unsigned char const *)arguments->output + offset) +
                                     head * depth;
        nk_f32_t const *gradient_row = (nk_f32_t const *)((unsigned char const *)arguments->output_gradient + offset) +
                                       head * depth;
        nk_f32_t *prepared_row = (nk_f32_t *)((unsigned char *)arguments->query_gradient +
                                              token * arguments->query_gradient_stride) +
                                 head * depth;
        nk_f32_t dot = 0;
        nk_u32_t *prepared_pairs = (nk_u32_t *)prepared_row;
        for (nk_size_t element = lane * 2; element < depth; element += 64) {
            nk_f32_t const low = gradient_row[element], high = gradient_row[element + 1];
            dot = fmaf(high, output_row[element + 1], fmaf(low, output_row[element], dot));
            prepared_pairs[element / 2] = nk_f32x2_to_bf16x2_ampere_(low, high);
        }
        for (unsigned offset = 16; offset != 0; offset >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, offset);
        if (lane == 0) prepared_row[depth - 1] = dot;
    }
}

/** Issues, from @p thread of @p threads, the asynchronous copies of @p rows folded rows of @p task
 *  from the one @p remainder heads into @p query_first, Q from the queries and dO from the prepared
 *  dQ rows unless @p copies is zero, in @p chunks 16-byte chunks per row of 128-byte column blocks
 *  of @p block_bytes, zero past the head and the task, and writes each row's state. */
NUMKONG_DEVICE void nk_attention_backward_copy_rows_blackwell_(
    nk_attention_backward_arguments_t const *arguments, nk_attention_backward_task_t const *task, nk_size_t query_first,
    unsigned remainder, unsigned rows, unsigned chunks, unsigned block_bytes, unsigned char *queries_shared,
    unsigned char *gradients_shared, nk_attention_backward_rows_ampere_t *state, int copies, unsigned thread,
    unsigned threads) {
    nk_size_t const depth = arguments->depth;
    unsigned const group = (unsigned)(arguments->head_count / arguments->key_value_head_count);
    nk_size_t const segment_first = arguments->query_offsets[task->segment];
    unsigned const row_chunks = (unsigned)(depth * 2 / 16);
    // A thread keeps one chunk column across rows, and rows divide by the group in 32 bits.
    unsigned const row_step = threads / chunks, chunk = thread % chunks;
    for (unsigned local = thread / chunks; copies && local < rows && thread < row_step * chunks; local += row_step) {
        unsigned const offset = remainder + local;
        nk_size_t const query = query_first + offset / group;
        nk_size_t const token = segment_first + query, head = task->key_value_head * group + offset % group;
        int const inside = query < task->rows && chunk < row_chunks;
        unsigned char const *query_row = arguments->queries + token * arguments->query_stride + head * depth * 2;
        unsigned char const *gradient_row = (unsigned char const *)arguments->query_gradient +
                                            token * arguments->query_gradient_stride + head * depth * 4;
        unsigned const destination = nk_swizzled_panel_offset_simt_(local, chunk * 16, 128, block_bytes);
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(queries_shared + destination),
                                   inside ? query_row + chunk * 16 : arguments->queries, inside ? 16 : 0);
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(gradients_shared + destination),
                                   inside ? gradient_row + chunk * 16 : arguments->queries, inside ? 16 : 0);
    }
    for (unsigned local = thread; local < rows; local += threads) {
        unsigned const offset = remainder + local;
        nk_size_t const query = query_first + offset / group;
        nk_size_t const token = segment_first + query, head = task->key_value_head * group + offset % group;
        int const live = query < task->rows;
        nk_f32_t const *prepared_row = (nk_f32_t const *)((unsigned char const *)arguments->query_gradient +
                                                          token * arguments->query_gradient_stride) +
                                       head * depth;
        nk_size_t key_begin = 0, key_end = 0;
        if (live)
            nk_diagonal_band_row_range_simt_(arguments->band, task->first_band_row + (nk_i64_t)query, task->length,
                                             &key_begin, &key_end);
        state->log_sum_exp2[local] =
            live ? arguments->log_sum_exp[token * arguments->head_count + head] * NUMKONG_F32_LOG2E_ : 0.0f;
        state->dot[local] = live ? prepared_row[depth - 1] : 0.0f;
        state->key_begin[local] = (unsigned)key_begin, state->key_end[local] = (unsigned)key_end;
    }
}

/** Publishes the operands the compute warps staged and their tensor-memory writes to the thread
 *  issuing the products, past the compute warps alone. */
NUMKONG_DEVICE void nk_attention_backward_publish_blackwell_(void) {
    nk_fence_async_shared_blackwell_();
    nk_tmem_fence_before_blackwell_();
    nk_barrier_sync_blackwell_(1, nk_attention_backward_compute_threads_blackwell_k);
    nk_tmem_fence_after_blackwell_();
}

/** Waits for the products issued last, flipping @p phase. */
NUMKONG_DEVICE void nk_attention_backward_wait_blackwell_(nk_attention_backward_control_blackwell_t *control,
                                                          nk_u32_t *phase) {
    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->done), *phase);
    *phase ^= 1;
    nk_tmem_fence_after_blackwell_();
}

/** Issues S = A · Bᵀ into the S columns and dP = C · Dᵀ into the dP ones over @p depth_steps
 *  32-byte steps, from warp 0, then signals the barrier: K-major operands, A and C of 128 rows and
 *  B and D of @p inner. */
NUMKONG_DEVICE void nk_attention_backward_scores_blackwell_(nk_attention_backward_control_blackwell_t *control,
                                                            nk_u32_t tensor_memory, nk_u32_t scores_a,
                                                            nk_u32_t scores_b, nk_u32_t gradients_a,
                                                            nk_u32_t gradients_b, unsigned depth_steps, unsigned inner,
                                                            nk_u32_t instruction) {
    if (threadIdx.x >> 5 != 0 || !nk_elect_one_blackwell_()) return;
    // K-major rows, whose leading offset goes unread, in 8-row groups 1024 bytes apart.
    nk_u64_t const descriptors[4] = {
        nk_smem_descriptor_blackwell_(scores_a, nk_smem_swizzle_128_blackwell_k, 16, 1024),
        nk_smem_descriptor_blackwell_(scores_b, nk_smem_swizzle_128_blackwell_k, 16, 1024),
        nk_smem_descriptor_blackwell_(gradients_a, nk_smem_swizzle_128_blackwell_k, 16, 1024),
        nk_smem_descriptor_blackwell_(gradients_b, nk_smem_swizzle_128_blackwell_k, 16, 1024)};
    for (unsigned step = 0; step < depth_steps; ++step) {
        // Each 32-byte step starts where row 0 holds it: a block further on every fourth step.
        nk_u32_t const offset = step / 4 * nk_attention_block_bytes_blackwell_k + step % 4 * 32;
        nk_u32_t const inner_offset = step / 4 * inner * 128 + step % 4 * 32;
        nk_mma_f16_blackwell_(tensor_memory + nk_attention_backward_scores_column_blackwell_k,
                              descriptors[0] + offset / 16, descriptors[1] + inner_offset / 16, instruction, step != 0);
        nk_mma_f16_blackwell_(tensor_memory + nk_attention_backward_gradients_column_blackwell_k,
                              descriptors[2] + offset / 16, descriptors[3] + inner_offset / 16, instruction, step != 0);
    }
    nk_mma_commit_blackwell_(nk_shared_address_ampere_(&control->done));
}

/** Issues the first accumulator's dS · B, and with @p weighs the second's P · C, over @p inner
 *  positions of tensor-memory A and MN-major B and C of @p inner rows, starting afresh unless
 *  @p accumulate, from warp 0, then signals the barrier and @p release, the slot's empty one. */
NUMKONG_DEVICE void nk_attention_backward_values_blackwell_(nk_attention_backward_control_blackwell_t *control,
                                                            nk_u32_t tensor_memory, nk_u32_t gradients_b,
                                                            nk_u32_t weights_b, int weighs, unsigned inner,
                                                            nk_u32_t instruction, int accumulate, nk_u64_t *release) {
    if (threadIdx.x >> 5 != 0 || !nk_elect_one_blackwell_()) return;
    // MN-major: 128 bytes of N per row, the next 128 a block on, 8-row groups 1024 bytes apart.
    nk_u64_t const gradients = nk_smem_descriptor_blackwell_(gradients_b, nk_smem_swizzle_128_blackwell_k, inner * 128,
                                                             1024);
    nk_u64_t const weights = nk_smem_descriptor_blackwell_(weights_b, nk_smem_swizzle_128_blackwell_k, inner * 128,
                                                           1024);
    for (unsigned step = 0; step < inner / 16; ++step) {
        nk_u32_t const accumulates = accumulate || step != 0;
        nk_mma_f16_tmem_blackwell_(tensor_memory + nk_attention_backward_first_column_blackwell_k,
                                   tensor_memory + nk_attention_backward_gradients_column_blackwell_k + 8 * step,
                                   gradients + step * 2048 / 16, instruction, accumulates);
        if (weighs)
            nk_mma_f16_tmem_blackwell_(tensor_memory + nk_attention_backward_second_column_blackwell_k,
                                       tensor_memory + nk_attention_backward_scores_column_blackwell_k + 8 * step,
                                       weights + step * 2048 / 16, instruction, accumulates);
    }
    nk_mma_commit_blackwell_(nk_shared_address_ampere_(&control->done));
    nk_mma_commit_blackwell_(nk_shared_address_ampere_(release));
}

/** Turns this thread's lane of S and dP, @p inner columns each, into BF16 dS over dP's first half
 *  and, when it @p weighs, P over S's: P = 2^(S · scale₂ − lse₂) and dS = P · (dP − D), positions a
 *  row does not see weighing 0. Lanes are keys from @p position_first and columns rows when
 *  @p transposed, else lanes are rows and columns positions from it. */
NUMKONG_DEVICE void nk_attention_backward_weigh_blackwell_(nk_u32_t tensor_memory,
                                                           nk_attention_backward_rows_ampere_t const *rows,
                                                           unsigned position_first, unsigned inner, int transposed,
                                                           int weighs, nk_f32_t scale2) {
    unsigned const lane_first = (threadIdx.x >> 5) * 32, lane = threadIdx.x % nk_attention_backward_rows_blackwell_k;
    nk_u32_t const scores = nk_tmem_offset_blackwell_(tensor_memory, lane_first,
                                                      nk_attention_backward_scores_column_blackwell_k);
    nk_u32_t const gradients = nk_tmem_offset_blackwell_(tensor_memory, lane_first,
                                                         nk_attention_backward_gradients_column_blackwell_k);
    // A lane's own row state, read once, when lanes are rows.
    nk_f32_t const lane_log_sum_exp2 = rows->log_sum_exp2[lane], lane_dot = rows->dot[lane];
    unsigned const lane_key_begin = rows->key_begin[lane], lane_key_end = rows->key_end[lane];
    // Each chunk writes the half-width columns that earlier chunks already read.
    for (unsigned chunk = 0; chunk < inner; chunk += 32) {
        nk_u32_t score_bits[32], gradient_bits[32], weight_words[16], gradient_words[16];
        nk_tmem_load_x32_blackwell_(scores + chunk, score_bits);
        nk_tmem_load_x32_blackwell_(gradients + chunk, gradient_bits);
#pragma unroll
        for (unsigned pair = 0; pair < 16; ++pair) {
            nk_f32_t weight[2], score_gradient[2];
#pragma unroll
            for (unsigned index = 0; index < 2; ++index) {
                unsigned const column = chunk + pair * 2 + index;
                unsigned const position = position_first + (transposed ? lane : column);
                nk_f32_t const log_sum_exp2 = transposed ? rows->log_sum_exp2[column] : lane_log_sum_exp2;
                nk_f32_t const dot = transposed ? rows->dot[column] : lane_dot;
                unsigned const key_begin = transposed ? rows->key_begin[column] : lane_key_begin;
                unsigned const key_end = transposed ? rows->key_end[column] : lane_key_end;
                // A select, not a branch, so the exponentials of a chunk issue back to back.
                nk_f32_t const power = nk_f32_exp2_cuda_(__uint_as_float(score_bits[pair * 2 + index]) * scale2 -
                                                         log_sum_exp2);
                weight[index] = (position >= key_begin) & (position < key_end) ? power : 0.0f;
                score_gradient[index] = weight[index] * (__uint_as_float(gradient_bits[pair * 2 + index]) - dot);
            }
            weight_words[pair] = nk_f32x2_to_bf16x2_ampere_(weight[0], weight[1]);
            gradient_words[pair] = nk_f32x2_to_bf16x2_ampere_(score_gradient[0], score_gradient[1]);
        }
        if (weighs) nk_tmem_store_x16_blackwell_(scores + chunk / 2, weight_words);
        nk_tmem_store_x16_blackwell_(gradients + chunk / 2, gradient_words);
    }
    nk_tmem_wait_store_blackwell_();
}

/** Writes this thread's lane of the accumulator at @p column times @p multiplier as @p depth F32
 *  into @p destination, or zeros unless it @p accumulated; a null @p destination skips the stores
 *  but still joins the warp's loads. */
NUMKONG_DEVICE void nk_attention_backward_store_blackwell_(nk_u32_t tensor_memory, unsigned column, nk_size_t depth,
                                                           nk_f32_t multiplier, int accumulated,
                                                           nk_f32_t *destination) {
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, (threadIdx.x >> 5) * 32, column);
    for (unsigned chunk = 0; chunk < depth; chunk += 32) {
        nk_u32_t bits[32];
        nk_tmem_load_x32_blackwell_(lanes + chunk, bits);
        if (!destination) continue;
        for (unsigned index = 0; index < 32 && chunk + index < depth; ++index)
            destination[chunk + index] = accumulated ? __uint_as_float(bits[index]) * multiplier : 0.0f;
    }
}

/** Copies Q and dO of @p rows whole-token folded rows of @p task from @p query_first through
 *  the maps, a box per 128-byte column block of @p staged_bytes, expecting their bytes on
 *  @p barrier. */
NUMKONG_DEVICE void nk_attention_backward_load_rows_blackwell_(nk_attention_backward_maps_blackwell_t const *maps,
                                                               nk_attention_backward_task_t const *task,
                                                               nk_size_t query_first, unsigned staged_bytes,
                                                               unsigned rows, unsigned char *queries_shared,
                                                               unsigned char *gradients_shared, nk_u64_t *barrier) {
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    nk_u32_t const address = nk_shared_address_ampere_(barrier);
    nk_i32_t const head = (nk_i32_t)(task->key_value_head * (arguments->head_count / arguments->key_value_head_count));
    nk_i32_t const token = (nk_i32_t)(arguments->query_offsets[task->segment] + query_first);
    nk_mbarrier_expect_bytes_blackwell_(address, 2 * rows * staged_bytes);
    for (unsigned block = 0; block < staged_bytes / 128; ++block) {
        nk_load_box_3d_blackwell_(nk_shared_address_ampere_(queries_shared + block * rows * 128), &maps->queries_map,
                                  address, (nk_i32_t)(block * 128), head, token);
        nk_load_box_3d_blackwell_(nk_shared_address_ampere_(gradients_shared + block * rows * 128),
                                  &maps->gradients_map, address, (nk_i32_t)(block * 128), head, token);
    }
}

/** Copies @p rows positions of the K and V planes from @p position_first through the packed map, a
 *  box per 64 positions of each 128-byte column block of @p staged_bytes, rereading the planes'
 *  last 64 positions past @p positions_padded, expecting their bytes on @p barrier. */
NUMKONG_DEVICE void nk_attention_backward_load_planes_blackwell_(nk_attention_backward_maps_blackwell_t const *maps,
                                                                 nk_size_t keys_row, nk_size_t values_row,
                                                                 nk_size_t position_first, nk_size_t positions_padded,
                                                                 unsigned staged_bytes, unsigned rows,
                                                                 unsigned char *keys_shared,
                                                                 unsigned char *values_shared, nk_u64_t *barrier) {
    nk_u32_t const address = nk_shared_address_ampere_(barrier);
    nk_mbarrier_expect_bytes_blackwell_(address, 2 * rows * staged_bytes);
    for (unsigned block = 0; block < staged_bytes / 128; ++block)
        for (unsigned box = 0; box < rows / nk_attention_box_rows_blackwell_k; ++box) {
            nk_size_t const first = position_first + box * nk_attention_box_rows_blackwell_k;
            nk_size_t const row = first < positions_padded ? first
                                                           : positions_padded - nk_attention_box_rows_blackwell_k;
            unsigned const offset = (block * rows + box * nk_attention_box_rows_blackwell_k) * 128;
            nk_load_box_blackwell_(nk_shared_address_ampere_(keys_shared + offset), &maps->planes_map, address,
                                   (nk_i32_t)(block * 128), (nk_i32_t)(keys_row + row));
            nk_load_box_blackwell_(nk_shared_address_ampere_(values_shared + offset), &maps->planes_map, address,
                                   (nk_i32_t)(block * 128), (nk_i32_t)(values_row + row));
        }
}

/** Finds the chunks of @p inner folded rows of @p task that see any of the 128 keys from
 *  @p key_first, by bisection, since the keys a row sees only move forward with the row. */
NUMKONG_DEVICE void nk_attention_backward_chunks_blackwell_(nk_attention_backward_arguments_t const *arguments,
                                                            nk_attention_backward_task_t const *task,
                                                            unsigned key_first, unsigned inner, nk_size_t folded_rows,
                                                            unsigned *chunk_begin, unsigned *chunk_end) {
    unsigned const chunk_count = (unsigned)nk_size_divide_round_up_(folded_rows, inner);
    unsigned low = 0, high = chunk_count;
    for (int ending = 0; ending < 2; ++ending) {
        while (low < high) {
            unsigned const middle = low + (high - low) / 2;
            nk_size_t const first = (nk_size_t)middle * inner;
            nk_size_t const last = (first + inner < folded_rows ? first + inner : folded_rows) - 1;
            nk_size_t span_begin, span_end;
            nk_attention_backward_span_ampere_(arguments, task, first, last, &span_begin, &span_end);
            int const past = ending ? span_begin >= key_first + nk_attention_backward_rows_blackwell_k
                                    : span_end > key_first;
            if (past) high = middle;
            else low = middle + 1;
        }
        if (!ending) *chunk_begin = low, high = chunk_count;
    }
    *chunk_end = low;
}

/** The key and value gradients of every task of the window, a block per 128 keys: Sᵀ = K · Qᵀ and
 *  dPᵀ = V · dOᵀ over each chunk of folded rows that sees the block, accumulating dV += Pᵀ · dO and
 *  dK += dSᵀ · Q with Pᵀ and dSᵀ in tensor memory, a lane per key, 128 gradient columns at a time.
 *  The compute warps hold K and V; the producer warps stage the next chunks' Q and dO. */
NUMKONG_DEVICE void nk_attention_backward_keys_blackwell_(nk_attention_backward_maps_blackwell_t const *maps) {
    extern __shared__ unsigned char nk_attention_backward_shared_blackwell_[];
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    nk_size_t const depth = arguments->depth, group = arguments->head_count / arguments->key_value_head_count;
    unsigned const group32 = (unsigned)group;
    unsigned const inner = nk_attention_backward_inner_blackwell_(depth);
    unsigned const slots = nk_attention_backward_slots_blackwell_(depth);
    unsigned const staged_bytes = nk_attention_backward_staged_bytes_blackwell_(depth);
    unsigned const slot_bytes = 2 * inner * staged_bytes;
    unsigned char *const keys_shared = nk_shared_aligned_ampere_(nk_attention_backward_shared_blackwell_, 1024);
    unsigned char *const values_shared = keys_shared + nk_attention_backward_rows_blackwell_k * staged_bytes;
    unsigned char *const ring = values_shared + nk_attention_backward_rows_blackwell_k * staged_bytes;
    nk_attention_backward_control_blackwell_t *const control =
        (nk_attention_backward_control_blackwell_t *)(ring + slots * slot_bytes);
    nk_u32_t const tensor_memory = nk_attention_backward_initialize_blackwell_(control);
    nk_u32_t const keys_address = nk_shared_address_ampere_(keys_shared),
                   values_address = nk_shared_address_ampere_(values_shared);

    unsigned const row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth * 2, nk_attention_step_bytes_k);
    unsigned const chunks = staged_bytes / 16, columns = row_bytes / 2;
    unsigned const slices = nk_u32_divide_round_up_(columns, nk_attention_backward_slice_blackwell_k);
    nk_u32_t const scores_instruction = nk_mma_instruction_blackwell_(1, 1, nk_attention_backward_rows_blackwell_k,
                                                                      inner, nk_major_k_k, nk_major_k_k);
    nk_size_t const gradient_floats = arguments->key_value_gradient_stride / sizeof(nk_f32_t);
    unsigned const compute_warps = nk_attention_backward_compute_threads_blackwell_k / 32;
    int const producer = threadIdx.x >= nk_attention_backward_compute_threads_blackwell_k;
    nk_u32_t phase = 0, sequence = 0;

    // Blocks take every grid-th 128-key block of the window's tasks in turn.
    nk_size_t segment_first = arguments->tasks_begin / arguments->key_value_head_count, items_before = 0;
    nk_size_t task_index, block;
    for (nk_size_t item = blockIdx.x;
         nk_attention_backward_next_cuda_(arguments, 1, nk_attention_backward_rows_blackwell_k, &segment_first,
                                          &items_before, item, &task_index, &block);
         item += gridDim.x) {
        nk_attention_backward_task_t const task = nk_attention_backward_task_simt_(arguments, task_index, row_bytes);
        nk_size_t const positions_padded = nk_size_round_up_to_multiple_(task.length, nk_attention_panel_k);
        nk_size_t const folded_rows = group * task.rows;
        unsigned const key_first = (unsigned)block * nk_attention_backward_rows_blackwell_k;
        // The previous block's keys and values are free, as its products' last wait shows.
        if (!producer) {
            nk_attention_backward_copy_plane_blackwell_(keys_shared, task.keys_plane, key_first, positions_padded,
                                                        nk_attention_backward_rows_blackwell_k, row_bytes, chunks,
                                                        threadIdx.x, nk_attention_backward_compute_threads_blackwell_k);
            nk_attention_backward_copy_plane_blackwell_(values_shared, task.values_plane, key_first, positions_padded,
                                                        nk_attention_backward_rows_blackwell_k, row_bytes, chunks,
                                                        threadIdx.x, nk_attention_backward_compute_threads_blackwell_k);
            nk_commit_async_ampere_();
            nk_wait_async_ampere_(0);
            nk_attention_backward_publish_blackwell_();
        }
        unsigned const position = key_first + threadIdx.x;
        nk_size_t const row_first = (task.key_first + position) * gradient_floats + task.key_value_head * depth;
        int const live = position < task.length;
        for (unsigned slice = 0; slice < slices; ++slice) {
            unsigned const column_first = slice * nk_attention_backward_slice_blackwell_k;
            unsigned const slice_columns = columns - column_first < nk_attention_backward_slice_blackwell_k
                                               ? columns - column_first
                                               : nk_attention_backward_slice_blackwell_k;
            // The slice's columns start this many column blocks into Q and dO.
            nk_u32_t const slice_offset = column_first * 2 / 128 * inner * 128;
            nk_u32_t const values_instruction = nk_mma_instruction_blackwell_(
                1, 1, nk_attention_backward_rows_blackwell_k, slice_columns, nk_major_k_k, nk_major_mn_k);
            int accumulated = 0;
            unsigned chunk_begin, chunk_end;
            nk_attention_backward_chunks_blackwell_(arguments, &task, key_first, inner, folded_rows, &chunk_begin,
                                                    &chunk_end);
            nk_size_t const rows_end = (nk_size_t)chunk_end * inner < folded_rows ? (nk_size_t)chunk_end * inner
                                                                                  : folded_rows;
            nk_size_t query_first = (nk_size_t)chunk_begin * inner / group;
            unsigned remainder = (unsigned)((nk_size_t)chunk_begin * inner % group);
            // Chunks advance the query and head they start at in 32 bits, past any 64-bit division.
            for (nk_size_t first = (nk_size_t)chunk_begin * inner; first < rows_end;
                 first += inner, remainder += inner, query_first += remainder / group32, remainder %= group32) {
                unsigned const count = folded_rows - first < inner ? (unsigned)(folded_rows - first) : inner;
                unsigned const slot = sequence % slots;
                nk_u32_t const slot_phase = (sequence / slots) & 1;
                ++sequence;
                unsigned char *const queries_shared = ring + slot * slot_bytes;
                unsigned char *const gradients_shared = queries_shared + inner * staged_bytes;
                if (producer) {
                    // Whole chunks come through the maps, their tokens all inside the segment.
                    int const boxed = maps->mapped && count == inner;
                    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->empty[slot]), slot_phase ^ 1);
                    if (arguments->prepared)
                        nk_attention_backward_copy_rows_blackwell_(
                            arguments, &task, query_first, remainder, inner, chunks, inner * 128, queries_shared,
                            gradients_shared, &control->rows[slot], !boxed, threadIdx.x - compute_warps * 32,
                            blockDim.x - compute_warps * 32);
                    else
                        nk_attention_backward_stage_rows_ampere_(arguments, &task, first, inner, chunks, inner * 128, 0,
                                                                 queries_shared, gradients_shared, &control->rows[slot],
                                                                 (threadIdx.x >> 5) - compute_warps,
                                                                 (blockDim.x >> 5) - compute_warps);
                    nk_commit_async_ampere_();
                    nk_wait_async_ampere_(0);
                    nk_fence_async_shared_blackwell_();
                    if (boxed && threadIdx.x == nk_attention_backward_compute_threads_blackwell_k)
                        nk_attention_backward_load_rows_blackwell_(maps, &task, query_first, staged_bytes, inner,
                                                                   queries_shared, gradients_shared,
                                                                   &control->full[slot]);
                    else nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->full[slot]));
                    continue;
                }
                nk_u32_t const queries_address = nk_shared_address_ampere_(queries_shared),
                               gradients_address = nk_shared_address_ampere_(gradients_shared);
                nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->full[slot]), slot_phase);
                nk_attention_backward_scores_blackwell_(control, tensor_memory, keys_address, queries_address,
                                                        values_address, gradients_address, row_bytes / 32, inner,
                                                        scores_instruction);
                nk_attention_backward_wait_blackwell_(control, &phase);
                nk_attention_backward_weigh_blackwell_(tensor_memory, &control->rows[slot], key_first, inner, 1, 1,
                                                       arguments->scale2);
                nk_attention_backward_publish_blackwell_();
                nk_attention_backward_values_blackwell_(control, tensor_memory, queries_address + slice_offset,
                                                        gradients_address + slice_offset, 1, inner, values_instruction,
                                                        accumulated, &control->empty[slot]);
                nk_attention_backward_wait_blackwell_(control, &phase);
                accumulated = 1;
            }
            if (producer) continue;
            nk_size_t const stored = depth - column_first < slice_columns ? depth - column_first : slice_columns;
            nk_attention_backward_store_blackwell_(
                tensor_memory, nk_attention_backward_first_column_blackwell_k, stored, arguments->scale, accumulated,
                live ? arguments->key_gradient + row_first + column_first : NUMKONG_NULL);
            nk_attention_backward_store_blackwell_(
                tensor_memory, nk_attention_backward_second_column_blackwell_k, stored, 1.0f, accumulated,
                live ? arguments->value_gradient + row_first + column_first : NUMKONG_NULL);
        }
    }
    nk_attention_backward_release_blackwell_(tensor_memory);
}

/** The query gradients of every task of the window, a block per 128 folded rows: for each panel of
 *  keys the rows see, S = Q · Kᵀ and dP = dO · Vᵀ, then dQ += dS · K with dS in tensor memory, a
 *  lane per row. The compute warps hold Q and dO; the producer warps copy the next K and V panels
 *  ahead of the products. */
NUMKONG_DEVICE void nk_attention_backward_queries_blackwell_(nk_attention_backward_maps_blackwell_t const *maps) {
    extern __shared__ unsigned char nk_attention_backward_shared_blackwell_[];
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    nk_size_t const depth = arguments->depth, group = arguments->head_count / arguments->key_value_head_count;
    unsigned const inner = nk_attention_backward_inner_blackwell_(depth);
    unsigned const slots = nk_attention_backward_slots_blackwell_(depth);
    unsigned const staged_bytes = nk_attention_backward_staged_bytes_blackwell_(depth);
    unsigned const slot_bytes = 2 * inner * staged_bytes;
    unsigned char *const queries_shared = nk_shared_aligned_ampere_(nk_attention_backward_shared_blackwell_, 1024);
    unsigned char *const gradients_shared = queries_shared + nk_attention_backward_rows_blackwell_k * staged_bytes;
    unsigned char *const ring = gradients_shared + nk_attention_backward_rows_blackwell_k * staged_bytes;
    nk_attention_backward_control_blackwell_t *const control =
        (nk_attention_backward_control_blackwell_t *)(ring + slots * slot_bytes);
    nk_u32_t const tensor_memory = nk_attention_backward_initialize_blackwell_(control);
    nk_u32_t const queries_address = nk_shared_address_ampere_(queries_shared),
                   gradients_address = nk_shared_address_ampere_(gradients_shared);

    unsigned const row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth * 2, nk_attention_step_bytes_k);
    unsigned const chunks = staged_bytes / 16;
    nk_u32_t const scores_instruction = nk_mma_instruction_blackwell_(1, 1, nk_attention_backward_rows_blackwell_k,
                                                                      inner, nk_major_k_k, nk_major_k_k);
    nk_u32_t const values_instruction = nk_mma_instruction_blackwell_(1, 1, nk_attention_backward_rows_blackwell_k,
                                                                      row_bytes / 2, nk_major_k_k, nk_major_mn_k);
    unsigned const compute_threads = nk_attention_backward_compute_threads_blackwell_k;
    int const producer = threadIdx.x >= compute_threads;
    nk_u32_t phase = 0, sequence = 0, stagings = 0;

    nk_size_t segment_first = arguments->tasks_begin / arguments->key_value_head_count, items_before = 0;
    nk_size_t task_index, chunk;
    for (nk_size_t item = blockIdx.x;
         nk_attention_backward_next_cuda_(arguments, 0, nk_attention_backward_rows_blackwell_k, &segment_first,
                                          &items_before, item, &task_index, &chunk);
         item += gridDim.x) {
        nk_attention_backward_task_t const task = nk_attention_backward_task_simt_(arguments, task_index, row_bytes);
        nk_size_t const positions_padded = nk_size_round_up_to_multiple_(task.length, nk_attention_panel_k);
        nk_size_t const folded_rows = group * task.rows;
        nk_size_t const first = chunk * nk_attention_backward_rows_blackwell_k;
        nk_size_t const last = (first + nk_attention_backward_rows_blackwell_k < folded_rows
                                    ? first + nk_attention_backward_rows_blackwell_k
                                    : folded_rows) -
                               1;
        nk_size_t span_begin, span_end;
        nk_attention_backward_span_ampere_(arguments, &task, first, last, &span_begin, &span_end);
        unsigned const panel_first = (unsigned)(span_begin / inner);
        unsigned const panel_end = span_begin < span_end ? (unsigned)nk_size_divide_round_up_(span_end, inner)
                                                         : panel_first;
        nk_size_t const keys_row = (nk_size_t)(task.keys_plane - arguments->packed) / row_bytes;
        nk_size_t const values_row = keys_row + arguments->key_value_head_count * positions_padded;
        // The products are done with the previous chunk's rows, as their last wait showed.
        if (!producer) {
            int const boxed = maps->mapped && first + nk_attention_backward_rows_blackwell_k <= folded_rows;
            if (arguments->prepared)
                nk_attention_backward_copy_rows_blackwell_(
                    arguments, &task, first / group, (unsigned)(first % group), nk_attention_backward_rows_blackwell_k,
                    chunks, nk_attention_block_bytes_blackwell_k, queries_shared, gradients_shared, &control->rows[0],
                    !boxed, threadIdx.x, compute_threads);
            else
                nk_attention_backward_stage_rows_ampere_(
                    arguments, &task, first, nk_attention_backward_rows_blackwell_k, chunks,
                    nk_attention_block_bytes_blackwell_k, 0, queries_shared, gradients_shared, &control->rows[0],
                    threadIdx.x >> 5, compute_threads / 32);
            if (boxed && threadIdx.x == 0)
                nk_attention_backward_load_rows_blackwell_(maps, &task, first / group, staged_bytes,
                                                           nk_attention_backward_rows_blackwell_k, queries_shared,
                                                           gradients_shared, &control->staged);
            nk_commit_async_ampere_();
            nk_wait_async_ampere_(0);
            nk_attention_backward_publish_blackwell_();
            if (boxed) nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->staged), stagings++ & 1);
        }
        int accumulated = 0;
        for (unsigned panel = panel_first; panel < panel_end; ++panel) {
            unsigned const position_first = panel * inner;
            unsigned const slot = sequence % slots;
            nk_u32_t const slot_phase = (sequence / slots) & 1;
            ++sequence;
            unsigned char *const keys_shared = ring + slot * slot_bytes;
            unsigned char *const values_shared = keys_shared + inner * staged_bytes;
            if (producer) {
                nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->empty[slot]), slot_phase ^ 1);
                if (maps->mapped) {
                    if (threadIdx.x == compute_threads)
                        nk_attention_backward_load_planes_blackwell_(maps, keys_row, values_row, position_first,
                                                                     positions_padded, staged_bytes, inner, keys_shared,
                                                                     values_shared, &control->full[slot]);
                    else nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->full[slot]));
                    continue;
                }
                nk_attention_backward_copy_plane_blackwell_(
                    keys_shared, task.keys_plane, position_first, positions_padded, inner, row_bytes, chunks,
                    threadIdx.x - compute_threads, blockDim.x - compute_threads);
                nk_attention_backward_copy_plane_blackwell_(
                    values_shared, task.values_plane, position_first, positions_padded, inner, row_bytes, chunks,
                    threadIdx.x - compute_threads, blockDim.x - compute_threads);
                nk_commit_async_ampere_();
                nk_wait_async_ampere_(0);
                nk_fence_async_shared_blackwell_();
                nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->full[slot]));
                continue;
            }
            nk_u32_t const keys_address = nk_shared_address_ampere_(keys_shared),
                           values_address = nk_shared_address_ampere_(values_shared);
            nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->full[slot]), slot_phase);
            nk_attention_backward_scores_blackwell_(control, tensor_memory, queries_address, keys_address,
                                                    gradients_address, values_address, row_bytes / 32, inner,
                                                    scores_instruction);
            nk_attention_backward_wait_blackwell_(control, &phase);
            nk_attention_backward_weigh_blackwell_(tensor_memory, &control->rows[0], position_first, inner, 0, 0,
                                                   arguments->scale2);
            nk_attention_backward_publish_blackwell_();
            nk_attention_backward_values_blackwell_(control, tensor_memory, keys_address, keys_address, 0, inner,
                                                    values_instruction, accumulated, &control->empty[slot]);
            nk_attention_backward_wait_blackwell_(control, &phase);
            accumulated = 1;
        }
        if (producer) continue;

        nk_size_t const folded = first + threadIdx.x, query = folded / group;
        nk_size_t const token = arguments->query_offsets[task.segment] + query;
        nk_size_t const head = task.key_value_head * group + folded % group;
        nk_f32_t *const destination = query < task.rows ? (nk_f32_t *)((unsigned char *)arguments->query_gradient +
                                                                       token * arguments->query_gradient_stride) +
                                                              head * depth
                                                        : NUMKONG_NULL;
        nk_attention_backward_store_blackwell_(tensor_memory, nk_attention_backward_first_column_blackwell_k, depth,
                                               arguments->scale, accumulated, destination);
    }
    nk_attention_backward_release_blackwell_(tensor_memory);
}

enum {
    // Single pass: S with P over its first half, dP then the dQ partial, dK, dV.
    nk_attention_backward_single_gradients_column_blackwell_k = 128,
    nk_attention_backward_single_key_column_blackwell_k = 256,
    nk_attention_backward_single_value_column_blackwell_k = 384,
    // Eight warps weigh and drain, two per lane quarter splitting the columns; four more stage.
    nk_attention_backward_single_compute_threads_blackwell_k = 256,
    nk_attention_backward_single_threads_blackwell_k = 384,
};

/** A chunk's folded rows as one key block sees them: −lse₂, −D · scale, the keys of the block each
 *  row sees as offsets from the block's first key clamped to it, and whether every row sees all. */
typedef struct __align__(16) {
    nk_f32_t negated_log_sum_exp2[128];
    nk_f32_t negated_scaled_dot[128];
    unsigned char key_begin[128];
    unsigned char key_end[128];
    int whole;
} nk_attention_backward_window_blackwell_t;

/** Where a staged dQ partial goes: its first token and head, its live rows, the head offset of its
 *  first row, and whether it went into the 128-byte swizzle for the sums map. */
typedef struct {
    nk_size_t token_first;
    unsigned head_first, live, remainder, boxed;
} nk_attention_backward_reduction_blackwell_t;

/** A single-pass block's state: the rows of the chunk the compute warps weigh next, both slots'
 *  staged partials, the products' barrier, the slots' full barriers, the drained ones the compute
 *  warps arrive on once a partial is staged, the one they arrive on once a chunk is weighed, the
 *  item's K and V planes landing, its last products leaving them, and the tensor-memory address. */
typedef struct {
    nk_attention_backward_window_blackwell_t rows;
    nk_attention_backward_reduction_blackwell_t reductions[2];
    nk_u64_t done, weighed, full[2], drained[2], planes, spent;
    nk_u32_t tensor_memory;
} nk_attention_backward_single_control_blackwell_t;

/** Whether the single pass takes heads of @p depth: whole halves of 32 columns up to 128. */
NUMKONG_CONSTEXPR int nk_attention_backward_single_depth_blackwell_(nk_size_t depth) {
    return depth % 64 == 0 && depth <= 128;
}

/** Dynamic shared memory of a single-pass block: K and V of 128 keys, two slots of Q and dO of 128
 *  rows, dSᵀ of 128 keys by 128 rows, the state, and the slack that aligns them to 1024 bytes. */
NUMKONG_CONSTEXPR nk_size_t nk_attention_backward_single_shared_bytes_blackwell_(nk_size_t depth) {
    return 1024 + (2 + 2 * 2) * 128 * nk_attention_backward_staged_bytes_blackwell_(depth) + 128 * 256 +
           sizeof(nk_attention_backward_single_control_blackwell_t);
}

enum { nk_attention_backward_prepared_loads_blackwell_k = 8 };

/** Sums this lane's share of @p row_lanes' D in the order of float2 loads at 2 · lane and 64 more,
 *  from the float4 at 4 · lane that the half warp from @p row_lanes holds. */
NUMKONG_DEVICE nk_f32_t nk_attention_backward_prepared_dot_blackwell_(float4 gradient, float4 output,
                                                                      unsigned row_lanes, int second_half) {
    unsigned const lane = threadIdx.x & 31, source = row_lanes + (lane >> 1);
    int const odd = lane & 1;
    nk_f32_t const even_sum = fmaf(gradient.y, output.y, fmaf(gradient.x, output.x, 0.0f));
    nk_f32_t const odd_sum = fmaf(gradient.w, output.w, fmaf(gradient.z, output.z, 0.0f));
    nk_f32_t const even_first = __shfl_sync(0xFFFFFFFFu, even_sum, source);
    nk_f32_t const odd_first = __shfl_sync(0xFFFFFFFFu, odd_sum, source);
    nk_f32_t dot = odd ? odd_first : even_first;
    if (second_half) {
        unsigned const second = 16 + (lane >> 1);
        float4 const gradient_second = make_float4(
            __shfl_sync(0xFFFFFFFFu, gradient.x, second), __shfl_sync(0xFFFFFFFFu, gradient.y, second),
            __shfl_sync(0xFFFFFFFFu, gradient.z, second), __shfl_sync(0xFFFFFFFFu, gradient.w, second));
        float4 const output_second = make_float4(
            __shfl_sync(0xFFFFFFFFu, output.x, second), __shfl_sync(0xFFFFFFFFu, output.y, second),
            __shfl_sync(0xFFFFFFFFu, output.z, second), __shfl_sync(0xFFFFFFFFu, output.w, second));
        dot = odd ? fmaf(gradient_second.w, output_second.w, fmaf(gradient_second.z, output_second.z, dot))
                  : fmaf(gradient_second.y, output_second.y, fmaf(gradient_second.x, output_second.x, dot));
    }
    for (unsigned offset = 16; offset != 0; offset >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, offset);
    return dot;
}

/** Writes each folded row's dO as BF16 and D = dO · O into the workspace the maps name, rows of
 *  @c head_count × depth values per token, and zeroes its dQ, which the single pass accumulates
 *  into. A warp takes runs of folded rows, a float4 per lane, so heads of 64 dimensions fill it
 *  with two rows at once, and issues every row's loads before the first sum. */
NUMKONG_DEVICE void nk_attention_backward_prepare_single_blackwell_(
    nk_attention_backward_maps_blackwell_t const *maps) {
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    unsigned const lane = threadIdx.x & 31;
    nk_size_t const depth = arguments->depth, group = arguments->head_count / arguments->key_value_head_count;
    unsigned const group32 = (unsigned)group, lanes_per_row = (unsigned)depth / 4;
    unsigned const rows_per_load = 32 / lanes_per_row, row_in_load = lane / lanes_per_row;
    unsigned const rows = nk_attention_backward_prepared_loads_blackwell_k * rows_per_load;
    nk_size_t const element = (lane % lanes_per_row) * 4;
    nk_size_t const warp = (blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x) >> 5;
    nk_size_t const warps = (gridDim.x * (nk_size_t)blockDim.x) >> 5;
    nk_size_t segment_first = arguments->tasks_begin / arguments->key_value_head_count, items_before = 0, task, run;
    for (nk_size_t item = warp;
         nk_attention_backward_next_cuda_(arguments, 0, rows, &segment_first, &items_before, item, &task, &run);
         item += warps) {
        nk_size_t const segment = task / arguments->key_value_head_count;
        nk_size_t const head_first = task % arguments->key_value_head_count * group;
        nk_size_t const token_first = arguments->query_offsets[segment];
        unsigned const folded_first = (unsigned)run * rows;
        unsigned const folded_end = (unsigned)(group * (arguments->query_offsets[segment + 1] - token_first));
        unsigned const live = folded_end - folded_first < rows ? folded_end - folded_first : rows;
        float4 gradients[nk_attention_backward_prepared_loads_blackwell_k],
            outputs[nk_attention_backward_prepared_loads_blackwell_k];
#pragma unroll
        for (unsigned load = 0; load < nk_attention_backward_prepared_loads_blackwell_k; ++load) {
            unsigned const row = load * rows_per_load + row_in_load;
            unsigned const folded = folded_first + (row < live ? row : 0);
            nk_size_t const offset = (token_first + folded / group32) * arguments->output_stride;
            nk_size_t const column = (head_first + folded % group32) * depth + element;
            nk_f32_t const *output_row = (nk_f32_t const *)((unsigned char const *)arguments->output + offset);
            nk_f32_t const *gradient_row = (nk_f32_t const *)((unsigned char const *)arguments->output_gradient +
                                                              offset);
            gradients[load] = *(float4 const *)(gradient_row + column);
            outputs[load] = *(float4 const *)(output_row + column);
        }
#pragma unroll
        for (unsigned load = 0; load < nk_attention_backward_prepared_loads_blackwell_k; ++load) {
            float4 const gradient = gradients[load], output = outputs[load];
            unsigned const row = load * rows_per_load + row_in_load, folded = folded_first + row;
            nk_size_t const token = token_first + folded / group32, head = head_first + folded % group32;
            nk_size_t const prepared = token * arguments->head_count + head;
            nk_f32_t const first_dot = nk_attention_backward_prepared_dot_blackwell_(gradient, output, 0,
                                                                                     rows_per_load == 1);
            nk_f32_t const second_dot = rows_per_load == 2
                                            ? nk_attention_backward_prepared_dot_blackwell_(gradient, output, 16, 0)
                                            : 0.0f;
            if (row >= live) continue;
            *(uint2 *)(maps->prepared_gradients + (prepared * depth + element) * 2) = make_uint2(
                nk_f32x2_to_bf16x2_ampere_(gradient.x, gradient.y), nk_f32x2_to_bf16x2_ampere_(gradient.z, gradient.w));
            *(float4 *)((nk_f32_t *)((unsigned char *)arguments->query_gradient +
                                     token * arguments->query_gradient_stride) +
                        head * depth + element) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (lane % lanes_per_row == 0) maps->prepared_dots[prepared] = row_in_load ? second_dot : first_dot;
        }
    }
}

/** Waits until every bulk group of this thread has read its shared memory. */
NUMKONG_DEVICE void nk_bulk_wait_reads_blackwell_(void) {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}

/** Expects @p bytes more on @p barrier from asynchronous copies, without arriving. */
NUMKONG_DEVICE void nk_mbarrier_expect_blackwell_(nk_u32_t barrier, nk_u32_t bytes) {
    asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(barrier), "r"(bytes) : "memory");
}

/** Adds the F32 box at @p source to the box at column @p column, head @p head and token @p token of
 *  the tensor @p map describes, as this thread's next bulk group member. */
NUMKONG_DEVICE void nk_reduce_box_3d_blackwell_(void const *map, nk_u32_t source, nk_i32_t column, nk_i32_t head,
                                                nk_i32_t token) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3}], [%4];\n" ::"l"(map),
        "r"(column), "r"(head), "r"(token), "r"(source)
        : "memory");
}

/** Adds @p bytes of F32 at @p source to @p destination, as this thread's next bulk group member. */
NUMKONG_DEVICE void nk_reduce_bulk_f32_blackwell_(nk_f32_t *destination, nk_u32_t source, nk_u32_t bytes) {
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n" ::"l"(destination),
                 "r"(source), "r"(bytes)
                 : "memory");
}

/** Initializes the single pass's barriers and allocates every tensor-memory column from warp 0. */
NUMKONG_DEVICE nk_u32_t
nk_attention_backward_initialize_single_blackwell_(nk_attention_backward_single_control_blackwell_t *control) {
    unsigned const producers = nk_attention_backward_single_threads_blackwell_k -
                               nk_attention_backward_single_compute_threads_blackwell_k;
    unsigned const compute_warps = nk_attention_backward_single_compute_threads_blackwell_k / 32;
    if (threadIdx.x == 0) {
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->done), 1);
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->weighed), compute_warps);
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->planes), 1);
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->spent), 1);
        for (unsigned slot = 0; slot < 2; ++slot) {
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->full[slot]), producers);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->drained[slot]), compute_warps);
        }
        nk_mbarrier_init_fence_blackwell_();
    }
    if (threadIdx.x >> 5 == 0)
        nk_tmem_alloc_blackwell_(nk_shared_address_ampere_(&control->tensor_memory),
                                 nk_attention_tensor_columns_blackwell_k);
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    nk_tmem_fence_after_blackwell_();
    return control->tensor_memory;
}

/** Waits for the products issued last, flipping @p phase. */
NUMKONG_DEVICE void nk_attention_backward_wait_single_blackwell_(nk_u64_t *done, nk_u32_t *phase) {
    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(done), *phase);
    *phase ^= 1;
    nk_tmem_fence_after_blackwell_();
}

/** Publishes the compute warps' shared-memory and tensor-memory writes past all eight of them. */
NUMKONG_DEVICE void nk_attention_backward_publish_single_blackwell_(void) {
    nk_fence_async_shared_blackwell_();
    nk_tmem_fence_before_blackwell_();
    nk_barrier_sync_blackwell_(1, nk_attention_backward_single_compute_threads_blackwell_k);
    nk_tmem_fence_after_blackwell_();
}

/** One folded row's state as a key block sees it, held in registers until the window frees. */
typedef struct {
    nk_f32_t negated_log_sum_exp2, negated_scaled_dot;
    unsigned key_begin, key_end;
} nk_attention_backward_row_blackwell_t;

/** Loads the state of the folded row @p local of a chunk from @p query_first and @p remainder as
 *  the block of keys from @p key_first sees it, D from the workspace. */
NUMKONG_DEVICE nk_attention_backward_row_blackwell_t nk_attention_backward_row_window_blackwell_(
    nk_attention_backward_maps_blackwell_t const *maps, nk_attention_backward_task_t const *task, nk_size_t query_first,
    unsigned remainder, unsigned local, unsigned key_first) {
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    unsigned const group = (unsigned)(arguments->head_count / arguments->key_value_head_count);
    unsigned const offset = remainder + local;
    nk_size_t const query = query_first + offset / group;
    nk_size_t const token = arguments->query_offsets[task->segment] + query;
    nk_size_t const head = task->key_value_head * group + offset % group;
    int const live = query < task->rows;
    nk_size_t key_begin = 0, key_end = 0;
    if (live)
        nk_diagonal_band_row_range_simt_(arguments->band, task->first_band_row + (nk_i64_t)query, task->length,
                                         &key_begin, &key_end);
    nk_size_t const key_last = key_first + nk_attention_backward_rows_blackwell_k;
    key_begin = key_begin < key_first ? key_first : key_begin > key_last ? key_last : key_begin;
    key_end = key_end < key_first ? key_first : key_end > key_last ? key_last : key_end;
    nk_attention_backward_row_blackwell_t row;
    row.negated_log_sum_exp2 = live ? -arguments->log_sum_exp[token * arguments->head_count + head] * NUMKONG_F32_LOG2E_
                                    : 0.0f;
    row.negated_scaled_dot = live ? -maps->prepared_dots[token * arguments->head_count + head] * arguments->scale
                                  : 0.0f;
    row.key_begin = (unsigned)(key_begin - key_first), row.key_end = (unsigned)(key_end - key_first);
    return row;
}

/** Whether every one of @p rows folded rows from @p query_first and @p remainder sees all 128 keys
 *  from @p key_first: as the keys a row sees only move forward with it, whether both ends do. */
NUMKONG_DEVICE int nk_attention_backward_whole_blackwell_(nk_attention_backward_arguments_t const *arguments,
                                                          nk_attention_backward_task_t const *task,
                                                          nk_size_t query_first, unsigned remainder, unsigned rows,
                                                          unsigned key_first) {
    unsigned const group = (unsigned)(arguments->head_count / arguments->key_value_head_count);
    unsigned const last_offset = remainder + rows - 1;
    nk_size_t const query_last = query_first + last_offset / group;
    nk_size_t first_begin, first_end, last_begin, last_end;
    nk_diagonal_band_row_range_simt_(arguments->band, task->first_band_row + (nk_i64_t)query_first, task->length,
                                     &first_begin, &first_end);
    nk_diagonal_band_row_range_simt_(arguments->band, task->first_band_row + (nk_i64_t)query_last, task->length,
                                     &last_begin, &last_end);
    return query_last < task->rows && last_begin <= key_first &&
           first_end >= key_first + nk_attention_backward_rows_blackwell_k;
}

/** Issues, from @p thread of @p threads, the asynchronous copies of @p rows folded rows of
 *  @p task from @p query_first and @p remainder out of @p source, BF16 head rows @p token_stride
 *  bytes apart per token, into @p shared, in @p chunks 16-byte chunks per row of 128-byte column
 *  blocks of @p block_bytes, zero past the head and the task. */
NUMKONG_DEVICE void nk_attention_backward_copy_tokens_blackwell_(
    nk_attention_backward_arguments_t const *arguments, nk_attention_backward_task_t const *task, nk_size_t query_first,
    unsigned remainder, unsigned rows, unsigned chunks, unsigned block_bytes, unsigned char const *source,
    nk_size_t token_stride, unsigned char *shared, unsigned thread, unsigned threads) {
    nk_size_t const depth = arguments->depth;
    unsigned const group = (unsigned)(arguments->head_count / arguments->key_value_head_count);
    nk_size_t const segment_first = arguments->query_offsets[task->segment];
    unsigned const row_chunks = (unsigned)(depth * 2 / 16);
    for (unsigned index = thread; index < rows * chunks; index += threads) {
        unsigned const local = index / chunks, chunk = index % chunks, offset = remainder + local;
        nk_size_t const query = query_first + offset / group;
        nk_size_t const token = segment_first + query, head = task->key_value_head * group + offset % group;
        int const inside = query < task->rows && chunk < row_chunks;
        unsigned char const *row = source + token * token_stride + head * depth * 2;
        nk_copy_b128_async_ampere_(
            nk_shared_address_ampere_(shared + nk_swizzled_panel_offset_simt_(local, chunk * 16, 128, block_bytes)),
            inside ? row + chunk * 16 : source, inside ? 16 : 0);
    }
}

/** Copies Q and dO of 128 whole-token folded rows of @p task from @p query_first through the maps,
 *  a box per 128-byte column block of @p staged_bytes, completing their bytes on @p barrier without
 *  arriving. Past @p reading, the slot's earlier reductions still read it: dO lands once the first
 *  of their two bulk groups is read, Q once both are. */
NUMKONG_DEVICE void nk_attention_backward_load_single_blackwell_(nk_attention_backward_maps_blackwell_t const *maps,
                                                                 nk_attention_backward_task_t const *task,
                                                                 nk_size_t query_first, unsigned staged_bytes,
                                                                 unsigned char *queries_shared,
                                                                 unsigned char *gradients_shared, int reading,
                                                                 nk_u64_t *barrier) {
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    nk_u32_t const address = nk_shared_address_ampere_(barrier);
    nk_i32_t const head = (nk_i32_t)(task->key_value_head * (arguments->head_count / arguments->key_value_head_count));
    nk_i32_t const token = (nk_i32_t)(arguments->query_offsets[task->segment] + query_first);
    unsigned const block_bytes = nk_attention_backward_rows_blackwell_k * 128;
    nk_mbarrier_expect_blackwell_(address, 2 * nk_attention_backward_rows_blackwell_k * staged_bytes);
    if (reading) nk_bulk_wait_reads_but_one_blackwell_();
    for (unsigned block = 0; block < staged_bytes / 128; ++block)
        nk_load_box_3d_blackwell_(nk_shared_address_ampere_(gradients_shared + block * block_bytes),
                                  &maps->gradients_map, address, (nk_i32_t)(block * 128), head, token);
    if (reading) nk_bulk_wait_reads_blackwell_();
    for (unsigned block = 0; block < staged_bytes / 128; ++block)
        nk_load_box_3d_blackwell_(nk_shared_address_ampere_(queries_shared + block * block_bytes), &maps->queries_map,
                                  address, (nk_i32_t)(block * 128), head, token);
}

/** Issues @p a · @p bᵀ of K-major operands into the accumulator at @p accumulator from warp 0 over
 *  @p depth_steps 32-byte steps, then signals @p done unless it is null. */
NUMKONG_DEVICE void nk_attention_backward_scores_single_blackwell_(nk_u64_t *done, nk_u32_t accumulator, nk_u32_t a,
                                                                   nk_u32_t b, unsigned depth_steps,
                                                                   nk_u32_t instruction) {
    if (threadIdx.x >> 5 != 0 || !nk_elect_one_blackwell_()) return;
    nk_u64_t const a_descriptor = nk_smem_descriptor_blackwell_(a, nk_smem_swizzle_128_blackwell_k, 16, 1024);
    nk_u64_t const b_descriptor = nk_smem_descriptor_blackwell_(b, nk_smem_swizzle_128_blackwell_k, 16, 1024);
    for (unsigned step = 0; step < depth_steps; ++step) {
        nk_u32_t const offset = (step / 4 * nk_attention_block_bytes_blackwell_k + step % 4 * 32) / 16;
        nk_mma_f16_blackwell_(accumulator, a_descriptor + offset, b_descriptor + offset, instruction, step != 0);
    }
    if (done) nk_mma_commit_blackwell_(nk_shared_address_ampere_(done));
}

/** Issues from warp 0, over 128 rows and 128 keys, dV += Pᵀ · dO from tensor memory, dK += dSᵀ · Q
 *  and dQ's partial dS · K into the dP columns, with dSᵀ in shared memory, then signals @p done. */
NUMKONG_DEVICE void nk_attention_backward_products_single_blackwell_(nk_u64_t *done, nk_u32_t tensor_memory,
                                                                     nk_u32_t keys, nk_u32_t queries,
                                                                     nk_u32_t gradients, nk_u32_t score_gradients,
                                                                     nk_u32_t values_instruction,
                                                                     nk_u32_t queries_instruction, int accumulate) {
    if (threadIdx.x >> 5 != 0 || !nk_elect_one_blackwell_()) return;
    // MN-major operands span 128 rows of 128-byte blocks; K-major dSᵀ steps 32 bytes along rows.
    nk_u64_t const gradients_descriptor = nk_smem_descriptor_blackwell_(gradients, nk_smem_swizzle_128_blackwell_k,
                                                                        128 * 128, 1024);
    nk_u64_t const queries_descriptor = nk_smem_descriptor_blackwell_(queries, nk_smem_swizzle_128_blackwell_k,
                                                                      128 * 128, 1024);
    nk_u64_t const keys_descriptor = nk_smem_descriptor_blackwell_(keys, nk_smem_swizzle_128_blackwell_k, 128 * 128,
                                                                   1024);
    nk_u64_t const transposed_descriptor = nk_smem_descriptor_blackwell_(score_gradients,
                                                                         nk_smem_swizzle_128_blackwell_k, 16, 1024);
    nk_u64_t const plain_descriptor = nk_smem_descriptor_blackwell_(score_gradients, nk_smem_swizzle_128_blackwell_k,
                                                                    128 * 128, 1024);
    for (unsigned step = 0; step < 8; ++step) {
        nk_u32_t const accumulates = accumulate || step != 0;
        nk_u32_t const transposed_offset = (step / 4 * nk_attention_block_bytes_blackwell_k + step % 4 * 32) / 16;
        nk_mma_f16_tmem_blackwell_(tensor_memory + nk_attention_backward_single_value_column_blackwell_k,
                                   tensor_memory + nk_attention_backward_scores_column_blackwell_k + 8 * step,
                                   gradients_descriptor + step * 2048 / 16, values_instruction, accumulates);
        nk_mma_f16_blackwell_(tensor_memory + nk_attention_backward_single_key_column_blackwell_k,
                              transposed_descriptor + transposed_offset, queries_descriptor + step * 2048 / 16,
                              values_instruction, accumulates);
        nk_mma_f16_blackwell_(tensor_memory + nk_attention_backward_single_gradients_column_blackwell_k,
                              plain_descriptor + step * 2048 / 16, keys_descriptor + step * 2048 / 16,
                              queries_instruction, step != 0);
    }
    nk_mma_commit_blackwell_(nk_shared_address_ampere_(done));
}

/** Weighs 32 columns of this thread's lane from @p column: P = 2^(S · scale₂ − lse₂) and dS =
 *  P · (dP · scale − D · scale), masked by each row's keys unless every row sees all. */
NUMKONG_DEVICE void nk_attention_backward_weigh_columns_blackwell_(nk_attention_backward_window_blackwell_t const *rows,
                                                                   unsigned column, unsigned lane, int masked,
                                                                   nk_u32_t const score_bits[32],
                                                                   nk_u32_t const gradient_bits[32], nk_u64_t scales2,
                                                                   nk_u64_t scales, nk_u32_t weight_words[16],
                                                                   nk_u32_t gradient_words[16]) {
#pragma unroll
    for (unsigned quad = 0; quad < 8; ++quad) {
        ulonglong2 const offsets = *(ulonglong2 const *)(rows->negated_log_sum_exp2 + column + quad * 4);
        ulonglong2 const dots = *(ulonglong2 const *)(rows->negated_scaled_dot + column + quad * 4);
        nk_u32_t const key_begin = masked ? *(nk_u32_t const *)(rows->key_begin + column + quad * 4) : 0;
        nk_u32_t const key_end = masked ? *(nk_u32_t const *)(rows->key_end + column + quad * 4) : 0;
#pragma unroll
        for (unsigned pair = 0; pair < 2; ++pair) {
            unsigned const bit = quad * 4 + pair * 2;
            nk_u64_t const exponents = nk_f32x2_fma_blackwell_(((nk_u64_t)score_bits[bit + 1] << 32) | score_bits[bit],
                                                               scales2, pair ? offsets.y : offsets.x);
            nk_f32_t weight[2];
#pragma unroll
            for (unsigned index = 0; index < 2; ++index) {
                nk_f32_t const power = nk_f32_exp2_cuda_(index ? nk_f32x2_high_blackwell_(exponents)
                                                               : nk_f32x2_low_blackwell_(exponents));
                unsigned const shift = 8 * (pair * 2 + index);
                weight[index] = !masked || ((lane >= ((key_begin >> shift) & 0xFFu)) &
                                            (lane < ((key_end >> shift) & 0xFFu)))
                                    ? power
                                    : 0.0f;
            }
            nk_u64_t const differences = nk_f32x2_fma_blackwell_(
                ((nk_u64_t)gradient_bits[bit + 1] << 32) | gradient_bits[bit], scales, pair ? dots.y : dots.x);
            nk_u64_t const score_gradients = nk_f32x2_mul_blackwell_(nk_f32x2_blackwell_(weight[0], weight[1]),
                                                                     differences);
            weight_words[quad * 2 + pair] = nk_f32x2_to_bf16x2_ampere_(weight[0], weight[1]);
            gradient_words[quad * 2 + pair] = nk_f32x2_to_bf16x2_ampere_(nk_f32x2_low_blackwell_(score_gradients),
                                                                         nk_f32x2_high_blackwell_(score_gradients));
        }
    }
}

/** Turns this warp's quarter of keys and half of the 128 rows of Sᵀ and dPᵀ into BF16 Pᵀ over the
 *  first half of the S columns and dSᵀ · scale into @p score_gradients, K-major in the 128-byte
 *  swizzle; positions a row does not see weigh 0. */
NUMKONG_DEVICE void nk_attention_backward_weigh_single_blackwell_(nk_u32_t tensor_memory,
                                                                  nk_attention_backward_window_blackwell_t const *rows,
                                                                  unsigned char *score_gradients, nk_f32_t scale2,
                                                                  nk_f32_t scale) {
    unsigned const warp = threadIdx.x >> 5, quarter = warp & 3, half = warp >> 2;
    unsigned const lane = quarter * 32 + (threadIdx.x & 31), column_first = half * 64;
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, quarter * 32, 0);
    nk_u64_t const scales2 = nk_f32x2_blackwell_(scale2, scale2), scales = nk_f32x2_blackwell_(scale, scale);
    int const masked = !rows->whole;
    nk_u32_t score_bits[64];
    nk_tmem_load_x32_blackwell_(lanes + nk_attention_backward_scores_column_blackwell_k + column_first, score_bits);
    nk_tmem_load_x32_blackwell_(lanes + nk_attention_backward_scores_column_blackwell_k + column_first + 32,
                                score_bits + 32);
    // The other half writes its Pᵀ over the S columns this half reads.
    nk_tmem_fence_before_blackwell_();
    nk_barrier_sync_blackwell_(1, nk_attention_backward_single_compute_threads_blackwell_k);
    nk_tmem_fence_after_blackwell_();
    unsigned char *const row_bytes = score_gradients + half * nk_attention_block_bytes_blackwell_k + lane * 128;
#pragma unroll
    for (unsigned chunk = 0; chunk < 2; ++chunk) {
        nk_u32_t gradient_bits[32], weight_words[16], gradient_words[16];
        nk_tmem_load_x32_blackwell_(
            lanes + nk_attention_backward_single_gradients_column_blackwell_k + column_first + chunk * 32,
            gradient_bits);
        if (masked)
            nk_attention_backward_weigh_columns_blackwell_(rows, column_first + chunk * 32, lane, 1,
                                                           score_bits + chunk * 32, gradient_bits, scales2, scales,
                                                           weight_words, gradient_words);
        else
            nk_attention_backward_weigh_columns_blackwell_(rows, column_first + chunk * 32, lane, 0,
                                                           score_bits + chunk * 32, gradient_bits, scales2, scales,
                                                           weight_words, gradient_words);
        nk_tmem_store_x16_blackwell_(
            lanes + nk_attention_backward_scores_column_blackwell_k + column_first / 2 + chunk * 16, weight_words);
#pragma unroll
        for (unsigned part = 0; part < 4; ++part)
            *(uint4 *)(row_bytes + (((chunk * 4 + part) ^ (lane & 7)) * 16)) = make_uint4(
                gradient_words[part * 4], gradient_words[part * 4 + 1], gradient_words[part * 4 + 2],
                gradient_words[part * 4 + 3]);
    }
    nk_tmem_wait_store_blackwell_();
}

/** Stores this warp's half of the dQ partial of its quarter's rows into @p staging: whole chunks as
 *  one 128-byte swizzled block per 32 columns for the sums map, partial ones as plain rows of
 *  @p depth floats. Then records where it goes and arrives on @p drained. */
NUMKONG_DEVICE void nk_attention_backward_drain_single_blackwell_(
    nk_attention_backward_maps_blackwell_t const *maps, nk_attention_backward_task_t const *task,
    nk_u32_t tensor_memory, unsigned char *staging, nk_size_t query_first, unsigned remainder, unsigned live,
    nk_attention_backward_reduction_blackwell_t *reduction, nk_u64_t *drained) {
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    unsigned const warp = threadIdx.x >> 5, quarter = warp & 3, half = warp >> 2;
    unsigned const lane = quarter * 32 + (threadIdx.x & 31);
    nk_size_t const depth = arguments->depth;
    unsigned const half_columns = (unsigned)depth / 2;
    int const boxed = live == nk_attention_backward_rows_blackwell_k;
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(
        tensor_memory, quarter * 32, nk_attention_backward_single_gradients_column_blackwell_k + half * half_columns);
    for (unsigned chunk = 0; chunk < half_columns; chunk += 32) {
        nk_u32_t bits[32];
        nk_tmem_load_x32_blackwell_(lanes + chunk, bits);
        unsigned const column = half * half_columns + chunk;
        unsigned char *const row = boxed ? staging + column / 32 * nk_attention_block_bytes_blackwell_k + lane * 128
                                         : staging + lane * depth * 4 + column * 4;
        unsigned const swizzle = boxed ? lane & 7 : 0;
#pragma unroll
        for (unsigned part = 0; part < 8; ++part)
            *(uint4 *)(row + ((part ^ swizzle) * 16)) = make_uint4(bits[part * 4], bits[part * 4 + 1],
                                                                   bits[part * 4 + 2], bits[part * 4 + 3]);
    }
    if (threadIdx.x == 0) {
        unsigned const group = (unsigned)(arguments->head_count / arguments->key_value_head_count);
        reduction->token_first = arguments->query_offsets[task->segment] + query_first;
        reduction->head_first = (unsigned)task->key_value_head * group;
        reduction->live = live, reduction->remainder = remainder, reduction->boxed = boxed;
    }
    nk_fence_async_shared_blackwell_();
    nk_tmem_fence_before_blackwell_();
    nk_attention_warp_arrive_blackwell_(drained);
}

/** Reduces the partial staged in @p staging into dQ from @p thread of the @p threads producers, as
 *  @p reduction says: whole chunks from one thread, their dO blocks in a bulk group ahead of their
 *  Q blocks, so the dO half frees first. */
NUMKONG_DEVICE void nk_attention_backward_reduce_single_blackwell_(
    nk_attention_backward_maps_blackwell_t const *maps, nk_attention_backward_reduction_blackwell_t const *reduction,
    unsigned char *staging, unsigned thread, unsigned threads) {
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    nk_size_t const depth = arguments->depth;
    unsigned const group = (unsigned)(arguments->head_count / arguments->key_value_head_count);
    if (reduction->boxed) {
        if (thread != 0) return;
        for (unsigned half = 0; half < 2; ++half) {
            unsigned const block_first = half ? 0 : (unsigned)depth / 64;
            for (unsigned block = block_first; block < block_first + depth / 64; ++block)
                nk_reduce_box_3d_blackwell_(
                    &maps->sums_map, nk_shared_address_ampere_(staging + block * nk_attention_block_bytes_blackwell_k),
                    (nk_i32_t)(block * 32), (nk_i32_t)reduction->head_first, (nk_i32_t)reduction->token_first);
            nk_bulk_commit_blackwell_();
        }
        return;
    }
    for (unsigned row = thread; row < reduction->live; row += threads) {
        unsigned const offset = reduction->remainder + row;
        if (row != 0 && offset % group != 0) continue;
        unsigned const run = group - offset % group < reduction->live - row ? group - offset % group
                                                                            : reduction->live - row;
        nk_size_t const token = reduction->token_first + offset / group;
        nk_f32_t *destination = (nk_f32_t *)((unsigned char *)arguments->query_gradient +
                                             token * arguments->query_gradient_stride) +
                                (reduction->head_first + offset % group) * depth;
        nk_reduce_bulk_f32_blackwell_(destination, nk_shared_address_ampere_(staging + row * depth * 4),
                                      (nk_u32_t)(run * depth * 4));
        nk_bulk_commit_blackwell_();
    }
}

/** Writes this warp's half of its quarter's lanes of the accumulator at @p column as @p depth F32
 *  into @p destination, or zeros unless it @p accumulated; null @p destination skips storing. */
NUMKONG_DEVICE void nk_attention_backward_store_single_blackwell_(nk_u32_t tensor_memory, unsigned column,
                                                                  nk_size_t depth, int accumulated,
                                                                  nk_f32_t *destination) {
    unsigned const warp = threadIdx.x >> 5, quarter = warp & 3, half = warp >> 2;
    unsigned const half_columns = (unsigned)depth / 2;
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, quarter * 32, column + half * half_columns);
    for (unsigned chunk = 0; chunk < half_columns; chunk += 32) {
        nk_u32_t bits[32];
        nk_tmem_load_x32_blackwell_(lanes + chunk, bits);
        if (!destination) continue;
#pragma unroll
        for (unsigned part = 0; part < 8; ++part)
            *(float4 *)(destination + half * half_columns + chunk + part * 4) =
                accumulated ? make_float4(__uint_as_float(bits[part * 4]), __uint_as_float(bits[part * 4 + 1]),
                                          __uint_as_float(bits[part * 4 + 2]), __uint_as_float(bits[part * 4 + 3]))
                            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

/** All three gradients in one pass, a block per 128 keys: per chunk of 128 folded rows that sees
 *  the block, Sᵀ and dPᵀ once, then dV += Pᵀ · dO and dK += dSᵀ · Q in tensor memory and the dQ
 *  partial dS · K over the dP columns. The partial stages in the chunk's spent slot, which the
 *  producers reduce into dQ before refilling it with Q and the prepared dO. */
NUMKONG_DEVICE void nk_attention_backward_single_blackwell_(nk_attention_backward_maps_blackwell_t const *maps) {
    extern __shared__ unsigned char nk_attention_backward_shared_blackwell_[];
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    nk_size_t const depth = arguments->depth, group = arguments->head_count / arguments->key_value_head_count;
    unsigned const group32 = (unsigned)group, inner = nk_attention_backward_rows_blackwell_k;
    unsigned const staged_bytes = nk_attention_backward_staged_bytes_blackwell_(depth);
    unsigned const slot_bytes = 2 * inner * staged_bytes;
    unsigned char *const keys_shared = nk_shared_aligned_ampere_(nk_attention_backward_shared_blackwell_, 1024);
    unsigned char *const values_shared = keys_shared + inner * staged_bytes;
    unsigned char *const ring = values_shared + inner * staged_bytes;
    unsigned char *const score_gradients = ring + 2 * slot_bytes;
    nk_attention_backward_single_control_blackwell_t *const control =
        (nk_attention_backward_single_control_blackwell_t *)(score_gradients + 128 * 256);
    nk_u32_t const tensor_memory = nk_attention_backward_initialize_single_blackwell_(control);
    nk_u32_t const keys_address = nk_shared_address_ampere_(keys_shared),
                   values_address = nk_shared_address_ampere_(values_shared),
                   score_gradients_address = nk_shared_address_ampere_(score_gradients);

    unsigned const row_bytes = (unsigned)depth * 2, chunks = staged_bytes / 16;
    nk_u32_t const scores_instruction = nk_mma_instruction_blackwell_(1, 1, 128, inner, nk_major_k_k, nk_major_k_k);
    nk_u32_t const values_instruction = nk_mma_instruction_blackwell_(1, 1, 128, (nk_u32_t)depth, nk_major_k_k,
                                                                      nk_major_mn_k);
    nk_u32_t const queries_instruction = nk_mma_instruction_blackwell_(1, 1, 128, (nk_u32_t)depth, nk_major_mn_k,
                                                                       nk_major_mn_k);
    nk_size_t const gradient_floats = arguments->key_value_gradient_stride / sizeof(nk_f32_t);
    unsigned const compute_threads = nk_attention_backward_single_compute_threads_blackwell_k;
    unsigned const producers = nk_attention_backward_single_threads_blackwell_k - compute_threads;
    int const producer = threadIdx.x >= compute_threads;
    unsigned const lane = (threadIdx.x >> 5 & 3) * 32 + (threadIdx.x & 31), local = threadIdx.x - compute_threads;
    nk_u32_t phase = 0, sequence = 0, planes_phase = 0;
    nk_u32_t items_done = 0;

    nk_size_t segment_first = arguments->tasks_begin / arguments->key_value_head_count, items_before = 0;
    nk_size_t task_index, block;
    for (nk_size_t item = blockIdx.x;
         nk_attention_backward_next_cuda_(arguments, 1, nk_attention_backward_rows_blackwell_k, &segment_first,
                                          &items_before, item, &task_index, &block);
         item += gridDim.x) {
        nk_attention_backward_task_t const task = nk_attention_backward_task_simt_(arguments, task_index, row_bytes);
        nk_size_t const positions_padded = nk_size_round_up_to_multiple_(task.length, nk_attention_panel_k);
        nk_size_t const folded_rows = group * task.rows;
        unsigned const key_first = (unsigned)block * nk_attention_backward_rows_blackwell_k;
        if (!producer) {
            nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->planes), planes_phase);
            planes_phase ^= 1;
        }
        int accumulated = 0, scored = 0;
        unsigned chunk_begin, chunk_end;
        nk_attention_backward_chunks_blackwell_(arguments, &task, key_first, inner, folded_rows, &chunk_begin,
                                                &chunk_end);
        nk_size_t const rows_end = (nk_size_t)chunk_end * inner < folded_rows ? (nk_size_t)chunk_end * inner
                                                                              : folded_rows;
        nk_size_t query_first = (nk_size_t)chunk_begin * inner / group;
        unsigned remainder = (unsigned)((nk_size_t)chunk_begin * inner % group);
        // A producer loads this item's planes once the previous item's products are done with them.
        if (producer && local == 0) {
            if (items_done != 0)
                nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->spent), (items_done - 1) & 1);
            nk_attention_backward_load_planes_blackwell_(
                maps, (nk_size_t)(task.keys_plane - arguments->packed) / row_bytes,
                (nk_size_t)(task.values_plane - arguments->packed) / row_bytes, key_first, positions_padded,
                staged_bytes, inner, keys_shared, values_shared, &control->planes);
        }
        ++items_done;
        if (!producer && threadIdx.x == 0 && (nk_size_t)chunk_begin * inner >= rows_end)
            nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->spent));
        for (nk_size_t first = (nk_size_t)chunk_begin * inner; first < rows_end;
             first += inner, remainder += inner, query_first += remainder / group32, remainder %= group32) {
            unsigned const count = folded_rows - first < inner ? (unsigned)(folded_rows - first) : inner;
            unsigned const slot = sequence % 2;
            nk_u32_t const slot_phase = (sequence / 2) & 1;
            ++sequence;
            unsigned char *const queries_shared = ring + slot * slot_bytes;
            unsigned char *const gradients_shared = queries_shared + inner * staged_bytes;
            if (producer) {
                int const boxed = count == inner;
                // The rows' state loads fly while the slot still drains.
                nk_attention_backward_row_blackwell_t const row = nk_attention_backward_row_window_blackwell_(
                    maps, &task, query_first, remainder, local, key_first);
                int const whole = nk_attention_backward_whole_blackwell_(arguments, &task, query_first, remainder,
                                                                         inner, key_first);
                nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->drained[slot]), slot_phase ^ 1);
                int const reduces = sequence > 2;
                if (reduces)
                    nk_attention_backward_reduce_single_blackwell_(maps, &control->reductions[slot], queries_shared,
                                                                   local, producers);
                // Boxed partials reduce from one thread, which alone then waits for them.
                int const reading = reduces && control->reductions[slot].boxed;
                if (!(boxed && reading)) {
                    nk_bulk_wait_reads_blackwell_();
                    nk_barrier_sync_blackwell_(2, producers);
                }
                if (boxed && local == 0)
                    nk_attention_backward_load_single_blackwell_(maps, &task, query_first, staged_bytes, queries_shared,
                                                                 gradients_shared, reading, &control->full[slot]);
                if (!boxed) {
                    nk_attention_backward_copy_tokens_blackwell_(
                        arguments, &task, query_first, remainder, inner, chunks, inner * 128, arguments->queries,
                        arguments->query_stride, queries_shared, local, producers);
                    nk_attention_backward_copy_tokens_blackwell_(
                        arguments, &task, query_first, remainder, inner, chunks, inner * 128, maps->prepared_gradients,
                        arguments->head_count * depth * 2, gradients_shared, local, producers);
                    nk_commit_async_ampere_();
                    nk_wait_async_ampere_(0);
                }
                // The window frees once the compute warps weighed the previous chunk.
                if (sequence > 1)
                    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->weighed), (sequence - 2) & 1);
                control->rows.negated_log_sum_exp2[local] = row.negated_log_sum_exp2;
                control->rows.negated_scaled_dot[local] = row.negated_scaled_dot;
                control->rows.key_begin[local] = (unsigned char)row.key_begin;
                control->rows.key_end[local] = (unsigned char)row.key_end;
                if (local == 0) control->rows.whole = whole;
                nk_fence_async_shared_blackwell_();
                nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->full[slot]));
                continue;
            }
            nk_u32_t const queries_address = nk_shared_address_ampere_(queries_shared),
                           gradients_address = nk_shared_address_ampere_(gradients_shared);
            nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->full[slot]), slot_phase);
            if (threadIdx.x < 32) {
                if (!scored)
                    nk_attention_backward_scores_single_blackwell_(
                        NUMKONG_NULL, tensor_memory + nk_attention_backward_scores_column_blackwell_k, keys_address,
                        queries_address, row_bytes / 32, scores_instruction);
                // The previous chunk's dQ partial leaves the dP columns before this dP lands there.
                if (sequence > 1) {
                    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->drained[slot ^ 1]),
                                                ((sequence - 2) / 2) & 1);
                    nk_tmem_fence_after_blackwell_();
                }
                nk_attention_backward_scores_single_blackwell_(
                    &control->done, tensor_memory + nk_attention_backward_single_gradients_column_blackwell_k,
                    values_address, gradients_address, row_bytes / 32, scores_instruction);
            }
            nk_attention_backward_wait_single_blackwell_(&control->done, &phase);
            nk_attention_backward_weigh_single_blackwell_(tensor_memory, &control->rows, score_gradients,
                                                          arguments->scale2, arguments->scale);
            nk_attention_warp_arrive_blackwell_(&control->weighed);
            nk_attention_backward_publish_single_blackwell_();
            nk_attention_backward_products_single_blackwell_(
                &control->done, tensor_memory, keys_address, queries_address, gradients_address,
                score_gradients_address, values_instruction, queries_instruction, accumulated);
            nk_attention_backward_wait_single_blackwell_(&control->done, &phase);
            // The next chunk's S overwrites the spent Pᵀ while this chunk's dQ drains from dP.
            scored = first + inner < rows_end;
            // The products are done with K and V, so the next item's planes load during the stores.
            if (!scored && threadIdx.x == 0) nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->spent));
            if (scored && threadIdx.x < 32) {
                unsigned const next = sequence % 2;
                nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->full[next]), (sequence / 2) & 1);
                nk_attention_backward_scores_single_blackwell_(
                    NUMKONG_NULL, tensor_memory + nk_attention_backward_scores_column_blackwell_k, keys_address,
                    nk_shared_address_ampere_(ring + next * slot_bytes), row_bytes / 32, scores_instruction);
            }
            nk_attention_backward_drain_single_blackwell_(maps, &task, tensor_memory, queries_shared, query_first,
                                                          remainder, count, &control->reductions[slot],
                                                          &control->drained[slot]);
            accumulated = 1;
        }
        if (producer) continue;
        unsigned const position = key_first + lane;
        nk_size_t const row_first = (task.key_first + position) * gradient_floats + task.key_value_head * depth;
        int const live = position < task.length;
        nk_attention_backward_store_single_blackwell_(
            tensor_memory, nk_attention_backward_single_key_column_blackwell_k, depth, accumulated,
            live ? arguments->key_gradient + row_first : NUMKONG_NULL);
        nk_attention_backward_store_single_blackwell_(
            tensor_memory, nk_attention_backward_single_value_column_blackwell_k, depth, accumulated,
            live ? arguments->value_gradient + row_first : NUMKONG_NULL);
    }
    // The last two partials still wait in their slots.
    for (nk_u32_t staged = sequence > 2 ? sequence - 2 : 0; producer && staged < sequence; ++staged) {
        nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->drained[staged % 2]), (staged / 2) & 1);
        nk_attention_backward_reduce_single_blackwell_(maps, &control->reductions[staged % 2],
                                                       ring + staged % 2 * slot_bytes, local, producers);
        nk_bulk_wait_blackwell_();
    }
    nk_attention_backward_release_blackwell_(tensor_memory);
}

#pragma endregion Backward

#pragma region Launch

/** Describes @p tokens tokens of @p head_count heads of @p head_bytes bytes, @p query_stride bytes
 *  apart from @p queries, to the Tensor Memory Accelerator as boxes of 128 bytes of @p group heads
 *  of 128 / @p group tokens: one 128-byte column block of a Q tile in the 128-byte swizzle.
 *  Returns zero when the driver refuses. */
NUMKONG_INLINE int nk_attention_queries_map_blackwell_(CUtensorMap *map, void const *queries, nk_size_t tokens,
                                                       nk_size_t head_count, nk_size_t head_bytes,
                                                       nk_size_t query_stride, nk_size_t group) {
    static nk_cross_encode_blackwell_t const encode = nk_cross_encoder_blackwell_();
    if (!encode) return 0;
    cuuint64_t const dimensions[3] = {head_bytes, head_count, tokens};
    cuuint64_t const strides[2] = {head_bytes, query_stride};
    cuuint32_t const box[3] = {128, (cuuint32_t)group, (cuuint32_t)(nk_attention_tile_rows_blackwell_k / group)};
    cuuint32_t const element_strides[3] = {1, 1, 1};
    return encode(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, (void *)queries, dimensions, strides, box, element_strides,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

/** Describes the F32 dQ of @p head_count heads of @p depth floats, rows @p stride bytes apart, as
 *  boxes of 32 floats of @p group heads of 128 / @p group tokens in the 128-byte swizzle, which the
 *  single pass reduces into. Returns zero when the driver refuses. */
NUMKONG_INLINE int nk_attention_sums_map_blackwell_(CUtensorMap *map, void *sums, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t stride, nk_size_t group) {
    static nk_cross_encode_blackwell_t const encode = nk_cross_encoder_blackwell_();
    if (!encode) return 0;
    cuuint64_t const dimensions[3] = {depth, head_count, 0x7FFFFFFF};
    cuuint64_t const strides[2] = {depth * 4, stride};
    cuuint32_t const box[3] = {32, (cuuint32_t)group, (cuuint32_t)(nk_attention_backward_rows_blackwell_k / group)};
    cuuint32_t const element_strides[3] = {1, 1, 1};
    return encode(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, sums, dimensions, strides, box, element_strides,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

/**
 *  @brief Validates the contract, describes the packed buffer to the Tensor Memory Accelerator and
 *      launches the kernel for the depth's width, one block per multiprocessor.
 *  @param[in] element_bytes Bytes of one input element.
 *  @param[in] widens Whether Q widens to F16, as I8's does against the F16 K and V of its pack.
 *
 *  The payload starts a whole number of rows past the pack's start, wherever the directory the host
 *  cannot see ends, so one map at the start reaches every plane.
 */
NUMKONG_INLINE nk_status_t nk_attention_launch_tma_blackwell_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, unsigned element_bytes, int widens,
    void const *queries, void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count,
    nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (tasks_begin >= tasks_end || depth == 0) return nk_success_k;
    nk_attention_tile_arguments_blackwell_t arguments;
    arguments.attention = nk_attention_arguments_init_simt_(
        queries, packed, output, log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_token_count,
        query_stride, output_stride, scale, 1, 1, keys_before, keys_after, tasks_begin, tasks_end);
    if (depth > nk_attention_wide_depth_blackwell_k)
        return nk_launch_resident_cuda_(fallback_kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX,
                                        &arguments.attention, stream);
    nk_attention_width_t const width = depth <= nk_attention_narrow_depth_blackwell_k ? nk_attention_width_128_k
                                                                                      : nk_attention_width_256_k;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(
        depth * nk_attention_operand_bytes_blackwell_(element_bytes, widens), nk_attention_step_bytes_k);
    if (!nk_cross_map_blackwell_(&arguments.map, packed, 0x7FFFFFFF, row_bytes, row_bytes, 128,
                                 nk_attention_box_rows_blackwell_k))
        return nk_device_memory_mismatch_k;
    // Q tiles copy as whole head-group tokens when 16-byte strides reach every head of every token.
    nk_size_t const group = head_count / key_value_head_count, head_bytes = depth * element_bytes;
    arguments.query_tokens = 0;
    if (!widens && nk_attention_tile_rows_blackwell_k % group == 0 && query_token_count <= 0xFFFFFFFFu &&
        (((nk_size_t)queries | query_stride | head_bytes) & 15) == 0 &&
        nk_attention_queries_map_blackwell_(&arguments.queries_map, queries, query_token_count, head_count, head_bytes,
                                            query_stride, group))
        arguments.query_tokens = query_token_count;
    unsigned const shared_bytes = nk_attention_shared_bytes_blackwell_(element_bytes, widens, width);
    void const *kernel = width == nk_attention_width_128_k ? narrow_kernel : wide_kernel;
    // Too few items to fill half the multiprocessors split their keys across clusters of two.
    nk_size_t const items_most = key_value_head_count * (query_token_count - tasks_begin / head_count);
    int multiprocessors = 0;
    nk_status_t const status = nk_device_attribute_cuda_(cudaDevAttrMultiProcessorCount, &multiprocessors, stream);
    if (status != nk_success_k) return status;
    arguments.paired = items_most != 0 && 2 * items_most <= (nk_size_t)multiprocessors;
    if (arguments.paired)
        return nk_cross_launch_pairs_blackwell_(kernel, nk_attention_threads_blackwell_k, shared_bytes, items_most,
                                                &arguments, stream);
    return nk_launch_resident_cuda_(kernel, nk_attention_threads_blackwell_k, shared_bytes, shared_bytes,
                                    NUMKONG_SIZE_MAX, &arguments, stream);
}

/** The launch for BF16. */
NUMKONG_INLINE nk_status_t nk_attention_launch_bf16_blackwell_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_tma_blackwell_(narrow_kernel, wide_kernel, fallback_kernel, 2, 0, queries, packed,
                                              output, log_sum_exp, head_count, key_value_head_count, depth,
                                              query_offsets, query_token_count, query_stride, output_stride, scale,
                                              keys_before, keys_after, tasks_begin, tasks_end, stream);
}

/** The launch for F16. */
NUMKONG_INLINE nk_status_t nk_attention_launch_f16_blackwell_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_tma_blackwell_(narrow_kernel, wide_kernel, fallback_kernel, 2, 0, queries, packed,
                                              output, log_sum_exp, head_count, key_value_head_count, depth,
                                              query_offsets, query_token_count, query_stride, output_stride, scale,
                                              keys_before, keys_after, tasks_begin, tasks_end, stream);
}

/** The launch for E4M3. */
NUMKONG_INLINE nk_status_t nk_attention_launch_e4m3_blackwell_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_tma_blackwell_(narrow_kernel, wide_kernel, fallback_kernel, 1, 0, queries, packed,
                                              output, log_sum_exp, head_count, key_value_head_count, depth,
                                              query_offsets, query_token_count, query_stride, output_stride, scale,
                                              keys_before, keys_after, tasks_begin, tasks_end, stream);
}

/** The launch for I8, whose Q widens to F16 against the F16 K and V of its pack. */
NUMKONG_INLINE nk_status_t nk_attention_launch_i8_blackwell_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_tma_blackwell_(narrow_kernel, wide_kernel, fallback_kernel, 1, 1, queries, packed,
                                              output, log_sum_exp, head_count, key_value_head_count, depth,
                                              query_offsets, query_token_count, query_stride, output_stride, scale,
                                              keys_before, keys_after, tasks_begin, tasks_end, stream);
}

static __global__ void __launch_bounds__(nk_attention_threads_k)
    nk_attention_backward_prepare_blackwell_kernel_(nk_attention_backward_arguments_t arguments) {
    nk_attention_backward_prepare_blackwell_(&arguments);
}

static __global__ void __launch_bounds__(nk_attention_threads_k) nk_attention_backward_prepare_single_blackwell_kernel_(
    __grid_constant__ nk_attention_backward_maps_blackwell_t const maps) {
    nk_attention_backward_prepare_single_blackwell_(&maps);
}

/** Frees @p workspace in the order of @p stream. */
NUMKONG_INLINE void nk_attention_backward_free_blackwell_(void *workspace, nk_stream_t stream) {
    int caller = 0;
    if (nk_device_enter_cuda_(stream, &caller) != nk_success_k) return;
    cudaFreeAsync(workspace, (cudaStream_t)stream);
    nk_device_leave_cuda_(caller);
}

/** Allocates on @p stream the single pass's workspace for @p maps and describes its dO, returning
 *  null when either fails, so the launch falls back to the two passes. */
NUMKONG_INLINE void *nk_attention_backward_workspace_blackwell_(nk_attention_backward_maps_blackwell_t *maps,
                                                                nk_stream_t stream) {
    nk_attention_backward_arguments_t const *arguments = &maps->arguments;
    nk_size_t const depth = arguments->depth, group = arguments->head_count / arguments->key_value_head_count;
    nk_size_t const rows = arguments->query_token_count * arguments->head_count;
    void *workspace = NUMKONG_NULL;
    int caller = 0;
    if (nk_device_enter_cuda_(stream, &caller) != nk_success_k) return NUMKONG_NULL;
    cudaError_t const allocated = cudaMallocAsync(&workspace, rows * (depth * 2 + 4), (cudaStream_t)stream);
    if (allocated != cudaSuccess) cudaGetLastError();
    nk_device_leave_cuda_(caller);
    if (allocated != cudaSuccess) return NUMKONG_NULL;
    if (!nk_attention_queries_map_blackwell_(&maps->gradients_map, workspace, arguments->query_token_count,
                                             arguments->head_count, depth * 2, arguments->head_count * depth * 2,
                                             group)) {
        nk_attention_backward_free_blackwell_(workspace, stream);
        return NUMKONG_NULL;
    }
    maps->prepared_gradients = (unsigned char *)workspace;
    maps->prepared_dots = (nk_f32_t *)((unsigned char *)workspace + rows * depth * 2);
    return workspace;
}

/** Launches the BF16 backward kernels, a block per multiprocessor, after preparing dQ where Q and
 *  dQ rows align for 16-byte copies, with the maps of Q, dO and the planes where whole 128-byte
 *  column blocks reach them, or, with no shared memory, a block per task of the @c cuda loops for
 *  heads too deep for the tensor cores. */
NUMKONG_INLINE nk_status_t nk_attention_backward_launch_blackwell_(void const *keys_kernel, void const *queries_kernel,
                                                                   nk_attention_backward_arguments_t arguments,
                                                                   nk_stream_t stream) {
    nk_status_t status = nk_attention_backward_validate_cuda_(&arguments);
    if (status != nk_success_k) return status;
    if (arguments.tasks_begin >= arguments.tasks_end || arguments.depth == 0) return nk_success_k;
    nk_size_t const depth = arguments.depth, group = arguments.head_count / arguments.key_value_head_count;
    int const tensor_cores = depth <= nk_attention_backward_depth_blackwell_k;
    arguments.prepared = tensor_cores && depth % 8 == 0 &&
                         (((nk_size_t)arguments.queries | arguments.query_stride | (nk_size_t)arguments.query_gradient |
                           arguments.query_gradient_stride) &
                          15) == 0;
    nk_attention_backward_maps_blackwell_t maps;
    maps.arguments = arguments;
    // Boxes span whole 128-byte column blocks of 128 folded rows, so heads of 64 or 128 dimensions.
    maps.mapped = arguments.prepared && depth * 2 % 128 == 0 && depth <= nk_attention_narrow_depth_blackwell_k &&
                  nk_attention_backward_rows_blackwell_k % group == 0 &&
                  nk_attention_queries_map_blackwell_(&maps.queries_map, arguments.queries, arguments.query_token_count,
                                                      arguments.head_count, depth * 2, arguments.query_stride, group) &&
                  nk_attention_queries_map_blackwell_(&maps.gradients_map, arguments.query_gradient,
                                                      arguments.query_token_count, arguments.head_count, depth * 4,
                                                      arguments.query_gradient_stride, group) &&
                  nk_cross_map_blackwell_(&maps.planes_map, arguments.packed, 0x7FFFFFFF, depth * 2, depth * 2, 128,
                                          nk_attention_box_rows_blackwell_k);
    // The single pass reads O and dO by float4 and reduces dQ through TMA, both on 16-byte rows.
    maps.single = maps.mapped && nk_attention_backward_single_depth_blackwell_(depth) &&
                  arguments.query_token_count != 0 &&
                  (((nk_size_t)arguments.key_gradient | (nk_size_t)arguments.value_gradient |
                    arguments.key_value_gradient_stride | (nk_size_t)arguments.output |
                    (nk_size_t)arguments.output_gradient | arguments.output_stride) &
                   15) == 0 &&
                  nk_attention_sums_map_blackwell_(&maps.sums_map, arguments.query_gradient, arguments.head_count,
                                                   depth, arguments.query_gradient_stride, group);
    void *workspace = maps.single ? nk_attention_backward_workspace_blackwell_(&maps, stream) : NUMKONG_NULL;
    if (workspace) {
        nk_size_t const single_bytes = nk_attention_backward_single_shared_bytes_blackwell_(depth);
        status = nk_launch_resident_cuda_((void const *)nk_attention_backward_prepare_single_blackwell_kernel_,
                                          nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &maps, stream);
        if (status == nk_success_k)
            status = nk_launch_resident_cuda_(keys_kernel, nk_attention_backward_single_threads_blackwell_k,
                                              single_bytes, single_bytes, NUMKONG_SIZE_MAX, &maps, stream);
        nk_attention_backward_free_blackwell_(workspace, stream);
        return status;
    }
    maps.single = 0;
    status = arguments.prepared
                 ? nk_launch_resident_cuda_((void const *)nk_attention_backward_prepare_blackwell_kernel_,
                                            nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &maps.arguments, stream)
                 : nk_success_k;
    unsigned const threads = tensor_cores ? nk_attention_backward_threads_blackwell_k : nk_attention_threads_k;
    nk_size_t const shared_bytes = tensor_cores ? nk_attention_backward_shared_bytes_blackwell_(depth) : 0;
    nk_size_t const shared_ceiling =
        tensor_cores ? nk_attention_backward_shared_bytes_blackwell_(nk_attention_backward_depth_blackwell_k) : 0;
    nk_size_t const blocks_wanted = tensor_cores ? NUMKONG_SIZE_MAX : arguments.tasks_end - arguments.tasks_begin;
    if (status == nk_success_k)
        status = nk_launch_resident_cuda_(keys_kernel, threads, shared_bytes, shared_ceiling, blocks_wanted, &maps,
                                          stream);
    if (status == nk_success_k)
        status = nk_launch_resident_cuda_(queries_kernel, threads, shared_bytes, shared_ceiling, blocks_wanted, &maps,
                                          stream);
    return status;
}

#pragma endregion Launch

#pragma region Pack

/** Copies the K and V planes of each @b (segment,kv_head) task, both position-major, one block per
 *  task, at offsets the key counts alone give, zeroing every padded element, so one tensor map
 *  reaches every plane. */
NUMKONG_DEVICE void nk_attention_pack_rows_blackwell_(
    nk_size_t element_bytes, unsigned char const *keys, unsigned char const *values, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count,
    nk_size_t key_stride, nk_size_t value_stride, unsigned char *packed, nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_size_t const depth_bytes = depth * element_bytes;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth_bytes, nk_attention_step_bytes_k);
    unsigned char *payload = packed + nk_attention_payload_offset_simt_(segment_count, row_bytes);
    // Planes start on 16 bytes, so 16-byte chunks move whenever every source head row does too.
    int const chunked = (((nk_size_t)keys | (nk_size_t)values | key_stride | value_stride | depth_bytes) & 15) == 0;
    // Each thread keeps one 16-byte column and walks rows, so no index divides per chunk.
    unsigned const row_chunks = (unsigned)(row_bytes / 16), depth_chunks = (unsigned)(depth_bytes / 16);
    unsigned const rows_per_pass = blockDim.x / row_chunks, lane_row = threadIdx.x / row_chunks;
    unsigned const chunk = threadIdx.x % row_chunks;
    uint4 const zero = make_uint4(0, 0, 0, 0);
    nk_size_t cursor_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task = tasks_begin + blockIdx.x; task < tasks_end; task += gridDim.x) {
        nk_size_t const segment = task / key_value_head_count, head = task % key_value_head_count;
        nk_attention_pack_advance_simt_(key_offsets, key_lengths, key_value_head_count, row_bytes, segment,
                                        &cursor_segment, &payload_offset);
        nk_size_t const length = nk_attention_pack_key_count_simt_(key_offsets, key_lengths, segment);
        if (length == 0) continue;
        nk_size_t const plane_bytes = nk_size_round_up_to_multiple_(length, nk_attention_panel_k) * row_bytes;
        unsigned char *keys_plane = payload + payload_offset + head * plane_bytes;
        unsigned char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        unsigned char const *keys_first = keys + key_offsets[segment] * key_stride + head * depth_bytes;
        unsigned char const *values_first = values + key_offsets[segment] * value_stride + head * depth_bytes;
        if (chunked) {
            if (lane_row >= rows_per_pass) continue;
            unsigned const padded = (unsigned)(plane_bytes / row_bytes);
            int const live_column = chunk < depth_chunks;
            unsigned char *keys_column = keys_plane + chunk * 16, *values_column = values_plane + chunk * 16;
            unsigned char const *keys_source = keys_first + chunk * 16, *values_source = values_first + chunk * 16;
            for (unsigned position = lane_row; position < padded; position += 4 * rows_per_pass) {
                uint4 keys_quartets[4], values_quartets[4];
#pragma unroll
                for (unsigned step = 0; step < 4; ++step) {
                    nk_size_t const row = position + step * rows_per_pass;
                    int const inside = live_column && row < length;
                    keys_quartets[step] = inside ? __ldg((uint4 const *)(keys_source + row * key_stride)) : zero;
                    values_quartets[step] = inside ? __ldg((uint4 const *)(values_source + row * value_stride)) : zero;
                }
#pragma unroll
                for (unsigned step = 0; step < 4; ++step) {
                    nk_size_t const row = position + step * rows_per_pass;
                    if (row >= padded) break;
                    *(uint4 *)(keys_column + row * row_bytes) = keys_quartets[step];
                    *(uint4 *)(values_column + row * row_bytes) = values_quartets[step];
                }
            }
        }
        else
            for (nk_size_t index = threadIdx.x; index < plane_bytes; index += blockDim.x) {
                nk_size_t const position = index / row_bytes, byte = index % row_bytes;
                int const inside = position < length && byte < depth_bytes;
                keys_plane[index] = inside ? keys_first[position * key_stride + byte] : (unsigned char)0;
                values_plane[index] = inside ? values_first[position * value_stride + byte] : (unsigned char)0;
            }
    }
}

/** Writes the K and V planes of each @b (segment,kv_head) task as F16 rows of I8 codes, laid out as
 *  @c nk_attention_pack_rows_blackwell_ lays out F16, so the products read them as they land. */
NUMKONG_DEVICE void nk_attention_pack_rows_i8_blackwell_(unsigned char const *keys, unsigned char const *values,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_size_t key_stride,
                                                         nk_size_t value_stride, unsigned char *packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * 2, nk_attention_step_bytes_k);
    unsigned char *payload = packed + nk_attention_payload_offset_simt_(segment_count, row_bytes);
    // Each thread widens one 16-byte column of codes into 32 bytes of a row, four rows at a time.
    int const chunked = (((nk_size_t)keys | (nk_size_t)values | key_stride | value_stride | depth) & 15) == 0;
    unsigned const row_chunks = (unsigned)(row_bytes / 32), depth_chunks = (unsigned)(depth / 16);
    unsigned const rows_per_pass = blockDim.x / row_chunks, lane_row = threadIdx.x / row_chunks;
    unsigned const chunk = threadIdx.x % row_chunks;
    nk_u32_t const subtrahend = (0x6400u | 128u) * 0x00010001u;
    nk_size_t cursor_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task = tasks_begin + blockIdx.x; task < tasks_end; task += gridDim.x) {
        nk_size_t const segment = task / key_value_head_count, head = task % key_value_head_count;
        nk_attention_pack_advance_simt_(key_offsets, key_lengths, key_value_head_count, row_bytes, segment,
                                        &cursor_segment, &payload_offset);
        nk_size_t const length = nk_attention_pack_key_count_simt_(key_offsets, key_lengths, segment);
        if (length == 0) continue;
        nk_size_t const plane_bytes = nk_size_round_up_to_multiple_(length, nk_attention_panel_k) * row_bytes;
        unsigned char *keys_plane = payload + payload_offset + head * plane_bytes;
        unsigned char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        unsigned char const *keys_first = keys + key_offsets[segment] * key_stride + head * depth;
        unsigned char const *values_first = values + key_offsets[segment] * value_stride + head * depth;
        if (chunked) {
            if (lane_row >= rows_per_pass) continue;
            unsigned const padded = (unsigned)(plane_bytes / row_bytes);
            int const live_column = chunk < depth_chunks;
            unsigned char *keys_column = keys_plane + chunk * 32, *values_column = values_plane + chunk * 32;
            unsigned char const *keys_source = keys_first + chunk * 16, *values_source = values_first + chunk * 16;
            uint4 const zero = make_uint4(0, 0, 0, 0);
            for (unsigned position = lane_row; position < padded; position += 4 * rows_per_pass) {
                uint4 keys_codes[4], values_codes[4];
#pragma unroll
                for (unsigned step = 0; step < 4; ++step) {
                    nk_size_t const row = position + step * rows_per_pass;
                    int const inside = live_column && row < length;
                    keys_codes[step] = inside ? __ldg((uint4 const *)(keys_source + row * key_stride)) : zero;
                    values_codes[step] = inside ? __ldg((uint4 const *)(values_source + row * value_stride)) : zero;
                }
#pragma unroll
                for (unsigned step = 0; step < 4; ++step) {
                    nk_size_t const row = position + step * rows_per_pass;
                    if (row >= padded) break;
                    nk_u32_t const key_words[4] = {keys_codes[step].x, keys_codes[step].y, keys_codes[step].z,
                                                   keys_codes[step].w};
                    nk_u32_t const value_words[4] = {values_codes[step].x, values_codes[step].y, values_codes[step].z,
                                                     values_codes[step].w};
                    nk_u32_t key_halves[8], value_halves[8];
#pragma unroll
                    for (unsigned word = 0; word < 4; ++word) {
                        nk_u8x4_to_f16x4_blackwell_(key_words[word] ^ 0x80808080u, subtrahend, &key_halves[2 * word],
                                                    &key_halves[2 * word + 1]);
                        nk_u8x4_to_f16x4_blackwell_(value_words[word] ^ 0x80808080u, subtrahend,
                                                    &value_halves[2 * word], &value_halves[2 * word + 1]);
                    }
                    uint4 *const keys_row = (uint4 *)(keys_column + row * row_bytes);
                    uint4 *const values_row = (uint4 *)(values_column + row * row_bytes);
                    keys_row[0] = make_uint4(key_halves[0], key_halves[1], key_halves[2], key_halves[3]);
                    keys_row[1] = make_uint4(key_halves[4], key_halves[5], key_halves[6], key_halves[7]);
                    values_row[0] = make_uint4(value_halves[0], value_halves[1], value_halves[2], value_halves[3]);
                    values_row[1] = make_uint4(value_halves[4], value_halves[5], value_halves[6], value_halves[7]);
                }
            }
        }
        else {
            __half *const keys_halves = (__half *)keys_plane, *const values_halves = (__half *)values_plane;
            nk_i8_t const *const keys_codes = (nk_i8_t const *)keys_first;
            nk_i8_t const *const values_codes = (nk_i8_t const *)values_first;
            for (nk_size_t index = threadIdx.x; index < plane_bytes / 2; index += blockDim.x) {
                nk_size_t const position = index / (row_bytes / 2), element = index % (row_bytes / 2);
                int const inside = position < length && element < depth;
                keys_halves[index] = __int2half_rn(inside ? keys_codes[position * key_stride + element] : 0);
                values_halves[index] = __int2half_rn(inside ? values_codes[position * value_stride + element] : 0);
            }
        }
    }
}

/** Writes an I8 pack's payload: F16 rows for the tile's depths, the @c cuda layout of codes past
 *  them for the fallback kernel. */
NUMKONG_DEVICE void nk_attention_pack_payload_i8_blackwell_(unsigned char const *keys, unsigned char const *values,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                            nk_size_t segment_count, nk_size_t key_stride,
                                                            nk_size_t value_stride, unsigned char *packed,
                                                            nk_size_t tasks_begin, nk_size_t tasks_end) {
    if (depth <= nk_attention_wide_depth_blackwell_k)
        nk_attention_pack_rows_i8_blackwell_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                             segment_count, key_stride, value_stride, packed, tasks_begin, tasks_end);
    else
        nk_attention_pack_payload_i8_simt_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                           segment_count, key_stride, value_stride, packed, tasks_begin, tasks_end);
}

/** Bytes per element of an I8 pack's rows: 2 where the tile reads F16, 1 where the fallback reads
 *  codes. */
NUMKONG_CONSTEXPR nk_size_t nk_attention_pack_element_bytes_i8_blackwell_(nk_size_t depth) {
    return depth <= nk_attention_wide_depth_blackwell_k ? 2 : 1;
}

#pragma endregion Pack

#pragma region Attention Macros

/** Generates a device pack recording the @p isa_suffix capability: both planes position-major for
 *  the tile's depths, the @c cuda layout past them for the fallback kernel. */
#define nk_define_attention_pack_blackwell_(input_type_name, isa_suffix, input_value_type)                            \
    static __global__ void nk_attention_pack_##input_type_name##_##isa_suffix##_kernel_(                              \
        unsigned char const *keys, unsigned char const *values, nk_size_t key_value_head_count, nk_size_t depth,      \
        nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count, nk_size_t key_stride,      \
        nk_size_t value_stride, unsigned char *packed, nk_size_t tasks_begin, nk_size_t tasks_end) {                  \
        if (depth <= nk_attention_wide_depth_blackwell_k)                                                             \
            nk_attention_pack_rows_blackwell_(sizeof(nk_##input_value_type##_t), keys, values, key_value_head_count,  \
                                              depth, key_offsets, key_lengths, segment_count, key_stride,             \
                                              value_stride, packed, tasks_begin, tasks_end);                          \
        else                                                                                                          \
            nk_attention_pack_payload_##input_value_type##_simt_(keys, values, key_value_head_count, depth,           \
                                                                 key_offsets, key_lengths, segment_count, key_stride, \
                                                                 value_stride, packed, tasks_begin, tasks_end);       \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_attention_pack_##input_type_name##_##isa_suffix(                                       \
        nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,    \
        nk_size_t segment_count, nk_##input_value_type##_t const *keys, nk_size_t key_stride,                         \
        nk_##input_value_type##_t const *values, nk_size_t value_stride, void *key_value_packed,                      \
        nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {                                             \
        return nk_attention_pack_launch_cuda_(                                                                        \
            (void const *)nk_attention_pack_##input_type_name##_##isa_suffix##_kernel_, nk_cap_##isa_suffix##_k,      \
            sizeof(nk_##input_value_type##_t), keys, values, key_value_head_count, depth, key_offsets, key_lengths,   \
            segment_count, key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, stream);               \
    }

/**
 *  @brief Generates one dtype's tile roles, kernels and public entry point for the TMA attention.
 *  @param[in] isa_suffix The capability the roles, kernels and entry point are named for, whose
 *      @c nk_tmem_load_extreme_x32_ helper finds each row's maximum.
 *  @param[in] scores_fn Issues one 32-byte depth step of S = Q · Kᵀ from shared-memory
 *      descriptors of both operands.
 *  @param[in] values_fn Issues one step of O += P · V, with P at a tensor-memory address.
 *  @param[in] weights_fn Packs four probabilities of consecutive positions into words of P, a word
 *      per 16-bit pair for @c kind::f16 or one for four E4M3 codes, and adds the weights P · V sees
 *      to the F32 pair of running sums.
 *  @param[in] format The operand format code of both products.
 *
 *  The roles issue the products or weigh the panels of one tile. The tile runs over every work
 *  item of a launch in narrow and wide kernels, unmasked and masked by the band, plus a fallback
 *  kernel. The entry point picks the masked pair unless the band reaches every key, and passes
 *  them to @c nk_attention_launch_tma_blackwell_.
 *
 *  Warps 0 to 7 run the softmax of the two tiles, 8 to 11 correct and store O, 12 multiplies,
 *  and 13 schedules and loads.
 */
#define nk_define_attention_tma_blackwell_(input_type_name, isa_suffix, input_value_type, input_bytes, input_widens,   \
                                           scores_fn, values_fn, weights_fn, format)                                   \
    /* Issues S = Q · Kᵀ of a tile against the K panel at keys into slot's S columns and barrier,                   \
     * from the elected lane of the converged MMA warp. */                                                             \
    NUMKONG_DEVICE void nk_attention_issue_scores_##input_type_name##_##isa_suffix##_(                                 \
        nk_attention_kernel_blackwell_t const *kernel, nk_attention_schedule_blackwell_t const *schedule,              \
        nk_attention_control_blackwell_t *control, nk_u32_t queries_address, nk_u32_t keys, nk_u32_t tensor_memory,    \
        unsigned tile, unsigned slot, int elected) {                                                                   \
        nk_u32_t const accumulator = tensor_memory + nk_attention_scores_column_blackwell_k +                          \
                                     (slot ? nk_attention_tile_rows_blackwell_k : 0);                                  \
        /* K-major rows, whose leading offset goes unread, in 8-row groups 1024 bytes apart */                         \
        nk_u64_t const a = nk_smem_descriptor_blackwell_(queries_address + tile * schedule->queries_bytes,             \
                                                         nk_smem_swizzle_128_blackwell_k, 16, 1024);                   \
        nk_u64_t const b = nk_smem_descriptor_blackwell_(keys, nk_smem_swizzle_128_blackwell_k, 16, 1024);             \
        unsigned const steps =                                                                                         \
            nk_attention_staged_bytes_blackwell_(kernel->element_bytes, kernel->widens, kernel->width) / 32;           \
        _Pragma("unroll") for (unsigned step = 0; step < steps; ++step) {                                              \
            /* Each 32-byte step starts where row 0 holds it: a block on every fourth step */                          \
            nk_u32_t const offset = step / 4 * nk_attention_block_bytes_blackwell_k + step % 4 * 32;                   \
            if (step < schedule->depth_steps && elected)                                                               \
                scores_fn(accumulator, a + offset / 16, b + offset / 16, schedule->scores_instruction, step != 0);     \
        }                                                                                                              \
        if (elected) nk_attention_commit_blackwell_(&control->scored[slot]);                                           \
    }                                                                                                                  \
    /* Issues O += P · V of a tile against the V panel at values with P in slot, each half of the                     \
     * panel once its half of P is written and its O corrected, starting O afresh on an item's first                   \
     * panel and signalling its last. The S that follows signals the others, as the tensor cores run                   \
     * products in order. */                                                                                           \
    NUMKONG_DEVICE void nk_attention_issue_values_##input_type_name##_##isa_suffix##_(                                 \
        nk_attention_schedule_blackwell_t const *schedule, nk_attention_control_blackwell_t *control, nk_u32_t values, \
        nk_u32_t tensor_memory, unsigned tile, unsigned slot, int first, int last, int elected,                        \
        nk_attention_progress_blackwell_t *progress) {                                                                 \
        nk_u32_t const accumulator = tensor_memory + nk_attention_output_column_blackwell_k +                          \
                                     tile * nk_attention_tile_rows_blackwell_k;                                        \
        nk_u32_t const weights = tensor_memory + nk_attention_scores_column_blackwell_k +                              \
                                 (slot ? nk_attention_tile_rows_blackwell_k : 0);                                      \
        nk_u32_t const weighing = progress->weighings[slot]++ & 1;                                                     \
        /* MN-major V: 128 bytes of N per row, next 128 a block on, 8-row groups 1024 apart */                         \
        nk_u64_t const b = nk_smem_descriptor_blackwell_(values, nk_smem_swizzle_128_blackwell_k,                      \
                                                         nk_attention_block_bytes_blackwell_k, 1024);                  \
        nk_attention_wait_blackwell_(&control->weighed_half[slot], weighing);                                          \
        if (progress->panels[tile])                                                                                    \
            nk_attention_wait_blackwell_(&control->corrected[tile], (progress->panels[tile] - 1) & 1);                 \
        nk_tmem_fence_after_blackwell_();                                                                              \
        _Pragma("unroll") for (unsigned step = 0; step < schedule->value_steps; ++step) {                              \
            if (step == schedule->value_steps / 2) {                                                                   \
                nk_attention_wait_blackwell_(&control->weighed[slot], weighing);                                       \
                nk_tmem_fence_after_blackwell_();                                                                      \
            }                                                                                                          \
            if (elected)                                                                                               \
                values_fn(accumulator, weights + 8 * step, b + step * schedule->value_step_bytes / 16,                 \
                          schedule->values_instruction, !first || step != 0);                                          \
        }                                                                                                              \
        if (last && elected) nk_attention_commit_blackwell_(&control->accumulated[tile]);                              \
        /* A one-stage ring, I8's at wide depths, can't hold V until the next S needs the stage */                     \
        if (schedule->ring_stages == 1 && elected) nk_attention_commit_blackwell_(&control->consumed[0]);              \
        ++progress->panels[tile];                                                                                      \
    }                                                                                                                  \
    /* Issues one item's products from one elected lane of the converged MMA warp, which keeps the                     \
     * operands in uniform registers. Two tiles take turns, so one tile's softmax runs while the                       \
     * other's products do: S of tile 0, P · V of tile 1 a panel behind, S of tile 1, and then the                    \
     * P · V of tile 0. */                                                                                            \
    NUMKONG_DEVICE void nk_attention_multiply_item_##input_type_name##_##isa_suffix##_(                                \
        nk_attention_kernel_blackwell_t const *kernel, nk_attention_schedule_blackwell_t const *schedule,              \
        nk_attention_control_blackwell_t *control, nk_attention_item_blackwell_t const *item,                          \
        nk_u32_t queries_address, nk_u32_t stages_address, nk_u32_t tensor_memory,                                     \
        nk_attention_progress_blackwell_t *progress) {                                                                 \
        int const elected = nk_elect_one_blackwell_();                                                                 \
        for (unsigned tile = 0; tile < item->tiles; ++tile)                                                            \
            nk_attention_wait_blackwell_(&control->queries_staged[tile], progress->items[tile]++ & 1);                 \
        nk_u32_t keys = 0, values = 0;                                                                                 \
        int const doubled = nk_attention_double_scores_blackwell_(schedule, item->tiles);                              \
        if (doubled) {                                                                                                 \
            /* The next panel's S lands in the other slot while this one's softmax runs */                             \
            keys = nk_attention_acquire_stage_blackwell_(schedule, control, stages_address, progress);                 \
            nk_attention_issue_scores_##input_type_name##_##isa_suffix##_(kernel, schedule, control, queries_address,  \
                                                                          keys, tensor_memory, 0, 0, elected);         \
            for (unsigned panel = 0; panel < item->panels; ++panel) {                                                  \
                values = nk_attention_acquire_stage_blackwell_(schedule, control, stages_address, progress);           \
                if (panel + 1 < item->panels) {                                                                        \
                    keys = nk_attention_acquire_stage_blackwell_(schedule, control, stages_address, progress);         \
                    nk_attention_issue_scores_##input_type_name##_##isa_suffix##_(                                     \
                        kernel, schedule, control, queries_address, keys, tensor_memory, 0, (panel + 1) % 2 * 2,       \
                        elected);                                                                                      \
                }                                                                                                      \
                nk_attention_issue_values_##input_type_name##_##isa_suffix##_(                                         \
                    schedule, control, values, tensor_memory, 0, panel % 2 * 2, panel == 0, panel + 1 == item->panels, \
                    elected, progress);                                                                                \
                if (panel + 1 < item->panels && elected) nk_attention_commit_blackwell_(&control->multiplied);         \
            }                                                                                                          \
        }                                                                                                              \
        for (unsigned panel = 0; !doubled && panel <= item->panels; ++panel) {                                         \
            int const scoring = panel < item->panels;                                                                  \
            if (scoring) {                                                                                             \
                keys = nk_attention_acquire_stage_blackwell_(schedule, control, stages_address, progress);             \
                nk_attention_issue_scores_##input_type_name##_##isa_suffix##_(                                         \
                    kernel, schedule, control, queries_address, keys, tensor_memory, 0, 0, elected);                   \
            }                                                                                                          \
            if (item->tiles == 2 && panel > 0)                                                                         \
                nk_attention_issue_values_##input_type_name##_##isa_suffix##_(                                         \
                    schedule, control, values, tensor_memory, 1, 1, panel == 1, panel == item->panels, elected,        \
                    progress);                                                                                         \
            if (!scoring) break;                                                                                       \
            if (item->tiles == 2)                                                                                      \
                nk_attention_issue_scores_##input_type_name##_##isa_suffix##_(                                         \
                    kernel, schedule, control, queries_address, keys, tensor_memory, 1, 1, elected);                   \
            values = nk_attention_acquire_stage_blackwell_(schedule, control, stages_address, progress);               \
            nk_attention_issue_values_##input_type_name##_##isa_suffix##_(schedule, control, values, tensor_memory, 0, \
                                                                          0, panel == 0, panel + 1 == item->panels,    \
                                                                          elected, progress);                          \
        }                                                                                                              \
        for (unsigned tile = 0; elected && tile < item->tiles; ++tile)                                                 \
            nk_attention_commit_blackwell_(&control->queries_read[tile]);                                              \
    }                                                                                                                  \
    /* Issues every item's products from the MMA warp. */                                                              \
    NUMKONG_DEVICE void nk_attention_multiply_##input_type_name##_##isa_suffix##_(                                     \
        nk_attention_kernel_blackwell_t const *kernel, nk_attention_schedule_blackwell_t const *schedule,              \
        nk_attention_control_blackwell_t *control, unsigned char *queries, unsigned char *stages,                      \
        nk_u32_t tensor_memory, nk_attention_tile_arguments_blackwell_t const *arguments) {                            \
        nk_attention_progress_blackwell_t progress = {0, 0, {0, 0}, {0, 0}, {0, 0}};                                   \
        nk_u32_t const queries_address = nk_shared_address_ampere_(queries);                                           \
        nk_u32_t const stages_address = nk_shared_address_ampere_(stages);                                             \
        for (nk_u32_t sequence = 0;; ++sequence) {                                                                     \
            nk_attention_work_t work;                                                                                  \
            nk_attention_take_work_blackwell_(control, sequence, &work);                                               \
            if (!work.row_count) return;                                                                               \
            nk_attention_item_blackwell_t item;                                                                        \
            nk_attention_item_blackwell_(&arguments->attention, schedule, &work, &item);                               \
            if (item.panels)                                                                                           \
                nk_attention_multiply_item_##input_type_name##_##isa_suffix##_(                                        \
                    kernel, schedule, control, &item, queries_address, stages_address, tensor_memory, &progress);      \
        }                                                                                                              \
    }                                                                                                                  \
    /* The largest of this thread's 128 scores of a panel at tensor-memory address scores, scaled to                   \
     * base 2, over the keys set in visible when the panel is masked and over all of them otherwise,                   \
     * where only the extreme score gets scaled. */                                                                    \
    NUMKONG_DEVICE nk_f32_t nk_attention_panel_maximum_##input_type_name##_##isa_suffix##_(                            \
        nk_u32_t scores, nk_f32_t scale2, int masked, nk_u32_t const visible[4]) {                                     \
        if (masked) return nk_attention_masked_maximum_blackwell_(scores, scale2, visible);                            \
        /* A negative scale makes the smallest score the largest */                                                    \
        int const smallest = scale2 < 0;                                                                               \
        nk_f32_t extreme = nk_tmem_load_extreme_x32_##isa_suffix##_(scores, smallest);                                 \
        _Pragma("unroll") for (unsigned chunk = 32; chunk < nk_attention_panel_blackwell_k; chunk += 32) {             \
            nk_f32_t const other = nk_tmem_load_extreme_x32_##isa_suffix##_(scores + chunk, smallest);                 \
            extreme = smallest ? fminf(extreme, other) : fmaxf(extreme, other);                                        \
        }                                                                                                              \
        return extreme * scale2;                                                                                       \
    }                                                                                                                  \
    /* Writes P over the first columns of this thread's row of one panel of S, at tensor-memory                        \
     * address scores, as 2^(s · scale2 − subtrahend), zero past the keys set in visible when the                   \
     * panel is masked, signalling weighed_half once its first half lands, and returning the F32                       \
     * pair of sums of the weights P · V sees. */                                                                     \
    NUMKONG_DEVICE nk_u64_t nk_attention_weigh_chunks_##input_type_name##_##isa_suffix##_(                             \
        nk_attention_kernel_blackwell_t const *kernel, nk_u64_t *weighed_half, nk_f32_t scale2, nk_f32_t subtrahend,   \
        nk_u32_t scores, int masked, nk_u32_t const visible[4]) {                                                      \
        nk_u64_t const multiplier = nk_f32x2_blackwell_(scale2, scale2);                                               \
        /* E4M3 weights take 256 · p, as an exponent 8 higher */                                                       \
        nk_f32_t const amplifier = kernel->element_bytes == 1 && !kernel->widens ? 8.0f : 0.0f;                        \
        nk_u64_t const addend = nk_f32x2_blackwell_(amplifier - subtrahend, amplifier - subtrahend);                   \
        nk_u64_t sums = 0;                                                                                             \
        _Pragma("unroll 1") for (unsigned chunk = 0; chunk < nk_attention_panel_blackwell_k; chunk += 32) {            \
            nk_u32_t bits[32], words[16];                                                                              \
            nk_tmem_load_x32_blackwell_(scores + chunk, bits);                                                         \
            _Pragma("unroll") for (unsigned group = 0; group < 8; ++group) {                                           \
                nk_f32_t probabilities[4];                                                                             \
                nk_u32_t packed[2];                                                                                    \
                _Pragma("unroll") for (unsigned pair = 0; pair < 2; ++pair) {                                          \
                    unsigned const offset = group * 4 + pair * 2;                                                      \
                    nk_u64_t const exponents = nk_f32x2_fma_blackwell_(                                                \
                        ((nk_u64_t)bits[offset + 1] << 32) | bits[offset], multiplier, addend);                        \
                    probabilities[pair * 2] = nk_f32_exp2_cuda_(nk_f32x2_low_blackwell_(exponents));                   \
                    probabilities[pair * 2 + 1] = nk_f32_exp2_cuda_(nk_f32x2_high_blackwell_(exponents));              \
                }                                                                                                      \
                _Pragma("unroll") for (unsigned index = 0; masked && index < 4; ++index) {                             \
                    if (!((visible[chunk / 32] >> (group * 4 + index)) & 1)) probabilities[index] = 0;                 \
                }                                                                                                      \
                weights_fn(probabilities, packed, &sums);                                                              \
                if (kernel->element_bytes == 1 && !kernel->widens) words[group] = packed[0];                           \
                else words[group * 2] = packed[0], words[group * 2 + 1] = packed[1];                                   \
            }                                                                                                          \
            if (kernel->element_bytes == 1 && !kernel->widens) nk_tmem_store_x8_blackwell_(scores + chunk / 4, words); \
            else nk_tmem_store_x16_blackwell_(scores + chunk / 2, words);                                              \
            if (chunk + 32 == nk_attention_panel_blackwell_k / 2) {                                                    \
                nk_tmem_wait_store_blackwell_();                                                                       \
                nk_tmem_fence_before_blackwell_();                                                                     \
                nk_attention_warp_arrive_blackwell_(weighed_half);                                                     \
            }                                                                                                          \
        }                                                                                                              \
        return sums;                                                                                                   \
    }                                                                                                                  \
    /* Turns this thread's row of one panel of S, at tensor-memory address scores, into P over S's                     \
     * first columns: finds the row's new maximum, hands the factor that rescales O to the                             \
     * correction warps when the panel rescales, and adds the weights P · V sees to row_sum. A                        \
     * masked panel, which the whole warp agrees on, sees only the keys set in visible. */                             \
    NUMKONG_DEVICE void nk_attention_weigh_panel_##input_type_name##_##isa_suffix##_(                                  \
        nk_attention_kernel_blackwell_t const *kernel, nk_attention_schedule_blackwell_t const *schedule,              \
        nk_attention_control_blackwell_t *control, nk_u32_t scores, unsigned slot, int masked,                         \
        nk_u32_t const visible[4], int rescales, nk_f32_t *row_max, nk_f32_t *row_sum) {                               \
        unsigned const tile_row = threadIdx.x % nk_attention_tile_rows_blackwell_k;                                    \
        nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();                                     \
        nk_f32_t const panel_max = nk_attention_panel_maximum_##input_type_name##_##isa_suffix##_(                     \
            scores, schedule->scale2, masked, visible);                                                                \
        nk_f32_t const new_max = panel_max > *row_max + schedule->rescale_threshold ? panel_max : *row_max;            \
        /* With every key so far masked, subtracting 0 keeps exp2(−∞ − max) from being NaN */                          \
        nk_f32_t const subtrahend = new_max == negative_infinity ? 0.0f : new_max;                                     \
        nk_f32_t const factor = new_max == *row_max ? 1.0f : nk_f32_exp2_cuda_(*row_max - subtrahend);                 \
        *row_max = new_max;                                                                                            \
        if (rescales) {                                                                                                \
            control->factors[slot][tile_row] = factor;                                                                 \
            nk_attention_warp_arrive_blackwell_(&control->rescaling[slot]);                                            \
        }                                                                                                              \
        nk_u64_t *const weighed_half = &control->weighed_half[slot];                                                   \
        nk_u64_t const sums = masked ? nk_attention_weigh_chunks_##input_type_name##_##isa_suffix##_(                  \
                                           kernel, weighed_half, schedule->scale2, subtrahend, scores, 1, visible)     \
                                     : nk_attention_weigh_chunks_##input_type_name##_##isa_suffix##_(                  \
                                           kernel, weighed_half, schedule->scale2, subtrahend, scores, 0, visible);    \
        *row_sum = *row_sum * factor + (nk_f32x2_low_blackwell_(sums) + nk_f32x2_high_blackwell_(sums));               \
        nk_tmem_wait_store_blackwell_();                                                                               \
        nk_tmem_fence_before_blackwell_();                                                                             \
        nk_attention_warp_arrive_blackwell_(&control->weighed[slot]);                                                  \
    }                                                                                                                  \
    /* Stages Q and turns every panel of S into P for one tile of every item, a row per thread, from                   \
     * the tile's softmax warpgroup, then hands the row sums to the correction warps. */                               \
    NUMKONG_DEVICE void nk_attention_softmax_##input_type_name##_##isa_suffix##_(                                      \
        nk_attention_kernel_blackwell_t const *kernel, nk_attention_schedule_blackwell_t const *schedule,              \
        nk_attention_control_blackwell_t *control, unsigned char *queries, nk_u32_t tensor_memory,                     \
        nk_attention_tile_arguments_blackwell_t const *arguments) {                                                    \
        nk_attention_arguments_t const *const attention = &arguments->attention;                                       \
        unsigned const tile = threadIdx.x / nk_attention_tile_rows_blackwell_k;                                        \
        unsigned const tile_row = threadIdx.x % nk_attention_tile_rows_blackwell_k;                                    \
        nk_u32_t const scores = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32,                           \
                                                          nk_attention_scores_column_blackwell_k);                     \
        nk_u32_t items = 0, uses = 0, scorings[3] = {0, 0, 0};                                                         \
        for (nk_u32_t sequence = 0;; ++sequence) {                                                                     \
            nk_attention_work_t work;                                                                                  \
            nk_attention_take_work_blackwell_(control, sequence, &work);                                               \
            if (!work.row_count) return;                                                                               \
            nk_attention_item_blackwell_t item;                                                                        \
            nk_attention_item_blackwell_(attention, schedule, &work, &item);                                           \
            nk_u32_t const first_use = uses;                                                                           \
            uses += 2 * item.panels;                                                                                   \
            int const doubled = nk_attention_double_scores_blackwell_(schedule, item.tiles);                           \
            nk_u32_t const scorings_before[3] = {scorings[0], scorings[1], scorings[2]};                               \
            if (doubled) scorings[0] += item.panels - item.panels / 2, scorings[2] += item.panels / 2;                 \
            else                                                                                                       \
                for (unsigned index = 0; index < item.tiles; ++index) scorings[index] += item.panels;                  \
            if (!item.panels || tile >= item.tiles) continue;                                                          \
            /* The last tile's S of a panel ends every read of its K */                                                \
            int const releases_keys = tile + 1 == item.tiles && tile_row == 0;                                         \
            nk_u32_t const queries_parity = (items++ & 1) ^ 1;                                                         \
            if (!nk_attention_queries_mapped_blackwell_(schedule, &work)) {                                            \
                nk_attention_wait_blackwell_(&control->queries_read[tile], queries_parity);                            \
                nk_attention_stage_queries_blackwell_(schedule, attention, &work,                                      \
                                                      queries + tile * schedule->queries_bytes, tile);                 \
                nk_attention_warp_arrive_blackwell_(&control->queries_staged[tile]);                                   \
            }                                                                                                          \
            nk_size_t const local = tile * nk_attention_tile_rows_blackwell_k + tile_row;                              \
            nk_i64_t const row = nk_attention_row_position_simt_(&work, local, item.length);                           \
            /* Tensor-memory accesses take the whole warp, so its rows classify a panel jointly */                     \
            nk_size_t const warp_local = local - tile_row % 32;                                                        \
            nk_size_t const warp_rows = work.row_count <= warp_local       ? 0                                         \
                                        : work.row_count - warp_local < 32 ? work.row_count - warp_local               \
                                                                           : 32;                                       \
            nk_i64_t const warp_first = nk_attention_row_position_simt_(&work, warp_local, item.length);               \
            nk_f32_t row_max = nk_attention_negative_infinity_simt_(), row_sum = 0;                                    \
            for (unsigned panel = 0; panel < item.panels; ++panel) {                                                   \
                unsigned const slot = doubled ? panel % 2 * 2 : tile;                                                  \
                nk_u32_t const scoring = scorings_before[slot] + (doubled ? panel / 2 : panel);                        \
                nk_size_t const position = (nk_size_t)(item.panel_first + panel) * nk_attention_panel_blackwell_k;     \
                int const masked = nk_attention_tile_coverage_simt_(schedule->band, warp_first, warp_rows, position,   \
                                                                    nk_attention_panel_blackwell_k,                    \
                                                                    item.length) != nk_diagonal_band_inside_k;         \
                nk_u32_t visible[4] = {0, 0, 0, 0};                                                                    \
                for (unsigned chunk = 0; masked && local < work.row_count && chunk < 4; ++chunk)                       \
                    visible[chunk] = nk_diagonal_band_row_mask_simt_(schedule->band, row, position + chunk * 32,       \
                                                                     item.length);                                     \
                nk_attention_wait_blackwell_(&control->scored[slot], scoring & 1);                                     \
                nk_tmem_fence_after_blackwell_();                                                                      \
                if (releases_keys) nk_attention_release_stage_blackwell_(schedule, control, first_use + 2 * panel);    \
                nk_attention_weigh_panel_##input_type_name##_##isa_suffix##_(                                          \
                    kernel, schedule, control, scores + (slot ? nk_attention_tile_rows_blackwell_k : 0), slot, masked, \
                    visible, panel != 0, &row_max, &row_sum);                                                          \
            }                                                                                                          \
            /* A row's weights sum in units of the weights a probability of one takes */                               \
            nk_f32_t const unit = kernel->element_bytes == 1 && !kernel->widens ? 256.0f : 1.0f;                       \
            nk_f32_t const unit_probabilities[4] = {unit, 0, 0, 0};                                                    \
            nk_u32_t unit_packed[2];                                                                                   \
            nk_u64_t unit_sums = 0;                                                                                    \
            weights_fn(unit_probabilities, unit_packed, &unit_sums);                                                   \
            nk_f32_t const unit_sum = nk_f32x2_low_blackwell_(unit_sums) + nk_f32x2_high_blackwell_(unit_sums);        \
            /* A one-panel item may finish before correction warps read the last sums */                               \
            nk_attention_wait_blackwell_(&control->sums_read[tile], items & 1);                                        \
            control->sums[tile][tile_row] = row_sum;                                                                   \
            /* A paired item's correction warps store the joint log-sum-exp */                                         \
            if (item.shared)                                                                                           \
                control->log_sum_exps[tile][tile_row] = nk_attention_log_sum_exp_simt_(row_max, row_sum, unit_sum);    \
            nk_attention_warp_arrive_blackwell_(&control->summed[tile]);                                               \
            nk_f32_t *const log_sum_exp_slot = local < work.row_count && !item.shared                                  \
                                                   ? nk_attention_log_sum_exp_slot_simt_(attention, &work, local)      \
                                                   : NUMKONG_NULL;                                                     \
            if (log_sum_exp_slot) *log_sum_exp_slot = nk_attention_log_sum_exp_simt_(row_max, row_sum, unit_sum);      \
        }                                                                                                              \
    }                                                                                                                  \
    /* Every work item of one launch: the roles above, the loads and the corrections. */                               \
    NUMKONG_DEVICE void nk_attention_tile_##input_type_name##_##isa_suffix##_(                                         \
        nk_attention_width_t width, nk_attention_mask_t mask,                                                          \
        nk_attention_tile_arguments_blackwell_t const *arguments) {                                                    \
        extern __shared__ unsigned char nk_attention_dynamic_shared_blackwell_[];                                      \
        nk_attention_kernel_blackwell_t const kernel = {input_bytes, input_widens, width, mask, format};               \
        nk_attention_schedule_blackwell_t const schedule = nk_attention_schedule_blackwell_(&kernel, arguments);       \
        /* The swizzled tiles need 1024-byte alignment, which dynamic shared memory lacks */                           \
        unsigned char *const queries = nk_shared_aligned_ampere_(nk_attention_dynamic_shared_blackwell_, 1024);        \
        unsigned char *const stages = queries + schedule.tiles * schedule.queries_bytes;                               \
        nk_attention_control_blackwell_t *const control =                                                              \
            (nk_attention_control_blackwell_t *)(stages + schedule.ring_stages * schedule.stage_bytes);                \
        nk_u32_t const tensor_memory = nk_attention_initialize_blackwell_(&schedule, control);                         \
        unsigned const warp = threadIdx.x >> 5;                                                                        \
        if (warp < nk_attention_correct_warp_blackwell_k)                                                              \
            nk_attention_softmax_##input_type_name##_##isa_suffix##_(&kernel, &schedule, control, queries,             \
                                                                     tensor_memory, arguments);                        \
        else if (warp < nk_attention_multiply_warp_blackwell_k)                                                        \
            nk_attention_correct_blackwell_(&schedule, control, tensor_memory, arguments);                             \
        else if (warp == nk_attention_multiply_warp_blackwell_k)                                                       \
            nk_attention_multiply_##input_type_name##_##isa_suffix##_(&kernel, &schedule, control, queries, stages,    \
                                                                      tensor_memory, arguments);                       \
        else nk_attention_load_blackwell_(&schedule, control, queries, stages, arguments);                             \
        nk_attention_release_blackwell_(tensor_memory);                                                                \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_attention_threads_blackwell_k, 1)                                      \
        nk_attention_packed_##input_type_name##_narrow_##isa_suffix##_kernel_(                                         \
            __grid_constant__ nk_attention_tile_arguments_blackwell_t const arguments) {                               \
        nk_attention_tile_##input_type_name##_##isa_suffix##_(nk_attention_width_128_k, nk_attention_mask_none_k,      \
                                                              &arguments);                                             \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_attention_threads_blackwell_k, 1)                                      \
        nk_attention_packed_##input_type_name##_wide_##isa_suffix##_kernel_(                                           \
            __grid_constant__ nk_attention_tile_arguments_blackwell_t const arguments) {                               \
        nk_attention_tile_##input_type_name##_##isa_suffix##_(nk_attention_width_256_k, nk_attention_mask_none_k,      \
                                                              &arguments);                                             \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_attention_threads_blackwell_k, 1)                                      \
        nk_attention_packed_##input_type_name##_narrow_masked_##isa_suffix##_kernel_(                                  \
            __grid_constant__ nk_attention_tile_arguments_blackwell_t const arguments) {                               \
        nk_attention_tile_##input_type_name##_##isa_suffix##_(nk_attention_width_128_k,                                \
                                                              nk_attention_mask_diagonal_band_k, &arguments);          \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_attention_threads_blackwell_k, 1)                                      \
        nk_attention_packed_##input_type_name##_wide_masked_##isa_suffix##_kernel_(                                    \
            __grid_constant__ nk_attention_tile_arguments_blackwell_t const arguments) {                               \
        nk_attention_tile_##input_type_name##_##isa_suffix##_(nk_attention_width_256_k,                                \
                                                              nk_attention_mask_diagonal_band_k, &arguments);          \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                   \
        nk_attention_packed_##input_type_name##_fallback_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {  \
        nk_attention_fallback_##input_value_type##_cuda_(&arguments);                                                  \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_attention_packed_##input_type_name##_##isa_suffix(                                      \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,          \
        nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,                      \
        nk_##input_value_type##_t const *queries, nk_size_t query_stride, void const *key_value_packed,                \
        nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,  \
        nk_stream_t stream) {                                                                                          \
        int const masked = keys_before != NUMKONG_SIZE_MAX || keys_after != NUMKONG_SIZE_MAX;                          \
        return nk_attention_launch_##input_type_name##_blackwell_(                                                     \
            masked ? (void const *)nk_attention_packed_##input_type_name##_narrow_masked_##isa_suffix##_kernel_        \
                   : (void const *)nk_attention_packed_##input_type_name##_narrow_##isa_suffix##_kernel_,              \
            masked ? (void const *)nk_attention_packed_##input_type_name##_wide_masked_##isa_suffix##_kernel_          \
                   : (void const *)nk_attention_packed_##input_type_name##_wide_##isa_suffix##_kernel_,                \
            (void const *)nk_attention_packed_##input_type_name##_fallback_##isa_suffix##_kernel_, queries,            \
            key_value_packed, output, log_sum_exp, head_count, key_value_head_count, depth, query_offsets,             \
            query_token_count, query_stride, output_stride, scale, keys_before, keys_after, tasks_begin, tasks_end,    \
            stream);                                                                                                   \
    }

#pragma endregion Attention Macros

/*  Later generations read the helpers above and emit only their own kernels. */
#if NUMKONG_TARGET_BLACKWELL

#pragma region BF16

nk_define_attention_pack_size_simt_(bf16, blackwell, 2)
nk_define_attention_packed_shape_cuda_(bf16, blackwell)
nk_define_attention_pack_blackwell_(bf16, blackwell, bf16)
nk_define_attention_tma_blackwell_(bf16, blackwell, bf16, 2, 0, nk_mma_f16_blackwell_, nk_mma_f16_tmem_blackwell_,
                                   nk_attention_weights_bf16_blackwell_, /*format=*/1)

/** Both BF16 backward kernels: tensor cores when the launch gives them shared memory, the
 *  @c cuda kernels' loops when it does not. */
static __global__ void __launch_bounds__(nk_attention_backward_threads_blackwell_k, 1)
    nk_attention_backward_keys_bf16_blackwell_kernel_(
        __grid_constant__ nk_attention_backward_maps_blackwell_t const maps) {
    if (maps.single) nk_attention_backward_single_blackwell_(&maps);
    else if (nk_dynamic_shared_bytes_ampere_()) nk_attention_backward_keys_blackwell_(&maps);
    else nk_attention_backward_keys_bf16_cuda_(&maps.arguments);
}
static __global__ void __launch_bounds__(nk_attention_backward_threads_blackwell_k, 1)
    nk_attention_backward_queries_bf16_blackwell_kernel_(
        __grid_constant__ nk_attention_backward_maps_blackwell_t const maps) {
    if (nk_dynamic_shared_bytes_ampere_()) nk_attention_backward_queries_blackwell_(&maps);
    else nk_attention_backward_queries_bf16_cuda_(&maps.arguments);
}
nk_define_attention_backward_cuda_(bf16, blackwell, bf16, nk_attention_backward_launch_blackwell_)

#pragma endregion BF16

#pragma region F16

nk_define_attention_pack_size_simt_(f16, blackwell, 2)
nk_define_attention_packed_shape_cuda_(f16, blackwell)
nk_define_attention_pack_blackwell_(f16, blackwell, f16)
nk_define_attention_tma_blackwell_(f16, blackwell, f16, 2, 0, nk_mma_f16_blackwell_, nk_mma_f16_tmem_blackwell_,
                                   nk_attention_weights_f16_blackwell_, /*format=*/0)

#pragma endregion F16

#pragma region E4M3

nk_define_attention_pack_size_simt_(e4m3, blackwell, 1)
nk_define_attention_packed_shape_cuda_(e4m3, blackwell)
nk_define_attention_pack_blackwell_(e4m3, blackwell, e4m3)
nk_define_attention_tma_blackwell_(e4m3, blackwell, e4m3, 1, 0, nk_mma_f8f6f4_blackwell_, nk_mma_f8f6f4_tmem_blackwell_,
                                   nk_attention_weights_e4m3_blackwell_, /*format=*/0)

#pragma endregion E4M3

#pragma region I8

NUMKONG_API nk_status_t nk_attention_pack_size_i8_blackwell(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    *bytes = nk_attention_pack_size_simt_(key_value_head_count, depth, token_count, segment_count,
                                          nk_attention_pack_element_bytes_i8_blackwell_(depth));
    return nk_success_k;
}

static __global__ void nk_attention_pack_i8_blackwell_kernel_(unsigned char const *keys, unsigned char const *values,
                                                              nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                              nk_size_t segment_count, nk_size_t key_stride,
                                                              nk_size_t value_stride, unsigned char *packed,
                                                              nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_pack_payload_i8_blackwell_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                            segment_count, key_stride, value_stride, packed, tasks_begin, tasks_end);
}

NUMKONG_API nk_status_t nk_attention_pack_i8_blackwell(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_i8_t const *keys,
                                                       nk_size_t key_stride, nk_i8_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_pack_launch_cuda_((void const *)nk_attention_pack_i8_blackwell_kernel_, nk_cap_blackwell_k,
                                          nk_attention_pack_element_bytes_i8_blackwell_(depth), keys, values,
                                          key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                          key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, stream);
}

nk_define_attention_packed_shape_cuda_(i8, blackwell)
nk_define_attention_tma_blackwell_(i8, blackwell, i8, 1, 1, nk_mma_f16_blackwell_, nk_mma_f16_tmem_blackwell_,
                                   nk_attention_weights_u8_blackwell_, /*format=*/0)

#pragma endregion I8

#endif // NUMKONG_TARGET_BLACKWELL

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_BLACKWELL_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_ATTENTION_BLACKWELL_CUH
