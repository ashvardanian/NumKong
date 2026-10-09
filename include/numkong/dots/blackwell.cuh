/**
 *  @file include/numkong/dots/blackwell.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated Batched Dot Products for the NVIDIA compute capability 10.x family.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/ampere.cuh
 *
 *  Persistent blocks of ten warps own @b [128,128] output tiles. One thread streams 128-byte depth
 *  slabs of A and B through a six-stage ring with the Tensor Memory Accelerator, which swizzles
 *  them into the canonical 128-byte layout and zero-fills every edge, and one thread issues
 *  `tcgen05.mma` over shared-memory descriptors into one of two F32 accumulators in tensor memory,
 *  so four epilogue warps drain one tile while the next accumulates, and four stage warps prepare
 *  the stages the products read next. Every dtype multiplies 32 bytes of depth per instruction:
 *  BF16 and F16 as @c kind::f16, Float8 as @c kind::f8f6f4, E2M1 as the block-scaled
 *  @c kind::mxf4 with every scale at 2⁰, and Float6 as the Float8 of the next exponent width,
 *  which is its own code with the sign moved up and a power of two the output undoes. The stage
 *  warps move A's signs in shared memory, and the pack moves B's. Integers, which this family has
 *  no tensor-core kind for, load as narrower slabs that the stage warps widen to exact F16 values
 *  for @c kind::f16, in chunks of depth short enough for F32 to sum them exactly.
 *  Wider floats stay on the Ampere and @c cuda capabilities. Only the "100f" family code carries
 *  these instructions, so every other device pass traps.
 */
#ifndef NUMKONG_DOTS_BLACKWELL_CUH
#define NUMKONG_DOTS_BLACKWELL_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_CUDA_BLACKWELL_

#include <cuda.h> // `CUtensorMap`, `cuTensorMapEncodeTiled`

#include "numkong/dots/ada.cuh" // `nk_f6x4_to_f8x4_ada_`
#include "numkong/dots/ampere.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_cross_threads_blackwell_k = 320,
    nk_cross_rows_blackwell_k = 128,
    nk_cross_columns_blackwell_k = 128,
    nk_cross_slab_bytes_blackwell_k = 128,
    nk_cross_a_bytes_blackwell_k = nk_cross_rows_blackwell_k * nk_cross_slab_bytes_blackwell_k,
    nk_cross_stage_bytes_blackwell_k = (nk_cross_rows_blackwell_k + nk_cross_columns_blackwell_k) *
                                       nk_cross_slab_bytes_blackwell_k,
    nk_cross_ring_bytes_blackwell_k = 196608,
    nk_cross_stages_blackwell_k = nk_cross_ring_bytes_blackwell_k / nk_cross_stage_bytes_blackwell_k,
    nk_cross_tensor_columns_blackwell_k = 512,
    // Tensor-memory columns: two accumulators, then per-stage block scales or running integer sums,
    // the statistics of two tiles in turn, and in the last 32 the unit scales of unscaled kinds.
    nk_cross_scale_slots_column_blackwell_k = 2 * nk_cross_columns_blackwell_k,
    nk_cross_unit_scales_column_blackwell_k = nk_cross_tensor_columns_blackwell_k - 32,
    nk_cross_statistics_column_blackwell_k = nk_cross_unit_scales_column_blackwell_k - 32,
    // FP4 pairs keep two 224-column accumulators, leaving 64 columns for one slab's scales.
    nk_cross_pair_scales_column_blackwell_k = 2 * 224,
};

nk_static_assert_(nk_cross_scale_slots_column_blackwell_k + 32 * nk_cross_stages_blackwell_k <=
                      nk_cross_statistics_column_blackwell_k,
                  nk_cross_scale_slots_fit_tensor_memory_blackwell);
nk_static_assert_(nk_cross_rows_blackwell_k == nk_cross_columns_blackwell_k,
                  nk_cross_drain_threads_own_one_column_blackwell);

/** Barriers and the epilogue warps' exchange, past the ring of stages. Per stage, @c loaded
 *  completes with the copies, @c ready with the stage warps that stage it, and @c consumed with the
 *  products and any stage warps that only read it; per accumulator, @c accumulated completes with
 *  the products and @c drained with the epilogue; per statistics slot, @c measured completes with
 *  the stage warps that fill it and @c collected with the epilogue warps that read it. */
typedef struct {
    nk_u64_t loaded[nk_cross_stages_blackwell_k];
    nk_u64_t ready[nk_cross_stages_blackwell_k];
    nk_u64_t consumed[nk_cross_stages_blackwell_k];
    nk_u64_t accumulated[2];
    nk_u64_t drained[2];
    nk_u64_t measured[2];
    nk_u64_t collected[2];
    nk_u32_t tensor_memory;
    nk_u32_t column_norms[nk_cross_columns_blackwell_k];
    nk_u32_t column_byte_sums[nk_cross_columns_blackwell_k];
} nk_cross_control_blackwell_t;

enum {
    // Each epilogue warp stages 32 rows of 32 outputs, 128 swizzled bytes each, in two boxes.
    nk_cross_output_box_bytes_blackwell_k = 32 * 128,
    nk_cross_output_staging_bytes_blackwell_k = 4 * 2 * nk_cross_output_box_bytes_blackwell_k,
    nk_cross_shared_bytes_blackwell_k = 1024 + nk_cross_ring_bytes_blackwell_k +
                                        nk_cross_output_staging_bytes_blackwell_k +
                                        sizeof(nk_cross_control_blackwell_t),
};

/** Everything one launch shares: tensor maps of A and B rows, 128-byte boxes of 128 rows, the
 *  output's map in boxes of 32 rows of 32 outputs, unless it is misaligned for one, the shapes and
 *  output every tile takes, and how many row tiles sweep the columns together. Passed as a grid
 *  constant, so the maps keep an address. */
typedef struct {
    CUtensorMap a_map;
    CUtensorMap b_map;
    CUtensorMap c_map;
    int c_mapped;
    nk_cross_tile_arguments_t tile;
    nk_size_t group_rows;
} nk_cross_tile_arguments_blackwell_t;

/** How the stage warps prepare a stage before the products read it, fixed by each dtype's tile. */
typedef enum {

    /** The products read the copies as they land. */
    nk_cross_staging_none_k = 0,

    /** Codes are rewritten in place into the codes the tensor cores multiply. */
    nk_cross_staging_widen_k,

    /** Narrower boxes of integer codes widen into F16 slabs ahead of them. */
    nk_cross_staging_expand_k,

    /** The slab's block scales are copied into tensor memory. */
    nk_cross_staging_block_scales_k,
} nk_cross_staging_t;

/** What a kernel and its launch shape make of the ring and the depth loop. */
typedef struct {
    nk_cross_staging_t staging;

    /** What the tile computes, and over which band of the output. */
    nk_cross_metric_t metric;
    nk_diagonal_band_t band;

    /** Undoes the power of two a widening introduced, or 1. */
    nk_f32_t output_scale;

    /** How the squared norms are stored, and the factor undoing their widening's power of two. */
    nk_cross_norm_t norm;
    nk_f32_t norm_scale;

    /** Whether the stage warps visit every stage, to prepare it or to square its rows. */
    int visits_stages;

    /** Whether U8 codes widen offset by −128, which both rows' byte sums restore. */
    int offsets;

    /** Whether the stage warps hand each tile's norms or byte sums to the epilogue warps. */
    int publishes;

    /** Bytes of depth one copy lands per row, and where in a stage the copies land. */
    unsigned box_bytes, box_offset;

    /** Bytes of a stage, block scales' atoms past both boxes included, and the ring's stages. */
    unsigned stage_bytes, ring_stages;

    /** The scale bytes per row and slab of a block-scaled dtype, and its blocks per row. */
    unsigned scale_bytes;
    nk_size_t blocks;

    /** Slabs per tile, slabs per chunk whose F32 sums stay exact, and chunks per tile. */
    nk_size_t slabs, chunk_slabs, chunks;

    /** Whether a pair of blocks shares each tile, which the dot products of dtypes the tensor cores
     *  read as loaded do, and this block's rank in it, which picks its 128 rows and its @c b_rows
     *  of the B rows. */
    int paired;
    nk_u32_t pair_rank;
    unsigned b_rows;

    /** The shape of one output tile, whose columns each accumulator spans, and the tiles this block
     *  walks: from @c first_tile, @c tile_stride apart. */
    unsigned tile_rows, tile_columns;
    nk_size_t first_tile, tile_stride;
} nk_cross_schedule_blackwell_t;

/** Squares and byte sums one stage thread gathers over its A row and its B row. */
typedef struct {
    nk_u32_t row_integer_norm, column_integer_norm;
    nk_f32_t row_real_norm, column_real_norm;
    nk_u32_t row_byte_sum, column_byte_sum;
} nk_cross_statistics_blackwell_t;

#pragma endregion Configuration

#pragma region Instructions

NUMKONG_DEVICE void nk_mbarrier_init_blackwell_(nk_u32_t barrier, nk_u32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier), "r"(count) : "memory");
}

NUMKONG_DEVICE void nk_mbarrier_init_fence_blackwell_(void) {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

/* Spins until the phase of parity @p parity completes; a fresh barrier reads parity 1 as done. */
NUMKONG_DEVICE void nk_mbarrier_wait_blackwell_(nk_u32_t barrier, nk_u32_t parity) {
    nk_u32_t done;
    do {
        asm volatile("{\n.reg .pred ready;\n"                                      //
                     "mbarrier.try_wait.parity.shared::cta.b64 ready, [%1], %2;\n" //
                     "selp.u32 %0, 1, 0, ready;\n}\n"
                     : "=r"(done)
                     : "r"(barrier), "r"(parity)
                     : "memory");
    } while (!done);
}

NUMKONG_DEVICE void nk_mbarrier_arrive_blackwell_(nk_u32_t barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(barrier) : "memory");
}

/* Arrives and expects @p bytes more from asynchronous copies before the phase completes. */
NUMKONG_DEVICE void nk_mbarrier_expect_bytes_blackwell_(nk_u32_t barrier, nk_u32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier), "r"(bytes) : "memory");
}

/* Copies the box at byte @p column and row @p row of the tensor @p map describes, zero-filling
 *  what lies outside it, and completes its bytes on @p barrier. */
NUMKONG_DEVICE void nk_load_box_blackwell_(nk_u32_t destination, void const *map, nk_u32_t barrier, nk_i32_t column,
                                           nk_i32_t row) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes " //
                 "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(destination),
                 "l"(map), "r"(column), "r"(row), "r"(barrier)
                 : "memory");
}

NUMKONG_DEVICE void nk_prefetch_map_blackwell_(void const *map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(map) : "memory");
}

/* Copies the box at byte @p column and row @p row of the tensor @p map describes from @p source,
 *  clipping what lies outside it, as this thread's next bulk group member. */
NUMKONG_DEVICE void nk_store_box_blackwell_(void const *map, nk_u32_t source, nk_i32_t column, nk_i32_t row) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n" ::"l"(map),
                 "r"(column), "r"(row), "r"(source)
                 : "memory");
}

/* Closes this thread's pending bulk stores into a group. */
NUMKONG_DEVICE void nk_bulk_commit_blackwell_(void) { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }

/* Waits until at most one of this thread's bulk groups still reads shared memory. */
NUMKONG_DEVICE void nk_bulk_wait_reads_but_one_blackwell_(void) {
    asm volatile("cp.async.bulk.wait_group.read 1;\n" ::: "memory");
}

/* Waits until every bulk group of this thread completes. */
NUMKONG_DEVICE void nk_bulk_wait_blackwell_(void) { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }

/* This block's rank in its cluster. */
NUMKONG_DEVICE nk_u32_t nk_cluster_rank_blackwell_(void) {
    nk_u32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    return rank;
}

/* Waits until every thread of every block in the cluster arrives, ordering memory across them. */
NUMKONG_DEVICE void nk_cluster_sync_blackwell_(void) {
    asm volatile("barrier.cluster.arrive.release.aligned;\n" //
                 "barrier.cluster.wait.acquire.aligned;\n" ::
                     : "memory");
}

/* The cluster address of shared address @p shared in the block of rank @p rank. */
NUMKONG_DEVICE nk_u32_t nk_cluster_address_blackwell_(nk_u32_t shared, nk_u32_t rank) {
    nk_u32_t address;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(address) : "r"(shared), "r"(rank));
    return address;
}

/* Arrives on the barrier at cluster address @p barrier, in any block of the cluster. */
NUMKONG_DEVICE void nk_mbarrier_arrive_cluster_blackwell_(nk_u32_t barrier) {
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" ::"r"(barrier) : "memory");
}

/* Copies a box into this block's @p destination like @c nk_load_box_blackwell_, completing its
 *  bytes on the barrier at cluster address @p barrier, the pair leader's. */
NUMKONG_DEVICE void nk_load_box_pair_blackwell_(nk_u32_t destination, void const *map, nk_u32_t barrier,
                                                nk_i32_t column, nk_i32_t row) {
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes " //
                 "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(destination),
                 "l"(map), "r"(column), "r"(row), "r"(barrier)
                 : "memory");
}

/* Allocates @p columns of tensor memory in both blocks of the pair, which call it from the same
 *  warp with the same @p holder, and gives up the right to allocate more. */
NUMKONG_DEVICE void nk_tmem_alloc_pair_blackwell_(nk_u32_t holder, nk_u32_t columns) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n" //
                 "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::"r"(holder),
                 "r"(columns)
                 : "memory");
}

NUMKONG_DEVICE void nk_tmem_dealloc_pair_blackwell_(nk_u32_t address, nk_u32_t columns) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n" ::"r"(address), "r"(columns) : "memory");
}

/* A 256-row step across the pair: rows 0 to 127 from this block's operands into its lanes, rows
 *  128 to 255 from the same addresses in the other block into its lanes. */
NUMKONG_DEVICE void nk_mma_f16_pair_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                               nk_u32_t accumulate) {
    asm volatile("{\n.reg .pred accumulate;\n"      //
                 "setp.ne.b32 accumulate, %4, 0;\n" //
                 "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate)
                 : "memory");
}

NUMKONG_DEVICE void nk_mma_f8f6f4_pair_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                                  nk_u32_t accumulate) {
    asm volatile("{\n.reg .pred accumulate;\n"      //
                 "setp.ne.b32 accumulate, %4, 0;\n" //
                 "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate)
                 : "memory");
}

/* A 256-row block-scaled step across the pair over packed nibble pairs, 32 codes per UE8M0 scale,
 *  each block reading its own rows' scales at @p a_scales and all 256 columns' at @p b_scales. */
NUMKONG_DEVICE void nk_mma_mxf4_pair_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                                nk_u32_t accumulate, nk_u32_t a_scales, nk_u32_t b_scales) {
    asm volatile("{\n.reg .pred accumulate;\n"                                                //
                 "setp.ne.b32 accumulate, %4, 0;\n"                                           //
                 "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, " //
                 "[%5], [%6], accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate), "r"(a_scales), "r"(b_scales)
                 : "memory");
}

/* A 256-row block-scaled step across the pair over nibble pairs, 16 codes per UE4M3 scale. */
NUMKONG_DEVICE void nk_mma_nvf4_pair_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                                nk_u32_t accumulate, nk_u32_t a_scales, nk_u32_t b_scales) {
    asm volatile("{\n.reg .pred accumulate;\n"                                                    //
                 "setp.ne.b32 accumulate, %4, 0;\n"                                               //
                 "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, " //
                 "[%5], [%6], accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate), "r"(a_scales), "r"(b_scales)
                 : "memory");
}

/* A 256-row block-scaled step across the pair over FP8 codes, 32 codes per UE8M0 scale. */
NUMKONG_DEVICE void nk_mma_mxf8f6f4_pair_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                                    nk_u32_t accumulate, nk_u32_t a_scales, nk_u32_t b_scales) {
    asm volatile("{\n.reg .pred accumulate;\n"                                                    //
                 "setp.ne.b32 accumulate, %4, 0;\n"                                               //
                 "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, " //
                 "[%5], [%6], accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate), "r"(a_scales), "r"(b_scales)
                 : "memory");
}

/* Copies a 512-byte scale atom like @c nk_tmem_copy_atom_blackwell_ in both blocks of the pair,
 *  each from its own shared memory at the same address into its own tensor memory. */
NUMKONG_DEVICE void nk_tmem_copy_atom_pair_blackwell_(nk_u32_t columns, nk_u64_t atom) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n" ::"r"(columns), "l"(atom) : "memory");
}

/* Arrives on the barrier at shared address @p barrier in both blocks of the pair once every pair
 *  operation issued so far completes. */
NUMKONG_DEVICE void nk_mma_commit_pair_blackwell_(nk_u32_t barrier) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n" ::"r"(
            barrier),
        "h"((unsigned short)3)
        : "memory");
}

/* Orders this thread's shared-memory stores before later tensor-core reads of them. */
NUMKONG_DEVICE void nk_fence_async_shared_blackwell_(void) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

NUMKONG_DEVICE void nk_barrier_sync_blackwell_(nk_u32_t barrier, nk_u32_t threads) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(barrier), "r"(threads) : "memory");
}

/* Allocates @p columns of tensor memory, a power of two from 32, writing their address to
 *  @p holder, and gives up the right to allocate more. */
NUMKONG_DEVICE void nk_tmem_alloc_blackwell_(nk_u32_t holder, nk_u32_t columns) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" //
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::"r"(holder),
                 "r"(columns)
                 : "memory");
}

NUMKONG_DEVICE void nk_tmem_dealloc_blackwell_(nk_u32_t address, nk_u32_t columns) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(address), "r"(columns) : "memory");
}

NUMKONG_DEVICE void nk_tmem_fence_before_blackwell_(void) {
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}

NUMKONG_DEVICE void nk_tmem_fence_after_blackwell_(void) {
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}

/* Reads 32 consecutive columns of this thread's lane, waiting for them in the same block so no
 *  register is read early. */
NUMKONG_DEVICE void nk_tmem_load_x32_blackwell_(nk_u32_t address, nk_u32_t values[32]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "                                                   //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "                   //
                 "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n" //
                 "tcgen05.wait::ld.sync.aligned;\n"
                 : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
                   "=r"(values[6]), "=r"(values[7]), "=r"(values[8]), "=r"(values[9]), "=r"(values[10]),
                   "=r"(values[11]), "=r"(values[12]), "=r"(values[13]), "=r"(values[14]), "=r"(values[15]),
                   "=r"(values[16]), "=r"(values[17]), "=r"(values[18]), "=r"(values[19]), "=r"(values[20]),
                   "=r"(values[21]), "=r"(values[22]), "=r"(values[23]), "=r"(values[24]), "=r"(values[25]),
                   "=r"(values[26]), "=r"(values[27]), "=r"(values[28]), "=r"(values[29]), "=r"(values[30]),
                   "=r"(values[31])
                 : "r"(address)
                 : "memory");
}

/* Reads 4 consecutive columns of this thread's lane and waits for them. */
NUMKONG_DEVICE void nk_tmem_load_x4_blackwell_(nk_u32_t address, nk_u32_t values[4]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n" //
                 "tcgen05.wait::ld.sync.aligned;\n"
                 : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
                 : "r"(address)
                 : "memory");
}

/* Writes @p value into 8 consecutive columns of this thread's lane and waits for it. */
NUMKONG_DEVICE void nk_tmem_fill_x8_blackwell_(nk_u32_t address, nk_u32_t value) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %1, %1, %1, %1, %1, %1, %1};\n" //
                 "tcgen05.wait::st.sync.aligned;\n" ::"r"(address),
                 "r"(value)
                 : "memory");
}

NUMKONG_DEVICE void nk_mma_f16_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                          nk_u32_t accumulate) {
    asm volatile("{\n.reg .pred accumulate;\n"      //
                 "setp.ne.b32 accumulate, %4, 0;\n" //
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate)
                 : "memory");
}

NUMKONG_DEVICE void nk_mma_f8f6f4_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                             nk_u32_t accumulate) {
    asm volatile("{\n.reg .pred accumulate;\n"      //
                 "setp.ne.b32 accumulate, %4, 0;\n" //
                 "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate)
                 : "memory");
}

/* Writes four consecutive columns of this thread's lane, without waiting. */
NUMKONG_DEVICE void nk_tmem_store_x4_blackwell_(nk_u32_t address, nk_u32_t first, nk_u32_t second, nk_u32_t third,
                                                nk_u32_t fourth) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(address), "r"(first),
                 "r"(second), "r"(third), "r"(fourth)
                 : "memory");
}

/* Copies the 512-byte scale atom @p atom describes, 32 rows of 16 bytes, into 4 columns at
 *  @p columns of every lane quarter, ordered before this thread's later products. */
NUMKONG_DEVICE void nk_tmem_copy_atom_blackwell_(nk_u32_t columns, nk_u64_t atom) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n" ::"r"(columns), "l"(atom) : "memory");
}

/* Subtracts packed F16 pairs, exact for the integers the widenings produce. */
NUMKONG_DEVICE nk_u32_t nk_f16x2_subtract_blackwell_(nk_u32_t minuend, nk_u32_t subtrahend) {
    nk_u32_t difference;
    asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(difference) : "r"(minuend), "r"(subtrahend));
    return difference;
}

/* Waits until every tensor-memory store this thread issued lands. */
NUMKONG_DEVICE void nk_tmem_wait_store_blackwell_(void) {
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
}

/* One block-scaled step over packed nibble pairs, 32 codes per UE8M0 scale, A's scales read from
 *  tensor memory at @p a_scales and B's at @p b_scales. */
NUMKONG_DEVICE void nk_mma_mxf4_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                           nk_u32_t accumulate, nk_u32_t a_scales, nk_u32_t b_scales) {
    asm volatile("{\n.reg .pred accumulate;\n"                                                //
                 "setp.ne.b32 accumulate, %4, 0;\n"                                           //
                 "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, " //
                 "[%5], [%6], accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate), "r"(a_scales), "r"(b_scales)
                 : "memory");
}

/* One block-scaled step over packed nibble pairs, 16 codes per UE4M3 scale. */
NUMKONG_DEVICE void nk_mma_nvf4_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                           nk_u32_t accumulate, nk_u32_t a_scales, nk_u32_t b_scales) {
    asm volatile("{\n.reg .pred accumulate;\n"                                                    //
                 "setp.ne.b32 accumulate, %4, 0;\n"                                               //
                 "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, " //
                 "[%5], [%6], accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate), "r"(a_scales), "r"(b_scales)
                 : "memory");
}

/* One block-scaled step over FP8 codes, 32 codes per UE8M0 scale. */
NUMKONG_DEVICE void nk_mma_mxf8f6f4_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                               nk_u32_t accumulate, nk_u32_t a_scales, nk_u32_t b_scales) {
    asm volatile("{\n.reg .pred accumulate;\n"                                                    //
                 "setp.ne.b32 accumulate, %4, 0;\n"                                               //
                 "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, " //
                 "[%5], [%6], accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate), "r"(a_scales), "r"(b_scales)
                 : "memory");
}

/* Arrives on @p barrier once every tensor-core operation this thread issued so far completes. */
NUMKONG_DEVICE void nk_mma_commit_blackwell_(nk_u32_t barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(barrier)
                 : "memory");
}

#pragma endregion Instructions

#pragma region Descriptors

/** The layout code of shared-memory descriptors for the 128-byte swizzle of staged operands. */
enum { nk_smem_swizzle_128_blackwell_k = 2 };

/** Which way an MMA operand's consecutive bytes run in shared memory. */
typedef enum {

    /** Along the depth. */
    nk_major_k_k = 0,

    /** Along the rows of A or the columns of B. */
    nk_major_mn_k = 1,
} nk_major_t;

/** The shared-memory descriptor of an operand at @p address in @p layout, @p leading_bytes between
 *  its groups along the leading dimension and @p stride_bytes between its 8-row groups: bits 0-13
 *  hold the start in 16-byte units, bits 16-29 and 32-45 both offsets alike, bits 46-48 the fixed
 *  version 1 and bits 61-63 the layout. Adding n moves the start 16n bytes on, which within a
 *  swizzled row reads deeper along the same rows, because the swizzle permutes final addresses. */
NUMKONG_DEVICE nk_u64_t nk_smem_descriptor_blackwell_(nk_u32_t address, nk_u32_t layout, nk_u32_t leading_bytes,
                                                      nk_u32_t stride_bytes) {
    return (nk_u64_t)((address & 0x3FFFFu) >> 4) | ((nk_u64_t)(leading_bytes >> 4) << 16) |
           ((nk_u64_t)(stride_bytes >> 4) << 32) | ((nk_u64_t)1 << 46) | ((nk_u64_t)layout << 61);
}

/** The instruction descriptor of a dense @p rows × @p columns step with F32 accumulators, from the
 *  format codes of A and B, 0 and 1 for F16 and BF16 under @c kind::f16 or for E4M3 and E5M2 under
 *  @c kind::f8f6f4, and the majors of both. */
NUMKONG_DEVICE nk_u32_t nk_mma_instruction_blackwell_(nk_u32_t a_format, nk_u32_t b_format, nk_u32_t rows,
                                                      nk_u32_t columns, nk_major_t a_major, nk_major_t b_major) {
    return (1u << 4) | (a_format << 7) | (b_format << 10) | ((nk_u32_t)a_major << 15) | ((nk_u32_t)b_major << 16) |
           ((columns >> 3) << 17) | ((rows >> 4) << 24);
}

/** The instruction descriptor of a block-scaled @p rows × @p columns step over K-major codes of
 *  @p a_format and @p b_format, 1 for E2M1 and 0 or 1 for E4M3 or E5M2, reading UE8M0 scales when
 *  @p ue8m0 is set and UE4M3 ones otherwise from byte @p scale_byte of their columns. */
NUMKONG_DEVICE nk_u32_t nk_mma_scaled_instruction_blackwell_(nk_u32_t a_format, nk_u32_t b_format, nk_u32_t rows,
                                                             nk_u32_t columns, nk_u32_t ue8m0, nk_u32_t scale_byte) {
    return (scale_byte << 4) | (a_format << 7) | (b_format << 10) | ((columns >> 3) << 17) | (ue8m0 << 23) |
           ((rows >> 7) << 27) | (scale_byte << 29);
}

/** The tensor-memory address @p lane lanes down and @p column columns across from @p base. */
NUMKONG_DEVICE nk_u32_t nk_tmem_offset_blackwell_(nk_u32_t base, unsigned lane, unsigned column) {
    return base + (lane << 16) + column;
}

#pragma endregion Descriptors

#pragma region Multiplies

NUMKONG_DEVICE void nk_dots_bf16_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                nk_u32_t scales, nk_u32_t step) {
    nk_mma_f16_blackwell_(accumulator, a, b,
                          nk_mma_instruction_blackwell_(1, 1, nk_cross_rows_blackwell_k, nk_cross_columns_blackwell_k,
                                                        nk_major_k_k, nk_major_k_k),
                          accumulate);
}

NUMKONG_DEVICE void nk_dots_f16_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                               nk_u32_t scales, nk_u32_t step) {
    nk_mma_f16_blackwell_(accumulator, a, b,
                          nk_mma_instruction_blackwell_(0, 0, nk_cross_rows_blackwell_k, nk_cross_columns_blackwell_k,
                                                        nk_major_k_k, nk_major_k_k),
                          accumulate);
}

NUMKONG_DEVICE void nk_dots_e4m3_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                nk_u32_t scales, nk_u32_t step) {
    nk_mma_f8f6f4_blackwell_(accumulator, a, b,
                             nk_mma_instruction_blackwell_(0, 0, nk_cross_rows_blackwell_k,
                                                           nk_cross_columns_blackwell_k, nk_major_k_k, nk_major_k_k),
                             accumulate);
}

NUMKONG_DEVICE void nk_dots_e5m2_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                nk_u32_t scales, nk_u32_t step) {
    nk_mma_f8f6f4_blackwell_(accumulator, a, b,
                             nk_mma_instruction_blackwell_(1, 1, nk_cross_rows_blackwell_k,
                                                           nk_cross_columns_blackwell_k, nk_major_k_k, nk_major_k_k),
                             accumulate);
}

NUMKONG_DEVICE void nk_dots_e2m1_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                nk_u32_t scales, nk_u32_t step) {
    nk_mma_mxf4_blackwell_(
        accumulator, a, b,
        nk_mma_scaled_instruction_blackwell_(1, 1, nk_cross_rows_blackwell_k, nk_cross_columns_blackwell_k, 1, 0),
        accumulate, scales, scales + 16);
}

/*  A slab's scale columns hold 4 bytes per row in each: 16 NVFP4, 8 MXFP4 or 4 MXFP8 bytes per row
 *  and slab, so a step of 32 bytes starts at byte step × slab bytes / 4, in the column that byte
 *  falls in and at its offset within it. */
NUMKONG_DEVICE void nk_dots_nvfp4_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                 nk_u32_t scales, nk_u32_t step) {
    nk_mma_nvf4_blackwell_(
        accumulator, a, b,
        nk_mma_scaled_instruction_blackwell_(1, 1, nk_cross_rows_blackwell_k, nk_cross_columns_blackwell_k, 0, 0),
        accumulate, scales + 4 * step, scales + 16 + 4 * step);
}

NUMKONG_DEVICE void nk_dots_mxfp4_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                 nk_u32_t scales, nk_u32_t step) {
    nk_u32_t const column = 4 * (step / 2);
    nk_mma_mxf4_blackwell_(accumulator, a, b,
                           nk_mma_scaled_instruction_blackwell_(1, 1, nk_cross_rows_blackwell_k,
                                                                nk_cross_columns_blackwell_k, 1, (step & 1) * 2),
                           accumulate, scales + column, scales + 16 + column);
}

NUMKONG_DEVICE void nk_dots_mxfp8e4m3_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                     nk_u32_t scales, nk_u32_t step) {
    nk_mma_mxf8f6f4_blackwell_(
        accumulator, a, b,
        nk_mma_scaled_instruction_blackwell_(0, 0, nk_cross_rows_blackwell_k, nk_cross_columns_blackwell_k, 1, step),
        accumulate, scales, scales + 16);
}

NUMKONG_DEVICE void nk_dots_mxfp8e5m2_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                     nk_u32_t scales, nk_u32_t step) {
    nk_mma_mxf8f6f4_blackwell_(
        accumulator, a, b,
        nk_mma_scaled_instruction_blackwell_(1, 1, nk_cross_rows_blackwell_k, nk_cross_columns_blackwell_k, 1, step),
        accumulate, scales, scales + 16);
}

#pragma endregion Multiplies

#pragma region Tile

/** The origin of output tile @p tile, shaped as @p schedule says, or false for a tile the band of
 *  the output does not reach. Tiles run down a group of @c group_rows row tiles before moving
 *  right, so the group's A rows stay in L2 while every B tile streams from memory once per group
 *  rather than once per row tile. */
NUMKONG_DEVICE int nk_cross_tile_origin_blackwell_(nk_cross_tile_arguments_blackwell_t const *arguments,
                                                   nk_cross_schedule_blackwell_t const *schedule, nk_size_t tile,
                                                   nk_size_t *first_row, nk_size_t *first_column) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    nk_size_t const row_tiles = shape->tiles / shape->column_tiles;
    nk_size_t const group_tiles = arguments->group_rows * shape->column_tiles;
    nk_size_t const group_first = tile / group_tiles * arguments->group_rows, within = tile % group_tiles;
    nk_size_t const group_rows = row_tiles - group_first < arguments->group_rows ? row_tiles - group_first
                                                                                 : arguments->group_rows;
    *first_row = shape->rows_begin + (group_first + within % group_rows) * schedule->tile_rows;
    *first_column = within / group_rows * schedule->tile_columns;
    // Only symmetric bands skip tiles; classifying each tile against the full band costs I8 7%.
    return !nk_cross_symmetric_simt_(schedule->band) ||
           nk_diagonal_band_tile_coverage_simt_(schedule->band, (nk_i64_t)*first_row, schedule->tile_rows,
                                                *first_column, schedule->tile_columns) != nk_diagonal_band_outside_k;
}

/*  Each norm words function adds the squares of one 16-byte chunk of a staged row, in any order,
 *  since a sum of squares has none. Float8 and Float6 square their F16 widenings, and E2M3 and E2M1
 *  their scaled integer ones, so the tile scales them back. */

NUMKONG_DEVICE void nk_cross_norm_words_e5m2_blackwell_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                        nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_e5m2x4_to_f16x4_ampere_(words[word], &low, &high);
        nk_f16x2_norm_update_simt_(low, real_sum);
        nk_f16x2_norm_update_simt_(high, real_sum);
    }
}

NUMKONG_DEVICE void nk_cross_norm_words_e4m3_blackwell_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                        nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_e4m3x4_to_f16x4_ampere_(words[word], &low, &high);
        nk_f16x2_norm_update_simt_(low, real_sum);
        nk_f16x2_norm_update_simt_(high, real_sum);
    }
}

NUMKONG_DEVICE void nk_cross_norm_words_e3m2_blackwell_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                        nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_e3m2x4_to_f16x4_ampere_(words[word], &low, &high);
        nk_f16x2_norm_update_simt_(low, real_sum);
        nk_f16x2_norm_update_simt_(high, real_sum);
    }
}

NUMKONG_DEVICE void nk_cross_norm_words_e2m3_blackwell_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                        nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const magnitudes = nk_e2m3x4_to_u8x4_magnitudes_simt_(words[word]);
        *integer_sum = __dp4a(magnitudes, magnitudes, *integer_sum);
    }
}

NUMKONG_DEVICE void nk_cross_norm_words_e2m1_blackwell_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                        nk_f32_t *real_sum) {
    // Squares of twice each magnitude, {0, 1, 4, 9, 16, 36, 64, 144}, 4 nibbles per lookup.
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const low = __byte_perm(0x09040100u, 0x90402410u, words[word] & 0x7777u);
        nk_u32_t const high = __byte_perm(0x09040100u, 0x90402410u, (words[word] >> 16) & 0x7777u);
        *integer_sum = __dp4a(low, 0x01010101u, *integer_sum);
        *integer_sum = __dp4a(high, 0x01010101u, *integer_sum);
    }
}

NUMKONG_DEVICE void nk_cross_norm_words_i8_blackwell_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                      nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word)
        *integer_sum = (nk_u32_t)__dp4a((int)words[word], (int)words[word], (int)*integer_sum);
}

NUMKONG_DEVICE void nk_cross_norm_words_i4_blackwell_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                      nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_i4x8_to_i8x8_simt_(words[word], &low, &high);
        *integer_sum = (nk_u32_t)__dp4a((int)low, (int)low, (int)*integer_sum);
        *integer_sum = (nk_u32_t)__dp4a((int)high, (int)high, (int)*integer_sum);
    }
}

NUMKONG_DEVICE void nk_cross_norm_words_u8_blackwell_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                      nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) *integer_sum = __dp4a(words[word], words[word], *integer_sum);
}

NUMKONG_DEVICE void nk_cross_norm_words_u4_blackwell_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                      nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_u4x8_to_u8x8_simt_(words[word], &low, &high);
        *integer_sum = __dp4a(low, low, *integer_sum);
        *integer_sum = __dp4a(high, high, *integer_sum);
    }
}

/*  Each stage row function adds the squares of staged 128-byte row @p tile_row of @p operand, when
 *  @p squares, and for Float6 widens it in place. The 16-byte chunks go in swizzle order, so a
 *  quarter-warp's reads of 8 rows land on distinct banks of shared memory. */
NUMKONG_DEVICE void nk_cross_stage_row_bf16_blackwell_(unsigned char *operand, unsigned tile_row, int squares,
                                                       nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < nk_cross_slab_bytes_blackwell_k / 16; ++chunk) {
        uint4 *const address = (uint4 *)(operand + nk_swizzled_offset_simt_(tile_row, chunk * 16, 128));
        uint4 bytes = *address;
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_bf16_norm_update_simt_(words, integer_sum, &slab_sum);
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_row_f16_blackwell_(unsigned char *operand, unsigned tile_row, int squares,
                                                      nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < nk_cross_slab_bytes_blackwell_k / 16; ++chunk) {
        uint4 *const address = (uint4 *)(operand + nk_swizzled_offset_simt_(tile_row, chunk * 16, 128));
        uint4 bytes = *address;
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_f16_norm_update_simt_(words, integer_sum, &slab_sum);
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_row_e5m2_blackwell_(unsigned char *operand, unsigned tile_row, int squares,
                                                       nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < nk_cross_slab_bytes_blackwell_k / 16; ++chunk) {
        uint4 *const address = (uint4 *)(operand + nk_swizzled_offset_simt_(tile_row, chunk * 16, 128));
        uint4 bytes = *address;
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_cross_norm_words_e5m2_blackwell_(words, integer_sum, &slab_sum);
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_row_e4m3_blackwell_(unsigned char *operand, unsigned tile_row, int squares,
                                                       nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < nk_cross_slab_bytes_blackwell_k / 16; ++chunk) {
        uint4 *const address = (uint4 *)(operand + nk_swizzled_offset_simt_(tile_row, chunk * 16, 128));
        uint4 bytes = *address;
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_cross_norm_words_e4m3_blackwell_(words, integer_sum, &slab_sum);
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_row_e2m1_blackwell_(unsigned char *operand, unsigned tile_row, int squares,
                                                       nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < nk_cross_slab_bytes_blackwell_k / 16; ++chunk) {
        uint4 *const address = (uint4 *)(operand + nk_swizzled_offset_simt_(tile_row, chunk * 16, 128));
        uint4 bytes = *address;
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_cross_norm_words_e2m1_blackwell_(words, integer_sum, &slab_sum);
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_row_e3m2_blackwell_(unsigned char *operand, unsigned tile_row, int squares,
                                                       nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < nk_cross_slab_bytes_blackwell_k / 16; ++chunk) {
        uint4 *const address = (uint4 *)(operand + nk_swizzled_offset_simt_(tile_row, chunk * 16, 128));
        uint4 bytes = *address;
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_cross_norm_words_e3m2_blackwell_(words, integer_sum, &slab_sum);
        bytes.x = nk_f6x4_to_f8x4_ada_(words[0]), bytes.y = nk_f6x4_to_f8x4_ada_(words[1]),
        bytes.z = nk_f6x4_to_f8x4_ada_(words[2]), bytes.w = nk_f6x4_to_f8x4_ada_(words[3]);
        *address = bytes;
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_row_e2m3_blackwell_(unsigned char *operand, unsigned tile_row, int squares,
                                                       nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < nk_cross_slab_bytes_blackwell_k / 16; ++chunk) {
        uint4 *const address = (uint4 *)(operand + nk_swizzled_offset_simt_(tile_row, chunk * 16, 128));
        uint4 bytes = *address;
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_cross_norm_words_e2m3_blackwell_(words, integer_sum, &slab_sum);
        bytes.x = nk_f6x4_to_f8x4_ada_(words[0]), bytes.y = nk_f6x4_to_f8x4_ada_(words[1]),
        bytes.z = nk_f6x4_to_f8x4_ada_(words[2]), bytes.w = nk_f6x4_to_f8x4_ada_(words[3]);
        *address = bytes;
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

/** Four byte codes, each a value plus a bias, as two F16 pairs of the values: 0x64 over a byte is
 *  1024 plus it, and @p subtrahend holds 1024 plus the bias in both halves. */
NUMKONG_DEVICE void nk_u8x4_to_f16x4_blackwell_(nk_u32_t codes, nk_u32_t subtrahend, nk_u32_t *low, nk_u32_t *high) {
    *low = nk_f16x2_subtract_blackwell_(__byte_perm(codes, 0x64646464u, 0x4140), subtrahend);
    *high = nk_f16x2_subtract_blackwell_(__byte_perm(codes, 0x64646464u, 0x4342), subtrahend);
}

/*  Each expand row function widens staged row @p tile_row of integer codes, rows narrower than 128
 *  bytes in the swizzle of their width at @p raw, into the same row of the 128-byte F16 rows at
 *  @p wide, adding its squares when @p squares. U8 widens to its value minus 128 and counts its
 *  bytes for the offset, the rest to their values. */
NUMKONG_DEVICE void nk_cross_expand_row_i8_blackwell_(unsigned char const *raw, unsigned char *wide, unsigned tile_row,
                                                      int squares, nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_u32_t const subtrahend = (0x6400u | 128) * 0x00010001u;
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        uint4 const bytes = *(uint4 const *)(raw + nk_swizzled_offset_simt_(tile_row, chunk * 16, 64));
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_cross_norm_words_i8_blackwell_(words, integer_sum, real_sum);
        nk_u32_t halves[8];
#pragma unroll
        for (unsigned word = 0; word < 4; ++word)
            nk_u8x4_to_f16x4_blackwell_(words[word] ^ 0x80808080u, subtrahend, &halves[2 * word],
                                        &halves[2 * word + 1]);
#pragma unroll
        for (unsigned part = 0; part < 2; ++part)
            *(uint4 *)(wide + nk_swizzled_offset_simt_(tile_row, (chunk * 2 + part) * 16, 128)) = make_uint4(
                halves[4 * part], halves[4 * part + 1], halves[4 * part + 2], halves[4 * part + 3]);
    }
}

NUMKONG_DEVICE void nk_cross_expand_row_u8_blackwell_(unsigned char const *raw, unsigned char *wide, unsigned tile_row,
                                                      int squares, nk_u32_t *integer_sum, nk_f32_t *real_sum,
                                                      nk_u32_t *byte_sum) {
    nk_u32_t const subtrahend = (0x6400u | 128) * 0x00010001u;
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        uint4 const bytes = *(uint4 const *)(raw + nk_swizzled_offset_simt_(tile_row, chunk * 16, 64));
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_cross_norm_words_u8_blackwell_(words, integer_sum, real_sum);
        nk_u32_t halves[8];
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            *byte_sum = __dp4a(words[word], 0x01010101u, *byte_sum);
            nk_u8x4_to_f16x4_blackwell_(words[word], subtrahend, &halves[2 * word], &halves[2 * word + 1]);
        }
#pragma unroll
        for (unsigned part = 0; part < 2; ++part)
            *(uint4 *)(wide + nk_swizzled_offset_simt_(tile_row, (chunk * 2 + part) * 16, 128)) = make_uint4(
                halves[4 * part], halves[4 * part + 1], halves[4 * part + 2], halves[4 * part + 3]);
    }
}

NUMKONG_DEVICE void nk_cross_expand_row_i4_blackwell_(unsigned char const *raw, unsigned char *wide, unsigned tile_row,
                                                      int squares, nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_u32_t const subtrahend = (0x6400u | 8) * 0x00010001u;
#pragma unroll
    for (unsigned chunk = 0; chunk < 2; ++chunk) {
        uint4 const bytes = *(uint4 const *)(raw + nk_swizzled_offset_simt_(tile_row, chunk * 16, 32));
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_cross_norm_words_i4_blackwell_(words, integer_sum, real_sum);
        nk_u32_t halves[16];
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            // Element 2i sits in the high nibble of byte i; flipping I4's sign bit adds 8.
            nk_u32_t const odd = (words[word] & 0x0F0F0F0Fu) ^ 0x08080808u,
                           even = ((words[word] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            nk_u8x4_to_f16x4_blackwell_(__byte_perm(even, odd, 0x5140), subtrahend, &halves[4 * word],
                                        &halves[4 * word + 1]);
            nk_u8x4_to_f16x4_blackwell_(__byte_perm(even, odd, 0x7362), subtrahend, &halves[4 * word + 2],
                                        &halves[4 * word + 3]);
        }
#pragma unroll
        for (unsigned part = 0; part < 4; ++part)
            *(uint4 *)(wide + nk_swizzled_offset_simt_(tile_row, (chunk * 4 + part) * 16, 128)) = make_uint4(
                halves[4 * part], halves[4 * part + 1], halves[4 * part + 2], halves[4 * part + 3]);
    }
}

NUMKONG_DEVICE void nk_cross_expand_row_u4_blackwell_(unsigned char const *raw, unsigned char *wide, unsigned tile_row,
                                                      int squares, nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    nk_u32_t const subtrahend = (0x6400u | 0) * 0x00010001u;
#pragma unroll
    for (unsigned chunk = 0; chunk < 2; ++chunk) {
        uint4 const bytes = *(uint4 const *)(raw + nk_swizzled_offset_simt_(tile_row, chunk * 16, 32));
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (squares) nk_cross_norm_words_u4_blackwell_(words, integer_sum, real_sum);
        nk_u32_t halves[16];
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            // Element 2i sits in the high nibble of byte i.
            nk_u32_t const odd = (words[word] & 0x0F0F0F0Fu) ^ 0, even = ((words[word] >> 4) & 0x0F0F0F0Fu) ^ 0;
            nk_u8x4_to_f16x4_blackwell_(__byte_perm(even, odd, 0x5140), subtrahend, &halves[4 * word],
                                        &halves[4 * word + 1]);
            nk_u8x4_to_f16x4_blackwell_(__byte_perm(even, odd, 0x7362), subtrahend, &halves[4 * word + 2],
                                        &halves[4 * word + 3]);
        }
#pragma unroll
        for (unsigned part = 0; part < 4; ++part)
            *(uint4 *)(wide + nk_swizzled_offset_simt_(tile_row, (chunk * 4 + part) * 16, 128)) = make_uint4(
                halves[4 * part], halves[4 * part + 1], halves[4 * part + 2], halves[4 * part + 3]);
    }
}

/** The 16 scale bytes at @p scales, those at or past @p count as zeros, in one load when whole and
 *  aligned, since dense scale rows sit at any byte. */
NUMKONG_DEVICE uint4 nk_cross_scale_group_blackwell_(unsigned char const *scales, nk_size_t count) {
    if (count >= 16 && ((nk_size_t)scales & 15) == 0) return __ldg((uint4 const *)scales);
    nk_u32_t words[4] = {0, 0, 0, 0};
#pragma unroll
    for (unsigned byte = 0; byte < 16; ++byte)
        if (byte < count) words[byte / 4] |= (nk_u32_t)scales[byte] << (8 * (byte % 4));
    return make_uint4(words[0], words[1], words[2], words[3]);
}

/** Loads 16-byte group @p group of the scales of A row @p row and B row @p column, bytes at or past
 *  @p blocks and rows past either edge as zeros. */
NUMKONG_DEVICE void nk_cross_load_scale_groups_blackwell_(nk_cross_tile_arguments_t const *shape, nk_size_t row,
                                                          nk_size_t column, nk_size_t blocks, nk_size_t group,
                                                          uint4 *a_group, uint4 *b_group) {
    nk_size_t const offset = group * 16, count = offset < blocks ? blocks - offset : 0;
    *a_group = nk_cross_scale_group_blackwell_(shape->a_scales + row * shape->a_scales_stride + offset,
                                               row < shape->rows_end ? count : 0);
    *b_group = nk_cross_scale_group_blackwell_(shape->b_scales + column * shape->b_scales_stride + offset,
                                               column < shape->column_count ? count : 0);
}

/** Word @p word of a 16-byte scale group. */
NUMKONG_DEVICE nk_u32_t nk_cross_scale_word_blackwell_(uint4 group, unsigned word) {
    return word == 0 ? group.x : word == 1 ? group.y : word == 2 ? group.z : group.w;
}

/** Writes this thread's 4 A and 4 B scales of atom @p atom into the stage's atoms at @p atoms,
 *  where byte 16l + 4c of an atom holds the scales of row l + 32c, as @c tcgen05.cp multicasts
 *  them into every lane quarter; B's @p atom_count atoms follow A's. */
NUMKONG_DEVICE void nk_cross_stage_scale_atom_blackwell_(unsigned char *atoms, unsigned tile_row, unsigned atom,
                                                         unsigned atom_count, nk_u32_t a_word, nk_u32_t b_word) {
    unsigned char *const a_lane = atoms + atom * 512 + tile_row % 32 * 16 + tile_row / 32 * 4;
    *(nk_u32_t *)a_lane = a_word;
    *(nk_u32_t *)(a_lane + atom_count * 512) = b_word;
}

/** Copies the slab's @p atom_count scale atoms of A and as many of B from shared address @p atoms
 *  into the scale columns at @p scales, A's from column 0 and B's from column 16. */
NUMKONG_DEVICE void nk_cross_copy_scales_blackwell_(nk_u32_t scales, nk_u32_t atoms, unsigned atom_count) {
    for (unsigned atom = 0; atom < atom_count; ++atom) {
        nk_tmem_copy_atom_blackwell_(scales + 4 * atom, nk_smem_descriptor_blackwell_(atoms + 512 * atom, 0, 128, 128));
        nk_tmem_copy_atom_blackwell_(scales + 16 + 4 * atom,
                                     nk_smem_descriptor_blackwell_(atoms + 512 * (atom_count + atom), 0, 128, 128));
    }
}

/** Writes this thread's 4 A scales and the 4 of its two B rows, @p tile_row and 128 below, of atom
 *  @p atom into a pair stage's atoms at @p atoms, where B's 2 · @p atom_count atoms follow A's, the
 *  two halves of each atom group side by side, as the pair's 256 B rows read them. */
NUMKONG_DEVICE void nk_cross_stage_scale_pair_atom_blackwell_(unsigned char *atoms, unsigned tile_row, unsigned atom,
                                                              unsigned atom_count, nk_u32_t a_word, nk_u32_t b_low_word,
                                                              nk_u32_t b_high_word) {
    unsigned const lane = tile_row % 32 * 16 + tile_row / 32 * 4;
    *(nk_u32_t *)(atoms + atom * 512 + lane) = a_word;
    *(nk_u32_t *)(atoms + (atom_count + 2 * atom) * 512 + lane) = b_low_word;
    *(nk_u32_t *)(atoms + (atom_count + 2 * atom + 1) * 512 + lane) = b_high_word;
}

/** Copies a pair slab's @p atom_count scale atoms of A from shared address @p atoms into the scale
 *  columns at @p scales, and B's twice as many into the columns from 16, in both blocks. */
NUMKONG_DEVICE void nk_cross_copy_scales_pair_blackwell_(nk_u32_t scales, nk_u32_t atoms, unsigned atom_count) {
    for (unsigned atom = 0; atom < atom_count; ++atom)
        nk_tmem_copy_atom_pair_blackwell_(scales + 4 * atom,
                                          nk_smem_descriptor_blackwell_(atoms + 512 * atom, 0, 128, 128));
    for (unsigned atom = 0; atom < 2 * atom_count; ++atom)
        nk_tmem_copy_atom_pair_blackwell_(
            scales + 16 + 4 * atom, nk_smem_descriptor_blackwell_(atoms + 512 * (atom_count + atom), 0, 128, 128));
}

/** Adds the squares of the first @p blocks blocks of staged 128-byte nvfp4 row @p tile_row of
 *  @p operand, each block's times its scale squared, from the slab's @p scales, reading the 16-byte
 *  chunks through the swizzle. */
NUMKONG_DEVICE void nk_cross_stage_scaled_row_nvfp4_blackwell_(unsigned char const *operand, unsigned tile_row,
                                                               unsigned char const *scales, nk_size_t blocks,
                                                               nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
    for (unsigned block = 0; block < blocks; ++block) {
        nk_f32_t block_sum = 0;
        for (unsigned element = block * 16; element != (block + 1) * 16; ++element) {
            unsigned char const *bytes = operand + nk_swizzled_offset_simt_(tile_row, element / 32 * 16, 128);
            nk_f32_t const value = nk_e2m1_load_f32_simt_(bytes, element % 32);
            block_sum += value * value;
        }
        nk_f32_t const scale = nk_block_scaled_decode_scale_serial_(scales[block], nk_ue4m3_k);
        slab_sum += block_sum * scale * scale;
    }
    *real_sum += slab_sum;
}

/** Adds the squares of the first @p blocks blocks of staged 128-byte mxfp4 row @p tile_row of
 *  @p operand, each block's times its scale squared, from the slab's @p scales, reading the 16-byte
 *  chunks through the swizzle. */
NUMKONG_DEVICE void nk_cross_stage_scaled_row_mxfp4_blackwell_(unsigned char const *operand, unsigned tile_row,
                                                               unsigned char const *scales, nk_size_t blocks,
                                                               nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
    for (unsigned block = 0; block < blocks; ++block) {
        nk_f32_t block_sum = 0;
        for (unsigned element = block * 32; element != (block + 1) * 32; ++element) {
            unsigned char const *bytes = operand + nk_swizzled_offset_simt_(tile_row, element / 32 * 16, 128);
            nk_f32_t const value = nk_e2m1_load_f32_simt_(bytes, element % 32);
            block_sum += value * value;
        }
        nk_f32_t const scale = nk_block_scaled_decode_scale_serial_(scales[block], nk_ue8m0_k);
        slab_sum += block_sum * scale * scale;
    }
    *real_sum += slab_sum;
}

/** Adds the squares of the first @p blocks blocks of staged 128-byte mxfp8e4m3 row @p tile_row of
 *  @p operand, each block's times its scale squared, from the slab's @p scales, reading the 16-byte
 *  chunks through the swizzle. */
NUMKONG_DEVICE void nk_cross_stage_scaled_row_mxfp8e4m3_blackwell_(unsigned char const *operand, unsigned tile_row,
                                                                   unsigned char const *scales, nk_size_t blocks,
                                                                   nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
    for (unsigned block = 0; block < blocks; ++block) {
        nk_f32_t block_sum = 0;
        for (unsigned element = block * 32; element != (block + 1) * 32; ++element) {
            unsigned char const *bytes = operand + nk_swizzled_offset_simt_(tile_row, element / 16 * 16, 128);
            nk_f32_t const value = nk_e4m3_load_f32_simt_(bytes, element % 16);
            block_sum += value * value;
        }
        nk_f32_t const scale = nk_block_scaled_decode_scale_serial_(scales[block], nk_ue8m0_k);
        slab_sum += block_sum * scale * scale;
    }
    *real_sum += slab_sum;
}

/** Adds the squares of the first @p blocks blocks of staged 128-byte mxfp8e5m2 row @p tile_row of
 *  @p operand, each block's times its scale squared, from the slab's @p scales, reading the 16-byte
 *  chunks through the swizzle. */
NUMKONG_DEVICE void nk_cross_stage_scaled_row_mxfp8e5m2_blackwell_(unsigned char const *operand, unsigned tile_row,
                                                                   unsigned char const *scales, nk_size_t blocks,
                                                                   nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
    for (unsigned block = 0; block < blocks; ++block) {
        nk_f32_t block_sum = 0;
        for (unsigned element = block * 32; element != (block + 1) * 32; ++element) {
            unsigned char const *bytes = operand + nk_swizzled_offset_simt_(tile_row, element / 16 * 16, 128);
            nk_f32_t const value = nk_e5m2_load_f32_simt_(bytes, element % 16);
            block_sum += value * value;
        }
        nk_f32_t const scale = nk_block_scaled_decode_scale_serial_(scales[block], nk_ue8m0_k);
        slab_sum += block_sum * scale * scale;
    }
    *real_sum += slab_sum;
}

/** Zeros the codes of every block of staged 128-byte row @p tile_row of @p operand, each
 *  @p block_chunks 16-byte chunks from block @p first_block on, whose UE8M0 scale in @p word is
 *  0x00, which the tensor cores read as 2⁻¹²⁷ and the serial backends as zero. */
NUMKONG_DEVICE void nk_cross_zero_unscaled_blocks_blackwell_(unsigned char *operand, unsigned tile_row, nk_u32_t word,
                                                             unsigned first_block, unsigned block_chunks) {
    for (nk_u32_t zeros = __vcmpeq4(word, 0) & 0x01010101u; zeros; zeros &= zeros - 1) {
        unsigned const block = first_block + (__ffs(zeros) - 1) / 8;
        for (unsigned chunk = block * block_chunks; chunk != (block + 1) * block_chunks; ++chunk)
            *(uint4 *)(operand + nk_swizzled_offset_simt_(tile_row, chunk * 16, 128)) = make_uint4(0, 0, 0, 0);
    }
}

/** The tile row a stage or epilogue thread owns. Warp w reaches lanes from 32 · (w mod 4) on, so
 *  warps 2 to 9 put thread t on lane t mod 128: stage warp w and epilogue warp w + 4 share rows. */
NUMKONG_DEVICE unsigned nk_cross_drain_row_blackwell_(void) { return threadIdx.x % 128; }

/** Fixes what a dtype's kernel computes: the @p staging its stage warps do and, for integers, the
 *  @p raw_bytes of depth per staged row they widen into a 128-byte row of F16 and the
 *  @p exact_slabs whose F32 sums of widened products stay within 2²⁴, so exact, or zero for no
 *  bound: I8 and offset U8 products reach 2¹⁴, U4 ones 225 and I4 ones 64. A block-scaled dtype
 *  gives its @p block_size and @p scale_bytes per row and slab: 16 for NVFP4, 8 for MXFP4, 4 for
 *  MXFP8. @p offsets says U8 codes widen offset by −128, and a nonzero @p pair_columns that a pair
 *  of blocks multiplies tiles of 256 rows and that many columns together with @c cta_group::2, each
 *  loading half the rows of both operands: 256, whose two accumulators fill tensor memory, or 224,
 *  leaving 64 columns for the scales FP4 kinds read. */
NUMKONG_DEVICE nk_cross_schedule_blackwell_t nk_cross_schedule_blackwell_(
    nk_cross_tile_arguments_t const *shape, nk_cross_metric_t metric, nk_diagonal_band_t band,
    nk_cross_staging_t staging, unsigned raw_bytes, nk_size_t exact_slabs, nk_size_t block_size, unsigned scale_bytes,
    int offsets, unsigned pair_columns, nk_f32_t output_scale, nk_cross_norm_t norm, nk_f32_t norm_scale) {
    nk_cross_schedule_blackwell_t schedule;
    schedule.staging = staging;
    schedule.metric = metric, schedule.band = band;
    schedule.output_scale = output_scale, schedule.norm = norm, schedule.norm_scale = norm_scale;
    schedule.visits_stages = staging != nk_cross_staging_none_k || metric != nk_cross_metric_dot_k;
    schedule.offsets = offsets;
    schedule.publishes = metric != nk_cross_metric_dot_k || offsets;
    schedule.box_bytes = raw_bytes ? raw_bytes : nk_cross_slab_bytes_blackwell_k;
    schedule.box_offset = raw_bytes ? nk_cross_stage_bytes_blackwell_k : 0;
    schedule.paired = pair_columns && metric == nk_cross_metric_dot_k;
    schedule.stage_bytes = schedule.box_offset +
                           (nk_cross_rows_blackwell_k + nk_cross_columns_blackwell_k) * schedule.box_bytes +
                           (nk_cross_rows_blackwell_k + (nk_cross_columns_blackwell_k << schedule.paired)) *
                               scale_bytes;
    // Every stage starts 1024-byte aligned, as the 128-byte swizzle of copies and products needs.
    schedule.stage_bytes = (unsigned)nk_size_round_up_to_multiple_(schedule.stage_bytes, 1024);
    schedule.ring_stages = nk_cross_ring_bytes_blackwell_k / schedule.stage_bytes;
    schedule.scale_bytes = scale_bytes;
    schedule.blocks = block_size ? shape->depth / block_size : 0;
    schedule.slabs = shape->depth_slabs;
    schedule.chunk_slabs = exact_slabs && schedule.slabs > exact_slabs ? exact_slabs
                                                                       : (schedule.slabs ? schedule.slabs : 1);
    schedule.chunks = nk_size_divide_round_up_(schedule.slabs ? schedule.slabs : 1, schedule.chunk_slabs);
    schedule.pair_rank = schedule.paired ? nk_cluster_rank_blackwell_() : 0;
    schedule.tile_rows = nk_cross_rows_blackwell_k << schedule.paired;
    schedule.tile_columns = schedule.paired ? pair_columns : nk_cross_columns_blackwell_k;
    schedule.b_rows = schedule.tile_columns >> schedule.paired;
    schedule.first_tile = blockIdx.x >> schedule.paired;
    schedule.tile_stride = gridDim.x >> schedule.paired;
    return schedule;
}

/** One 32-byte depth step of a @b [256,256] pair tile over K-major codes the tensor cores read as
 *  loaded, into F32 sums: BF16 and F16 as @c kind::f16, E4M3 and E5M2 as @c kind::f8f6f4. */
NUMKONG_DEVICE void nk_cross_mma_pair_bf16_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b,
                                                      nk_u32_t accumulate) {
    nk_mma_f16_pair_blackwell_(accumulator, a, b,
                               nk_mma_instruction_blackwell_(1, 1, 256, 256, nk_major_k_k, nk_major_k_k), accumulate);
}

NUMKONG_DEVICE void nk_cross_mma_pair_f16_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b,
                                                     nk_u32_t accumulate) {
    nk_mma_f16_pair_blackwell_(accumulator, a, b,
                               nk_mma_instruction_blackwell_(0, 0, 256, 256, nk_major_k_k, nk_major_k_k), accumulate);
}

NUMKONG_DEVICE void nk_cross_mma_pair_e5m2_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b,
                                                      nk_u32_t accumulate) {
    nk_mma_f8f6f4_pair_blackwell_(
        accumulator, a, b, nk_mma_instruction_blackwell_(1, 1, 256, 256, nk_major_k_k, nk_major_k_k), accumulate);
}

NUMKONG_DEVICE void nk_cross_mma_pair_e4m3_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b,
                                                      nk_u32_t accumulate) {
    nk_mma_f8f6f4_pair_blackwell_(
        accumulator, a, b, nk_mma_instruction_blackwell_(0, 0, 256, 256, nk_major_k_k, nk_major_k_k), accumulate);
}

/** One 32-byte depth step of a @b [256,224] pair tile over E2M1 nibble pairs as @c kind::mxf4 with
 *  unit scales, or over NVFP4 or MXFP4 with the slab's scales from column @p scales on. */
NUMKONG_DEVICE void nk_cross_mma_pair_e2m1_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                      nk_u32_t scales) {
    nk_mma_mxf4_pair_blackwell_(accumulator, a, b, nk_mma_scaled_instruction_blackwell_(1, 1, 256, 224, 1, 0),
                                accumulate, scales, scales + 16);
}

NUMKONG_DEVICE void nk_cross_mma_pair_nvfp4_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b,
                                                       nk_u32_t accumulate, nk_u32_t scales, nk_u32_t step) {
    nk_mma_nvf4_pair_blackwell_(accumulator, a, b, nk_mma_scaled_instruction_blackwell_(1, 1, 256, 224, 0, 0),
                                accumulate, scales + 4 * step, scales + 16 + 8 * step);
}

NUMKONG_DEVICE void nk_cross_mma_pair_mxfp4_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b,
                                                       nk_u32_t accumulate, nk_u32_t scales, nk_u32_t step) {
    nk_mma_mxf4_pair_blackwell_(accumulator, a, b,
                                nk_mma_scaled_instruction_blackwell_(1, 1, 256, 224, 1, (step & 1) * 2), accumulate,
                                scales + 4 * (step / 2), scales + 16 + 8 * (step / 2));
}

/** One 32-byte depth step of a @b [256,224] pair tile over MXFP8 codes, whose one scale atom per
 *  slab holds the scale of step @p step in byte @p step of every column. */
NUMKONG_DEVICE void nk_cross_mma_pair_mxfp8e4m3_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b,
                                                           nk_u32_t accumulate, nk_u32_t scales, nk_u32_t step) {
    nk_mma_mxf8f6f4_pair_blackwell_(accumulator, a, b, nk_mma_scaled_instruction_blackwell_(0, 0, 256, 224, 1, step),
                                    accumulate, scales, scales + 16);
}

NUMKONG_DEVICE void nk_cross_mma_pair_mxfp8e5m2_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b,
                                                           nk_u32_t accumulate, nk_u32_t scales, nk_u32_t step) {
    nk_mma_mxf8f6f4_pair_blackwell_(accumulator, a, b, nk_mma_scaled_instruction_blackwell_(1, 1, 256, 224, 1, step),
                                    accumulate, scales, scales + 16);
}

/** Arrives on @p barrier once the products issued so far complete: this block's, or both blocks'
 *  of a pair. */
NUMKONG_DEVICE void nk_cross_commit_blackwell_(nk_cross_schedule_blackwell_t const *schedule, nk_u32_t barrier) {
    if (schedule->paired) nk_mma_commit_pair_blackwell_(barrier);
    else nk_mma_commit_blackwell_(barrier);
}

NUMKONG_DEVICE void nk_cross_advance_blackwell_(nk_cross_schedule_blackwell_t const *schedule, nk_u32_t *stage,
                                                nk_u32_t *phase) {
    if (++*stage == schedule->ring_stages) *stage = 0, *phase ^= 1;
}

/** Waits for every thread of the block, or of the pair when it shares tiles. */
NUMKONG_DEVICE void nk_cross_sync_blackwell_(nk_cross_schedule_blackwell_t const *schedule) {
    if (schedule->paired) nk_cluster_sync_blackwell_();
    else __syncthreads();
}

/** Initializes the barriers, allocates every tensor-memory column, of both blocks for a pair, and
 *  fills the unit scales where the accumulators leave room, returning the columns' address. Every
 *  thread of the block calls it. */
NUMKONG_DEVICE nk_u32_t nk_cross_initialize_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                       nk_cross_control_blackwell_t *control) {
    unsigned const warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) {
        unsigned const consumers = schedule->staging != nk_cross_staging_none_k ? 1 : 1 + 4 * schedule->visits_stages;
        for (unsigned stage = 0; stage < schedule->ring_stages; ++stage) {
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->loaded[stage]), 1);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->ready[stage]), 4 << schedule->paired);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->consumed[stage]), consumers);
        }
        // The leader's `drained` counts the epilogue warps of both blocks of a pair.
        for (unsigned buffer = 0; buffer < 2; ++buffer) {
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->accumulated[buffer]), 1);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->drained[buffer]), 4 << schedule->paired);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->measured[buffer]), 4);
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->collected[buffer]), 4);
        }
        nk_mbarrier_init_fence_blackwell_();
    }
    nk_u32_t const holder = nk_shared_address_ampere_(&control->tensor_memory);
    if (warp == 1 && schedule->paired) nk_tmem_alloc_pair_blackwell_(holder, nk_cross_tensor_columns_blackwell_k);
    else if (warp == 1) nk_tmem_alloc_blackwell_(holder, nk_cross_tensor_columns_blackwell_k);
    nk_tmem_fence_before_blackwell_();
    nk_cross_sync_blackwell_(schedule);
    nk_tmem_fence_after_blackwell_();
    nk_u32_t const tensor_memory = control->tensor_memory;
    // UE8M0 code 127 is 2⁰; every byte of the 32 columns holds it, whatever layout is read.
    if (warp >= 6)
        for (unsigned column = 0; column < 32; column += 8)
            nk_tmem_fill_x8_blackwell_(nk_tmem_offset_blackwell_(tensor_memory, warp % 4 * 32,
                                                                 nk_cross_unit_scales_column_blackwell_k + column),
                                       0x7F7F7F7Fu);
    nk_tmem_fence_before_blackwell_();
    nk_cross_sync_blackwell_(schedule);
    nk_tmem_fence_after_blackwell_();
    return tensor_memory;
}

/** Frees the tensor memory once every thread, of both blocks for a pair, is done with it. */
NUMKONG_DEVICE void nk_cross_release_blackwell_(nk_cross_schedule_blackwell_t const *schedule, nk_u32_t tensor_memory) {
    nk_tmem_fence_before_blackwell_();
    nk_cross_sync_blackwell_(schedule);
    if (threadIdx.x >> 5 != 1) return;
    __syncwarp();
    nk_tmem_fence_after_blackwell_();
    if (schedule->paired) nk_tmem_dealloc_pair_blackwell_(tensor_memory, nk_cross_tensor_columns_blackwell_k);
    else nk_tmem_dealloc_blackwell_(tensor_memory, nk_cross_tensor_columns_blackwell_k);
}

/** Streams every tile's slabs of A and B rows into the ring, from one thread. Each block of a pair
 *  loads its own 128 rows of both, completing them on the leader's barrier, which alone expects
 *  the bytes of both blocks. */
NUMKONG_DEVICE void nk_cross_load_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                             nk_cross_control_blackwell_t *control, unsigned char *stages,
                                             nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_u32_t const stages_address = nk_shared_address_ampere_(stages);
    nk_u32_t const copy_bytes = (nk_cross_rows_blackwell_k + schedule->b_rows) * schedule->box_bytes;
    nk_size_t const a_half = schedule->pair_rank * nk_cross_rows_blackwell_k;
    nk_size_t const b_half = schedule->pair_rank * schedule->b_rows;
    nk_prefetch_map_blackwell_(&arguments->a_map);
    nk_prefetch_map_blackwell_(&arguments->b_map);
    nk_u32_t stage = 0, phase = 0;
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            nk_u32_t const loaded = nk_shared_address_ampere_(&control->loaded[stage]);
            nk_u32_t const a_box = stages_address + stage * schedule->stage_bytes + schedule->box_offset;
            nk_u32_t const b_box = a_box + nk_cross_rows_blackwell_k * schedule->box_bytes;
            nk_i32_t const column = (nk_i32_t)(slab * schedule->box_bytes);
            nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->consumed[stage]), phase ^ 1);
            // Stage warps wait on their own block's copies; only unstaged pairs share one barrier.
            if (schedule->paired && schedule->staging == nk_cross_staging_none_k) {
                if (schedule->pair_rank == 0) nk_mbarrier_expect_bytes_blackwell_(loaded, 2 * copy_bytes);
                nk_u32_t const leader_loaded = nk_cluster_address_blackwell_(loaded, 0);
                nk_load_box_pair_blackwell_(a_box, &arguments->a_map, leader_loaded, column,
                                            (nk_i32_t)(first_row + a_half));
                nk_load_box_pair_blackwell_(b_box, &arguments->b_map, leader_loaded, column,
                                            (nk_i32_t)(first_column + b_half));
            }
            else {
                nk_mbarrier_expect_bytes_blackwell_(loaded, copy_bytes);
                nk_load_box_blackwell_(a_box, &arguments->a_map, loaded, column, (nk_i32_t)(first_row + a_half));
                nk_load_box_blackwell_(b_box, &arguments->b_map, loaded, column, (nk_i32_t)(first_column + b_half));
            }
            nk_cross_advance_blackwell_(schedule, &stage, &phase);
        }
    }
}

/** Where the multiply and stage warps stand in the ring of stages and the two accumulators. */
typedef struct {
    nk_u32_t stage, phase, iteration, tiles;
} nk_cross_cursor_blackwell_t;

/** Waits until the accumulator of the next chunk is drained, and returns its tensor-memory address,
 *  with the slabs @p first_slab to @p end_slab of the chunk. */
NUMKONG_DEVICE nk_u32_t nk_cross_chunk_begin_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                        nk_cross_control_blackwell_t *control, nk_u32_t tensor_memory,
                                                        nk_cross_cursor_blackwell_t const *cursor, nk_size_t chunk,
                                                        nk_size_t *first_slab, nk_size_t *end_slab) {
    nk_u32_t const buffer = cursor->iteration & 1, buffer_phase = (cursor->iteration >> 1) & 1;
    *first_slab = chunk * schedule->chunk_slabs;
    *end_slab = *first_slab + schedule->chunk_slabs < schedule->slabs ? *first_slab + schedule->chunk_slabs
                                                                      : schedule->slabs;
    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->drained[buffer]), buffer_phase ^ 1);
    nk_tmem_fence_after_blackwell_();
    return tensor_memory + buffer * schedule->tile_columns;
}

/** The shared-memory descriptors of the next slab's A and B stages, the tensor-memory address of
 *  its block scales, and the shared address of their atoms, once the slab is ready for products. */
typedef struct {
    nk_u64_t a, b;
    nk_u32_t scales, scale_atoms;
} nk_cross_operands_blackwell_t;

NUMKONG_DEVICE nk_cross_operands_blackwell_t nk_cross_slab_begin_blackwell_(
    nk_cross_schedule_blackwell_t const *schedule, nk_cross_control_blackwell_t *control, unsigned char *stages,
    nk_u32_t tensor_memory, nk_cross_cursor_blackwell_t const *cursor) {
    int const staged = schedule->staging != nk_cross_staging_none_k;
    int const scaled = schedule->staging == nk_cross_staging_block_scales_k;
    nk_u64_t *const arrival = staged ? &control->ready[cursor->stage] : &control->loaded[cursor->stage];
    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(arrival), cursor->phase);
    nk_tmem_fence_after_blackwell_();
    nk_u32_t const a_stage = nk_shared_address_ampere_(stages) + cursor->stage * schedule->stage_bytes;
    nk_cross_operands_blackwell_t operands;
    // K-major rows, whose leading offset goes unread, in 8-row groups 1024 bytes apart.
    operands.a = nk_smem_descriptor_blackwell_(a_stage, nk_smem_swizzle_128_blackwell_k, 16, 1024);
    operands.b = nk_smem_descriptor_blackwell_(a_stage + nk_cross_a_bytes_blackwell_k, nk_smem_swizzle_128_blackwell_k,
                                               16, 1024);
    // A pair copies each slab's scales over the last's, after the products that read them.
    operands.scales = tensor_memory + (scaled && schedule->paired ? nk_cross_pair_scales_column_blackwell_k
                                       : scaled ? nk_cross_scale_slots_column_blackwell_k + 32 * cursor->stage
                                                : nk_cross_unit_scales_column_blackwell_k);
    operands.scale_atoms = a_stage + nk_cross_stage_bytes_blackwell_k;
    return operands;
}

/** Releases the slab's stage once the products issued so far complete, and moves on to the next. */
NUMKONG_DEVICE void nk_cross_slab_end_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                 nk_cross_control_blackwell_t *control,
                                                 nk_cross_cursor_blackwell_t *cursor) {
    nk_cross_commit_blackwell_(schedule, nk_shared_address_ampere_(&control->consumed[cursor->stage]));
    nk_cross_advance_blackwell_(schedule, &cursor->stage, &cursor->phase);
}

/** Hands the finished chunk's accumulator to the epilogue warps, and moves on to the next one. */
NUMKONG_DEVICE void nk_cross_chunk_end_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                  nk_cross_control_blackwell_t *control,
                                                  nk_cross_cursor_blackwell_t *cursor) {
    nk_cross_commit_blackwell_(schedule, nk_shared_address_ampere_(&control->accumulated[cursor->iteration & 1]));
    ++cursor->iteration;
}

/** Issues every tile's products of BF16 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_bf16_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                      nk_u32_t tensor_memory,
                                                      nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_bf16_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                          accumulate);
                    else
                        nk_dots_bf16_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                    accumulate, operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of F16 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_f16_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                     nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                     nk_u32_t tensor_memory,
                                                     nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_f16_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                         accumulate);
                    else
                        nk_dots_f16_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                   accumulate, operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of E5M2 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_e5m2_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                      nk_u32_t tensor_memory,
                                                      nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_e5m2_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                          accumulate);
                    else
                        nk_dots_e5m2_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                    accumulate, operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of E4M3 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_e4m3_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                      nk_u32_t tensor_memory,
                                                      nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_e4m3_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                          accumulate);
                    else
                        nk_dots_e4m3_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                    accumulate, operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of E3M2 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_e3m2_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                      nk_u32_t tensor_memory,
                                                      nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    nk_dots_e5m2_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step, accumulate,
                                                operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of E2M3 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_e2m3_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                      nk_u32_t tensor_memory,
                                                      nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    nk_dots_e4m3_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step, accumulate,
                                                operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of E2M1 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_e2m1_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                      nk_u32_t tensor_memory,
                                                      nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_e2m1_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                          accumulate, operands.scales);
                    else
                        nk_dots_e2m1_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                    accumulate, operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of I8 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_i8_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                    nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                    nk_u32_t tensor_memory,
                                                    nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    nk_dots_f16_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step, accumulate,
                                               operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of U8 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_u8_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                    nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                    nk_u32_t tensor_memory,
                                                    nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    nk_dots_f16_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step, accumulate,
                                               operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of I4 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_i4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                    nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                    nk_u32_t tensor_memory,
                                                    nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    nk_dots_f16_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step, accumulate,
                                               operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of U4 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_u4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                    nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                    nk_u32_t tensor_memory,
                                                    nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    nk_dots_f16_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step, accumulate,
                                               operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of NVFP4 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_nvfp4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                       nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                       nk_u32_t tensor_memory,
                                                       nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
                if (schedule->paired) nk_cross_copy_scales_pair_blackwell_(operands.scales, operands.scale_atoms, 4);
                else nk_cross_copy_scales_blackwell_(operands.scales, operands.scale_atoms, 4);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_nvfp4_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                           accumulate, operands.scales, step);
                    else
                        nk_dots_nvfp4_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                     accumulate, operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of MXFP4 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_mxfp4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                       nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                       nk_u32_t tensor_memory,
                                                       nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
                if (schedule->paired) nk_cross_copy_scales_pair_blackwell_(operands.scales, operands.scale_atoms, 2);
                else nk_cross_copy_scales_blackwell_(operands.scales, operands.scale_atoms, 2);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_mxfp4_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                           accumulate, operands.scales, step);
                    else
                        nk_dots_mxfp4_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                     accumulate, operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of MXFP8E4M3 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_mxfp8e4m3_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                           nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                           nk_u32_t tensor_memory,
                                                           nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
                if (schedule->paired) nk_cross_copy_scales_pair_blackwell_(operands.scales, operands.scale_atoms, 1);
                else nk_cross_copy_scales_blackwell_(operands.scales, operands.scale_atoms, 1);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_mxfp8e4m3_blackwell_(accumulator, operands.a + 2 * step,
                                                               operands.b + 2 * step, accumulate, operands.scales,
                                                               step);
                    else
                        nk_dots_mxfp8e4m3_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                         accumulate, operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Issues every tile's products of MXFP8E5M2 from one thread, a chunk of slabs per accumulator,
 *  alternating the two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_mxfp8e5m2_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                           nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                           nk_u32_t tensor_memory,
                                                           nk_cross_tile_arguments_blackwell_t const *arguments) {
    if (schedule->pair_rank != 0) return;
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk) {
            nk_size_t first_slab, end_slab;
            nk_u32_t const accumulator = nk_cross_chunk_begin_blackwell_(schedule, control, tensor_memory, &cursor,
                                                                         chunk, &first_slab, &end_slab);
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_cross_operands_blackwell_t const operands = nk_cross_slab_begin_blackwell_(schedule, control, stages,
                                                                                              tensor_memory, &cursor);
                if (schedule->paired) nk_cross_copy_scales_pair_blackwell_(operands.scales, operands.scale_atoms, 1);
                else nk_cross_copy_scales_blackwell_(operands.scales, operands.scale_atoms, 1);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_mxfp8e5m2_blackwell_(accumulator, operands.a + 2 * step,
                                                               operands.b + 2 * step, accumulate, operands.scales,
                                                               step);
                    else
                        nk_dots_mxfp8e5m2_mma_blackwell_(accumulator, operands.a + 2 * step, operands.b + 2 * step,
                                                         accumulate, operands.scales, step);
                }
                nk_cross_slab_end_blackwell_(schedule, control, &cursor);
            }
            nk_cross_chunk_end_blackwell_(schedule, control, &cursor);
        }
    }
}

/** Writes this thread's rows' scales of the slab, from their 16-byte groups, into the stage's
 *  atoms, and adds their scaled squares for a metric. */
NUMKONG_DEVICE void nk_cross_prepare_nvfp4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_tile_arguments_t const *shape, unsigned char *stage,
                                                      nk_size_t slab, nk_size_t first_row, nk_size_t first_column,
                                                      uint4 a_group, uint4 b_group,
                                                      nk_cross_statistics_blackwell_t *statistics) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_size_t const row = first_row + tile_row, column = first_column + tile_row;
    nk_size_t const scales_offset = slab * schedule->scale_bytes, remaining_blocks = schedule->blocks - scales_offset;
    nk_size_t const slab_blocks = remaining_blocks < schedule->scale_bytes ? remaining_blocks : schedule->scale_bytes;
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    unsigned const first_word = 0;
    nk_u32_t a_words[4], b_words[4];
#pragma unroll
    for (unsigned atom = 0; atom < 4; ++atom) {
        a_words[atom] = nk_cross_scale_word_blackwell_(a_group, first_word + atom);
        b_words[atom] = nk_cross_scale_word_blackwell_(b_group, first_word + atom);
        nk_cross_stage_scale_atom_blackwell_(stage + nk_cross_stage_bytes_blackwell_k, tile_row, atom, 4, a_words[atom],
                                             b_words[atom]);
    }
    if (squares && row < shape->rows_end)
        nk_cross_stage_scaled_row_nvfp4_blackwell_(stage, tile_row,
                                                   shape->a_scales + row * shape->a_scales_stride + scales_offset,
                                                   slab_blocks, &statistics->row_real_norm);
    if (squares && nk_cross_symmetric_simt_(schedule->band) && column < shape->column_count)
        nk_cross_stage_scaled_row_nvfp4_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                                   shape->b_scales + column * shape->b_scales_stride + scales_offset,
                                                   slab_blocks, &statistics->column_real_norm);
}

/** Writes this thread's rows' scales of the slab from their 16-byte groups into the stage's atoms,
 *  zeros its rows' blocks of zero UE8M0 scales, and adds their scaled squares for a metric. */
NUMKONG_DEVICE void nk_cross_prepare_mxfp4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_tile_arguments_t const *shape, unsigned char *stage,
                                                      nk_size_t slab, nk_size_t first_row, nk_size_t first_column,
                                                      uint4 a_group, uint4 b_group,
                                                      nk_cross_statistics_blackwell_t *statistics) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_size_t const row = first_row + tile_row, column = first_column + tile_row;
    nk_size_t const scales_offset = slab * schedule->scale_bytes, remaining_blocks = schedule->blocks - scales_offset;
    nk_size_t const slab_blocks = remaining_blocks < schedule->scale_bytes ? remaining_blocks : schedule->scale_bytes;
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    unsigned const first_word = slab % 2 * 2;
    nk_u32_t a_words[2], b_words[2];
#pragma unroll
    for (unsigned atom = 0; atom < 2; ++atom) {
        a_words[atom] = nk_cross_scale_word_blackwell_(a_group, first_word + atom);
        b_words[atom] = nk_cross_scale_word_blackwell_(b_group, first_word + atom);
        nk_cross_stage_scale_atom_blackwell_(stage + nk_cross_stage_bytes_blackwell_k, tile_row, atom, 2, a_words[atom],
                                             b_words[atom]);
    }
    nk_cross_zero_unscaled_blocks_blackwell_(stage, tile_row, a_words[0], 0, 1);
    nk_cross_zero_unscaled_blocks_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, b_words[0], 0, 1);
    nk_cross_zero_unscaled_blocks_blackwell_(stage, tile_row, a_words[1], 4, 1);
    nk_cross_zero_unscaled_blocks_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, b_words[1], 4, 1);
    if (squares && row < shape->rows_end)
        nk_cross_stage_scaled_row_mxfp4_blackwell_(stage, tile_row,
                                                   shape->a_scales + row * shape->a_scales_stride + scales_offset,
                                                   slab_blocks, &statistics->row_real_norm);
    if (squares && nk_cross_symmetric_simt_(schedule->band) && column < shape->column_count)
        nk_cross_stage_scaled_row_mxfp4_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                                   shape->b_scales + column * shape->b_scales_stride + scales_offset,
                                                   slab_blocks, &statistics->column_real_norm);
}

/** Writes this thread's rows' scales of the slab from their 16-byte groups into the stage's atoms,
 *  zeros its rows' blocks of zero UE8M0 scales, and adds their scaled squares for a metric. */
NUMKONG_DEVICE void nk_cross_prepare_mxfp8e4m3_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                          nk_cross_tile_arguments_t const *shape, unsigned char *stage,
                                                          nk_size_t slab, nk_size_t first_row, nk_size_t first_column,
                                                          uint4 a_group, uint4 b_group,
                                                          nk_cross_statistics_blackwell_t *statistics) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_size_t const row = first_row + tile_row, column = first_column + tile_row;
    nk_size_t const scales_offset = slab * schedule->scale_bytes, remaining_blocks = schedule->blocks - scales_offset;
    nk_size_t const slab_blocks = remaining_blocks < schedule->scale_bytes ? remaining_blocks : schedule->scale_bytes;
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    unsigned const first_word = slab % 4 * 1;
    nk_u32_t a_words[1], b_words[1];
#pragma unroll
    for (unsigned atom = 0; atom < 1; ++atom) {
        a_words[atom] = nk_cross_scale_word_blackwell_(a_group, first_word + atom);
        b_words[atom] = nk_cross_scale_word_blackwell_(b_group, first_word + atom);
        nk_cross_stage_scale_atom_blackwell_(stage + nk_cross_stage_bytes_blackwell_k, tile_row, atom, 1, a_words[atom],
                                             b_words[atom]);
    }
    nk_cross_zero_unscaled_blocks_blackwell_(stage, tile_row, a_words[0], 0, 2);
    nk_cross_zero_unscaled_blocks_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, b_words[0], 0, 2);
    if (squares && row < shape->rows_end)
        nk_cross_stage_scaled_row_mxfp8e4m3_blackwell_(stage, tile_row,
                                                       shape->a_scales + row * shape->a_scales_stride + scales_offset,
                                                       slab_blocks, &statistics->row_real_norm);
    if (squares && nk_cross_symmetric_simt_(schedule->band) && column < shape->column_count)
        nk_cross_stage_scaled_row_mxfp8e4m3_blackwell_(
            stage + nk_cross_a_bytes_blackwell_k, tile_row,
            shape->b_scales + column * shape->b_scales_stride + scales_offset, slab_blocks,
            &statistics->column_real_norm);
}

/** Writes this thread's rows' scales of the slab from their 16-byte groups into the stage's atoms,
 *  zeros its rows' blocks of zero UE8M0 scales, and adds their scaled squares for a metric. */
NUMKONG_DEVICE void nk_cross_prepare_mxfp8e5m2_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                          nk_cross_tile_arguments_t const *shape, unsigned char *stage,
                                                          nk_size_t slab, nk_size_t first_row, nk_size_t first_column,
                                                          uint4 a_group, uint4 b_group,
                                                          nk_cross_statistics_blackwell_t *statistics) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_size_t const row = first_row + tile_row, column = first_column + tile_row;
    nk_size_t const scales_offset = slab * schedule->scale_bytes, remaining_blocks = schedule->blocks - scales_offset;
    nk_size_t const slab_blocks = remaining_blocks < schedule->scale_bytes ? remaining_blocks : schedule->scale_bytes;
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    unsigned const first_word = slab % 4 * 1;
    nk_u32_t a_words[1], b_words[1];
#pragma unroll
    for (unsigned atom = 0; atom < 1; ++atom) {
        a_words[atom] = nk_cross_scale_word_blackwell_(a_group, first_word + atom);
        b_words[atom] = nk_cross_scale_word_blackwell_(b_group, first_word + atom);
        nk_cross_stage_scale_atom_blackwell_(stage + nk_cross_stage_bytes_blackwell_k, tile_row, atom, 1, a_words[atom],
                                             b_words[atom]);
    }
    nk_cross_zero_unscaled_blocks_blackwell_(stage, tile_row, a_words[0], 0, 2);
    nk_cross_zero_unscaled_blocks_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, b_words[0], 0, 2);
    if (squares && row < shape->rows_end)
        nk_cross_stage_scaled_row_mxfp8e5m2_blackwell_(stage, tile_row,
                                                       shape->a_scales + row * shape->a_scales_stride + scales_offset,
                                                       slab_blocks, &statistics->row_real_norm);
    if (squares && nk_cross_symmetric_simt_(schedule->band) && column < shape->column_count)
        nk_cross_stage_scaled_row_mxfp8e5m2_blackwell_(
            stage + nk_cross_a_bytes_blackwell_k, tile_row,
            shape->b_scales + column * shape->b_scales_stride + scales_offset, slab_blocks,
            &statistics->column_real_norm);
}

/** Writes this pair thread's scales of the slab, from the 16-byte groups of its A row and its two B
 *  rows, into the stage's atoms. */
NUMKONG_DEVICE void nk_cross_prepare_pair_nvfp4_blackwell_(unsigned char *stage, uint4 a_group, uint4 b_low_group,
                                                           uint4 b_high_group) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
#pragma unroll
    for (unsigned atom = 0; atom < 4; ++atom)
        nk_cross_stage_scale_pair_atom_blackwell_(
            stage + nk_cross_stage_bytes_blackwell_k, tile_row, atom, 4, nk_cross_scale_word_blackwell_(a_group, atom),
            nk_cross_scale_word_blackwell_(b_low_group, atom), nk_cross_scale_word_blackwell_(b_high_group, atom));
}

/** Writes this pair thread's scales of the slab into the stage's atoms, and zeros the blocks of
 *  zero UE8M0 scales in its A row and in its row of this block's B rows, from @p b_own_group. */
NUMKONG_DEVICE void nk_cross_prepare_pair_mxfp4_blackwell_(unsigned char *stage, nk_size_t slab, uint4 a_group,
                                                           uint4 b_low_group, uint4 b_high_group, uint4 b_own_group) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    unsigned const first_word = slab % 2 * 2;
#pragma unroll
    for (unsigned atom = 0; atom < 2; ++atom) {
        nk_u32_t const a_word = nk_cross_scale_word_blackwell_(a_group, first_word + atom);
        nk_u32_t const b_low_word = nk_cross_scale_word_blackwell_(b_low_group, first_word + atom);
        nk_u32_t const b_high_word = nk_cross_scale_word_blackwell_(b_high_group, first_word + atom);
        nk_cross_stage_scale_pair_atom_blackwell_(stage + nk_cross_stage_bytes_blackwell_k, tile_row, atom, 2, a_word,
                                                  b_low_word, b_high_word);
        nk_cross_zero_unscaled_blocks_blackwell_(stage, tile_row, a_word, 4 * atom, 1);
        nk_cross_zero_unscaled_blocks_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                                 nk_cross_scale_word_blackwell_(b_own_group, first_word + atom),
                                                 4 * atom, 1);
    }
}

/** Writes this pair thread's scales of the slab into the stage's atom, and zeros the blocks of zero
 *  UE8M0 scales in its A row and in its row of this block's B rows, from @p b_own_group. */
NUMKONG_DEVICE void nk_cross_prepare_pair_mxfp8e4m3_blackwell_(unsigned char *stage, nk_size_t slab, uint4 a_group,
                                                               uint4 b_low_group, uint4 b_high_group,
                                                               uint4 b_own_group) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    unsigned const word = slab % 4;
    nk_u32_t const a_word = nk_cross_scale_word_blackwell_(a_group, word);
    nk_cross_stage_scale_pair_atom_blackwell_(stage + nk_cross_stage_bytes_blackwell_k, tile_row, 0, 1, a_word,
                                              nk_cross_scale_word_blackwell_(b_low_group, word),
                                              nk_cross_scale_word_blackwell_(b_high_group, word));
    nk_cross_zero_unscaled_blocks_blackwell_(stage, tile_row, a_word, 0, 2);
    nk_cross_zero_unscaled_blocks_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                             nk_cross_scale_word_blackwell_(b_own_group, word), 0, 2);
}

NUMKONG_DEVICE void nk_cross_prepare_pair_mxfp8e5m2_blackwell_(unsigned char *stage, nk_size_t slab, uint4 a_group,
                                                               uint4 b_low_group, uint4 b_high_group,
                                                               uint4 b_own_group) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    unsigned const word = slab % 4;
    nk_u32_t const a_word = nk_cross_scale_word_blackwell_(a_group, word);
    nk_cross_stage_scale_pair_atom_blackwell_(stage + nk_cross_stage_bytes_blackwell_k, tile_row, 0, 1, a_word,
                                              nk_cross_scale_word_blackwell_(b_low_group, word),
                                              nk_cross_scale_word_blackwell_(b_high_group, word));
    nk_cross_zero_unscaled_blocks_blackwell_(stage, tile_row, a_word, 0, 2);
    nk_cross_zero_unscaled_blocks_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                             nk_cross_scale_word_blackwell_(b_own_group, word), 0, 2);
}

/** Converts 32 exact F32 chunk sums to I32 and adds the running sums at @p running, unless this is
 *  the first chunk, keeping the total there unless this is the last. */
NUMKONG_DEVICE void nk_cross_fold_integer_sums_blackwell_(nk_u32_t bits[32], nk_u32_t running, int first, int last) {
#pragma unroll
    for (unsigned offset = 0; offset < 32; ++offset)
        bits[offset] = (nk_u32_t)__float2int_rn(__uint_as_float(bits[offset]));
    if (!first) {
        nk_u32_t partials[32];
        nk_tmem_load_x32_blackwell_(running, partials);
#pragma unroll
        for (unsigned offset = 0; offset < 32; ++offset) bits[offset] += partials[offset];
    }
    if (last) return;
#pragma unroll
    for (unsigned quartet = 0; quartet < 8; ++quartet)
        nk_tmem_store_x4_blackwell_(running + quartet * 4, bits[quartet * 4], bits[quartet * 4 + 1],
                                    bits[quartet * 4 + 2], bits[quartet * 4 + 3]);
    nk_tmem_wait_store_blackwell_();
}

/** Turns 32 drained sums of @p row from @p first_column into output bits: F32 dots times
 *  @p dot_scale, integer dots restored from the U8 offset, or the metric of either from the bits
 *  of both norms, F32 for floating types and exact U32 for integers. */
NUMKONG_DEVICE void nk_cross_finish_sums_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                    nk_cross_control_blackwell_t const *control, nk_u32_t bits[32],
                                                    nk_size_t row, nk_size_t first_column, unsigned tile_column,
                                                    nk_f32_t dot_scale, nk_u32_t row_norm, nk_u32_t row_byte_sum) {
    int const integer = schedule->staging == nk_cross_staging_expand_k;
    // Padding codes widen to −128 in both rows, which the byte sums never count.
    nk_u32_t const offset_correction = 16384u * (nk_u32_t)(schedule->slabs * schedule->box_bytes);
#pragma unroll
    for (unsigned offset = 0; offset < 32; ++offset) {
        if (schedule->offsets)
            bits[offset] += 128u * (row_byte_sum + control->column_byte_sums[tile_column + offset]) - offset_correction;
        nk_f32_t const dot = __uint_as_float(bits[offset]) * dot_scale;
        nk_u32_t const column_norm = control->column_norms[tile_column + offset];
        if (schedule->metric == nk_cross_metric_dot_k) {
            if (!integer) bits[offset] = __float_as_uint(dot);
        }
        else if (nk_cross_symmetric_simt_(schedule->band) && first_column + offset == row) bits[offset] = 0;
        else if (integer)
            bits[offset] = __float_as_uint(
                nk_cross_integer_metric_simt_(schedule->metric, schedule->norm, bits[offset], row_norm, column_norm));
        else if (schedule->metric == nk_cross_metric_angular_k)
            bits[offset] = __float_as_uint(
                nk_f32_angular_simt_(dot, __uint_as_float(row_norm), __uint_as_float(column_norm)));
        else
            bits[offset] = __float_as_uint(
                nk_f32_euclidean_simt_(dot, __uint_as_float(row_norm), __uint_as_float(column_norm)));
    }
}

/** Stores 32 output words of @p row from @p first_column, as 16-byte stores when all of them fit
 *  and align, skipping columns outside the band. */
NUMKONG_DEVICE void nk_cross_store_sums_blackwell_(nk_diagonal_band_t band, nk_cross_tile_arguments_t const *shape,
                                                   nk_u32_t const bits[32], nk_size_t row, nk_size_t first_column) {
    nk_u32_t *const output = (nk_u32_t *)((unsigned char *)shape->c + row * shape->c_stride) + first_column;
    nk_size_t column_begin, column_end;
    nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, shape->column_count, &column_begin, &column_end);
    nk_size_t const begin = column_begin > first_column ? column_begin - first_column : 0;
    nk_size_t const end = column_end - first_column < 32 ? column_end - first_column : 32;
    if (begin == 0 && end == 32 && ((nk_size_t)output & 15) == 0) {
        uint4 *output_quartets = (uint4 *)output;
#pragma unroll
        for (unsigned quartet = 0; quartet < 8; ++quartet)
            output_quartets[quartet] = make_uint4(bits[4 * quartet], bits[4 * quartet + 1], bits[4 * quartet + 2],
                                                  bits[4 * quartet + 3]);
        return;
    }
#pragma unroll
    for (unsigned offset = 0; offset < 32; ++offset)
        if (offset >= begin && offset < end) output[offset] = bits[offset];
}

/** Stores this warp's 32 rows of 32 outputs from @p first_column with one copy from @p box, its
 *  staging box in the 128-byte swizzle, which the map clips at the output's edges. Lane 0 issues
 *  the copy, once the copy issued two boxes ago has read the box. */
NUMKONG_DEVICE void nk_cross_store_box_blackwell_(nk_cross_tile_arguments_blackwell_t const *arguments,
                                                  unsigned char *box, nk_u32_t const bits[32], nk_size_t first_row,
                                                  nk_size_t first_column) {
    unsigned const lane = threadIdx.x % 32;
    if (lane == 0) nk_bulk_wait_reads_but_one_blackwell_();
    __syncwarp();
#pragma unroll
    for (unsigned quartet = 0; quartet < 8; ++quartet)
        *(uint4 *)(box + nk_swizzled_offset_simt_(lane, quartet * 16, 128)) = make_uint4(
            bits[4 * quartet], bits[4 * quartet + 1], bits[4 * quartet + 2], bits[4 * quartet + 3]);
    nk_fence_async_shared_blackwell_();
    __syncwarp();
    if (lane != 0) return;
    nk_store_box_blackwell_(&arguments->c_map, nk_shared_address_ampere_(box), (nk_i32_t)(first_column * 4),
                            (nk_i32_t)first_row);
    nk_bulk_commit_blackwell_();
}

/** Drains the accumulator at column @p accumulator of this warp's @p lanes into this thread's
 *  output row, 32 columns at a time and none past the output's last: integer chunk sums fold into
 *  the running sums first, and only the last chunk goes out, through this warp's two staging boxes
 *  at @p boxes, alternating with @p stores, or, where the output takes no map or a @c symmetric box
 *  would cross the diagonal, row by row. */
NUMKONG_DEVICE void nk_cross_drain_accumulator_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                          nk_cross_control_blackwell_t const *control,
                                                          nk_cross_tile_arguments_blackwell_t const *arguments,
                                                          unsigned char *boxes, nk_u32_t *stores, nk_u32_t lanes,
                                                          nk_u32_t accumulator, nk_size_t chunk, nk_size_t first_row,
                                                          nk_size_t first_column, nk_u32_t row_norm,
                                                          nk_u32_t row_byte_sum) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_size_t const row = first_row + tile_row, warp_first_row = first_row + tile_row / 32 * 32;
    unsigned const columns = shape->column_count - first_column < schedule->tile_columns
                                 ? (unsigned)(shape->column_count - first_column)
                                 : schedule->tile_columns;
    int const integer = schedule->staging == nk_cross_staging_expand_k, last = chunk + 1 == schedule->chunks;
    nk_f32_t const dot_scale = schedule->output_scale * (shape->a_tensor_scale ? *shape->a_tensor_scale : 1) *
                               (shape->b_tensor_scale ? *shape->b_tensor_scale : 1);
#pragma unroll 1
    for (unsigned tile_column = 0; tile_column < columns; tile_column += 32) {
        nk_u32_t bits[32];
        nk_tmem_load_x32_blackwell_(lanes + accumulator + tile_column, bits);
        // A depth of zero issues no products, leaving the accumulator unwritten.
        if (!schedule->slabs)
#pragma unroll
            for (unsigned offset = 0; offset < 32; ++offset) bits[offset] = 0;
        if (integer)
            nk_cross_fold_integer_sums_blackwell_(bits, lanes + nk_cross_scale_slots_column_blackwell_k + tile_column,
                                                  chunk == 0, last);
        if (!last) continue;
        nk_size_t const chunk_column = first_column + tile_column;
        nk_cross_finish_sums_blackwell_(schedule, control, bits, row, chunk_column, tile_column, dot_scale, row_norm,
                                        row_byte_sum);
        // Bulk stores clip a row's end only to 16 bytes, so a chunk crossing it goes out by rows.
        if (arguments->c_mapped && chunk_column + 32 <= shape->column_count &&
            (!nk_cross_symmetric_simt_(schedule->band) || chunk_column >= warp_first_row + 31)) {
            nk_cross_store_box_blackwell_(arguments, boxes + (*stores & 1) * nk_cross_output_box_bytes_blackwell_k,
                                          bits, warp_first_row, chunk_column);
            ++*stores;
        }
        else if (row < shape->rows_end) nk_cross_store_sums_blackwell_(schedule->band, shape, bits, row, chunk_column);
    }
}

/** Waits until the next stage of the ring lands, and returns it. */
NUMKONG_DEVICE unsigned char *nk_cross_stage_wait_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                             nk_cross_control_blackwell_t *control,
                                                             unsigned char *stages,
                                                             nk_cross_cursor_blackwell_t const *cursor) {
    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->loaded[cursor->stage]), cursor->phase);
    return stages + cursor->stage * schedule->stage_bytes;
}

/** Retires this thread's accesses of the stage before products or copies reuse it, tells the next
 *  consumer, and moves on to the next stage. */
NUMKONG_DEVICE void nk_cross_stage_done_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                   nk_cross_control_blackwell_t *control,
                                                   nk_cross_cursor_blackwell_t *cursor) {
    int const staged = schedule->staging != nk_cross_staging_none_k;
    nk_fence_async_shared_blackwell_();
    __syncwarp();
    nk_u32_t const done = nk_shared_address_ampere_(staged ? &control->ready[cursor->stage]
                                                           : &control->consumed[cursor->stage]);
    if ((threadIdx.x & 31) == 0 && schedule->paired)
        nk_mbarrier_arrive_cluster_blackwell_(nk_cluster_address_blackwell_(done, 0));
    else if ((threadIdx.x & 31) == 0) nk_mbarrier_arrive_blackwell_(done);
    nk_cross_advance_blackwell_(schedule, &cursor->stage, &cursor->phase);
}

/** Hands a tile's statistics to the epilogue warp of the same lanes through the tile's slot, once
 *  that warp has taken the slot's previous tile: four columns of tensor memory holding the row's
 *  norm and byte sum, then the column's, whose norm comes from its B row for @c symmetric and from
 *  the pack otherwise, block-scaled packs' rebased ones taking the B tensor scale here. */
NUMKONG_DEVICE void nk_cross_stage_publish_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_tile_arguments_t const *shape,
                                                      nk_cross_control_blackwell_t *control, nk_u32_t lanes,
                                                      nk_size_t first_column,
                                                      nk_cross_statistics_blackwell_t const *statistics,
                                                      nk_cross_cursor_blackwell_t *cursor) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    if (!schedule->publishes) return;
    nk_u32_t const slot = cursor->tiles & 1, slot_phase = (cursor->tiles >> 1) & 1;
    ++cursor->tiles;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_size_t const column = first_column + tile_row;
    nk_f32_t const a_tensor_scale = shape->a_tensor_scale ? *shape->a_tensor_scale : 1;
    nk_f32_t const column_tensor_scale = symmetric && shape->b_tensor_scale ? *shape->b_tensor_scale : 1;
    nk_fui32_t row_norm = nk_cross_norm_finalize_simt_(schedule->norm, statistics->row_integer_norm,
                                                       statistics->row_real_norm, schedule->norm_scale);
    nk_cross_wide_sum_t const *wide_column_norms = (nk_cross_wide_sum_t const *)shape->b_norms;
    nk_u32_t const *integer_column_norms = (nk_u32_t const *)shape->b_norms;
    nk_fui32_t column_norm = {0};
    if (symmetric)
        column_norm = nk_cross_norm_finalize_simt_(schedule->norm, statistics->column_integer_norm,
                                                   statistics->column_real_norm, schedule->norm_scale);
    else if (schedule->metric != nk_cross_metric_dot_k && column < shape->column_count &&
             schedule->staging == nk_cross_staging_block_scales_k) {
        // Block-scaled packs keep F32 norms relative to the column base; only the result rounds.
        nk_f32_t const b_tensor_scale = shape->b_tensor_scale ? *shape->b_tensor_scale : 1;
        nk_cross_wide_sum_t const packed = nk_cross_wide_times_simt_(
            wide_column_norms[column], nk_cross_tensor_factor_simt_(b_tensor_scale, b_tensor_scale));
        column_norm.f = nk_f32_scale_simt_(packed.sum, packed.exponent);
    }
    else if (schedule->metric != nk_cross_metric_dot_k && column < shape->column_count)
        column_norm.u = integer_column_norms[column];
    // Integer norms stay raw U32 bits, so the epilogue forms exact integer distances.
    if (schedule->norm == nk_cross_norm_f32_k)
        row_norm.f = row_norm.f * a_tensor_scale * a_tensor_scale,
        column_norm.f = column_norm.f * column_tensor_scale * column_tensor_scale;
    nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->collected[slot]), slot_phase ^ 1);
    nk_tmem_fence_after_blackwell_();
    nk_tmem_store_x4_blackwell_(lanes + nk_cross_statistics_column_blackwell_k + 4 * slot, row_norm.u,
                                statistics->row_byte_sum, column_norm.u, statistics->column_byte_sum);
    nk_tmem_wait_store_blackwell_();
    nk_tmem_fence_before_blackwell_();
    __syncwarp();
    if ((threadIdx.x & 31) == 0) nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->measured[slot]));
}

/** Prepares every stage of BF16 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_bf16_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                   nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                   nk_u32_t tensor_memory,
                                                   nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            nk_cross_stage_row_bf16_blackwell_(stage, tile_row, squares, &statistics.row_integer_norm,
                                               &statistics.row_real_norm);
            if (symmetric)
                nk_cross_stage_row_bf16_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, squares,
                                                   &statistics.column_integer_norm, &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of F16 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_f16_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                  nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                  nk_u32_t tensor_memory,
                                                  nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            nk_cross_stage_row_f16_blackwell_(stage, tile_row, squares, &statistics.row_integer_norm,
                                              &statistics.row_real_norm);
            if (symmetric)
                nk_cross_stage_row_f16_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, squares,
                                                  &statistics.column_integer_norm, &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of E5M2 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_e5m2_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                   nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                   nk_u32_t tensor_memory,
                                                   nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            nk_cross_stage_row_e5m2_blackwell_(stage, tile_row, squares, &statistics.row_integer_norm,
                                               &statistics.row_real_norm);
            if (symmetric)
                nk_cross_stage_row_e5m2_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, squares,
                                                   &statistics.column_integer_norm, &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of E4M3 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_e4m3_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                   nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                   nk_u32_t tensor_memory,
                                                   nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            nk_cross_stage_row_e4m3_blackwell_(stage, tile_row, squares, &statistics.row_integer_norm,
                                               &statistics.row_real_norm);
            if (symmetric)
                nk_cross_stage_row_e4m3_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, squares,
                                                   &statistics.column_integer_norm, &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of E3M2 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_e3m2_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                   nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                   nk_u32_t tensor_memory,
                                                   nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            nk_cross_stage_row_e3m2_blackwell_(stage, tile_row, squares, &statistics.row_integer_norm,
                                               &statistics.row_real_norm);
            if (symmetric)
                nk_cross_stage_row_e3m2_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, squares,
                                                   &statistics.column_integer_norm, &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of E2M3 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_e2m3_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                   nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                   nk_u32_t tensor_memory,
                                                   nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            nk_cross_stage_row_e2m3_blackwell_(stage, tile_row, squares, &statistics.row_integer_norm,
                                               &statistics.row_real_norm);
            if (symmetric)
                nk_cross_stage_row_e2m3_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, squares,
                                                   &statistics.column_integer_norm, &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of E2M1 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_e2m1_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                   nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                   nk_u32_t tensor_memory,
                                                   nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            nk_cross_stage_row_e2m1_blackwell_(stage, tile_row, squares, &statistics.row_integer_norm,
                                               &statistics.row_real_norm);
            if (symmetric)
                nk_cross_stage_row_e2m1_blackwell_(stage + nk_cross_a_bytes_blackwell_k, tile_row, squares,
                                                   &statistics.column_integer_norm, &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of I8 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_i8_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                 nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                 nk_u32_t tensor_memory,
                                                 nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            unsigned char const *const a_box = stage + schedule->box_offset;
            unsigned char const *const b_box = a_box + nk_cross_rows_blackwell_k * schedule->box_bytes;
            nk_cross_expand_row_i8_blackwell_(a_box, stage, tile_row, squares, &statistics.row_integer_norm,
                                              &statistics.row_real_norm);
            nk_cross_expand_row_i8_blackwell_(b_box, stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                              squares && symmetric, &statistics.column_integer_norm,
                                              &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of U8 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_u8_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                 nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                 nk_u32_t tensor_memory,
                                                 nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            unsigned char const *const a_box = stage + schedule->box_offset;
            unsigned char const *const b_box = a_box + nk_cross_rows_blackwell_k * schedule->box_bytes;
            nk_cross_expand_row_u8_blackwell_(a_box, stage, tile_row, squares, &statistics.row_integer_norm,
                                              &statistics.row_real_norm, &statistics.row_byte_sum);
            nk_cross_expand_row_u8_blackwell_(b_box, stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                              squares && symmetric, &statistics.column_integer_norm,
                                              &statistics.column_real_norm, &statistics.column_byte_sum);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of I4 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_i4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                 nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                 nk_u32_t tensor_memory,
                                                 nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            unsigned char const *const a_box = stage + schedule->box_offset;
            unsigned char const *const b_box = a_box + nk_cross_rows_blackwell_k * schedule->box_bytes;
            nk_cross_expand_row_i4_blackwell_(a_box, stage, tile_row, squares, &statistics.row_integer_norm,
                                              &statistics.row_real_norm);
            nk_cross_expand_row_i4_blackwell_(b_box, stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                              squares && symmetric, &statistics.column_integer_norm,
                                              &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of U4 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_u4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                 nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                 nk_u32_t tensor_memory,
                                                 nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const squares = schedule->metric != nk_cross_metric_dot_k;
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            unsigned char const *const a_box = stage + schedule->box_offset;
            unsigned char const *const b_box = a_box + nk_cross_rows_blackwell_k * schedule->box_bytes;
            nk_cross_expand_row_u4_blackwell_(a_box, stage, tile_row, squares, &statistics.row_integer_norm,
                                              &statistics.row_real_norm);
            nk_cross_expand_row_u4_blackwell_(b_box, stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                              squares && symmetric, &statistics.column_integer_norm,
                                              &statistics.column_real_norm);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of NVFP4 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_nvfp4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                    nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                    nk_u32_t tensor_memory,
                                                    nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        nk_size_t const row = first_row + tile_row, column = first_column + tile_row;
        uint4 a_group, b_group, a_next, b_next, b_high_group, b_high_next, unused;
        nk_cross_load_scale_groups_blackwell_(shape, row, column, schedule->blocks, 0, &a_next, &b_next);
        if (schedule->paired)
            nk_cross_load_scale_groups_blackwell_(shape, row, column + 128, schedule->blocks, 0, &unused, &b_high_next);
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            a_group = a_next, b_group = b_next, b_high_group = b_high_next;
            nk_cross_load_scale_groups_blackwell_(shape, row, column, schedule->blocks, slab + 1, &a_next, &b_next);
            if (schedule->paired)
                nk_cross_load_scale_groups_blackwell_(shape, row, column + 128, schedule->blocks, slab + 1, &unused,
                                                      &b_high_next);
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            if (schedule->paired) nk_cross_prepare_pair_nvfp4_blackwell_(stage, a_group, b_group, b_high_group);
            else
                nk_cross_prepare_nvfp4_blackwell_(schedule, shape, stage, slab, first_row, first_column, a_group,
                                                  b_group, &statistics);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of MXFP4 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_mxfp4_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                    nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                    nk_u32_t tensor_memory,
                                                    nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        nk_size_t const row = first_row + tile_row, column = first_column + tile_row;
        nk_size_t const own_column = column + schedule->pair_rank * schedule->b_rows;
        uint4 a_group, b_group, a_next, b_next, b_high_group, b_high_next, b_own_group, b_own_next, unused;
        nk_cross_load_scale_groups_blackwell_(shape, row, column, schedule->blocks, 0, &a_next, &b_next);
        if (schedule->paired) {
            nk_cross_load_scale_groups_blackwell_(shape, row, column + 128, schedule->blocks, 0, &unused, &b_high_next);
            nk_cross_load_scale_groups_blackwell_(shape, row, own_column, schedule->blocks, 0, &unused, &b_own_next);
        }
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            if (slab % 2 == 0) {
                a_group = a_next, b_group = b_next, b_high_group = b_high_next, b_own_group = b_own_next;
                nk_cross_load_scale_groups_blackwell_(shape, row, column, schedule->blocks, slab / 2 + 1, &a_next,
                                                      &b_next);
                if (schedule->paired) {
                    nk_cross_load_scale_groups_blackwell_(shape, row, column + 128, schedule->blocks, slab / 2 + 1,
                                                          &unused, &b_high_next);
                    nk_cross_load_scale_groups_blackwell_(shape, row, own_column, schedule->blocks, slab / 2 + 1,
                                                          &unused, &b_own_next);
                }
            }
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            if (schedule->paired)
                nk_cross_prepare_pair_mxfp4_blackwell_(stage, slab, a_group, b_group, b_high_group, b_own_group);
            else
                nk_cross_prepare_mxfp4_blackwell_(schedule, shape, stage, slab, first_row, first_column, a_group,
                                                  b_group, &statistics);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of MXFP8E4M3 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_mxfp8e4m3_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                        nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                        nk_u32_t tensor_memory,
                                                        nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        nk_size_t const row = first_row + tile_row, column = first_column + tile_row;
        nk_size_t const own_column = column + schedule->pair_rank * schedule->b_rows;
        uint4 a_group, b_group, a_next, b_next, b_high_group, b_high_next, b_own_group, b_own_next, unused;
        nk_cross_load_scale_groups_blackwell_(shape, row, column, schedule->blocks, 0, &a_next, &b_next);
        if (schedule->paired) {
            nk_cross_load_scale_groups_blackwell_(shape, row, column + 128, schedule->blocks, 0, &unused, &b_high_next);
            nk_cross_load_scale_groups_blackwell_(shape, row, own_column, schedule->blocks, 0, &unused, &b_own_next);
        }
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            if (slab % 4 == 0) {
                a_group = a_next, b_group = b_next, b_high_group = b_high_next, b_own_group = b_own_next;
                nk_cross_load_scale_groups_blackwell_(shape, row, column, schedule->blocks, slab / 4 + 1, &a_next,
                                                      &b_next);
                if (schedule->paired) {
                    nk_cross_load_scale_groups_blackwell_(shape, row, column + 128, schedule->blocks, slab / 4 + 1,
                                                          &unused, &b_high_next);
                    nk_cross_load_scale_groups_blackwell_(shape, row, own_column, schedule->blocks, slab / 4 + 1,
                                                          &unused, &b_own_next);
                }
            }
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            if (schedule->paired)
                nk_cross_prepare_pair_mxfp8e4m3_blackwell_(stage, slab, a_group, b_group, b_high_group, b_own_group);
            else
                nk_cross_prepare_mxfp8e4m3_blackwell_(schedule, shape, stage, slab, first_row, first_column, a_group,
                                                      b_group, &statistics);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Prepares every stage of MXFP8E5M2 the products need prepared from the four stage warps, each
 *  thread its row of A and of B, and publishes every tile's statistics. */
NUMKONG_DEVICE void nk_cross_stage_mxfp8e5m2_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                                        nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                        nk_u32_t tensor_memory,
                                                        nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const symmetric = nk_cross_symmetric_simt_(schedule->band);
    nk_cross_cursor_blackwell_t cursor = {0, 0, 0, 0};
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        nk_size_t const row = first_row + tile_row, column = first_column + tile_row;
        nk_size_t const own_column = column + schedule->pair_rank * schedule->b_rows;
        uint4 a_group, b_group, a_next, b_next, b_high_group, b_high_next, b_own_group, b_own_next, unused;
        nk_cross_load_scale_groups_blackwell_(shape, row, column, schedule->blocks, 0, &a_next, &b_next);
        if (schedule->paired) {
            nk_cross_load_scale_groups_blackwell_(shape, row, column + 128, schedule->blocks, 0, &unused, &b_high_next);
            nk_cross_load_scale_groups_blackwell_(shape, row, own_column, schedule->blocks, 0, &unused, &b_own_next);
        }
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            if (slab % 4 == 0) {
                a_group = a_next, b_group = b_next, b_high_group = b_high_next, b_own_group = b_own_next;
                nk_cross_load_scale_groups_blackwell_(shape, row, column, schedule->blocks, slab / 4 + 1, &a_next,
                                                      &b_next);
                if (schedule->paired) {
                    nk_cross_load_scale_groups_blackwell_(shape, row, column + 128, schedule->blocks, slab / 4 + 1,
                                                          &unused, &b_high_next);
                    nk_cross_load_scale_groups_blackwell_(shape, row, own_column, schedule->blocks, slab / 4 + 1,
                                                          &unused, &b_own_next);
                }
            }
            unsigned char *const stage = nk_cross_stage_wait_blackwell_(schedule, control, stages, &cursor);
            if (schedule->paired)
                nk_cross_prepare_pair_mxfp8e5m2_blackwell_(stage, slab, a_group, b_group, b_high_group, b_own_group);
            else
                nk_cross_prepare_mxfp8e5m2_blackwell_(schedule, shape, stage, slab, first_row, first_column, a_group,
                                                      b_group, &statistics);
            nk_cross_stage_done_blackwell_(schedule, control, &cursor);
        }
        nk_cross_stage_publish_blackwell_(schedule, shape, control, lanes, first_column, &statistics, &cursor);
    }
}

/** Drains every accumulator the products fill from the four epilogue warps: each owns the 32
 *  accumulator lanes its index modulo 4 may read, so one output row per thread, the row the stage
 *  warp of the same lanes prepares, whose statistics it takes before a tile's last chunk, sharing
 *  the column's norm and byte sum with the other epilogue warps. */
NUMKONG_DEVICE void nk_cross_drain_blackwell_(nk_cross_schedule_blackwell_t const *schedule,
                                              nk_cross_control_blackwell_t *control, unsigned char *stages,
                                              nk_u32_t tensor_memory,
                                              nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    unsigned char *const boxes = stages + nk_cross_ring_bytes_blackwell_k +
                                 tile_row / 32 * 2 * nk_cross_output_box_bytes_blackwell_k;
    nk_u32_t iteration = 0, tiles = 0, stores = 0;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, schedule, tile, &first_row, &first_column)) continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk, ++iteration) {
            nk_u32_t statistics[4] = {0, 0, 0, 0};
            if (chunk + 1 == schedule->chunks && schedule->publishes) {
                nk_u32_t const slot = tiles & 1, slot_phase = (tiles >> 1) & 1;
                nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->measured[slot]), slot_phase);
                nk_tmem_fence_after_blackwell_();
                nk_tmem_load_x4_blackwell_(lanes + nk_cross_statistics_column_blackwell_k + 4 * slot, statistics);
                nk_tmem_fence_before_blackwell_();
                __syncwarp();
                if ((threadIdx.x & 31) == 0)
                    nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->collected[slot]));
                // The first barrier retires the previous tile's reads, the second publishes these.
                nk_barrier_sync_blackwell_(1, 128);
                control->column_norms[tile_row] = statistics[2];
                control->column_byte_sums[tile_row] = statistics[3];
                nk_barrier_sync_blackwell_(1, 128);
            }
            nk_u32_t const buffer = iteration & 1, buffer_phase = (iteration >> 1) & 1;
            nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->accumulated[buffer]), buffer_phase);
            nk_tmem_fence_after_blackwell_();
            nk_cross_drain_accumulator_blackwell_(schedule, control, arguments, boxes, &stores, lanes,
                                                  buffer * schedule->tile_columns, chunk, first_row, first_column,
                                                  statistics[0], statistics[1]);
            nk_tmem_fence_before_blackwell_();
            __syncwarp();
            if ((threadIdx.x & 31) != 0) continue;
            // A pair's products reuse an accumulator once both blocks' epilogue warps are done.
            nk_u32_t const drained = nk_shared_address_ampere_(&control->drained[buffer]);
            if (schedule->paired) nk_mbarrier_arrive_cluster_blackwell_(nk_cluster_address_blackwell_(drained, 0));
            else nk_mbarrier_arrive_blackwell_(drained);
        }
        ++tiles;
    }
    if ((threadIdx.x & 31) == 0) nk_bulk_wait_blackwell_();
}

/** The 1024-byte aligned stages in dynamic shared memory, which promises no such alignment, then
 *  the output staging boxes, then the control block. */
NUMKONG_DEVICE unsigned char *nk_cross_stages_blackwell_(void) {
    extern __shared__ unsigned char nk_cross_dynamic_shared_blackwell_[];
    return nk_shared_aligned_ampere_(nk_cross_dynamic_shared_blackwell_, 1024);
}

NUMKONG_DEVICE nk_cross_control_blackwell_t *nk_cross_control_blackwell_(unsigned char *stages) {
    return (nk_cross_control_blackwell_t *)(stages + nk_cross_ring_bytes_blackwell_k +
                                            nk_cross_output_staging_bytes_blackwell_k);
}

/**
 *  @brief Every tile of one launch of one dtype: warp 0 loads, warp 1 multiplies, warps 2 to 5
 *      prepare stages, and warps 6 to 9 drain accumulators.
 *  @param[in] metric The dot product itself, or a distance from it and the two squared norms.
 *  @param[in] band Which outputs are computed; @c symmetric products take the upper triangle.
 *
 *  An integer dtype loads narrower slabs of codes that the stage warps widen to F16, and
 *  accumulates in chunks of depth whose F32 sums stay exact, adding each chunk's into I32 sums in
 *  tensor memory; U8 widens offset by −128, restored from the byte sums of both rows. The stage
 *  warps arrange a block-scaled slab's scales into 512-byte atoms past the stage, a 16-byte group
 *  per row ahead, which the multiply thread copies into tensor memory.
 */
NUMKONG_DEVICE void nk_cross_tile_bf16_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                  nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_none_k, 0, 0, 0, 0, 0, 256, 1.0f, nk_cross_norm_f32_k, 1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_bf16_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_bf16_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_f16_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                 nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_none_k, 0, 0, 0, 0, 0, 256, 1.0f, nk_cross_norm_f32_k, 1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_f16_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_f16_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_e5m2_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                  nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_none_k, 0, 0, 0, 0, 0, 256, 1.0f, nk_cross_norm_f32_k, 1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_e5m2_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_e5m2_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_e4m3_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                  nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_none_k, 0, 0, 0, 0, 0, 256, 1.0f, nk_cross_norm_f32_k,
        65536.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_e4m3_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_e4m3_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_e3m2_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                  nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_widen_k, 0, 0, 0, 0, 0, 0, 16777216.0f, nk_cross_norm_f32_k,
        16777216.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_e3m2_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_e3m2_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_e2m3_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                  nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_widen_k, 0, 0, 0, 0, 0, 0, 4096.0f, nk_cross_norm_f32_k,
        0.015625f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_e2m3_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_e2m3_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_e2m1_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                  nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_none_k, 0, 0, 0, 0, 0, 224, 1.0f, nk_cross_norm_f32_k, 0.25f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_e2m1_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_e2m1_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_i8_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_expand_k, 64, 16, 0, 0, 0, 0, 1.0f, nk_cross_norm_i32_k, 1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_i8_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_i8_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_u8_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_expand_k, 64, 16, 0, 0, 1, 0, 1.0f, nk_cross_norm_u32_k, 1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_u8_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_u8_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_i4_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_expand_k, 32, 4096, 0, 0, 0, 0, 1.0f, nk_cross_norm_i32_k,
        1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_i4_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_i4_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_u4_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_expand_k, 32, 1165, 0, 0, 0, 0, 1.0f, nk_cross_norm_u32_k,
        1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_u4_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_u4_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_nvfp4_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                   nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_block_scales_k, 0, 0, 16, 16, 0, 224, 1.0f,
        nk_cross_norm_f32_k, 1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_nvfp4_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_nvfp4_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_mxfp4_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                   nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_block_scales_k, 0, 0, 32, 8, 0, 224, 1.0f, nk_cross_norm_f32_k,
        1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_mxfp4_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_mxfp4_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_mxfp8e4m3_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                       nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_block_scales_k, 0, 0, 32, 4, 0, 224, 1.0f, nk_cross_norm_f32_k,
        1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_mxfp8e4m3_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_mxfp8e4m3_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

NUMKONG_DEVICE void nk_cross_tile_mxfp8e5m2_blackwell_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                       nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(
        &arguments->tile, metric, band, nk_cross_staging_block_scales_k, 0, 0, 32, 4, 0, 224, 1.0f, nk_cross_norm_f32_k,
        1.0f);
    unsigned char *const stages = nk_cross_stages_blackwell_();
    nk_cross_control_blackwell_t *const control = nk_cross_control_blackwell_(stages);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_mxfp8e5m2_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_mxfp8e5m2_blackwell_(&schedule, control, stages, tensor_memory, arguments);
    nk_cross_release_blackwell_(&schedule, tensor_memory);
}

/** Describes @p rows rows of @p row_bytes bytes, @p stride bytes apart from @p base, to the Tensor
 *  Memory Accelerator as boxes of @p box_rows rows of @p box_bytes bytes, 32, 64 or 128, in the
 *  swizzle of that width, zero-filling loads past either edge and clipping stores at it. Returns
 *  zero when the driver refuses. */
typedef CUresult (*nk_cross_encode_blackwell_t)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
                                                cuuint64_t const *, cuuint64_t const *, cuuint32_t const *,
                                                cuuint32_t const *, CUtensorMapInterleave, CUtensorMapSwizzle,
                                                CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

/** The driver's tiled tensor-map encoder, resolved through the runtime so nothing links against the
 *  driver library, or null when the driver lacks it. */
NUMKONG_INLINE nk_cross_encode_blackwell_t nk_cross_encoder_blackwell_(void) {
    void *entry = 0;
    enum cudaDriverEntryPointQueryResult found;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000, cudaEnableDefault, &found) !=
            cudaSuccess ||
        found != cudaDriverEntryPointSuccess)
        return 0;
    return (nk_cross_encode_blackwell_t)entry;
}

NUMKONG_INLINE int nk_cross_map_blackwell_(CUtensorMap *map, void const *base, nk_size_t rows, nk_size_t row_bytes,
                                           nk_size_t stride, unsigned box_bytes, unsigned box_rows) {
    // Each lookup costs as much as an encoding, so it runs once per process.
    static nk_cross_encode_blackwell_t const encode = nk_cross_encoder_blackwell_();
    if (!encode) return 0;
    // A zero-depth launch never reads its maps, but the driver refuses an empty dimension.
    cuuint64_t const dimensions[2] = {row_bytes ? row_bytes : 16, rows};
    cuuint64_t const strides[1] = {stride ? stride : 16};
    cuuint32_t const box[2] = {box_bytes, box_rows};
    cuuint32_t const element_strides[2] = {1, 1};
    CUtensorMapSwizzle const swizzle = box_bytes == 32   ? CU_TENSOR_MAP_SWIZZLE_32B
                                       : box_bytes == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                                         : CU_TENSOR_MAP_SWIZZLE_128B;
    return encode(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void *)base, dimensions, strides, box, element_strides,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

/** Launches as many clusters of two blocks of @p kernel as stay resident, at most @p pairs_wanted,
 *  passing the one argument struct at @p arguments by value. */
NUMKONG_INLINE nk_status_t nk_cross_launch_pairs_blackwell_(void const *kernel, nk_size_t pairs_wanted, void *arguments,
                                                            nk_stream_t stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 2, attribute.val.clusterDim.y = 1, attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t configuration;
    configuration.gridDim.x = 2, configuration.gridDim.y = 1, configuration.gridDim.z = 1;
    configuration.blockDim.x = nk_cross_threads_blackwell_k, configuration.blockDim.y = 1;
    configuration.blockDim.z = 1;
    configuration.dynamicSmemBytes = nk_cross_shared_bytes_blackwell_k;
    configuration.stream = (cudaStream_t)stream;
    configuration.attrs = &attribute, configuration.numAttrs = 1;
    int clusters = 0;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nk_cross_shared_bytes_blackwell_k) !=
            cudaSuccess ||
        cudaOccupancyMaxActiveClusters(&clusters, kernel, &configuration) != cudaSuccess || clusters == 0) {
        nk_device_leave_cuda_(caller);
        return nk_device_code_mismatch_k;
    }
    nk_size_t const pairs = (nk_size_t)clusters < pairs_wanted ? (nk_size_t)clusters : pairs_wanted;
    configuration.gridDim.x = (unsigned)(2 * pairs);
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    cudaError_t const status = cudaLaunchKernelExC(&configuration, kernel, launch_arguments);
    nk_device_leave_cuda_(caller);
    return status == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Writes every output of an empty depth, which the tensor maps cannot describe: dots of nothing
 *  times both tensor scales, so an infinite one gives NaN, and zero distances between zero
 *  vectors, over the upper triangle alone for @p upper. Every Blackwell output takes 32 bits. */
static __global__ void nk_cross_empty_blackwell_kernel_(nk_cross_tile_arguments_t arguments, nk_cross_metric_t metric,
                                                        int upper) {
    nk_f32_t const a_tensor_scale = arguments.a_tensor_scale ? *arguments.a_tensor_scale : 1;
    nk_f32_t const b_tensor_scale = arguments.b_tensor_scale ? *arguments.b_tensor_scale : 1;
    nk_f32_t const value = metric == nk_cross_metric_dot_k ? 0.0f * a_tensor_scale * b_tensor_scale : 0.0f;
    nk_size_t const outputs = (arguments.rows_end - arguments.rows_begin) * arguments.column_count;
    for (nk_size_t index = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; index < outputs;
         index += gridDim.x * (nk_size_t)blockDim.x) {
        nk_size_t const row = arguments.rows_begin + index / arguments.column_count;
        nk_size_t const column = index % arguments.column_count;
        if (upper && column < row) continue;
        ((nk_f32_t *)((unsigned char *)arguments.c + row * arguments.c_stride))[column] = value;
    }
}

/** Validates the contract, describes A's @p a_rows rows and B's @p column_count rows, and launches
 *  as many blocks of @p kernel as stay resident, each walking output tiles with a stride of the
 *  grid, dot products in pairs of blocks over tiles of @p pair_columns columns unless it is zero.
 *  @p box_bytes is the depth one copy lands per row, @p block_size the block of a block-scaled
 *  dtype, or zero. @p b_norms is the packed column norms a @c packed metric reads, or null for
 *  @c symmetric. A block-scaled @p dtype needs its block to divide @p depth and, past an empty
 *  depth, scales on both operands. Codes need 16-byte rows for the tensor maps, while scales may
 *  sit at any byte, like dense scale rows. */
NUMKONG_INLINE nk_status_t nk_cross_launch_blackwell_(
    void const *kernel, nk_cross_operand_t const *a, nk_size_t a_rows, nk_cross_operand_t const *b, void const *b_norms,
    void *c, nk_size_t result_bytes, nk_size_t rows_begin, nk_size_t rows_end, nk_size_t column_count, nk_size_t depth,
    nk_size_t block_size, unsigned box_bytes, unsigned pair_columns, nk_cross_metric_t metric, nk_size_t depth_bytes,
    nk_size_t a_stride, nk_size_t b_stride, nk_size_t c_stride, nk_stream_t stream) {
    if (block_size && (depth % block_size || (depth && (!a->scales || !b->scales)))) return nk_unexpected_dimensions_k;
    if ((((nk_size_t)a->elements) | a_stride | ((nk_size_t)b->elements) | b_stride) & 15 ||
        (((nk_size_t)c) | c_stride) & (result_bytes - 1))
        return nk_misaligned_k;
    if (rows_end <= rows_begin || column_count == 0) return nk_success_k;
    nk_cross_tile_arguments_blackwell_t arguments;
    if (depth == 0) {
        nk_cross_tile_arguments_t empty = {0};
        empty.c = c, empty.rows_begin = rows_begin, empty.rows_end = rows_end, empty.column_count = column_count;
        empty.c_stride = c_stride, empty.a_tensor_scale = a->tensor_scale, empty.b_tensor_scale = b->tensor_scale;
        int upper = b_norms == NUMKONG_NULL;
        void *empty_arguments[3] = {&empty, &metric, &upper};
        nk_size_t const blocks = nk_size_divide_round_up_((rows_end - rows_begin) * column_count, 256);
        return nk_launch_cuda_((void const *)nk_cross_empty_blackwell_kernel_, blocks < 1024 ? blocks : 1024, 256,
                               empty_arguments, 0, stream);
    }
    int const paired = pair_columns && metric == nk_cross_metric_dot_k;
    nk_size_t const tile_rows = nk_cross_rows_blackwell_k << paired,
                    tile_columns = paired ? pair_columns : nk_cross_columns_blackwell_k;
    if (!nk_cross_map_blackwell_(&arguments.a_map, a->elements, a_rows, depth_bytes, a_stride, box_bytes,
                                 nk_cross_rows_blackwell_k) ||
        !nk_cross_map_blackwell_(&arguments.b_map, b->elements, column_count, depth_bytes, b_stride, box_bytes,
                                 (unsigned)(tile_columns >> paired)))
        return nk_device_memory_mismatch_k;
    arguments.c_mapped = !((((nk_size_t)c) | c_stride) & 15) &&
                         nk_cross_map_blackwell_(&arguments.c_map, c, rows_end, column_count * result_bytes, c_stride,
                                                 128, 32);
    nk_cross_tile_arguments_t *const tile = &arguments.tile;
    nk_size_t const column_tiles = nk_size_divide_round_up_(column_count, tile_columns);
    nk_size_t const tiles = nk_size_divide_round_up_(rows_end - rows_begin, tile_rows) * column_tiles;
    tile->a = (unsigned char const *)a->elements, tile->b = (unsigned char const *)b->elements, tile->c = c;
    tile->rows_begin = rows_begin, tile->rows_end = rows_end, tile->column_count = column_count;
    tile->depth = depth, tile->depth_bytes = depth_bytes, tile->a_stride = a_stride, tile->b_stride = b_stride;
    tile->c_stride = c_stride, tile->column_tiles = column_tiles, tile->tiles = tiles;
    tile->depth_slabs = nk_size_divide_round_up_(depth_bytes, box_bytes);
    tile->b_norms = b_norms;
    tile->a_scales = a->scales, tile->b_scales = b->scales;
    tile->a_scales_stride = a->scales_stride, tile->b_scales_stride = b->scales_stride;
    tile->a_tensor_scale = a->tensor_scale, tile->b_tensor_scale = b->tensor_scale;
    // Resident tiles step through the depth together, so a near-square block of them shares each
    // slab of A and B through L2: as many row tiles sweep the columns together as the block's side.
    int multiprocessors = 0;
    nk_status_t const status = nk_device_attribute_cuda_(cudaDevAttrMultiProcessorCount, &multiprocessors, stream);
    if (status != nk_success_k) return status;
    nk_size_t const resident_tiles = (nk_size_t)multiprocessors >> paired;
    nk_size_t group_rows = 1;
    while (group_rows * group_rows < resident_tiles && group_rows < 16) ++group_rows;
    arguments.group_rows = group_rows;
    if (paired) return nk_cross_launch_pairs_blackwell_(kernel, tiles, &arguments, stream);
    return nk_launch_resident_cuda_(kernel, nk_cross_threads_blackwell_k, nk_cross_shared_bytes_blackwell_k,
                                    nk_cross_shared_bytes_blackwell_k, tiles, &arguments, stream);
}
#pragma endregion Tile

/*  The same shapes as @c nk_define_cross_packed_cuda_ and its kin, over the TMA launch. */
#pragma region Cross Macros

/**
 *  @brief Generates both shapes of one metric over the TMA launch: C = A × Bᵀ, or its angular or
 *      euclidean distances, over a B packed by @c nk_define_cross_pack_cuda_, and the Gram matrix
 *      C = A × Aᵀ, or its distances, over rows [rows_begin, rows_end).
 *  @param[in] metric @c dot, @c angular or @c euclidean, naming both the entry and the epilogue.
 *  @param[in] box_bytes Depth bytes one copy lands per row: 128, or 64 or 32 for widened integers.
 *  @param[in] pair_columns Columns of the 256-row tiles dot products multiply in pairs, or 0.
 *  @sa nk_define_cross_packed_ and nk_define_cross_symmetric_ for the host originals.
 *
 *  The Gram matrix covers the upper triangle, with the diagonal for dots and zeros on it for
 *  distances, skipping tiles wholly below it. Each shape is one block per resident slot walking
 *  @b [128,128] tiles.
 */
#define nk_define_cross_tma_blackwell_(metric, input_type_name, isa_suffix, input_value_type, packed_value_type,       \
                                       result_value_type, depth_simd_dimensions, dimensions_per_value, box_bytes,      \
                                       pair_columns)                                                                   \
    static __global__ void __launch_bounds__(nk_cross_threads_blackwell_k, 1)                                          \
        nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_(                                              \
            __grid_constant__ nk_cross_tile_arguments_blackwell_t const arguments) {                                   \
        nk_diagonal_band_t const band = {NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX};                                          \
        nk_cross_tile_##input_type_name##_blackwell_(nk_cross_metric_##metric##_k, band, &arguments);                  \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##metric##s_packed_##input_type_name##_##isa_suffix(                                    \
        nk_cross_##input_type_name##_operand_t const *a_operand, void const *b_packed_buffer,                          \
        nk_##result_value_type##_t *c_matrix, nk_size_t rows, nk_size_t column_count, nk_size_t depth,                 \
        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {                                                  \
        nk_size_t const row_bytes = nk_cross_padded_values_simt_(depth, depth_simd_dimensions, dimensions_per_value,   \
                                                                 sizeof(nk_##packed_value_type##_t)) *                 \
                                    sizeof(nk_##packed_value_type##_t);                                                \
        nk_size_t const scales_stride = nk_cross_scales_stride_serial_(nk_##input_type_name##_k, depth);               \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_u8_t const *b_rows = (nk_u8_t const *)(header + 1);                                                         \
        nk_cross_operand_t const a = nk_cross_operand_serial_(nk_##input_type_name##_k, a_operand, a_stride);          \
        nk_cross_operand_t const b = {b_rows, scales_stride ? b_rows + column_count * row_bytes : NUMKONG_NULL,        \
                                      scales_stride, &header->tensor_scale};                                           \
        return nk_cross_launch_blackwell_(                                                                             \
            (void const *)nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_, &a, rows, &b,              \
            b_rows + column_count * (row_bytes + scales_stride), c_matrix, sizeof(nk_##result_value_type##_t), 0,      \
            rows, column_count, depth, nk_block_scaled_format_of_dtype(nk_##input_type_name##_k).block_size,           \
            box_bytes, pair_columns, nk_cross_metric_##metric##_k,                                                     \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), a_stride, row_bytes, c_stride, stream);  \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_cross_threads_blackwell_k, 1)                                          \
        nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_(                                           \
            __grid_constant__ nk_cross_tile_arguments_blackwell_t const arguments) {                                   \
        nk_diagonal_band_t const band = {0, NUMKONG_SIZE_MAX};                                                         \
        nk_cross_tile_##input_type_name##_blackwell_(nk_cross_metric_##metric##_k, band, &arguments);                  \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##metric##s_symmetric_##input_type_name##_##isa_suffix(                                 \
        nk_cross_##input_type_name##_operand_t const *vectors_operand, nk_size_t vector_count, nk_size_t depth,        \
        nk_size_t stride, nk_##result_value_type##_t *result, nk_size_t result_stride, nk_size_t rows_begin,           \
        nk_size_t rows_end, nk_stream_t stream) {                                                                      \
        rows_end = nk_min_of_two(rows_end, vector_count);                                                              \
        nk_cross_operand_t const vectors = nk_cross_operand_serial_(nk_##input_type_name##_k, vectors_operand,         \
                                                                    stride);                                           \
        return nk_cross_launch_blackwell_(                                                                             \
            (void const *)nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_, &vectors, vector_count, \
            &vectors, 0, result, sizeof(nk_##result_value_type##_t), rows_begin, rows_end, vector_count, depth,        \
            nk_block_scaled_format_of_dtype(nk_##input_type_name##_k).block_size, box_bytes, pair_columns,             \
            nk_cross_metric_##metric##_k, depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), stride,    \
            stride, result_stride, stream);                                                                            \
    }

#pragma endregion Cross Macros

/*  Later generations read the helpers above and emit only their own kernels. */
#if NUMKONG_TARGET_BLACKWELL

#pragma region BF16

nk_define_cross_pack_cuda_(bf16, blackwell, bf16, bf16, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, bf16, blackwell, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*pair_columns=*/256)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_cuda_(f16, blackwell, f16, f16, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, f16, blackwell, f16, f16, f32, /*depth_simd_dimensions=*/8,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*pair_columns=*/256)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_pack_cuda_(e5m2, blackwell, e5m2, e5m2, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*pair_columns=*/256)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_pack_cuda_(e4m3, blackwell, e4m3, e4m3, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*pair_columns=*/256)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_pack_cuda_(e3m2, blackwell, e3m2, e5m2, nk_load_f6_to_f8_ada_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, e3m2, blackwell, e3m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*pair_columns=*/0)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_pack_cuda_(e2m3, blackwell, e2m3, e4m3, nk_load_f6_to_f8_ada_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, e2m3, blackwell, e2m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*pair_columns=*/0)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_cuda_(e2m1, blackwell, e2m1x2, e2m1x2, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, e2m1, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/128, /*pair_columns=*/224)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_pack_cuda_(i8, blackwell, i8, i8, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, i8, blackwell, i8, i8, i32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/64, /*pair_columns=*/0)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_cuda_(i4, blackwell, i4x2, i4x2, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, i4, blackwell, i4x2, i4x2, i32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/32, /*pair_columns=*/0)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_cuda_(u8, blackwell, u8, u8, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, u8, blackwell, u8, u8, u32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/64, /*pair_columns=*/0)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_cuda_(u4, blackwell, u4x2, u4x2, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, u4, blackwell, u4x2, u4x2, u32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/32, /*pair_columns=*/0)

#pragma endregion U4

#pragma region Block Scaled Floats

nk_define_cross_pack_size_simt_(nvfp4, blackwell, e2m1x2, cross_wide_sum, /*depth_simd_dimensions=*/32,
                                /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_cuda_(nvfp4, blackwell)
nk_define_cross_pack_rows_cuda_(nvfp4, blackwell, e2m1x2, e2m1x2, nk_load_b8_simt_,
                                /*norm_value_type=*/cross_wide_sum, nk_cross_scaled_pack_norm_nvfp4_simt_,
                                /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, nvfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/128, /*pair_columns=*/224)
nk_define_cross_pack_size_simt_(mxfp4, blackwell, e2m1x2, cross_wide_sum, /*depth_simd_dimensions=*/32,
                                /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_cuda_(mxfp4, blackwell)
nk_define_cross_pack_rows_cuda_(mxfp4, blackwell, e2m1x2, e2m1x2, nk_load_b8_simt_,
                                /*norm_value_type=*/cross_wide_sum, nk_cross_scaled_pack_norm_mxfp4_simt_,
                                /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, mxfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/128, /*pair_columns=*/224)
nk_define_cross_pack_size_simt_(mxfp8e4m3, blackwell, e4m3, cross_wide_sum, /*depth_simd_dimensions=*/16,
                                /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_cuda_(mxfp8e4m3, blackwell)
nk_define_cross_pack_rows_cuda_(mxfp8e4m3, blackwell, e4m3, e4m3, nk_load_b8_simt_,
                                /*norm_value_type=*/cross_wide_sum, nk_cross_scaled_pack_norm_mxfp8e4m3_simt_,
                                /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, mxfp8e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*pair_columns=*/224)
nk_define_cross_pack_size_simt_(mxfp8e5m2, blackwell, e5m2, cross_wide_sum, /*depth_simd_dimensions=*/16,
                                /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_cuda_(mxfp8e5m2, blackwell)
nk_define_cross_pack_rows_cuda_(mxfp8e5m2, blackwell, e5m2, e5m2, nk_load_b8_simt_,
                                /*norm_value_type=*/cross_wide_sum, nk_cross_scaled_pack_norm_mxfp8e5m2_simt_,
                                /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, mxfp8e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*pair_columns=*/224)

#pragma endregion Block Scaled Floats

#endif // NUMKONG_TARGET_BLACKWELL

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_BLACKWELL_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_DOTS_BLACKWELL_CUH
