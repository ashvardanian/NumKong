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

#if NUMKONG_TARGET_BLACKWELL

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
    nk_f32_t column_norms[nk_cross_columns_blackwell_k];
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

/** Bytes of depth per staged row of an integer dtype, which the stage warps widen into a 128-byte
 *  row of F16, or zero for dtypes the tensor cores read as loaded. */
NUMKONG_CONSTEXPR unsigned nk_cross_raw_bytes_blackwell_(nk_dtype_t dtype) {
    return dtype == nk_i8_k || dtype == nk_u8_k ? 64 : dtype == nk_i4_k || dtype == nk_u4_k ? 32 : 0;
}

/** Whether a pair of blocks multiplies @b [256,256] tiles together with @c cta_group::2, each
 *  loading half the rows of both operands: for dot products of the dtypes the tensor cores read as
 *  loaded and without block scales, whose two 256-column accumulators fill tensor memory. */
NUMKONG_CONSTEXPR int nk_cross_paired_blackwell_(nk_dtype_t dtype, nk_cross_metric_t metric) {
    return metric == nk_cross_metric_dot_k &&
           (dtype == nk_bf16_k || dtype == nk_f16_k || dtype == nk_e4m3_k || dtype == nk_e5m2_k);
}

/** Slabs whose F32 sums of widened integer products stay within 2²⁴, so exact, or zero for no
 *  bound: I8 and offset U8 products reach 2¹⁴, U4 ones 225 and I4 ones 64. */
NUMKONG_CONSTEXPR nk_size_t nk_cross_exact_slabs_blackwell_(nk_dtype_t dtype) {
    return dtype == nk_i8_k || dtype == nk_u8_k ? 16 : dtype == nk_u4_k ? 1165 : dtype == nk_i4_k ? 4096 : 0;
}

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

/** Issues depth step @p step of a slab, 32 bytes of a @b [128,128] tile, into the accumulator at
 *  tensor-memory address @p accumulator, adding to it unless @p accumulate is zero. @p scales
 *  addresses the slab's block scales, which only block-scaled kinds read: A's in its first 16
 *  columns, B's in the next 16. */
typedef void (*nk_cross_mma_blackwell_t)(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                         nk_u32_t scales, nk_u32_t step);

/** Rewrites four codes into the four codes the tensor cores multiply, exactly. */
typedef nk_u32_t (*nk_cross_widen_blackwell_t)(nk_u32_t codes);

/** What one kernel computes, fixed at compile time: the leading arguments of
 *  @c nk_cross_tile_blackwell_, documented there. */
typedef struct {
    nk_dtype_t dtype;
    nk_cross_mma_blackwell_t mma;
    nk_cross_widen_blackwell_t widen;
    nk_f32_t output_scale;
    nk_cross_norm_t norm;
    nk_cross_norm_update_t norm_update;
    nk_f32_t norm_scale;
    nk_cross_triangle_t triangle;
    nk_cross_metric_t metric;
} nk_cross_kernel_blackwell_t;

/** How the stage warps prepare a stage before the products read it. */
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

    /** Whether the stage warps visit every stage, to prepare it or to square its rows. */
    int visits_stages;

    /** Whether U8 codes widen offset by −128, which both rows' byte sums restore. */
    int offsets;

    /** Whether the stage warps hand each tile's norms or byte sums to the epilogue warps. */
    int publishes;

    /** Bytes of depth one copy lands per row, and where in a stage the copies land. */
    unsigned box_bytes, box_offset;

    /** Bytes of one stage, and the stages that fit the ring. */
    unsigned stage_bytes, ring_stages;

    /** The block-scaled format, its scale bytes per row and slab, and its blocks per row. */
    nk_block_scaled_format_t format;
    unsigned scale_bytes;
    nk_size_t blocks;

    /** Slabs per tile, slabs per chunk whose F32 sums stay exact, and chunks per tile. */
    nk_size_t slabs, chunk_slabs, chunks;

    /** Whether a pair of blocks shares each tile, see @c nk_cross_paired_blackwell_, and this
     *  block's rank in it, which picks its 128 rows and its 128 of the B rows. */
    int paired;
    nk_u32_t pair_rank;

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
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" ::"r"(barrier) : "memory");
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

#pragma region Tile

/** The origin of output tile @p tile, shaped as @p schedule says, or false for a tile wholly below
 *  the diagonal. Tiles run down a group of @c group_rows row tiles before moving right, so the
 *  group's A rows stay in L2 while every B tile streams from memory once per group rather than once
 *  per row tile. */
NUMKONG_DEVICE int nk_cross_tile_origin_blackwell_(nk_cross_tile_arguments_blackwell_t const *arguments,
                                                   nk_cross_triangle_t triangle,
                                                   nk_cross_schedule_blackwell_t const *schedule, nk_size_t tile,
                                                   nk_size_t *first_row, nk_size_t *first_column) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    nk_size_t const row_tiles = shape->tiles / shape->column_tiles;
    nk_size_t const group_tiles = arguments->group_rows * shape->column_tiles;
    nk_size_t const group_first = tile / group_tiles * arguments->group_rows, within = tile % group_tiles;
    nk_size_t const group_rows = row_tiles - group_first < arguments->group_rows ? row_tiles - group_first
                                                                                 : arguments->group_rows;
    *first_row = shape->row_start + (group_first + within % group_rows) * schedule->tile_rows;
    *first_column = within / group_rows * schedule->tile_columns;
    return triangle != nk_cross_triangle_upper_k || *first_column + schedule->tile_columns > *first_row;
}

/** Adds the squares of staged 128-byte row @p tile_row of @p operand and widens it in place, each
 *  optional. The 16-byte chunks go in swizzle order, so a quarter-warp's reads of 8 rows land on
 *  distinct banks of shared memory. */
NUMKONG_DEVICE void nk_cross_stage_row_blackwell_(nk_cross_norm_update_t norm_update, nk_cross_widen_blackwell_t widen,
                                                  unsigned char *operand, unsigned tile_row, nk_u32_t *integer_sum,
                                                  nk_f32_t *real_sum) {
    uint4 *const chunks = (uint4 *)(operand + tile_row * nk_cross_slab_bytes_blackwell_k);
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < nk_cross_slab_bytes_blackwell_k / 16; ++chunk) {
        uint4 *const address = chunks + (chunk ^ (tile_row & 7));
        uint4 bytes = *address;
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (norm_update) norm_update(words, integer_sum, &slab_sum);
        if (widen) {
            bytes.x = widen(words[0]), bytes.y = widen(words[1]), bytes.z = widen(words[2]), bytes.w = widen(words[3]);
            *address = bytes;
        }
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

/** Four byte codes, each a value plus a bias, as two F16 pairs of the values: 0x64 over a byte is
 *  1024 plus it, and @p subtrahend holds 1024 plus the bias in both halves. */
NUMKONG_DEVICE void nk_u8x4_to_f16x4_blackwell_(nk_u32_t codes, nk_u32_t subtrahend, nk_u32_t *low, nk_u32_t *high) {
    *low = nk_f16x2_subtract_blackwell_(__byte_perm(codes, 0x64646464u, 0x4140), subtrahend);
    *high = nk_f16x2_subtract_blackwell_(__byte_perm(codes, 0x64646464u, 0x4342), subtrahend);
}

/** Widens staged row @p tile_row of integer codes, rows @c nk_cross_raw_bytes_blackwell_ wide in
 *  the swizzle of that width at @p raw, into the same row of the 128-byte F16 rows at @p wide,
 *  adding its squares for a metric and its bytes for the U8 offset. U8 widens to its value minus
 *  128, the rest to their values. */
NUMKONG_DEVICE void nk_cross_expand_row_blackwell_(nk_dtype_t dtype, nk_cross_norm_update_t norm_update,
                                                   unsigned char const *raw, unsigned char *wide, unsigned tile_row,
                                                   nk_u32_t *integer_sum, nk_f32_t *real_sum, nk_u32_t *byte_sum) {
    unsigned const raw_bytes = nk_cross_raw_bytes_blackwell_(dtype), expansion = 128 / raw_bytes;
    nk_u32_t const bias = dtype == nk_u4_k ? 0 : dtype == nk_i4_k ? 8 : 128;
    nk_u32_t const subtrahend = (0x6400u | bias) * 0x00010001u;
    uint4 *const wide_chunks = (uint4 *)(wide + tile_row * nk_cross_slab_bytes_blackwell_k);
#pragma unroll
    for (unsigned chunk = 0; chunk < raw_bytes / 16; ++chunk) {
        uint4 const bytes = *(uint4 const *)(raw + nk_swizzled_offset_(tile_row, chunk * 16, raw_bytes));
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        if (norm_update) norm_update(words, integer_sum, real_sum);
        nk_u32_t halves[16];
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            if (expansion == 2) {
                if (dtype == nk_u8_k) *byte_sum = __dp4a(words[word], 0x01010101u, *byte_sum);
                nk_u32_t const codes = dtype == nk_i8_k ? words[word] ^ 0x80808080u : words[word];
                nk_u8x4_to_f16x4_blackwell_(codes, subtrahend, &halves[2 * word], &halves[2 * word + 1]);
                continue;
            }
            // Element 2i sits in the high nibble of byte i; flipping I4's sign bit adds 8.
            nk_u32_t const flip = dtype == nk_i4_k ? 0x08080808u : 0;
            nk_u32_t const odd = (words[word] & 0x0F0F0F0Fu) ^ flip, even = ((words[word] >> 4) & 0x0F0F0F0Fu) ^ flip;
            nk_u8x4_to_f16x4_blackwell_(__byte_perm(even, odd, 0x5140), subtrahend, &halves[4 * word],
                                        &halves[4 * word + 1]);
            nk_u8x4_to_f16x4_blackwell_(__byte_perm(even, odd, 0x7362), subtrahend, &halves[4 * word + 2],
                                        &halves[4 * word + 3]);
        }
#pragma unroll
        for (unsigned part = 0; part < expansion; ++part)
            wide_chunks[(chunk * expansion + part) ^ (tile_row & 7)] = make_uint4(
                halves[4 * part], halves[4 * part + 1], halves[4 * part + 2], halves[4 * part + 3]);
    }
}

/** The four scale bytes at @p scales that @p mask keeps, little-endian, read one byte at a time
 *  since dense scale rows sit at any byte. */
NUMKONG_DEVICE nk_u32_t nk_cross_scale_word_blackwell_(unsigned char const *scales, nk_u32_t mask) {
    nk_u32_t word = 0;
#pragma unroll
    for (unsigned byte = 0; byte < 4; ++byte)
        if (mask >> (8 * byte) & 0xFF) word |= (nk_u32_t)scales[byte] << (8 * byte);
    return word;
}

/** Writes the block scales of slab @p slab into the scale columns at @p slot_lane, this thread's
 *  lane of them: column `4j + c` holds the 4 scales from block `4j` of A row `lane + 32c`, and 16
 *  columns on, of B row `lane + 32c`, the layout block-scaled products read in every lane quarter.
 *  Bytes at or past a row's @p blocks are caller padding, perhaps NaN codes, and stage as zeros. */
NUMKONG_DEVICE void nk_cross_stage_scales_blackwell_(nk_cross_tile_arguments_t const *shape, nk_size_t first_row,
                                                     nk_size_t first_column, nk_size_t slab, unsigned scale_bytes,
                                                     nk_size_t blocks, nk_u32_t slot_lane) {
    unsigned const lane = threadIdx.x & 31;
    for (unsigned word = 0; word < scale_bytes / 4; ++word) {
        nk_size_t const offset = slab * scale_bytes + 4 * word;
        nk_u32_t const mask = offset >= blocks       ? 0
                              : blocks - offset >= 4 ? 0xFFFFFFFFu
                                                     : (1u << (8 * (blocks - offset))) - 1;
        nk_u32_t a_words[4], b_words[4];
#pragma unroll
        for (unsigned group = 0; group < 4; ++group) {
            nk_size_t const row = first_row + lane + 32 * group, column = first_column + lane + 32 * group;
            a_words[group] = mask && row < shape->row_end
                                 ? nk_cross_scale_word_blackwell_(
                                       shape->a_scales + row * shape->a_scales_stride + offset, mask)
                                 : 0;
            b_words[group] = mask && column < shape->column_count
                                 ? nk_cross_scale_word_blackwell_(
                                       shape->b_scales + column * shape->b_scales_stride + offset, mask)
                                 : 0;
        }
        nk_tmem_store_x4_blackwell_(slot_lane + 4 * word, a_words[0], a_words[1], a_words[2], a_words[3]);
        nk_tmem_store_x4_blackwell_(slot_lane + 16 + 4 * word, b_words[0], b_words[1], b_words[2], b_words[3]);
    }
}

/** Adds the squares of the first @p blocks blocks of staged 128-byte row @p tile_row of @p operand
 *  in block-scaled @p format, each block's times its scale squared, from the slab's @p scales,
 *  reading the 16-byte chunks through the swizzle. */
NUMKONG_DEVICE void nk_cross_stage_scaled_row_blackwell_(nk_block_scaled_format_t format, unsigned char const *operand,
                                                         unsigned tile_row, unsigned char const *scales,
                                                         nk_size_t blocks, nk_f32_t *real_sum) {
    unsigned const chunk_elements = format.element_dtype == nk_e2m1_k ? 32 : 16;
    unsigned const block_size = (unsigned)format.block_size;
    unsigned char const *const row = operand + tile_row * nk_cross_slab_bytes_blackwell_k;
    nk_f32_t slab_sum = 0;
    for (unsigned block = 0; block < blocks; ++block) {
        nk_f32_t block_sum = 0;
        for (unsigned element = block * block_size; element != (block + 1) * block_size; ++element) {
            unsigned char const *bytes = row + 16 * ((element / chunk_elements) ^ (tile_row & 7));
            nk_f32_t const value = nk_cross_load_f32_(format.element_dtype, bytes, element % chunk_elements);
            block_sum += value * value;
        }
        nk_f32_t const scale = nk_block_scaled_decode_scale_serial_(scales[block], format.scale_dtype);
        slab_sum += block_sum * scale * scale;
    }
    *real_sum += slab_sum;
}

/** The tile row a stage or epilogue thread owns. Warp w reaches lanes from 32 · (w mod 4) on, so
 *  warps 2 to 9 put thread t on lane t mod 128: stage warp w and epilogue warp w + 4 share rows. */
NUMKONG_DEVICE unsigned nk_cross_drain_row_blackwell_(void) { return threadIdx.x % 128; }

NUMKONG_DEVICE nk_cross_schedule_blackwell_t nk_cross_schedule_blackwell_(nk_cross_kernel_blackwell_t const *kernel,
                                                                          nk_cross_tile_arguments_t const *shape) {
    nk_cross_schedule_blackwell_t schedule;
    unsigned const raw_bytes = nk_cross_raw_bytes_blackwell_(kernel->dtype);
    nk_size_t const exact_slabs = nk_cross_exact_slabs_blackwell_(kernel->dtype);
    nk_size_t const block_size = nk_block_scaled_format_of_dtype(kernel->dtype).block_size;
    schedule.format = nk_block_scaled_format_of_dtype(kernel->dtype);
    schedule.staging = raw_bytes       ? nk_cross_staging_expand_k
                       : block_size    ? nk_cross_staging_block_scales_k
                       : kernel->widen ? nk_cross_staging_widen_k
                                       : nk_cross_staging_none_k;
    schedule.visits_stages = schedule.staging != nk_cross_staging_none_k || kernel->metric != nk_cross_metric_dot_k;
    schedule.offsets = kernel->dtype == nk_u8_k;
    schedule.publishes = kernel->metric != nk_cross_metric_dot_k || schedule.offsets;
    schedule.box_bytes = raw_bytes ? raw_bytes : nk_cross_slab_bytes_blackwell_k;
    schedule.box_offset = raw_bytes ? nk_cross_stage_bytes_blackwell_k : 0;
    schedule.stage_bytes = schedule.box_offset +
                           (nk_cross_rows_blackwell_k + nk_cross_columns_blackwell_k) * schedule.box_bytes;
    schedule.ring_stages = nk_cross_ring_bytes_blackwell_k / schedule.stage_bytes;
    // Scale bytes per row and slab: 16 for NVFP4, 8 for MXFP4, 4 for MXFP8.
    schedule.scale_bytes = block_size ? nk_cross_slab_bytes_blackwell_k *
                                            (schedule.format.element_dtype == nk_e2m1_k ? 2 : 1) / (unsigned)block_size
                                      : 0;
    schedule.blocks = block_size ? shape->depth / block_size : 0;
    schedule.slabs = shape->depth_slabs;
    schedule.chunk_slabs = exact_slabs && schedule.slabs > exact_slabs ? exact_slabs
                                                                       : (schedule.slabs ? schedule.slabs : 1);
    schedule.chunks = nk_size_divide_round_up_(schedule.slabs ? schedule.slabs : 1, schedule.chunk_slabs);
    schedule.paired = nk_cross_paired_blackwell_(kernel->dtype, kernel->metric);
    schedule.pair_rank = schedule.paired ? nk_cluster_rank_blackwell_() : 0;
    schedule.tile_rows = nk_cross_rows_blackwell_k << schedule.paired;
    schedule.tile_columns = nk_cross_columns_blackwell_k << schedule.paired;
    schedule.first_tile = blockIdx.x >> schedule.paired;
    schedule.tile_stride = gridDim.x >> schedule.paired;
    return schedule;
}

/** One 32-byte depth step of a @b [256,256] pair tile over K-major codes the tensor cores read as
 *  loaded, BF16 and F16 as @c kind::f16 and E4M3 and E5M2 as @c kind::f8f6f4, into F32 sums. */
NUMKONG_DEVICE void nk_cross_mma_pair_blackwell_(nk_dtype_t dtype, nk_u32_t accumulator, nk_u64_t a, nk_u64_t b,
                                                 nk_u32_t accumulate) {
    nk_u32_t const format = dtype == nk_bf16_k || dtype == nk_e5m2_k;
    nk_u32_t const instruction = nk_mma_instruction_blackwell_(format, format, 256, 256, nk_major_k_k, nk_major_k_k);
    if (dtype == nk_bf16_k || dtype == nk_f16_k) nk_mma_f16_pair_blackwell_(accumulator, a, b, instruction, accumulate);
    else nk_mma_f8f6f4_pair_blackwell_(accumulator, a, b, instruction, accumulate);
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
            nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&control->ready[stage]), 4);
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
    if (schedule->paired) return tensor_memory;
    // UE8M0 code 127 is 2⁰; every byte of the 32 columns holds it, whatever layout is read.
    if (warp >= 6)
        for (unsigned column = 0; column < 32; column += 8)
            nk_tmem_fill_x8_blackwell_(nk_tmem_offset_blackwell_(tensor_memory, warp % 4 * 32,
                                                                 nk_cross_unit_scales_column_blackwell_k + column),
                                       0x7F7F7F7Fu);
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
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
NUMKONG_DEVICE void nk_cross_load_blackwell_(nk_cross_kernel_blackwell_t const *kernel,
                                             nk_cross_schedule_blackwell_t const *schedule,
                                             nk_cross_control_blackwell_t *control, unsigned char *stages,
                                             nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_u32_t const stages_address = nk_shared_address_ampere_(stages);
    nk_u32_t const copy_bytes = (nk_cross_rows_blackwell_k + nk_cross_columns_blackwell_k) * schedule->box_bytes;
    nk_size_t const half = schedule->pair_rank * nk_cross_rows_blackwell_k;
    nk_prefetch_map_blackwell_(&arguments->a_map);
    nk_prefetch_map_blackwell_(&arguments->b_map);
    nk_u32_t stage = 0, phase = 0;
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, kernel->triangle, schedule, tile, &first_row, &first_column))
            continue;
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            nk_u32_t const loaded = nk_shared_address_ampere_(&control->loaded[stage]);
            nk_u32_t const a_box = stages_address + stage * schedule->stage_bytes + schedule->box_offset;
            nk_u32_t const b_box = a_box + nk_cross_rows_blackwell_k * schedule->box_bytes;
            nk_i32_t const column = (nk_i32_t)(slab * schedule->box_bytes);
            nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->consumed[stage]), phase ^ 1);
            if (schedule->paired) {
                if (schedule->pair_rank == 0) nk_mbarrier_expect_bytes_blackwell_(loaded, 2 * copy_bytes);
                nk_u32_t const leader_loaded = nk_cluster_address_blackwell_(loaded, 0);
                nk_load_box_pair_blackwell_(a_box, &arguments->a_map, leader_loaded, column,
                                            (nk_i32_t)(first_row + half));
                nk_load_box_pair_blackwell_(b_box, &arguments->b_map, leader_loaded, column,
                                            (nk_i32_t)(first_column + half));
            }
            else {
                nk_mbarrier_expect_bytes_blackwell_(loaded, copy_bytes);
                nk_load_box_blackwell_(a_box, &arguments->a_map, loaded, column, (nk_i32_t)first_row);
                nk_load_box_blackwell_(b_box, &arguments->b_map, loaded, column, (nk_i32_t)first_column);
            }
            nk_cross_advance_blackwell_(schedule, &stage, &phase);
        }
    }
}

/** Issues every tile's products from one thread, a chunk of slabs per accumulator, alternating the
 *  two so the epilogue drains one while the other fills. */
NUMKONG_DEVICE void nk_cross_multiply_blackwell_(nk_cross_kernel_blackwell_t const *kernel,
                                                 nk_cross_schedule_blackwell_t const *schedule,
                                                 nk_cross_control_blackwell_t *control, unsigned char *stages,
                                                 nk_u32_t tensor_memory,
                                                 nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_u32_t const stages_address = nk_shared_address_ampere_(stages);
    int const staged = schedule->staging != nk_cross_staging_none_k;
    int const scaled = schedule->staging == nk_cross_staging_block_scales_k;
    if (schedule->pair_rank != 0) return;
    nk_u32_t stage = 0, phase = 0, iteration = 0;
    for (nk_size_t tile = schedule->first_tile; tile < arguments->tile.tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, kernel->triangle, schedule, tile, &first_row, &first_column))
            continue;
        for (nk_size_t chunk = 0; chunk < schedule->chunks; ++chunk, ++iteration) {
            nk_u32_t const buffer = iteration & 1, buffer_phase = (iteration >> 1) & 1;
            nk_u32_t const accumulator = tensor_memory + buffer * schedule->tile_columns;
            nk_size_t const first_slab = chunk * schedule->chunk_slabs;
            nk_size_t const end_slab = first_slab + schedule->chunk_slabs < schedule->slabs
                                           ? first_slab + schedule->chunk_slabs
                                           : schedule->slabs;
            nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->drained[buffer]), buffer_phase ^ 1);
            nk_tmem_fence_after_blackwell_();
            for (nk_size_t slab = first_slab; slab < end_slab; ++slab) {
                nk_u64_t *const arrival = staged ? &control->ready[stage] : &control->loaded[stage];
                nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(arrival), phase);
                nk_tmem_fence_after_blackwell_();
                nk_u32_t const a_stage = stages_address + stage * schedule->stage_bytes;
                // K-major rows, whose leading offset goes unread, in 8-row groups 1024 bytes apart.
                nk_u64_t const a = nk_smem_descriptor_blackwell_(a_stage, nk_smem_swizzle_128_blackwell_k, 16, 1024);
                nk_u64_t const b = nk_smem_descriptor_blackwell_(a_stage + nk_cross_a_bytes_blackwell_k,
                                                                 nk_smem_swizzle_128_blackwell_k, 16, 1024);
                nk_u32_t const scales = tensor_memory + (scaled ? nk_cross_scale_slots_column_blackwell_k + 32 * stage
                                                                : nk_cross_unit_scales_column_blackwell_k);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step) {
                    nk_u32_t const accumulate = (nk_u32_t)((slab - first_slab) | step) != 0;
                    if (schedule->paired)
                        nk_cross_mma_pair_blackwell_(kernel->dtype, accumulator, a + 2 * step, b + 2 * step,
                                                     accumulate);
                    else kernel->mma(accumulator, a + 2 * step, b + 2 * step, accumulate, scales, step);
                }
                nk_cross_commit_blackwell_(schedule, nk_shared_address_ampere_(&control->consumed[stage]));
                nk_cross_advance_blackwell_(schedule, &stage, &phase);
            }
            nk_cross_commit_blackwell_(schedule, nk_shared_address_ampere_(&control->accumulated[buffer]));
        }
    }
}

/** Copies this thread's lane of the slab's block scales into the stage's scale slot and adds its
 *  rows' scaled squares for a metric. */
NUMKONG_DEVICE void nk_cross_prepare_scales_blackwell_(nk_cross_kernel_blackwell_t const *kernel,
                                                       nk_cross_schedule_blackwell_t const *schedule,
                                                       nk_cross_tile_arguments_t const *shape,
                                                       unsigned char const *stage, nk_u32_t scale_slot, nk_size_t slab,
                                                       nk_size_t first_row, nk_size_t first_column,
                                                       nk_cross_statistics_blackwell_t *statistics) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_size_t const row = first_row + tile_row, column = first_column + tile_row;
    nk_size_t const scales_offset = slab * schedule->scale_bytes, remaining_blocks = schedule->blocks - scales_offset;
    nk_size_t const slab_blocks = remaining_blocks < schedule->scale_bytes ? remaining_blocks : schedule->scale_bytes;
    int const squares = kernel->metric != nk_cross_metric_dot_k;
    nk_tmem_fence_after_blackwell_();
    nk_cross_stage_scales_blackwell_(shape, first_row, first_column, slab, schedule->scale_bytes, schedule->blocks,
                                     scale_slot);
    if (squares && row < shape->row_end)
        nk_cross_stage_scaled_row_blackwell_(schedule->format, stage, tile_row,
                                             shape->a_scales + row * shape->a_scales_stride + scales_offset,
                                             slab_blocks, &statistics->row_real_norm);
    if (squares && kernel->triangle == nk_cross_triangle_upper_k && column < shape->column_count)
        nk_cross_stage_scaled_row_blackwell_(schedule->format, stage + nk_cross_a_bytes_blackwell_k, tile_row,
                                             shape->b_scales + column * shape->b_scales_stride + scales_offset,
                                             slab_blocks, &statistics->column_real_norm);
    nk_tmem_wait_store_blackwell_();
    nk_tmem_fence_before_blackwell_();
}

/** Prepares this thread's A and B rows of one stage as the staging says, adding their squares for a
 *  metric, only B's for @c symmetric, which packs none, and their byte sums for the U8 offset. */
NUMKONG_DEVICE void nk_cross_prepare_stage_blackwell_(nk_cross_kernel_blackwell_t const *kernel,
                                                      nk_cross_schedule_blackwell_t const *schedule,
                                                      nk_cross_tile_arguments_t const *shape, unsigned char *stage,
                                                      nk_u32_t scale_slot, nk_size_t slab, nk_size_t first_row,
                                                      nk_size_t first_column,
                                                      nk_cross_statistics_blackwell_t *statistics) {
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_cross_norm_update_t const squares = kernel->metric != nk_cross_metric_dot_k ? kernel->norm_update : 0;
    nk_cross_norm_update_t const column_squares = kernel->triangle == nk_cross_triangle_upper_k ? squares : 0;
    unsigned char *const b_operand = stage + nk_cross_a_bytes_blackwell_k;
    unsigned char const *const a_box = stage + schedule->box_offset;
    unsigned char const *const b_box = a_box + nk_cross_rows_blackwell_k * schedule->box_bytes;
    switch (schedule->staging) {
    case nk_cross_staging_expand_k:
        nk_cross_expand_row_blackwell_(kernel->dtype, squares, a_box, stage, tile_row, &statistics->row_integer_norm,
                                       &statistics->row_real_norm, &statistics->row_byte_sum);
        nk_cross_expand_row_blackwell_(kernel->dtype, column_squares, b_box, b_operand, tile_row,
                                       &statistics->column_integer_norm, &statistics->column_real_norm,
                                       &statistics->column_byte_sum);
        break;
    case nk_cross_staging_block_scales_k:
        nk_cross_prepare_scales_blackwell_(kernel, schedule, shape, stage, scale_slot, slab, first_row, first_column,
                                           statistics);
        break;
    default:
        nk_cross_stage_row_blackwell_(squares, kernel->widen, stage, tile_row, &statistics->row_integer_norm,
                                      &statistics->row_real_norm);
        if (kernel->triangle == nk_cross_triangle_upper_k)
            nk_cross_stage_row_blackwell_(squares, kernel->widen, b_operand, tile_row, &statistics->column_integer_norm,
                                          &statistics->column_real_norm);
        break;
    }
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
 *  @p dot_scale, integer dots restored from the U8 offset, or the metric of either. */
NUMKONG_DEVICE void nk_cross_finish_sums_blackwell_(nk_cross_kernel_blackwell_t const *kernel,
                                                    nk_cross_schedule_blackwell_t const *schedule,
                                                    nk_cross_control_blackwell_t const *control, nk_u32_t bits[32],
                                                    nk_size_t row, nk_size_t first_column, unsigned tile_column,
                                                    nk_f32_t dot_scale, nk_f32_t row_norm, nk_u32_t row_byte_sum) {
    int const integer = schedule->staging == nk_cross_staging_expand_k;
    // Padding codes widen to −128 in both rows, which the byte sums never count.
    nk_u32_t const offset_correction = 16384u * (nk_u32_t)(schedule->slabs * schedule->box_bytes);
#pragma unroll
    for (unsigned offset = 0; offset < 32; ++offset) {
        if (schedule->offsets)
            bits[offset] += 128u * (row_byte_sum + control->column_byte_sums[tile_column + offset]) - offset_correction;
        nk_f32_t const dot = !integer                              ? __uint_as_float(bits[offset]) * dot_scale
                             : kernel->norm == nk_cross_norm_u32_k ? (nk_f32_t)bits[offset]
                                                                   : (nk_f32_t)(nk_i32_t)bits[offset];
        nk_f32_t const column_norm = control->column_norms[tile_column + offset];
        if (kernel->metric == nk_cross_metric_dot_k) {
            if (!integer) bits[offset] = __float_as_uint(dot);
        }
        else if (kernel->triangle == nk_cross_triangle_upper_k && first_column + offset == row) bits[offset] = 0;
        else if (kernel->metric == nk_cross_metric_angular_k)
            bits[offset] = __float_as_uint(nk_f32_angular_(dot, row_norm, column_norm));
        else bits[offset] = __float_as_uint(nk_f32_euclidean_(dot, row_norm, column_norm));
    }
}

/** Stores 32 output words of @p row from @p first_column, as 16-byte stores when all of them fit
 *  and align, skipping columns below the diagonal for @c symmetric. */
NUMKONG_DEVICE void nk_cross_store_sums_blackwell_(nk_cross_triangle_t triangle, nk_cross_tile_arguments_t const *shape,
                                                   nk_u32_t const bits[32], nk_size_t row, nk_size_t first_column) {
    nk_u32_t *const output = (nk_u32_t *)((unsigned char *)shape->c + row * shape->c_stride) + first_column;
    nk_size_t const begin = triangle == nk_cross_triangle_upper_k && row > first_column ? row - first_column : 0;
    nk_size_t const end = shape->column_count - first_column < 32 ? shape->column_count - first_column : 32;
    if (begin == 0 && end == 32 && ((nk_size_t)output & 15) == 0) {
#pragma unroll
        for (unsigned quartet = 0; quartet < 8; ++quartet)
            ((uint4 *)output)[quartet] = make_uint4(bits[4 * quartet], bits[4 * quartet + 1], bits[4 * quartet + 2],
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
    uint4 *const chunks = (uint4 *)(box + lane * 128);
    if (lane == 0) nk_bulk_wait_reads_but_one_blackwell_();
    __syncwarp();
#pragma unroll
    for (unsigned quartet = 0; quartet < 8; ++quartet)
        chunks[quartet ^ (lane & 7)] = make_uint4(bits[4 * quartet], bits[4 * quartet + 1], bits[4 * quartet + 2],
                                                  bits[4 * quartet + 3]);
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
NUMKONG_DEVICE void nk_cross_drain_accumulator_blackwell_(
    nk_cross_kernel_blackwell_t const *kernel, nk_cross_schedule_blackwell_t const *schedule,
    nk_cross_control_blackwell_t const *control, nk_cross_tile_arguments_blackwell_t const *arguments,
    unsigned char *boxes, nk_u32_t *stores, nk_u32_t lanes, nk_u32_t accumulator, nk_size_t chunk, nk_size_t first_row,
    nk_size_t first_column, nk_f32_t row_norm, nk_u32_t row_byte_sum) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_size_t const row = first_row + tile_row, warp_first_row = first_row + tile_row / 32 * 32;
    unsigned const columns = shape->column_count - first_column < schedule->tile_columns
                                 ? (unsigned)(shape->column_count - first_column)
                                 : schedule->tile_columns;
    int const integer = schedule->staging == nk_cross_staging_expand_k, last = chunk + 1 == schedule->chunks;
    nk_f32_t const dot_scale = kernel->output_scale * (shape->a_tensor_scale ? *shape->a_tensor_scale : 1) *
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
        nk_cross_finish_sums_blackwell_(kernel, schedule, control, bits, row, chunk_column, tile_column, dot_scale,
                                        row_norm, row_byte_sum);
        // Bulk stores clip a row's end only to 16 bytes, so a chunk crossing it goes out by rows.
        if (arguments->c_mapped && chunk_column + 32 <= shape->column_count &&
            (kernel->triangle != nk_cross_triangle_upper_k || chunk_column >= warp_first_row + 31)) {
            nk_cross_store_box_blackwell_(arguments, boxes + (*stores & 1) * nk_cross_output_box_bytes_blackwell_k,
                                          bits, warp_first_row, chunk_column);
            ++*stores;
        }
        else if (row < shape->row_end) nk_cross_store_sums_blackwell_(kernel->triangle, shape, bits, row, chunk_column);
    }
}

/** Prepares every stage the products need prepared from the four stage warps, each thread its row
 *  of A and of B, and hands every tile's statistics to the epilogue warp of the same lanes through
 *  the tile's slot, once that warp has taken the slot's previous tile: four columns of tensor
 *  memory holding the row's norm and byte sum, then the column's, whose norm comes from its B row
 *  for @c symmetric and from the pack otherwise. */
NUMKONG_DEVICE void nk_cross_stage_blackwell_(nk_cross_kernel_blackwell_t const *kernel,
                                              nk_cross_schedule_blackwell_t const *schedule,
                                              nk_cross_control_blackwell_t *control, unsigned char *stages,
                                              nk_u32_t tensor_memory,
                                              nk_cross_tile_arguments_blackwell_t const *arguments) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const tile_row = nk_cross_drain_row_blackwell_();
    nk_u32_t const lanes = nk_tmem_offset_blackwell_(tensor_memory, tile_row / 32 * 32, 0);
    int const staged = schedule->staging != nk_cross_staging_none_k;
    nk_u32_t stage = 0, phase = 0, tiles = 0;
    if (!schedule->visits_stages) return;
    for (nk_size_t tile = schedule->first_tile; tile < shape->tiles; tile += schedule->tile_stride) {
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_blackwell_(arguments, kernel->triangle, schedule, tile, &first_row, &first_column))
            continue;
        first_row += schedule->pair_rank * nk_cross_rows_blackwell_k;
        nk_cross_statistics_blackwell_t statistics = {0, 0, 0, 0, 0, 0};
        for (nk_size_t slab = 0; slab < schedule->slabs; ++slab) {
            nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->loaded[stage]), phase);
            nk_cross_prepare_stage_blackwell_(kernel, schedule, shape, stages + stage * schedule->stage_bytes,
                                              lanes + nk_cross_scale_slots_column_blackwell_k + 32 * stage, slab,
                                              first_row, first_column, &statistics);
            // Retires this thread's stage accesses before products or copies reuse the stage.
            nk_fence_async_shared_blackwell_();
            __syncwarp();
            nk_u64_t *const done = staged ? &control->ready[stage] : &control->consumed[stage];
            if ((threadIdx.x & 31) == 0) nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(done));
            nk_cross_advance_blackwell_(schedule, &stage, &phase);
        }
        if (!schedule->publishes) continue;
        nk_u32_t const slot = tiles & 1, slot_phase = (tiles >> 1) & 1;
        ++tiles;
        nk_size_t const column = first_column + tile_row;
        nk_f32_t const a_tensor_scale = shape->a_tensor_scale ? *shape->a_tensor_scale : 1;
        nk_f32_t const column_tensor_scale = kernel->triangle == nk_cross_triangle_upper_k && shape->b_tensor_scale
                                                 ? *shape->b_tensor_scale
                                                 : 1;
        nk_fui32_t const row_bits = nk_cross_norm_finalize_(kernel->norm, statistics.row_integer_norm,
                                                            statistics.row_real_norm, kernel->norm_scale);
        nk_fui32_t column_bits = {0};
        if (kernel->triangle == nk_cross_triangle_upper_k)
            column_bits = nk_cross_norm_finalize_(kernel->norm, statistics.column_integer_norm,
                                                  statistics.column_real_norm, kernel->norm_scale);
        else if (kernel->metric != nk_cross_metric_dot_k && column < shape->column_count)
            column_bits.u = ((nk_u32_t const *)shape->b_norms)[column];
        nk_f32_t const row_norm = nk_cross_norm_to_f32_(row_bits, kernel->norm) * a_tensor_scale * a_tensor_scale;
        nk_f32_t const column_norm = nk_cross_norm_to_f32_(column_bits, kernel->norm) * column_tensor_scale *
                                     column_tensor_scale;
        nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->collected[slot]), slot_phase ^ 1);
        nk_tmem_fence_after_blackwell_();
        nk_tmem_store_x4_blackwell_(lanes + nk_cross_statistics_column_blackwell_k + 4 * slot,
                                    __float_as_uint(row_norm), statistics.row_byte_sum, __float_as_uint(column_norm),
                                    statistics.column_byte_sum);
        nk_tmem_wait_store_blackwell_();
        nk_tmem_fence_before_blackwell_();
        __syncwarp();
        if ((threadIdx.x & 31) == 0) nk_mbarrier_arrive_blackwell_(nk_shared_address_ampere_(&control->measured[slot]));
    }
}

/** Drains every accumulator the products fill from the four epilogue warps: each owns the 32
 *  accumulator lanes its index modulo 4 may read, so one output row per thread, the row the stage
 *  warp of the same lanes prepares, whose statistics it takes before a tile's last chunk, sharing
 *  the column's norm and byte sum with the other epilogue warps. */
NUMKONG_DEVICE void nk_cross_drain_blackwell_(nk_cross_kernel_blackwell_t const *kernel,
                                              nk_cross_schedule_blackwell_t const *schedule,
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
        if (!nk_cross_tile_origin_blackwell_(arguments, kernel->triangle, schedule, tile, &first_row, &first_column))
            continue;
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
                control->column_norms[tile_row] = __uint_as_float(statistics[2]);
                control->column_byte_sums[tile_row] = statistics[3];
                nk_barrier_sync_blackwell_(1, 128);
            }
            nk_u32_t const buffer = iteration & 1, buffer_phase = (iteration >> 1) & 1;
            nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&control->accumulated[buffer]), buffer_phase);
            nk_tmem_fence_after_blackwell_();
            nk_cross_drain_accumulator_blackwell_(kernel, schedule, control, arguments, boxes, &stores, lanes,
                                                  buffer * schedule->tile_columns, chunk, first_row, first_column,
                                                  __uint_as_float(statistics[0]), statistics[1]);
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

/**
 *  @brief Every tile of one launch, shared by every dtype and metric.
 *  @param[in] dtype The input dtype; a block-scaled one has its slab's scales staged into tensor
 *      memory by the stage warps, between the load and the product, like a widening.
 *  @param[in] mma Issues one 32-byte depth step; inlined, being a constant.
 *  @param[in] widen Rewrites staged A codes, and B codes for @c symmetric, before the tensor cores
 *      read them; null where they read the codes as they are.
 *  @param[in] output_scale Undoes the power of two a widening introduced, or 1.
 *  @param[in] norm How the squared norms are stored: F32, or integer sums for integer dtypes.
 *  @param[in] norm_update Adds staged squares for a metric, see @c nk_cross_norm_update_t.
 *  @param[in] norm_scale Undoes the power of two the norm update's widening introduced, or 1.
 *  @param[in] triangle Whether tiles and outputs below the diagonal are skipped.
 *  @param[in] metric The dot product itself, or a distance from it and the two squared norms.
 *
 *  Warp 0 loads, warp 1 multiplies, warps 2 to 5 prepare stages, and warps 6 to 9 drain
 *  accumulators. An integer dtype loads narrower slabs of codes that the stage warps widen to F16,
 *  and accumulates in chunks of depth whose F32 sums stay exact, adding each chunk's into I32 sums
 *  in tensor memory; U8 widens offset by −128, restored from the byte sums of both rows.
 */
NUMKONG_DEVICE void nk_cross_tile_blackwell_(nk_dtype_t dtype, nk_cross_mma_blackwell_t mma,
                                             nk_cross_widen_blackwell_t widen, nk_f32_t output_scale,
                                             nk_cross_norm_t norm, nk_cross_norm_update_t norm_update,
                                             nk_f32_t norm_scale, nk_cross_triangle_t triangle,
                                             nk_cross_metric_t metric,
                                             nk_cross_tile_arguments_blackwell_t const *arguments) {
    extern __shared__ unsigned char nk_cross_dynamic_shared_blackwell_[];
    nk_cross_kernel_blackwell_t const kernel = {dtype,       mma,        widen,    output_scale, norm,
                                                norm_update, norm_scale, triangle, metric};
    nk_cross_schedule_blackwell_t const schedule = nk_cross_schedule_blackwell_(&kernel, &arguments->tile);
    // The swizzled stages need 1024-byte alignment, which dynamic shared memory does not promise.
    unsigned char *const stages = nk_shared_aligned_ampere_(nk_cross_dynamic_shared_blackwell_, 1024);
    nk_cross_control_blackwell_t *const control =
        (nk_cross_control_blackwell_t *)(stages + nk_cross_ring_bytes_blackwell_k +
                                         nk_cross_output_staging_bytes_blackwell_k);
    nk_u32_t const tensor_memory = nk_cross_initialize_blackwell_(&schedule, control);
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0 && lane == 0) nk_cross_load_blackwell_(&kernel, &schedule, control, stages, arguments);
    else if (warp == 1 && lane == 0)
        nk_cross_multiply_blackwell_(&kernel, &schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 6) nk_cross_drain_blackwell_(&kernel, &schedule, control, stages, tensor_memory, arguments);
    else if (warp >= 2) nk_cross_stage_blackwell_(&kernel, &schedule, control, stages, tensor_memory, arguments);
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
                                                            void *stream) {
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

/** Validates the contract, describes A's @p a_rows rows and B's @p column_count rows, and launches
 *  as many blocks of @p kernel as stay resident, each walking output tiles with a stride of the
 *  grid, in pairs for a @p dtype and @p metric that @c nk_cross_paired_blackwell_ pairs. @p b_norms
 *  is the packed column norms a @c packed metric reads, or null. A block-scaled @p dtype needs its
 *  block to divide @p depth and scales on both operands. Codes need 16-byte rows for the tensor
 *  maps, while scales may sit at any byte, as dense rows of them do. */
NUMKONG_INLINE nk_status_t nk_cross_launch_blackwell_(void const *kernel, nk_cross_operand_t const *a, nk_size_t a_rows,
                                                      nk_cross_operand_t const *b, void const *b_norms, void *c,
                                                      nk_size_t result_bytes, nk_size_t row_start, nk_size_t row_end,
                                                      nk_size_t column_count, nk_size_t depth, nk_dtype_t dtype,
                                                      nk_cross_metric_t metric, nk_size_t depth_bytes,
                                                      nk_size_t a_stride, nk_size_t b_stride, nk_size_t c_stride,
                                                      void *stream) {
    nk_size_t const block_size = nk_block_scaled_format_of_dtype(dtype).block_size;
    unsigned const raw_bytes = nk_cross_raw_bytes_blackwell_(dtype);
    unsigned const box_bytes = raw_bytes ? raw_bytes : nk_cross_slab_bytes_blackwell_k;
    if (block_size && (depth % block_size || !a->scales || !b->scales)) return nk_unexpected_dimensions_k;
    if ((((nk_size_t)a->elements) | a_stride | ((nk_size_t)b->elements) | b_stride) & 15 ||
        (((nk_size_t)c) | c_stride) & (result_bytes - 1))
        return nk_misaligned_k;
    if (row_end <= row_start || column_count == 0) return nk_success_k;
    nk_cross_tile_arguments_blackwell_t arguments;
    if (!nk_cross_map_blackwell_(&arguments.a_map, a->elements, a_rows, depth_bytes, a_stride, box_bytes,
                                 nk_cross_rows_blackwell_k) ||
        !nk_cross_map_blackwell_(&arguments.b_map, b->elements, column_count, depth_bytes, b_stride, box_bytes,
                                 nk_cross_columns_blackwell_k))
        return nk_device_memory_mismatch_k;
    arguments.c_mapped = !((((nk_size_t)c) | c_stride) & 15) &&
                         nk_cross_map_blackwell_(&arguments.c_map, c, row_end, column_count * result_bytes, c_stride,
                                                 128, 32);
    nk_cross_tile_arguments_t *const tile = &arguments.tile;
    int const paired = nk_cross_paired_blackwell_(dtype, metric);
    nk_size_t const tile_rows = nk_cross_rows_blackwell_k << paired,
                    tile_columns = nk_cross_columns_blackwell_k << paired;
    nk_size_t const column_tiles = nk_size_divide_round_up_(column_count, tile_columns);
    nk_size_t const tiles = nk_size_divide_round_up_(row_end - row_start, tile_rows) * column_tiles;
    tile->a = (unsigned char const *)a->elements, tile->b = (unsigned char const *)b->elements, tile->c = c;
    tile->row_start = row_start, tile->row_end = row_end, tile->column_count = column_count;
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

/*  The same shapes as @c nk_define_cross_packed_cuda_ and its kin, over the TMA launch: the site
 *  passes the tile's own leading arguments, from the MMA issue to the norm scale, last. */
#pragma region Cross Macros

/**
 *  @brief Generates C = A × Bᵀ, or its angular or euclidean distances, over a B packed by
 *      @c nk_define_cross_pack_cuda_, one block per resident slot walking @b [128,128] tiles.
 *  @param[in] metric @c dot, @c angular or @c euclidean, naming both the entry and the epilogue.
 *  @param[in] ... The tile's own leading arguments, see @c nk_cross_tile_blackwell_.
 *  @sa nk_define_cross_packed_ for the host original.
 */
#define nk_define_cross_tma_packed_blackwell_(metric, input_type_name, isa_suffix, input_value_type,                  \
                                              packed_value_type, result_value_type, depth_simd_dimensions,            \
                                              dimensions_per_value, ...)                                              \
    static __global__ void __launch_bounds__(nk_cross_threads_blackwell_k, 1)                                         \
        nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_(                                             \
            __grid_constant__ nk_cross_tile_arguments_blackwell_t const arguments) {                                  \
        nk_cross_tile_blackwell_(nk_##input_type_name##_k, __VA_ARGS__, nk_cross_triangle_full_k,                     \
                                 nk_cross_metric_##metric##_k, &arguments);                                           \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_##metric##s_packed_##input_type_name##_##isa_suffix(                                   \
        nk_cross_##input_type_name##_operand_t const *a_operand, void const *b_packed_buffer,                         \
        nk_##result_value_type##_t *c_matrix, nk_size_t row_count, nk_size_t column_count, nk_size_t depth,           \
        nk_size_t a_stride, nk_size_t c_stride, void *stream) {                                                       \
        nk_size_t const row_bytes = nk_cross_padded_values_simt_(depth, depth_simd_dimensions, dimensions_per_value,  \
                                                                 sizeof(nk_##packed_value_type##_t)) *                \
                                    sizeof(nk_##packed_value_type##_t);                                               \
        nk_size_t const scales_stride = nk_cross_scales_stride_(nk_##input_type_name##_k, depth);                     \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;     \
        nk_u8_t const *b_rows = (nk_u8_t const *)(header + 1);                                                        \
        nk_cross_operand_t const a = nk_cross_operand_(nk_##input_type_name##_k, a_operand, a_stride);                \
        nk_cross_operand_t const b = {b_rows, scales_stride ? b_rows + column_count * row_bytes : NUMKONG_NULL,       \
                                      scales_stride, &header->tensor_scale};                                          \
        return nk_cross_launch_blackwell_(                                                                            \
            (void const *)nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_, &a, row_count, &b,        \
            b_rows + column_count * (row_bytes + scales_stride), c_matrix, sizeof(nk_##result_value_type##_t), 0,     \
            row_count, column_count, depth, nk_##input_type_name##_k, nk_cross_metric_##metric##_k,                   \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), a_stride, row_bytes, c_stride, stream); \
    }

/**
 *  @brief Generates the Gram matrix C = A × Aᵀ, or its angular or euclidean distances, over rows
 *      [row_start, row_start + row_count): the upper triangle, with the diagonal for dots and zeros
 *      on it for distances, skipping tiles wholly below it.
 *
 *  Takes the parameters of @c nk_define_cross_tma_packed_blackwell_, so one bundle feeds both.
 *
 *  @sa nk_define_cross_symmetric_ for the host original.
 */
#define nk_define_cross_tma_symmetric_blackwell_(metric, input_type_name, isa_suffix, input_value_type,                \
                                                 packed_value_type, result_value_type, depth_simd_dimensions,          \
                                                 dimensions_per_value, ...)                                            \
    static __global__ void __launch_bounds__(nk_cross_threads_blackwell_k, 1)                                          \
        nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_(                                           \
            __grid_constant__ nk_cross_tile_arguments_blackwell_t const arguments) {                                   \
        nk_cross_tile_blackwell_(nk_##input_type_name##_k, __VA_ARGS__, nk_cross_triangle_upper_k,                     \
                                 nk_cross_metric_##metric##_k, &arguments);                                            \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##metric##s_symmetric_##input_type_name##_##isa_suffix(                                 \
        nk_cross_##input_type_name##_operand_t const *vectors_operand, nk_size_t vectors_count, nk_size_t depth,       \
        nk_size_t stride, nk_##result_value_type##_t *result, nk_size_t result_stride, nk_size_t row_start,            \
        nk_size_t row_count, void *stream) {                                                                           \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;       \
        nk_cross_operand_t const vectors = nk_cross_operand_(nk_##input_type_name##_k, vectors_operand, stride);       \
        return nk_cross_launch_blackwell_(                                                                             \
            (void const *)nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_, &vectors,               \
            vectors_count, &vectors, 0, result, sizeof(nk_##result_value_type##_t), row_start, row_end, vectors_count, \
            depth, nk_##input_type_name##_k, nk_cross_metric_##metric##_k,                                             \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), stride, stride, result_stride, stream);  \
    }

/** Both shapes of one metric over the TMA launch, packed and symmetric. */
#define nk_define_cross_tma_blackwell_(metric, input_type_name, isa_suffix, input_value_type, packed_value_type,       \
                                       result_value_type, depth_simd_dimensions, dimensions_per_value, ...)            \
    nk_define_cross_tma_packed_blackwell_(metric, input_type_name, isa_suffix, input_value_type, packed_value_type,    \
                                          result_value_type, depth_simd_dimensions, dimensions_per_value, __VA_ARGS__) \
    nk_define_cross_tma_symmetric_blackwell_(metric, input_type_name, isa_suffix, input_value_type, packed_value_type, \
                                             result_value_type, depth_simd_dimensions, dimensions_per_value,           \
                                             __VA_ARGS__)

#pragma endregion Cross Macros

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

#pragma region BF16

nk_define_cross_pack_cuda_(bf16, blackwell, bf16, bf16, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, bf16, blackwell, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                               /*dimensions_per_value=*/1, nk_dots_bf16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_cuda_(f16, blackwell, f16, f16, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, f16, blackwell, f16, f16, f32, /*depth_simd_dimensions=*/8,
                               /*dimensions_per_value=*/1, nk_dots_f16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_pack_cuda_(e5m2, blackwell, e5m2, e5m2, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, nk_dots_e5m2_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_pack_cuda_(e4m3, blackwell, e4m3, e4m3, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, nk_dots_e4m3_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_pack_cuda_(e3m2, blackwell, e3m2, e5m2, nk_load_f6_to_f8_ada_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, e3m2, blackwell, e3m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, nk_dots_e5m2_mma_blackwell_, nk_f6x4_to_f8x4_ada_,
                               /*output_scale=*/16777216.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_pack_cuda_(e2m3, blackwell, e2m3, e4m3, nk_load_f6_to_f8_ada_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, e2m3, blackwell, e2m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, nk_dots_e4m3_mma_blackwell_, nk_f6x4_to_f8x4_ada_,
                               /*output_scale=*/4096.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_cuda_(e2m1, blackwell, e2m1x2, e2m1x2, nk_load_b8_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, e2m1, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, nk_dots_e2m1_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_pack_cuda_(i8, blackwell, i8, i8, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, i8, blackwell, i8, i8, i32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, nk_dots_f16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_cuda_(i4, blackwell, i4x2, i4x2, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, i4, blackwell, i4x2, i4x2, i32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, nk_dots_f16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_cuda_(u8, blackwell, u8, u8, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, u8, blackwell, u8, u8, u32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, nk_dots_f16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_cuda_(u4, blackwell, u4x2, u4x2, nk_load_b8_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, u4, blackwell, u4x2, u4x2, u32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, nk_dots_f16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion U4

#pragma region Block Scaled Floats

nk_define_cross_pack_size_simt_(nvfp4, blackwell, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                                /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_cuda_(nvfp4, blackwell)
nk_define_cross_pack_rows_cuda_(nvfp4, blackwell, e2m1x2, e2m1x2, nk_load_b8_, /*norm_value_type=*/f32,
                                nk_e2m1_lane_sumsq_,
                                /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, nvfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, nk_dots_nvfp4_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)
nk_define_cross_pack_size_simt_(mxfp4, blackwell, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                                /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_cuda_(mxfp4, blackwell)
nk_define_cross_pack_rows_cuda_(mxfp4, blackwell, e2m1x2, e2m1x2, nk_load_b8_, /*norm_value_type=*/f32,
                                nk_e2m1_lane_sumsq_,
                                /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_tma_blackwell_(dot, mxfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, nk_dots_mxfp4_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)
nk_define_cross_pack_size_simt_(mxfp8e4m3, blackwell, e4m3, f32, /*depth_simd_dimensions=*/16,
                                /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_cuda_(mxfp8e4m3, blackwell)
nk_define_cross_pack_rows_cuda_(mxfp8e4m3, blackwell, e4m3, e4m3, nk_load_b8_, /*norm_value_type=*/f32,
                                nk_e4m3_lane_sumsq_,
                                /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, mxfp8e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, nk_dots_mxfp8e4m3_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)
nk_define_cross_pack_size_simt_(mxfp8e5m2, blackwell, e5m2, f32, /*depth_simd_dimensions=*/16,
                                /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_cuda_(mxfp8e5m2, blackwell)
nk_define_cross_pack_rows_cuda_(mxfp8e5m2, blackwell, e5m2, e5m2, nk_load_b8_, /*norm_value_type=*/f32,
                                nk_e5m2_lane_sumsq_,
                                /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_tma_blackwell_(dot, mxfp8e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, nk_dots_mxfp8e5m2_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                               /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion Block Scaled Floats

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_BLACKWELL
#endif // NUMKONG_DOTS_BLACKWELL_CUH
