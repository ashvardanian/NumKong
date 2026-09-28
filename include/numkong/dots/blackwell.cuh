/**
 *  @file include/numkong/dots/blackwell.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated Batched Dot Products for the NVIDIA compute capability 10.x family.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/ampere.cuh
 *
 *  Persistent blocks of six warps own @b [128,128] output tiles. One thread streams 128-byte depth
 *  slabs of A and B through a six-stage ring with the Tensor Memory Accelerator, which swizzles
 *  them into the canonical 128-byte layout and zero-fills every edge, and one thread issues
 *  `tcgen05.mma` over shared-memory descriptors into one of two F32 accumulators in tensor memory,
 *  so four epilogue warps drain one tile while the next accumulates, except where they must first
 *  widen the next tile's codes, which the products then wait for. Every dtype multiplies 32 bytes
 *  of depth per instruction: BF16 and F16 as @c kind::f16, Float8 as @c kind::f8f6f4, E2M1 as the
 *  block-scaled @c kind::mxf4 with every scale at 2⁰, and Float6 as the Float8 of the next exponent
 *  width, which is its own code with the sign moved up and a power of two the output undoes. The
 *  epilogue warps move A's signs in shared memory, and the pack moves B's. Integers and wider
 *  floats stay on the Ampere and @c cuda capabilities. Only the "100f" family code carries these
 *  instructions, so every other device pass traps.
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
    nk_cross_threads_blackwell_k = 192,
    nk_cross_rows_blackwell_k = 128,
    nk_cross_columns_blackwell_k = 128,
    nk_cross_slab_bytes_blackwell_k = 128,
    nk_cross_a_bytes_blackwell_k = nk_cross_rows_blackwell_k * nk_cross_slab_bytes_blackwell_k,
    nk_cross_stage_bytes_blackwell_k = (nk_cross_rows_blackwell_k + nk_cross_columns_blackwell_k) *
        nk_cross_slab_bytes_blackwell_k,
    nk_cross_stages_blackwell_k = 196608 / nk_cross_stage_bytes_blackwell_k,
    nk_cross_barriers_blackwell_k = 3 * nk_cross_stages_blackwell_k + 4,
    nk_cross_shared_bytes_blackwell_k = 1024 + nk_cross_stages_blackwell_k * nk_cross_stage_bytes_blackwell_k +
        nk_cross_barriers_blackwell_k * 8 + 16 + nk_cross_columns_blackwell_k * 4,
};

/** Everything one launch shares: tensor maps of A and B rows, 128-byte boxes of 128 rows, the
 *  shapes and output every tile takes, and how many row tiles sweep the columns together. Passed as
 *  a grid constant, so the maps keep an address. */
typedef struct {
    CUtensorMap a_map;
    CUtensorMap b_map;
    nk_cross_tile_arguments_t tile;
    nk_size_t group_rows;
} nk_cross_tile_arguments_blackwell_t;

/** Issues one 32-byte depth step of a @b [128,128] tile into the accumulator at tensor-memory
 *  address @p accumulator, adding to it unless @p accumulate is zero. @p scales addresses the block
 *  scales, which only block-scaled kinds read. */
typedef void (*nk_cross_mma_blackwell_t)(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                         nk_u32_t scales);

/** Rewrites four codes into the four codes the tensor cores multiply, exactly. */
typedef nk_u32_t (*nk_cross_widen_blackwell_t)(nk_u32_t codes);

#pragma endregion Configuration

#pragma region Instructions

#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 && \
    __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100

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

/* One block-scaled step over packed nibble pairs, 32 codes per scale, A's and B's scales read from
 *  tensor memory at @p scales and 16 columns past it. */
NUMKONG_DEVICE void nk_mma_mxf4_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                           nk_u32_t accumulate, nk_u32_t scales) {
    asm volatile("{\n.reg .pred accumulate;\n"                                                //
                 "setp.ne.b32 accumulate, %4, 0;\n"                                           //
                 "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, " //
                 "[%5], [%6], accumulate;\n}\n" ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate), "r"(scales), "r"(scales + 16)
                 : "memory");
}

/* Arrives on @p barrier once every tensor-core operation this thread issued so far completes. */
NUMKONG_DEVICE void nk_mma_commit_blackwell_(nk_u32_t barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(barrier)
                 : "memory");
}

#else

NUMKONG_DEVICE void nk_mbarrier_init_blackwell_(nk_u32_t barrier, nk_u32_t count) { __trap(); }
NUMKONG_DEVICE void nk_mbarrier_init_fence_blackwell_(void) { __trap(); }
NUMKONG_DEVICE void nk_mbarrier_wait_blackwell_(nk_u32_t barrier, nk_u32_t parity) { __trap(); }
NUMKONG_DEVICE void nk_mbarrier_arrive_blackwell_(nk_u32_t barrier) { __trap(); }
NUMKONG_DEVICE void nk_mbarrier_expect_bytes_blackwell_(nk_u32_t barrier, nk_u32_t bytes) { __trap(); }
NUMKONG_DEVICE void nk_load_box_blackwell_(nk_u32_t destination, void const *map, nk_u32_t barrier, nk_i32_t column,
                                           nk_i32_t row) {
    __trap();
}
NUMKONG_DEVICE void nk_prefetch_map_blackwell_(void const *map) { __trap(); }
NUMKONG_DEVICE void nk_fence_async_shared_blackwell_(void) { __trap(); }
NUMKONG_DEVICE void nk_barrier_sync_blackwell_(nk_u32_t barrier, nk_u32_t threads) { __trap(); }
NUMKONG_DEVICE void nk_tmem_alloc_blackwell_(nk_u32_t holder, nk_u32_t columns) { __trap(); }
NUMKONG_DEVICE void nk_tmem_dealloc_blackwell_(nk_u32_t address, nk_u32_t columns) { __trap(); }
NUMKONG_DEVICE void nk_tmem_fence_before_blackwell_(void) { __trap(); }
NUMKONG_DEVICE void nk_tmem_fence_after_blackwell_(void) { __trap(); }
NUMKONG_DEVICE void nk_tmem_load_x32_blackwell_(nk_u32_t address, nk_u32_t values[32]) { __trap(); }
NUMKONG_DEVICE void nk_tmem_fill_x8_blackwell_(nk_u32_t address, nk_u32_t value) { __trap(); }
NUMKONG_DEVICE void nk_mma_f16_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                          nk_u32_t accumulate) {
    __trap();
}
NUMKONG_DEVICE void nk_mma_f8f6f4_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                             nk_u32_t accumulate) {
    __trap();
}
NUMKONG_DEVICE void nk_mma_mxf4_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t instruction,
                                           nk_u32_t accumulate, nk_u32_t scales) {
    __trap();
}
NUMKONG_DEVICE void nk_mma_commit_blackwell_(nk_u32_t barrier) { __trap(); }

#endif // __CUDA_ARCH_FAMILY_SPECIFIC__ in the 10.x family

#pragma endregion Instructions

#pragma region Descriptors

/** The shared-memory descriptor of a K-major operand at @p shared in the 128-byte swizzle, whose
 *  eight-row groups lie 1024 bytes apart. Adding 2 moves it 32 bytes deeper along the swizzled
 *  rows, as the swizzle is applied to the final address bits. */
NUMKONG_DEVICE nk_u64_t nk_cross_descriptor_blackwell_(nk_u32_t shared) {
    return (nk_u64_t)((shared & 0x3FFFFu) >> 4) | ((nk_u64_t)1 << 16) | ((nk_u64_t)(1024 >> 4) << 32) |
           ((nk_u64_t)1 << 46) | ((nk_u64_t)2 << 61);
}

/** The instruction descriptor of a dense @b [128,128] step with F32 accumulators and K-major A and
 *  B, from the format code of both: 0 and 1 for F16 and BF16 under @c kind::f16, 0 and 1 for E4M3
 *  and E5M2 under @c kind::f8f6f4. */
NUMKONG_DEVICE nk_u32_t nk_cross_instruction_blackwell_(nk_u32_t format) {
    return (1u << 4) | (format << 7) | (format << 10) | ((nk_u32_t)(nk_cross_columns_blackwell_k >> 3) << 17) |
           ((nk_u32_t)(nk_cross_rows_blackwell_k >> 4) << 24);
}

/** The instruction descriptor of a block-scaled @b [128,128] step over E2M1 codes with UE8M0
 *  scales, both scale sets starting at their first column. */
NUMKONG_DEVICE nk_u32_t nk_cross_scaled_instruction_blackwell_(void) {
    return (1u << 7) | (1u << 10) | ((nk_u32_t)(nk_cross_columns_blackwell_k >> 3) << 17) | (1u << 23) |
           ((nk_u32_t)(nk_cross_rows_blackwell_k >> 4) << 24);
}

#pragma endregion Descriptors

#pragma region Tile

/** The origin of output tile @p tile, or false for a tile wholly below the diagonal. Tiles run down
 *  a group of @c group_rows row tiles before moving right, so the group's A rows stay in L2 while
 *  every B tile streams from memory once per group rather than once per row tile. */
NUMKONG_DEVICE int nk_cross_tile_origin_blackwell_(nk_cross_tile_arguments_blackwell_t const *arguments,
                                                   nk_cross_triangle_t triangle, nk_size_t tile, nk_size_t *first_row,
                                                   nk_size_t *first_column) {
    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    nk_size_t const row_tiles = shape->tiles / shape->column_tiles;
    nk_size_t const group_tiles = arguments->group_rows * shape->column_tiles;
    nk_size_t const group_first = tile / group_tiles * arguments->group_rows, within = tile % group_tiles;
    nk_size_t const group_rows = row_tiles - group_first < arguments->group_rows ? row_tiles - group_first
                                                                                 : arguments->group_rows;
    *first_row = shape->row_start + (group_first + within % group_rows) * nk_cross_rows_blackwell_k;
    *first_column = within / group_rows * nk_cross_columns_blackwell_k;
    return triangle != nk_cross_triangle_upper_k || *first_column + nk_cross_columns_blackwell_k > *first_row;
}

/** Adds the squares of one staged 128-byte row and widens it in place, each optional. Visiting its
 *  16-byte chunks in swizzle order puts a quarter-warp's reads of 8 rows on distinct banks. */
NUMKONG_DEVICE void nk_cross_stage_row_blackwell_(nk_cross_norm_update_t norm_update, nk_cross_widen_blackwell_t widen,
                                                  unsigned char *row, unsigned tile_row, nk_u32_t *integer_sum,
                                                  nk_f32_t *real_sum) {
    uint4 *const chunks = (uint4 *)row;
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

/**
 *  @brief Every tile of one launch, shared by every dtype and metric.
 *  @param[in] mma Issues one 32-byte depth step; inlined, being a constant.
 *  @param[in] widen Rewrites staged A codes, and B codes for @c symmetric, before the tensor cores
 *      read them; null where they read the codes as they are.
 *  @param[in] output_scale Undoes the power of two a widening introduced, or 1.
 *  @param[in] norm How the squared norms are stored; only @c nk_cross_norm_f32_k here.
 *  @param[in] norm_update Adds staged squares for a metric, see @c nk_cross_norm_update_t.
 *  @param[in] norm_scale Undoes the power of two the norm update's widening introduced, or 1.
 *  @param[in] triangle Whether tiles and outputs below the diagonal are skipped.
 *  @param[in] metric The dot product itself, or a distance from it and the two squared norms.
 *
 *  Warp 0 loads, warp 1 multiplies, and warps 2 to 5 drain: each owns the 32 accumulator lanes its
 *  index modulo 4 may read, one output row per thread. Where a stage needs widening or squaring,
 *  the draining warps do it between the load and the product, each on the rows it drains, so row
 *  norms never leave their thread; column norms of @c symmetric pass through shared memory.
 */
NUMKONG_DEVICE void nk_cross_tile_blackwell_(nk_cross_mma_blackwell_t mma, nk_cross_widen_blackwell_t widen,
                                             nk_f32_t output_scale, nk_cross_norm_t norm,
                                             nk_cross_norm_update_t norm_update, nk_f32_t norm_scale,
                                             nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                             nk_cross_tile_arguments_blackwell_t const *arguments) {
    extern __shared__ unsigned char nk_cross_shared_blackwell_[];
    nk_u32_t const shared_origin = nk_shared_address_ampere_(nk_cross_shared_blackwell_);
    nk_u32_t const padding = (1024 - (shared_origin & 1023)) & 1023;
    unsigned char *const stages = nk_cross_shared_blackwell_ + padding;
    nk_u32_t const stages_address = shared_origin + padding;
    nk_u32_t const barriers = stages_address + nk_cross_stages_blackwell_k * nk_cross_stage_bytes_blackwell_k;
    // Per stage: `loaded` by the copies, `ready` by drain warps that widen, and `consumed` by the
    // products and any drain warps that only read. Per accumulator: `accumulated` by the products,
    // `drained` by the epilogue.
    nk_u32_t const loaded = barriers, ready = loaded + 8 * nk_cross_stages_blackwell_k;
    nk_u32_t const consumed = ready + 8 * nk_cross_stages_blackwell_k;
    nk_u32_t const accumulated = consumed + 8 * nk_cross_stages_blackwell_k, drained = accumulated + 16;
    nk_u32_t const holder = drained + 16;
    unsigned char *const control = stages + (barriers - stages_address);
    nk_u32_t const *const holder_pointer = (nk_u32_t const *)(control + (holder - barriers));
    nk_fui32_t *const column_norms = (nk_fui32_t *)(control + (holder - barriers) + 16);

    nk_cross_tile_arguments_t const *const shape = &arguments->tile;
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int const helped = metric != nk_cross_metric_dot_k || widen != 0;
    nk_u32_t const columns = 512;
    nk_size_t const slabs = shape->depth_slabs;

    if (threadIdx.x == 0) {
        for (unsigned stage = 0; stage < nk_cross_stages_blackwell_k; ++stage) {
            nk_mbarrier_init_blackwell_(loaded + 8 * stage, 1);
            nk_mbarrier_init_blackwell_(ready + 8 * stage, 4);
            nk_mbarrier_init_blackwell_(consumed + 8 * stage, widen ? 1 : 1 + 4 * helped);
        }
        for (unsigned buffer = 0; buffer < 2; ++buffer) {
            nk_mbarrier_init_blackwell_(accumulated + 8 * buffer, 1);
            nk_mbarrier_init_blackwell_(drained + 8 * buffer, 4);
        }
        nk_mbarrier_init_fence_blackwell_();
    }
    if (warp == 1) nk_tmem_alloc_blackwell_(holder, columns);
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    nk_tmem_fence_after_blackwell_();
    nk_u32_t const tmem = *holder_pointer;
    nk_u32_t const scales = tmem + columns - 32;
    // UE8M0 code 127 is 2⁰; every byte of the 32 scale columns holds it, whatever layout is read.
    if (warp >= 2)
        for (unsigned column = 0; column < 32; column += 8)
            nk_tmem_fill_x8_blackwell_(scales + (((warp & 3) * 32) << 16) + column, 0x7F7F7F7Fu);
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    nk_tmem_fence_after_blackwell_();

    if (warp == 0 && lane == 0) {
        nk_prefetch_map_blackwell_(&arguments->a_map);
        nk_prefetch_map_blackwell_(&arguments->b_map);
        nk_u32_t stage = 0, phase = 0;
        for (nk_size_t tile = blockIdx.x; tile < shape->tiles; tile += gridDim.x) {
            nk_size_t first_row, first_column;
            if (!nk_cross_tile_origin_blackwell_(arguments, triangle, tile, &first_row, &first_column)) continue;
            for (nk_size_t slab = 0; slab < slabs; ++slab) {
                nk_u32_t const a_stage = stages_address + stage * nk_cross_stage_bytes_blackwell_k;
                nk_mbarrier_wait_blackwell_(consumed + 8 * stage, phase ^ 1);
                nk_mbarrier_expect_bytes_blackwell_(loaded + 8 * stage, nk_cross_stage_bytes_blackwell_k);
                nk_i32_t const column = (nk_i32_t)(slab * nk_cross_slab_bytes_blackwell_k);
                nk_load_box_blackwell_(a_stage, &arguments->a_map, loaded + 8 * stage, column, (nk_i32_t)first_row);
                nk_load_box_blackwell_(a_stage + nk_cross_a_bytes_blackwell_k, &arguments->b_map, loaded + 8 * stage,
                                       column, (nk_i32_t)first_column);
                if (++stage == nk_cross_stages_blackwell_k) stage = 0, phase ^= 1;
            }
        }
    }
    else if (warp == 1 && lane == 0) {
        nk_u32_t stage = 0, phase = 0, iteration = 0;
        for (nk_size_t tile = blockIdx.x; tile < shape->tiles; tile += gridDim.x) {
            nk_size_t first_row, first_column;
            if (!nk_cross_tile_origin_blackwell_(arguments, triangle, tile, &first_row, &first_column)) continue;
            nk_u32_t const buffer = iteration & 1, buffer_phase = (iteration >> 1) & 1;
            nk_mbarrier_wait_blackwell_(drained + 8 * buffer, buffer_phase ^ 1);
            nk_tmem_fence_after_blackwell_();
            nk_u32_t const accumulator = tmem + buffer * nk_cross_columns_blackwell_k;
            for (nk_size_t slab = 0; slab < slabs; ++slab) {
                nk_mbarrier_wait_blackwell_((widen ? ready : loaded) + 8 * stage, phase);
                nk_tmem_fence_after_blackwell_();
                nk_u32_t const a_stage = stages_address + stage * nk_cross_stage_bytes_blackwell_k;
                nk_u64_t const a = nk_cross_descriptor_blackwell_(a_stage);
                nk_u64_t const b = nk_cross_descriptor_blackwell_(a_stage + nk_cross_a_bytes_blackwell_k);
#pragma unroll
                for (unsigned step = 0; step < nk_cross_slab_bytes_blackwell_k / 32; ++step)
                    mma(accumulator, a + 2 * step, b + 2 * step, (nk_u32_t)(slab | step) != 0, scales);
                nk_mma_commit_blackwell_(consumed + 8 * stage);
                if (++stage == nk_cross_stages_blackwell_k) stage = 0, phase ^= 1;
            }
            nk_mma_commit_blackwell_(accumulated + 8 * buffer);
            ++iteration;
        }
    }
    else if (warp >= 2) {
        unsigned const quarter = warp & 3, tile_row = quarter * 32 + lane;
        nk_u32_t const lane_address = tmem + ((quarter * 32) << 16);
        nk_u32_t stage = 0, phase = 0, iteration = 0;
        for (nk_size_t tile = blockIdx.x; tile < shape->tiles; tile += gridDim.x) {
            nk_size_t first_row, first_column;
            if (!nk_cross_tile_origin_blackwell_(arguments, triangle, tile, &first_row, &first_column)) continue;
            nk_u32_t row_integer_norm = 0,
                     column_integer_norm[nk_cross_columns_blackwell_k / nk_cross_rows_blackwell_k];
            nk_f32_t row_real_norm = 0, column_real_norm[nk_cross_columns_blackwell_k / nk_cross_rows_blackwell_k];
#pragma unroll
            for (unsigned part = 0; part < nk_cross_columns_blackwell_k / nk_cross_rows_blackwell_k; ++part)
                column_integer_norm[part] = 0, column_real_norm[part] = 0;
            nk_cross_norm_update_t const squares = metric != nk_cross_metric_dot_k ? norm_update : 0;
            if (helped)
                for (nk_size_t slab = 0; slab < slabs; ++slab) {
                    nk_mbarrier_wait_blackwell_(loaded + 8 * stage, phase);
                    unsigned char *const a_row = stages + stage * nk_cross_stage_bytes_blackwell_k +
                                                 tile_row * nk_cross_slab_bytes_blackwell_k;
                    nk_cross_stage_row_blackwell_(squares, widen, a_row, tile_row, &row_integer_norm, &row_real_norm);
                    if (triangle == nk_cross_triangle_upper_k)
#pragma unroll
                        for (unsigned part = 0; part < nk_cross_columns_blackwell_k / nk_cross_rows_blackwell_k; ++part)
                            nk_cross_stage_row_blackwell_(
                                squares, widen,
                                a_row + nk_cross_a_bytes_blackwell_k + part * nk_cross_a_bytes_blackwell_k, tile_row,
                                &column_integer_norm[part], &column_real_norm[part]);
                    if (widen) nk_fence_async_shared_blackwell_();
                    __syncwarp();
                    if (lane == 0) nk_mbarrier_arrive_blackwell_((widen ? ready : consumed) + 8 * stage);
                    if (++stage == nk_cross_stages_blackwell_k) stage = 0, phase ^= 1;
                }

            nk_f32_t row_norm = 0;
            if (metric != nk_cross_metric_dot_k) {
                row_norm = nk_cross_norm_finalize_(norm, row_integer_norm, row_real_norm, norm_scale).f;
                // The first barrier retires the previous tile's reads, the second publishes these.
                nk_barrier_sync_blackwell_(1, 128);
#pragma unroll
                for (unsigned part = 0; part < nk_cross_columns_blackwell_k / nk_cross_rows_blackwell_k; ++part) {
                    unsigned const tile_column = part * nk_cross_rows_blackwell_k + tile_row;
                    nk_size_t const column = first_column + tile_column;
                    nk_fui32_t column_norm;
                    if (triangle == nk_cross_triangle_upper_k)
                        column_norm = nk_cross_norm_finalize_(norm, column_integer_norm[part], column_real_norm[part],
                                                              norm_scale);
                    else column_norm.u = column < shape->column_count ? ((nk_u32_t const *)shape->b_norms)[column] : 0;
                    column_norms[tile_column] = column_norm;
                }
                nk_barrier_sync_blackwell_(1, 128);
            }

            nk_u32_t const buffer = iteration & 1, buffer_phase = (iteration >> 1) & 1;
            nk_mbarrier_wait_blackwell_(accumulated + 8 * buffer, buffer_phase);
            nk_tmem_fence_after_blackwell_();
            nk_size_t const row = first_row + tile_row;
            unsigned char *const output = (unsigned char *)shape->c + row * shape->c_stride;
            int const aligned = ((((nk_size_t)shape->c) | shape->c_stride) & 15) == 0;
#pragma unroll 1
            for (unsigned chunk = 0; chunk < nk_cross_columns_blackwell_k / 32; ++chunk) {
                nk_fui32_t sums[32];
                nk_tmem_load_x32_blackwell_(lane_address + buffer * nk_cross_columns_blackwell_k + chunk * 32,
                                            (nk_u32_t *)sums);
                if (row >= shape->row_end) continue;
                nk_size_t const first_chunk_column = first_column + chunk * 32;
#pragma unroll
                for (unsigned offset = 0; offset < 32; ++offset) {
                    nk_f32_t const dot = slabs ? sums[offset].f * output_scale : 0.0f;
                    if (metric == nk_cross_metric_dot_k) sums[offset].f = dot;
                    else if (triangle == nk_cross_triangle_upper_k && first_chunk_column + offset == row)
                        sums[offset].f = 0;
                    else if (metric == nk_cross_metric_angular_k)
                        sums[offset].f = nk_f32_angular_(dot, row_norm, column_norms[chunk * 32 + offset].f);
                    else sums[offset].f = nk_f32_euclidean_(dot, row_norm, column_norms[chunk * 32 + offset].f);
                }
                // Whole chunks go out as 16-byte stores, 128 contiguous bytes per thread.
                if (aligned && first_chunk_column + 32 <= shape->column_count &&
                    (triangle != nk_cross_triangle_upper_k || first_chunk_column >= row)) {
                    uint4 *const destination = (uint4 *)(output + first_chunk_column * sizeof(nk_f32_t));
#pragma unroll
                    for (unsigned quartet = 0; quartet < 8; ++quartet)
                        destination[quartet] = make_uint4(sums[quartet * 4].u, sums[quartet * 4 + 1].u,
                                                          sums[quartet * 4 + 2].u, sums[quartet * 4 + 3].u);
                    continue;
                }
#pragma unroll
                for (unsigned offset = 0; offset < 32; ++offset) {
                    nk_size_t const column = first_chunk_column + offset;
                    if (column < shape->column_count && (triangle != nk_cross_triangle_upper_k || column >= row))
                        ((nk_f32_t *)output)[column] = sums[offset].f;
                }
            }
            nk_tmem_fence_before_blackwell_();
            __syncwarp();
            if (lane == 0) nk_mbarrier_arrive_blackwell_(drained + 8 * buffer);
            ++iteration;
        }
    }

    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    if (warp == 1) {
        __syncwarp();
        nk_tmem_fence_after_blackwell_();
        nk_tmem_dealloc_blackwell_(tmem, columns);
    }
}

/** Describes @p rows rows of @p row_bytes bytes, @p stride bytes apart from @p base, to the Tensor
 *  Memory Accelerator as boxes of 128 rows of 128 bytes in the 128-byte swizzle, zero-filling past
 *  either edge. Returns zero when the driver refuses. */
NUMKONG_INLINE int nk_cross_map_blackwell_(CUtensorMap *map, void const *base, nk_size_t rows, nk_size_t row_bytes,
                                           nk_size_t stride, unsigned box_rows) {
    typedef CUresult (*encode_t)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, cuuint64_t const *,
                                 cuuint64_t const *, cuuint32_t const *, cuuint32_t const *, CUtensorMapInterleave,
                                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    // Resolved through the runtime, so nothing links against the driver library.
    void *entry = 0;
    enum cudaDriverEntryPointQueryResult found;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000, cudaEnableDefault, &found) !=
            cudaSuccess ||
        found != cudaDriverEntryPointSuccess)
        return 0;
    // A zero-depth launch never reads its maps, but the driver refuses an empty dimension.
    cuuint64_t const dimensions[2] = {row_bytes ? row_bytes : 16, rows};
    cuuint64_t const strides[1] = {stride ? stride : 16};
    cuuint32_t const box[2] = {nk_cross_slab_bytes_blackwell_k, box_rows};
    cuuint32_t const element_strides[2] = {1, 1};
    return ((encode_t)entry)(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void *)base, dimensions, strides, box,
                             element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

/** Validates the contract, describes A's @p a_rows rows and B's @p column_count rows, and launches
 *  as many blocks of @p kernel as stay resident, each walking output tiles with a stride of the
 *  grid. @p b_norms is the packed column norms a @c packed metric reads, or null. */
NUMKONG_INLINE nk_status_t nk_cross_launch_blackwell_(void const *kernel, void const *a, nk_size_t a_rows,
                                                      void const *b, void const *b_norms, void *c,
                                                      nk_size_t result_bytes, nk_size_t row_start, nk_size_t row_end,
                                                      nk_size_t column_count, nk_size_t depth_bytes, nk_size_t a_stride,
                                                      nk_size_t b_stride, nk_size_t c_stride, void *stream) {
    if ((((nk_size_t)a) | a_stride | ((nk_size_t)b) | b_stride) & 15 ||
        (((nk_size_t)c) | c_stride) & (result_bytes - 1))
        return nk_misaligned_k;
    if (row_end <= row_start || column_count == 0) return nk_success_k;
    nk_cross_tile_arguments_blackwell_t arguments;
    if (!nk_cross_map_blackwell_(&arguments.a_map, a, a_rows, depth_bytes, a_stride, nk_cross_rows_blackwell_k) ||
        !nk_cross_map_blackwell_(&arguments.b_map, b, column_count, depth_bytes, b_stride,
                                 nk_cross_columns_blackwell_k))
        return nk_device_memory_mismatch_k;
    nk_cross_tile_arguments_t *const tile = &arguments.tile;
    nk_size_t const column_tiles = nk_size_divide_round_up_(column_count, nk_cross_columns_blackwell_k);
    nk_size_t const tiles = nk_size_divide_round_up_(row_end - row_start, nk_cross_rows_blackwell_k) * column_tiles;
    tile->a = (unsigned char const *)a, tile->b = (unsigned char const *)b, tile->c = c;
    tile->row_start = row_start, tile->row_end = row_end, tile->column_count = column_count;
    tile->depth = 0, tile->depth_bytes = depth_bytes, tile->a_stride = a_stride, tile->b_stride = b_stride;
    tile->c_stride = c_stride, tile->column_tiles = column_tiles, tile->tiles = tiles;
    tile->depth_slabs = nk_size_divide_round_up_(depth_bytes, nk_cross_slab_bytes_blackwell_k);
    tile->b_norms = b_norms;
    // Row tiles whose A rows fill half the L2 sweep the columns together, from 1 to 16 of them.
    int l2_bytes = 0;
    nk_status_t const status = nk_device_attribute_(cudaDevAttrL2CacheSize, &l2_bytes);
    if (status != nk_success_k) return status;
    nk_size_t const row_tile_bytes = nk_cross_rows_blackwell_k * (depth_bytes ? depth_bytes : 1);
    nk_size_t const group_rows = (nk_size_t)l2_bytes / 2 / row_tile_bytes;
    arguments.group_rows = group_rows < 1 ? 1 : group_rows > 16 ? 16 : group_rows;
    return nk_launch_resident_(kernel, nk_cross_threads_blackwell_k, nk_cross_shared_bytes_blackwell_k,
                               nk_cross_shared_bytes_blackwell_k, tiles, &arguments, stream);
}

#pragma endregion Tile

/*  The same shapes as @c nk_define_device_cross_packed_ and its kin, over the TMA launch: the site
 *  passes the tile's own leading arguments, from the MMA issue to the norm scale, last. */
#pragma region Cross Macros

/**
 *  @brief Generates C = A × Bᵀ, or its angular or euclidean distances, over a B packed by
 *      @c nk_define_device_cross_pack_, one block per resident slot walking @b [128,128] tiles.
 *  @param[in] metric @c dot, @c angular or @c euclidean, naming both the entry and the epilogue.
 *  @param[in] ... The tile's own leading arguments, see @c nk_cross_tile_blackwell_.
 *  @sa nk_define_cross_packed_ for the host original.
 */
#define nk_define_device_cross_tma_packed_(metric, input_type_name, isa_suffix, input_value_type, packed_value_type,   \
                                           result_value_type, depth_simd_dimensions, dimensions_per_value, ...)        \
    static __global__ void __launch_bounds__(nk_cross_threads_blackwell_k, 1)                                          \
        nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_(                                              \
            __grid_constant__ nk_cross_tile_arguments_blackwell_t const arguments) {                                   \
        nk_cross_tile_blackwell_(__VA_ARGS__, nk_cross_triangle_full_k, nk_cross_metric_##metric##_k, &arguments);     \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##metric##s_packed_##input_type_name##_##isa_suffix(                                    \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes, void *stream) {                                                                   \
        nk_size_t const row_bytes = nk_device_cross_padded_values_(depth, depth_simd_dimensions, dimensions_per_value, \
                                                                   sizeof(nk_##packed_value_type##_t)) *               \
                                    sizeof(nk_##packed_value_type##_t);                                                \
        unsigned char const *b_rows = (unsigned char const *)b_packed_buffer +                                         \
                                      sizeof(nk_cross_packed_buffer_header_t);                                         \
        return nk_cross_launch_blackwell_(                                                                             \
            (void const *)nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_, a_matrix, row_count,       \
            b_rows, b_rows + column_count * row_bytes, c_matrix, sizeof(nk_##result_value_type##_t), 0, row_count,     \
            column_count, depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), a_stride_in_bytes,         \
            row_bytes, c_stride_in_bytes, stream);                                                                     \
    }

/**
 *  @brief Generates the Gram matrix C = A × Aᵀ, or its angular or euclidean distances, over rows
 *      [row_start, row_start + row_count): the upper triangle, with the diagonal for dots and zeros
 *      on it for distances, skipping tiles wholly below it.
 *
 *  Takes the parameters of @c nk_define_device_cross_tma_packed_, so one bundle feeds both.
 *
 *  @sa nk_define_cross_symmetric_ for the host original.
 */
#define nk_define_device_cross_tma_symmetric_(metric, input_type_name, isa_suffix, input_value_type,                   \
                                              packed_value_type, result_value_type, depth_simd_dimensions,             \
                                              dimensions_per_value, ...)                                               \
    static __global__ void __launch_bounds__(nk_cross_threads_blackwell_k, 1)                                          \
        nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_(                                           \
            __grid_constant__ nk_cross_tile_arguments_blackwell_t const arguments) {                                   \
        nk_cross_tile_blackwell_(__VA_ARGS__, nk_cross_triangle_upper_k, nk_cross_metric_##metric##_k, &arguments);    \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##metric##s_symmetric_##input_type_name##_##isa_suffix(                                 \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, \
        nk_##result_value_type##_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start,                     \
        nk_size_t row_count, void *stream) {                                                                           \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;       \
        return nk_cross_launch_blackwell_(                                                                             \
            (void const *)nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_, vectors, vectors_count, \
            vectors, 0, result, sizeof(nk_##result_value_type##_t), row_start, row_end, vectors_count,                 \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), stride_in_bytes, stride_in_bytes,        \
            result_stride_in_bytes, stream);                                                                           \
    }

/** Both shapes of one metric over the TMA launch, packed and symmetric. */
#define nk_define_device_cross_tma_(metric, input_type_name, isa_suffix, input_value_type, packed_value_type,       \
                                    result_value_type, depth_simd_dimensions, dimensions_per_value, ...)            \
    nk_define_device_cross_tma_packed_(metric, input_type_name, isa_suffix, input_value_type, packed_value_type,    \
                                       result_value_type, depth_simd_dimensions, dimensions_per_value, __VA_ARGS__) \
    nk_define_device_cross_tma_symmetric_(metric, input_type_name, isa_suffix, input_value_type, packed_value_type, \
                                          result_value_type, depth_simd_dimensions, dimensions_per_value, __VA_ARGS__)

#pragma endregion Cross Macros

#pragma region Multiplies

NUMKONG_DEVICE void nk_dots_bf16_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                nk_u32_t scales) {
    nk_mma_f16_blackwell_(accumulator, a, b, nk_cross_instruction_blackwell_(1), accumulate);
}

NUMKONG_DEVICE void nk_dots_f16_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                               nk_u32_t scales) {
    nk_mma_f16_blackwell_(accumulator, a, b, nk_cross_instruction_blackwell_(0), accumulate);
}

NUMKONG_DEVICE void nk_dots_e4m3_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                nk_u32_t scales) {
    nk_mma_f8f6f4_blackwell_(accumulator, a, b, nk_cross_instruction_blackwell_(0), accumulate);
}

NUMKONG_DEVICE void nk_dots_e5m2_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                nk_u32_t scales) {
    nk_mma_f8f6f4_blackwell_(accumulator, a, b, nk_cross_instruction_blackwell_(1), accumulate);
}

NUMKONG_DEVICE void nk_dots_e2m1_mma_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t accumulate,
                                                nk_u32_t scales) {
    nk_mma_mxf4_blackwell_(accumulator, a, b, nk_cross_scaled_instruction_blackwell_(), accumulate, scales);
}

#pragma endregion Multiplies

#pragma region BF16

nk_define_device_cross_pack_(bf16, blackwell, bf16, bf16, nk_load_b8_, /*norm_value_type=*/f32,
                             /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_device_cross_tma_(dot, bf16, blackwell, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                            /*dimensions_per_value=*/1, nk_dots_bf16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion BF16

#pragma region F16

nk_define_device_cross_pack_(f16, blackwell, f16, f16, nk_load_b8_, /*norm_value_type=*/f32,
                             /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_device_cross_tma_(dot, f16, blackwell, f16, f16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1,
                            nk_dots_f16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL, /*output_scale=*/1.0f,
                            nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion F16

#pragma region E5M2

nk_define_device_cross_pack_(e5m2, blackwell, e5m2, e5m2, nk_load_b8_, /*norm_value_type=*/f32,
                             /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_device_cross_tma_(dot, e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e5m2_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E5M2

#pragma region E4M3

nk_define_device_cross_pack_(e4m3, blackwell, e4m3, e4m3, nk_load_b8_, /*norm_value_type=*/f32,
                             /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_device_cross_tma_(dot, e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e4m3_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E4M3

#pragma region E3M2

nk_define_device_cross_pack_(e3m2, blackwell, e3m2, e5m2, nk_load_f6_to_f8_ada_, /*norm_value_type=*/f32,
                             /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_device_cross_tma_(dot, e3m2, blackwell, e3m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e5m2_mma_blackwell_, nk_f6x4_to_f8x4_ada_,
                            /*output_scale=*/16777216.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E3M2

#pragma region E2M3

nk_define_device_cross_pack_(e2m3, blackwell, e2m3, e4m3, nk_load_f6_to_f8_ada_, /*norm_value_type=*/f32,
                             /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_device_cross_tma_(dot, e2m3, blackwell, e2m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e4m3_mma_blackwell_, nk_f6x4_to_f8x4_ada_,
                            /*output_scale=*/4096.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M3

#pragma region E2M1

nk_define_device_cross_pack_(e2m1, blackwell, e2m1x2, e2m1x2, nk_load_b8_, /*norm_value_type=*/f32,
                             /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_device_cross_tma_(dot, e2m1, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                            /*dimensions_per_value=*/2, nk_dots_e2m1_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M1

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_BLACKWELL
#endif // NUMKONG_DOTS_BLACKWELL_CUH
