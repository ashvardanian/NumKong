/**
 *  @file include/numkong/dots/hopper.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated Batched Dot Products for NVIDIA Hopper, compute capability 9.0.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/ampere.cuh
 *
 *  Warpgroup `wgmma.mma_async` over shared-memory descriptors, which only the "90a" code carries,
 *  so no other generation compiles these kernels. Two warpgroups own a @b [128,128] output tile, 64
 *  rows each, over the Ampere tile's three-stage `cp.async` ring of 64-byte depth slabs, whose
 *  XOR-swizzled rows are exactly the 64-byte-swizzle K-major layout the tensor cores read, so both
 *  operands stay in shared memory and each slab is two @c m64n128 steps per warpgroup. Once a slab
 *  lands, E2M3, E2M1, I4 and U4 widen in place into 8-bit integers, each thread rewriting one
 *  staged row. Float8 and E3M2 stay on Ampere's kernels, as Hopper's FP8 MMA keeps only 13 fraction
 *  bits of each 32-deep block. The pack pads rows to 16 bytes like the Ampere one, but keeps every
 *  code as it is, E2M3 too, since the widening happens in shared memory.
 */
#ifndef NUMKONG_DOTS_HOPPER_CUH
#define NUMKONG_DOTS_HOPPER_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_TARGET_HOPPER

#include "numkong/dots/ampere.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_cross_tile_hopper_k = 128,
    nk_cross_threads_hopper_k = 256,
    nk_cross_warpgroup_rows_hopper_k = 64,
};

/*  Each dtype launches the one shape under its own stem, so the generators paste it. */
enum {
    nk_cross_threads_bf16_hopper_k = nk_cross_threads_hopper_k,
    nk_cross_tile_bf16_hopper_k = nk_cross_tile_hopper_k,
    nk_cross_threads_f16_hopper_k = nk_cross_threads_hopper_k,
    nk_cross_tile_f16_hopper_k = nk_cross_tile_hopper_k,
    nk_cross_threads_e2m3_hopper_k = nk_cross_threads_hopper_k,
    nk_cross_tile_e2m3_hopper_k = nk_cross_tile_hopper_k,
    nk_cross_threads_e2m1_hopper_k = nk_cross_threads_hopper_k,
    nk_cross_tile_e2m1_hopper_k = nk_cross_tile_hopper_k,
    nk_cross_threads_i8_hopper_k = nk_cross_threads_hopper_k,
    nk_cross_tile_i8_hopper_k = nk_cross_tile_hopper_k,
    nk_cross_threads_i4_hopper_k = nk_cross_threads_hopper_k,
    nk_cross_tile_i4_hopper_k = nk_cross_tile_hopper_k,
    nk_cross_threads_u8_hopper_k = nk_cross_threads_hopper_k,
    nk_cross_tile_u8_hopper_k = nk_cross_tile_hopper_k,
    nk_cross_threads_u4_hopper_k = nk_cross_threads_hopper_k,
    nk_cross_tile_u4_hopper_k = nk_cross_tile_hopper_k,
};

#pragma endregion Configuration

#pragma region Instructions

NUMKONG_DEVICE void nk_wgmma_fence_hopper_(void) { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }

/*  Orders this thread's generic-proxy writes of shared memory, `cp.async` ones included, before
 *  the async-proxy reads of a later @c wgmma. */
NUMKONG_DEVICE void nk_fence_proxy_async_hopper_(void) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

NUMKONG_DEVICE void nk_wgmma_commit_hopper_(void) { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }

/*  Waits for every @c wgmma group this warp committed; another warp of the warpgroup may still be
 *  reading shared memory until a barrier. */
NUMKONG_DEVICE void nk_wgmma_wait_hopper_(void) { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

NUMKONG_DEVICE void nk_wgmma_bf16_hopper_(nk_fui32_t accumulators[64], nk_u64_t a_descriptor, nk_u64_t b_descriptor,
                                          int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"                                        //
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "                            //
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
        "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "  //
        "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "  //
        "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}"             //
        ", %64, %65, p, 1, 1, 0, 0;\n}\n"
        : "+f"(accumulators[0].f), "+f"(accumulators[1].f), "+f"(accumulators[2].f), "+f"(accumulators[3].f),
          "+f"(accumulators[4].f), "+f"(accumulators[5].f), "+f"(accumulators[6].f), "+f"(accumulators[7].f),
          "+f"(accumulators[8].f), "+f"(accumulators[9].f), "+f"(accumulators[10].f), "+f"(accumulators[11].f),
          "+f"(accumulators[12].f), "+f"(accumulators[13].f), "+f"(accumulators[14].f), "+f"(accumulators[15].f),
          "+f"(accumulators[16].f), "+f"(accumulators[17].f), "+f"(accumulators[18].f), "+f"(accumulators[19].f),
          "+f"(accumulators[20].f), "+f"(accumulators[21].f), "+f"(accumulators[22].f), "+f"(accumulators[23].f),
          "+f"(accumulators[24].f), "+f"(accumulators[25].f), "+f"(accumulators[26].f), "+f"(accumulators[27].f),
          "+f"(accumulators[28].f), "+f"(accumulators[29].f), "+f"(accumulators[30].f), "+f"(accumulators[31].f),
          "+f"(accumulators[32].f), "+f"(accumulators[33].f), "+f"(accumulators[34].f), "+f"(accumulators[35].f),
          "+f"(accumulators[36].f), "+f"(accumulators[37].f), "+f"(accumulators[38].f), "+f"(accumulators[39].f),
          "+f"(accumulators[40].f), "+f"(accumulators[41].f), "+f"(accumulators[42].f), "+f"(accumulators[43].f),
          "+f"(accumulators[44].f), "+f"(accumulators[45].f), "+f"(accumulators[46].f), "+f"(accumulators[47].f),
          "+f"(accumulators[48].f), "+f"(accumulators[49].f), "+f"(accumulators[50].f), "+f"(accumulators[51].f),
          "+f"(accumulators[52].f), "+f"(accumulators[53].f), "+f"(accumulators[54].f), "+f"(accumulators[55].f),
          "+f"(accumulators[56].f), "+f"(accumulators[57].f), "+f"(accumulators[58].f), "+f"(accumulators[59].f),
          "+f"(accumulators[60].f), "+f"(accumulators[61].f), "+f"(accumulators[62].f), "+f"(accumulators[63].f)
        : "l"(a_descriptor), "l"(b_descriptor), "r"(accumulate)
        : "memory");
}

NUMKONG_DEVICE void nk_wgmma_f16_hopper_(nk_fui32_t accumulators[64], nk_u64_t a_descriptor, nk_u64_t b_descriptor,
                                         int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"                                        //
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "                              //
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
        "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "  //
        "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "  //
        "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}"             //
        ", %64, %65, p, 1, 1, 0, 0;\n}\n"
        : "+f"(accumulators[0].f), "+f"(accumulators[1].f), "+f"(accumulators[2].f), "+f"(accumulators[3].f),
          "+f"(accumulators[4].f), "+f"(accumulators[5].f), "+f"(accumulators[6].f), "+f"(accumulators[7].f),
          "+f"(accumulators[8].f), "+f"(accumulators[9].f), "+f"(accumulators[10].f), "+f"(accumulators[11].f),
          "+f"(accumulators[12].f), "+f"(accumulators[13].f), "+f"(accumulators[14].f), "+f"(accumulators[15].f),
          "+f"(accumulators[16].f), "+f"(accumulators[17].f), "+f"(accumulators[18].f), "+f"(accumulators[19].f),
          "+f"(accumulators[20].f), "+f"(accumulators[21].f), "+f"(accumulators[22].f), "+f"(accumulators[23].f),
          "+f"(accumulators[24].f), "+f"(accumulators[25].f), "+f"(accumulators[26].f), "+f"(accumulators[27].f),
          "+f"(accumulators[28].f), "+f"(accumulators[29].f), "+f"(accumulators[30].f), "+f"(accumulators[31].f),
          "+f"(accumulators[32].f), "+f"(accumulators[33].f), "+f"(accumulators[34].f), "+f"(accumulators[35].f),
          "+f"(accumulators[36].f), "+f"(accumulators[37].f), "+f"(accumulators[38].f), "+f"(accumulators[39].f),
          "+f"(accumulators[40].f), "+f"(accumulators[41].f), "+f"(accumulators[42].f), "+f"(accumulators[43].f),
          "+f"(accumulators[44].f), "+f"(accumulators[45].f), "+f"(accumulators[46].f), "+f"(accumulators[47].f),
          "+f"(accumulators[48].f), "+f"(accumulators[49].f), "+f"(accumulators[50].f), "+f"(accumulators[51].f),
          "+f"(accumulators[52].f), "+f"(accumulators[53].f), "+f"(accumulators[54].f), "+f"(accumulators[55].f),
          "+f"(accumulators[56].f), "+f"(accumulators[57].f), "+f"(accumulators[58].f), "+f"(accumulators[59].f),
          "+f"(accumulators[60].f), "+f"(accumulators[61].f), "+f"(accumulators[62].f), "+f"(accumulators[63].f)
        : "l"(a_descriptor), "l"(b_descriptor), "r"(accumulate)
        : "memory");
}

NUMKONG_DEVICE void nk_wgmma_i8_hopper_(nk_fui32_t accumulators[64], nk_u64_t a_descriptor, nk_u64_t b_descriptor,
                                        int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"                                        //
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "                                //
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
        "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "  //
        "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "  //
        "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}"             //
        ", %64, %65, p;\n}\n"
        : "+r"(accumulators[0].u), "+r"(accumulators[1].u), "+r"(accumulators[2].u), "+r"(accumulators[3].u),
          "+r"(accumulators[4].u), "+r"(accumulators[5].u), "+r"(accumulators[6].u), "+r"(accumulators[7].u),
          "+r"(accumulators[8].u), "+r"(accumulators[9].u), "+r"(accumulators[10].u), "+r"(accumulators[11].u),
          "+r"(accumulators[12].u), "+r"(accumulators[13].u), "+r"(accumulators[14].u), "+r"(accumulators[15].u),
          "+r"(accumulators[16].u), "+r"(accumulators[17].u), "+r"(accumulators[18].u), "+r"(accumulators[19].u),
          "+r"(accumulators[20].u), "+r"(accumulators[21].u), "+r"(accumulators[22].u), "+r"(accumulators[23].u),
          "+r"(accumulators[24].u), "+r"(accumulators[25].u), "+r"(accumulators[26].u), "+r"(accumulators[27].u),
          "+r"(accumulators[28].u), "+r"(accumulators[29].u), "+r"(accumulators[30].u), "+r"(accumulators[31].u),
          "+r"(accumulators[32].u), "+r"(accumulators[33].u), "+r"(accumulators[34].u), "+r"(accumulators[35].u),
          "+r"(accumulators[36].u), "+r"(accumulators[37].u), "+r"(accumulators[38].u), "+r"(accumulators[39].u),
          "+r"(accumulators[40].u), "+r"(accumulators[41].u), "+r"(accumulators[42].u), "+r"(accumulators[43].u),
          "+r"(accumulators[44].u), "+r"(accumulators[45].u), "+r"(accumulators[46].u), "+r"(accumulators[47].u),
          "+r"(accumulators[48].u), "+r"(accumulators[49].u), "+r"(accumulators[50].u), "+r"(accumulators[51].u),
          "+r"(accumulators[52].u), "+r"(accumulators[53].u), "+r"(accumulators[54].u), "+r"(accumulators[55].u),
          "+r"(accumulators[56].u), "+r"(accumulators[57].u), "+r"(accumulators[58].u), "+r"(accumulators[59].u),
          "+r"(accumulators[60].u), "+r"(accumulators[61].u), "+r"(accumulators[62].u), "+r"(accumulators[63].u)
        : "l"(a_descriptor), "l"(b_descriptor), "r"(accumulate)
        : "memory");
}

NUMKONG_DEVICE void nk_wgmma_u8_hopper_(nk_fui32_t accumulators[64], nk_u64_t a_descriptor, nk_u64_t b_descriptor,
                                        int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"                                        //
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.u8 "                                //
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
        "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "  //
        "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "  //
        "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}"             //
        ", %64, %65, p;\n}\n"
        : "+r"(accumulators[0].u), "+r"(accumulators[1].u), "+r"(accumulators[2].u), "+r"(accumulators[3].u),
          "+r"(accumulators[4].u), "+r"(accumulators[5].u), "+r"(accumulators[6].u), "+r"(accumulators[7].u),
          "+r"(accumulators[8].u), "+r"(accumulators[9].u), "+r"(accumulators[10].u), "+r"(accumulators[11].u),
          "+r"(accumulators[12].u), "+r"(accumulators[13].u), "+r"(accumulators[14].u), "+r"(accumulators[15].u),
          "+r"(accumulators[16].u), "+r"(accumulators[17].u), "+r"(accumulators[18].u), "+r"(accumulators[19].u),
          "+r"(accumulators[20].u), "+r"(accumulators[21].u), "+r"(accumulators[22].u), "+r"(accumulators[23].u),
          "+r"(accumulators[24].u), "+r"(accumulators[25].u), "+r"(accumulators[26].u), "+r"(accumulators[27].u),
          "+r"(accumulators[28].u), "+r"(accumulators[29].u), "+r"(accumulators[30].u), "+r"(accumulators[31].u),
          "+r"(accumulators[32].u), "+r"(accumulators[33].u), "+r"(accumulators[34].u), "+r"(accumulators[35].u),
          "+r"(accumulators[36].u), "+r"(accumulators[37].u), "+r"(accumulators[38].u), "+r"(accumulators[39].u),
          "+r"(accumulators[40].u), "+r"(accumulators[41].u), "+r"(accumulators[42].u), "+r"(accumulators[43].u),
          "+r"(accumulators[44].u), "+r"(accumulators[45].u), "+r"(accumulators[46].u), "+r"(accumulators[47].u),
          "+r"(accumulators[48].u), "+r"(accumulators[49].u), "+r"(accumulators[50].u), "+r"(accumulators[51].u),
          "+r"(accumulators[52].u), "+r"(accumulators[53].u), "+r"(accumulators[54].u), "+r"(accumulators[55].u),
          "+r"(accumulators[56].u), "+r"(accumulators[57].u), "+r"(accumulators[58].u), "+r"(accumulators[59].u),
          "+r"(accumulators[60].u), "+r"(accumulators[61].u), "+r"(accumulators[62].u), "+r"(accumulators[63].u)
        : "l"(a_descriptor), "l"(b_descriptor), "r"(accumulate)
        : "memory");
}

#pragma endregion Instructions

#pragma region Descriptors

/** The layout code of a @c wgmma descriptor for the 64-byte swizzle every staged operand takes. */
enum { nk_smem_swizzle_64_hopper_k = 2 };

/**
 *  @brief The shared-memory descriptor of a @c wgmma operand at @p address in @p layout, with
 *      @p leading_bytes between its groups along the leading dimension and @p stride_bytes between
 *      its 8-row groups.
 *
 *  Bits 0-13 hold the start address in 16-byte units, bits 16-29 and 32-45 both offsets alike, and
 *  bits 62-63 the layout. A K-major swizzled operand ignores the leading offset, and an N-major one
 *  steps it between column groups of the swizzle's width. The 64-byte swizzle XORs address bits 4-5
 *  with bits 7-8, so a start 32 bytes into a row reads the second half of the same rows, and the
 *  base-offset bits stay zero while every staged block starts on a 512-byte boundary.
 */
NUMKONG_DEVICE nk_u64_t nk_smem_descriptor_hopper_(nk_u32_t address, nk_u32_t layout, nk_u32_t leading_bytes,
                                                   nk_u32_t stride_bytes) {
    nk_u64_t const start = (address & 0x3FFFFu) >> 4, leading = leading_bytes >> 4, stride = stride_bytes >> 4;
    return start | (leading << 16) | (stride << 32) | ((nk_u64_t)layout << 62);
}

/** Pins @p count registers between the asynchronous @c wgmma steps and the code around them, so the
 *  compiler moves no read or write of them across a fence or a wait. They are pinned as F32 when
 *  @p sums is @c nk_cross_epilogue_f32_k and as integers otherwise, their type in the MMA, since a
 *  reinterpreting move would read them early. */
NUMKONG_DEVICE void nk_wgmma_fence_operands_hopper_(nk_fui32_t *registers, unsigned count, nk_cross_epilogue_t sums) {
#pragma unroll
    for (unsigned index = 0; index < count; ++index)
        if (sums == nk_cross_epilogue_f32_k) asm volatile("" : "+f"(registers[index].f)::"memory");
        else asm volatile("" : "+r"(registers[index].u)::"memory");
}

#pragma endregion Descriptors

#pragma region Multiplies

/** Adds the two 32-byte depth steps of one slab, the executing warpgroup's 64 A rows of @p stage
 *  against all its B rows, to @p accumulators, as BF16 sums. */
NUMKONG_DEVICE void nk_cross_issue_slab_bf16_hopper_(nk_fui32_t accumulators[64], unsigned char const *stage) {
    nk_u32_t const a_address = nk_shared_address_ampere_(stage + (threadIdx.x >> 7) * nk_cross_warpgroup_rows_hopper_k *
                                                                     nk_cross_slab_bytes_ampere_k);
    nk_u32_t const b_address = nk_shared_address_ampere_(stage + nk_cross_stage_bytes_ampere_k);
    nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_f32_k);
    nk_wgmma_fence_hopper_();
    // K-major rows of 64 bytes, the leading offset unread, in 8-row groups 512 bytes apart.
#pragma unroll
    for (unsigned step = 0; step < 2; ++step)
        nk_wgmma_bf16_hopper_(
            accumulators, nk_smem_descriptor_hopper_(a_address + 32 * step, nk_smem_swizzle_64_hopper_k, 16, 512),
            nk_smem_descriptor_hopper_(b_address + 32 * step, nk_smem_swizzle_64_hopper_k, 16, 512), 1);
    nk_wgmma_commit_hopper_();
}

/** Adds the two 32-byte depth steps of one slab, the executing warpgroup's 64 A rows of @p stage
 *  against all its B rows, to @p accumulators, as F16 sums. */
NUMKONG_DEVICE void nk_cross_issue_slab_f16_hopper_(nk_fui32_t accumulators[64], unsigned char const *stage) {
    nk_u32_t const a_address = nk_shared_address_ampere_(stage + (threadIdx.x >> 7) * nk_cross_warpgroup_rows_hopper_k *
                                                                     nk_cross_slab_bytes_ampere_k);
    nk_u32_t const b_address = nk_shared_address_ampere_(stage + nk_cross_stage_bytes_ampere_k);
    nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_f32_k);
    nk_wgmma_fence_hopper_();
    // K-major rows of 64 bytes, the leading offset unread, in 8-row groups 512 bytes apart.
#pragma unroll
    for (unsigned step = 0; step < 2; ++step)
        nk_wgmma_f16_hopper_(
            accumulators, nk_smem_descriptor_hopper_(a_address + 32 * step, nk_smem_swizzle_64_hopper_k, 16, 512),
            nk_smem_descriptor_hopper_(b_address + 32 * step, nk_smem_swizzle_64_hopper_k, 16, 512), 1);
    nk_wgmma_commit_hopper_();
}

/** Adds the two 32-byte depth steps of one slab, the executing warpgroup's 64 A rows of @p stage
 *  against all its B rows, to @p accumulators, as I8 sums. */
NUMKONG_DEVICE void nk_cross_issue_slab_i8_hopper_(nk_fui32_t accumulators[64], unsigned char const *stage) {
    nk_u32_t const a_address = nk_shared_address_ampere_(stage + (threadIdx.x >> 7) * nk_cross_warpgroup_rows_hopper_k *
                                                                     nk_cross_slab_bytes_ampere_k);
    nk_u32_t const b_address = nk_shared_address_ampere_(stage + nk_cross_stage_bytes_ampere_k);
    nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_i32_k);
    nk_wgmma_fence_hopper_();
    // K-major rows of 64 bytes, the leading offset unread, in 8-row groups 512 bytes apart.
#pragma unroll
    for (unsigned step = 0; step < 2; ++step)
        nk_wgmma_i8_hopper_(accumulators,
                            nk_smem_descriptor_hopper_(a_address + 32 * step, nk_smem_swizzle_64_hopper_k, 16, 512),
                            nk_smem_descriptor_hopper_(b_address + 32 * step, nk_smem_swizzle_64_hopper_k, 16, 512), 1);
    nk_wgmma_commit_hopper_();
}

/** Adds the two 32-byte depth steps of one slab, the executing warpgroup's 64 A rows of @p stage
 *  against all its B rows, to @p accumulators, as U8 sums. */
NUMKONG_DEVICE void nk_cross_issue_slab_u8_hopper_(nk_fui32_t accumulators[64], unsigned char const *stage) {
    nk_u32_t const a_address = nk_shared_address_ampere_(stage + (threadIdx.x >> 7) * nk_cross_warpgroup_rows_hopper_k *
                                                                     nk_cross_slab_bytes_ampere_k);
    nk_u32_t const b_address = nk_shared_address_ampere_(stage + nk_cross_stage_bytes_ampere_k);
    nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_i32_k);
    nk_wgmma_fence_hopper_();
    // K-major rows of 64 bytes, the leading offset unread, in 8-row groups 512 bytes apart.
#pragma unroll
    for (unsigned step = 0; step < 2; ++step)
        nk_wgmma_u8_hopper_(accumulators,
                            nk_smem_descriptor_hopper_(a_address + 32 * step, nk_smem_swizzle_64_hopper_k, 16, 512),
                            nk_smem_descriptor_hopper_(b_address + 32 * step, nk_smem_swizzle_64_hopper_k, 16, 512), 1);
    nk_wgmma_commit_hopper_();
}

/** Widens this thread's row of @p stage, 64 E2M3 codes, in place into i8 times 8, and publishes the
 *  stage to the tensor cores once every row is done. */
NUMKONG_DEVICE void nk_cross_widen_e2m3_hopper_(unsigned char *stage) {
    uint4 *row = (uint4 *)(stage + threadIdx.x * nk_cross_slab_bytes_ampere_k);
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        uint4 const codes = row[chunk];
        row[chunk] = make_uint4(nk_e2m3x4_to_i8x4_ampere_(codes.x), nk_e2m3x4_to_i8x4_ampere_(codes.y),
                                nk_e2m3x4_to_i8x4_ampere_(codes.z), nk_e2m3x4_to_i8x4_ampere_(codes.w));
    }
    nk_fence_proxy_async_hopper_();
    __syncthreads();
}

/**
 *  @brief Folds one slab of E2M1 nibble pairs, 128 values per row, in two passes: each widens half
 *      of this thread's row in place into a whole row of 1-byte codes, and issues its two steps.
 *
 *  Widened chunk @c j of a pass comes from raw bytes 8j to 8j + 7 of that half, in the order the
 *  widening leaves its nibbles, which A and B share. The second pass waits for every warp's first
 *  steps before it rewrites the stage.
 */
NUMKONG_DEVICE void nk_cross_multiply_e2m1_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    uint4 raw[4];
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk)
        raw[chunk] = *(uint4 const *)(stage +
                                      nk_swizzled_offset_simt_(threadIdx.x, chunk << 4, nk_cross_slab_bytes_ampere_k));
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        if (half) {
            nk_wgmma_wait_hopper_();
            __syncthreads();
        }
#pragma unroll
        for (unsigned chunk = 0; chunk < 4; ++chunk) {
            uint4 const codes = raw[half * 2 + chunk / 2];
            uint4 widened;
            nk_e2m1x8_to_i8x8_ampere_(chunk & 1 ? codes.z : codes.x, &widened.x, &widened.y);
            nk_e2m1x8_to_i8x8_ampere_(chunk & 1 ? codes.w : codes.y, &widened.z, &widened.w);
            *(uint4 *)(stage +
                       nk_swizzled_offset_simt_(threadIdx.x, chunk << 4, nk_cross_slab_bytes_ampere_k)) = widened;
        }
        nk_fence_proxy_async_hopper_();
        __syncthreads();
        nk_cross_issue_slab_i8_hopper_(accumulators, stage);
    }
}

/**
 *  @brief Folds one slab of I4 nibble pairs, 128 values per row, in two passes: each widens half of
 *      this thread's row in place into a whole row of 1-byte codes, and issues its two steps.
 *
 *  Widened chunk @c j of a pass comes from raw bytes 8j to 8j + 7 of that half, in the order the
 *  widening leaves its nibbles, which A and B share. The second pass waits for every warp's first
 *  steps before it rewrites the stage.
 */
NUMKONG_DEVICE void nk_cross_multiply_i4_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    uint4 raw[4];
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk)
        raw[chunk] = *(uint4 const *)(stage +
                                      nk_swizzled_offset_simt_(threadIdx.x, chunk << 4, nk_cross_slab_bytes_ampere_k));
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        if (half) {
            nk_wgmma_wait_hopper_();
            __syncthreads();
        }
#pragma unroll
        for (unsigned chunk = 0; chunk < 4; ++chunk) {
            uint4 const codes = raw[half * 2 + chunk / 2];
            uint4 widened;
            nk_i4x8_to_i8x8_simt_(chunk & 1 ? codes.z : codes.x, &widened.x, &widened.y);
            nk_i4x8_to_i8x8_simt_(chunk & 1 ? codes.w : codes.y, &widened.z, &widened.w);
            *(uint4 *)(stage +
                       nk_swizzled_offset_simt_(threadIdx.x, chunk << 4, nk_cross_slab_bytes_ampere_k)) = widened;
        }
        nk_fence_proxy_async_hopper_();
        __syncthreads();
        nk_cross_issue_slab_i8_hopper_(accumulators, stage);
    }
}

/**
 *  @brief Folds one slab of U4 nibble pairs, 128 values per row, in two passes: each widens half of
 *      this thread's row in place into a whole row of 1-byte codes, and issues its two steps.
 *
 *  Widened chunk @c j of a pass comes from raw bytes 8j to 8j + 7 of that half, in the order the
 *  widening leaves its nibbles, which A and B share. The second pass waits for every warp's first
 *  steps before it rewrites the stage.
 */
NUMKONG_DEVICE void nk_cross_multiply_u4_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    uint4 raw[4];
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk)
        raw[chunk] = *(uint4 const *)(stage +
                                      nk_swizzled_offset_simt_(threadIdx.x, chunk << 4, nk_cross_slab_bytes_ampere_k));
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        if (half) {
            nk_wgmma_wait_hopper_();
            __syncthreads();
        }
#pragma unroll
        for (unsigned chunk = 0; chunk < 4; ++chunk) {
            uint4 const codes = raw[half * 2 + chunk / 2];
            uint4 widened;
            nk_u4x8_to_u8x8_simt_(chunk & 1 ? codes.z : codes.x, &widened.x, &widened.y);
            nk_u4x8_to_u8x8_simt_(chunk & 1 ? codes.w : codes.y, &widened.z, &widened.w);
            *(uint4 *)(stage +
                       nk_swizzled_offset_simt_(threadIdx.x, chunk << 4, nk_cross_slab_bytes_ampere_k)) = widened;
        }
        nk_fence_proxy_async_hopper_();
        __syncthreads();
        nk_cross_issue_slab_u8_hopper_(accumulators, stage);
    }
}

NUMKONG_DEVICE void nk_dots_bf16_multiply_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    nk_cross_issue_slab_bf16_hopper_(accumulators, stage);
}

NUMKONG_DEVICE void nk_dots_f16_multiply_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    nk_cross_issue_slab_f16_hopper_(accumulators, stage);
}

/** E2M3 as i8 times 8, exact on the integer MMA, so the products come out times 64. */
NUMKONG_DEVICE void nk_dots_e2m3_multiply_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    nk_cross_widen_e2m3_hopper_(stage);
    nk_cross_issue_slab_i8_hopper_(accumulators, stage);
}

/** E2M1 as i8 times 2, exact on the integer MMA, so the products come out times 4. */
NUMKONG_DEVICE void nk_dots_e2m1_multiply_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    nk_cross_multiply_e2m1_hopper_(accumulators, stage);
}

NUMKONG_DEVICE void nk_dots_i8_multiply_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    nk_cross_issue_slab_i8_hopper_(accumulators, stage);
}

NUMKONG_DEVICE void nk_dots_u8_multiply_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    nk_cross_issue_slab_u8_hopper_(accumulators, stage);
}

NUMKONG_DEVICE void nk_dots_i4_multiply_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    nk_cross_multiply_i4_hopper_(accumulators, stage);
}

NUMKONG_DEVICE void nk_dots_u4_multiply_hopper_(nk_fui32_t accumulators[64], unsigned char *stage) {
    nk_cross_multiply_u4_hopper_(accumulators, stage);
}

#pragma endregion Multiplies

#pragma region Tile

NUMKONG_DEVICE void nk_cross_accumulators_clear_hopper_(nk_fui32_t accumulators[64]) {
#pragma unroll
    for (unsigned index = 0; index < 64; ++index) accumulators[index].u = 0;
}

/** Fills all but one stage of the ring ahead of the first slab; the first warpgroup issues every
 *  copy, as the Ampere tile's 128 threads do. */
NUMKONG_DEVICE void nk_cross_prologue_hopper_(nk_cross_shared_ampere_t *shared,
                                              nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                              nk_size_t first_column, int copies) {
#pragma unroll
    for (unsigned stage = 0; stage + 1 < nk_cross_stages_ampere_k; ++stage) {
        if (copies && stage < arguments->depth_slabs)
            nk_cross_stage_ampere_(shared->stages[stage][0], shared->stages[stage][1], arguments, first_row,
                                   first_column, stage * nk_cross_slab_bytes_ampere_k);
        nk_commit_async_ampere_();
    }
}

/** Refills the stage every warp is done with, with the slab @p slab + stages − 1 ahead. */
NUMKONG_DEVICE void nk_cross_prefetch_hopper_(nk_cross_shared_ampere_t *shared,
                                              nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                              nk_size_t first_column, nk_size_t slab, unsigned write_stage,
                                              int copies) {
    nk_size_t const prefetch = slab + nk_cross_stages_ampere_k - 1;
    if (copies && prefetch < arguments->depth_slabs)
        nk_cross_stage_ampere_(shared->stages[write_stage][0], shared->stages[write_stage][1], arguments, first_row,
                               first_column, prefetch * nk_cross_slab_bytes_ampere_k);
    nk_commit_async_ampere_();
}

/**
 *  @brief Writes the output tile once every step is done.
 *  @param[in] epilogue What the accumulators hold and how they reach the output.
 *  @param[in] output_scale Undoes the power of two a widening introduced, or 1.
 *  @param[in] norm How the squared norms are stored and the metric is computed; unused for dots.
 *  @param[in] metric The dot product itself, or a distance from it and the two squared norms.
 *  @param[in] squares Whether this thread squared a staged row, so @p norm_value is its row's.
 *  @param[in] norm_value This thread's finalized squared norm of A row t, or of B row t − 128.
 *
 *  Warpgroup @c w multiplies A rows 64w to 64w + 63 by all 128 B rows, holding accumulator @c i at
 *  row 64w + 16 · warp + 8 · ((i / 2) mod 2) + lane / 4 and column 8 · (i / 4) + 2 · (lane mod 4) +
 *  i mod 2. Row norms, then column norms, go to the ring's first stage after every warp's steps.
 */
NUMKONG_DEVICE void nk_cross_tile_store_hopper_(nk_cross_shared_ampere_t *shared, nk_fui32_t accumulators[64],
                                                nk_cross_epilogue_t epilogue, nk_f32_t output_scale,
                                                nk_cross_norm_t norm, nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                nk_size_t first_row, nk_size_t first_column, int squares,
                                                nk_fui32_t norm_value, nk_cross_tile_arguments_t const *arguments) {
    unsigned const lane = threadIdx.x & 31, warp = (threadIdx.x >> 5) & 3;
    unsigned const warpgroup_row = (threadIdx.x >> 7) * nk_cross_warpgroup_rows_hopper_k;
    nk_fui32_t *norms = shared->norms;
    if (metric != nk_cross_metric_dot_k) {
        __syncthreads();
        nk_size_t const column = first_column + threadIdx.x - nk_cross_tile_ampere_k;
        nk_u32_t const *column_norms = (nk_u32_t const *)arguments->b_norms;
        if (squares) norms[threadIdx.x] = norm_value;
        else norms[threadIdx.x].u = column < arguments->column_count ? column_norms[column] : 0;
        __syncthreads();
    }

    unsigned const group = lane >> 2, quad = lane & 3;
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        unsigned const tile_row = warpgroup_row + warp * 16 + half * 8 + group;
        nk_size_t const row = first_row + tile_row;
        if (row >= arguments->rows_end) continue;
        nk_size_t column_begin, column_end;
        nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin, &column_end);
        unsigned char *output = (unsigned char *)arguments->c + row * arguments->c_stride;
        nk_f32_t *output_f32 = (nk_f32_t *)output;
        nk_u32_t *output_u32 = (nk_u32_t *)output;
        nk_fui32_t row_norm;
        row_norm.u = metric == nk_cross_metric_dot_k ? 0 : norms[tile_row].u;
#pragma unroll
        for (unsigned column_tile = 0; column_tile < nk_cross_tile_ampere_k / 8; ++column_tile)
#pragma unroll
            for (unsigned pair = 0; pair < 2; ++pair) {
                unsigned const tile_column = column_tile * 8 + quad * 2 + pair;
                nk_size_t const column = first_column + tile_column;
                if (column < column_begin || column >= column_end) continue;
                nk_fui32_t const sum = accumulators[column_tile * 4 + half * 2 + pair];
                if (metric == nk_cross_metric_dot_k) {
                    if (epilogue == nk_cross_epilogue_f32_k) output_f32[column] = sum.f * output_scale;
                    else if (epilogue == nk_cross_epilogue_i32_k) output_u32[column] = sum.u;
                    else output_f32[column] = (nk_f32_t)sum.i * output_scale;
                    continue;
                }
                if (nk_cross_symmetric_simt_(band) && column == row) {
                    output_f32[column] = 0.0f;
                    continue;
                }
                nk_fui32_t const column_norm = norms[nk_cross_tile_ampere_k + tile_column];
                nk_f32_t const dot = nk_cross_dot_to_f32_simt_(sum, epilogue, output_scale);
                if (norm != nk_cross_norm_f32_k)
                    output_f32[column] = nk_cross_integer_metric_simt_(metric, norm, sum.u, row_norm.u, column_norm.u);
                else if (metric == nk_cross_metric_angular_k)
                    output_f32[column] = nk_f32_angular_simt_(dot, row_norm.f, column_norm.f);
                else output_f32[column] = nk_f32_euclidean_simt_(dot, row_norm.f, column_norm.f);
            }
    }
}

NUMKONG_DEVICE void nk_cross_tile_bf16_hopper_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                               nk_cross_tile_arguments_t const *arguments) {
    // A 512-byte boundary starts every stage, as the 64-byte swizzle's zero base offset requires.
    __shared__ __align__(1024) nk_cross_shared_ampere_t shared;

    int const copies = threadIdx.x < nk_cross_threads_ampere_k;
    int const squares = metric != nk_cross_metric_dot_k && (copies || nk_cross_symmetric_simt_(band));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;

        nk_fui32_t accumulators[64];
        nk_cross_accumulators_clear_hopper_(accumulators);
        nk_f32_t real_norm = 0;
        nk_cross_prologue_hopper_(&shared, arguments, first_row, first_column, copies);

        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            nk_fence_proxy_async_hopper_();
            // This stage's copies are in, and every warp is done with the stage refilled below.
            __syncthreads();
            unsigned char *stage = shared.stages[read_stage][0];
            if (squares) nk_cross_stage_norm_bf16_ampere_(stage, &real_norm);
            nk_dots_bf16_multiply_hopper_(accumulators, stage);
            nk_cross_prefetch_hopper_(&shared, arguments, first_row, first_column, slab, write_stage, copies);
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_f32_k);
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_hopper_(&shared, accumulators, nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, metric,
                                    band, first_row, first_column, squares,
                                    nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, real_norm, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_f16_hopper_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                              nk_cross_tile_arguments_t const *arguments) {
    // A 512-byte boundary starts every stage, as the 64-byte swizzle's zero base offset requires.
    __shared__ __align__(1024) nk_cross_shared_ampere_t shared;

    int const copies = threadIdx.x < nk_cross_threads_ampere_k;
    int const squares = metric != nk_cross_metric_dot_k && (copies || nk_cross_symmetric_simt_(band));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;

        nk_fui32_t accumulators[64];
        nk_cross_accumulators_clear_hopper_(accumulators);
        nk_f32_t real_norm = 0;
        nk_cross_prologue_hopper_(&shared, arguments, first_row, first_column, copies);

        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            nk_fence_proxy_async_hopper_();
            // This stage's copies are in, and every warp is done with the stage refilled below.
            __syncthreads();
            unsigned char *stage = shared.stages[read_stage][0];
            if (squares) nk_cross_stage_norm_f16_ampere_(stage, &real_norm);
            nk_dots_f16_multiply_hopper_(accumulators, stage);
            nk_cross_prefetch_hopper_(&shared, arguments, first_row, first_column, slab, write_stage, copies);
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_f32_k);
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_hopper_(&shared, accumulators, nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, metric,
                                    band, first_row, first_column, squares,
                                    nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, real_norm, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_e2m3_hopper_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                               nk_cross_tile_arguments_t const *arguments) {
    // A 512-byte boundary starts every stage, as the 64-byte swizzle's zero base offset requires.
    __shared__ __align__(1024) nk_cross_shared_ampere_t shared;

    int const copies = threadIdx.x < nk_cross_threads_ampere_k;
    int const squares = metric != nk_cross_metric_dot_k && (copies || nk_cross_symmetric_simt_(band));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;

        nk_fui32_t accumulators[64];
        nk_cross_accumulators_clear_hopper_(accumulators);
        nk_u32_t integer_norm = 0;
        nk_cross_prologue_hopper_(&shared, arguments, first_row, first_column, copies);

        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            nk_fence_proxy_async_hopper_();
            // This stage's copies are in, and every warp is done with the stage refilled below.
            __syncthreads();
            unsigned char *stage = shared.stages[read_stage][0];
            if (squares) nk_cross_stage_norm_e2m3_ampere_(stage, &integer_norm);
            nk_dots_e2m3_multiply_hopper_(accumulators, stage);
            nk_cross_prefetch_hopper_(&shared, arguments, first_row, first_column, slab, write_stage, copies);
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_i32_to_f32_k);
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_hopper_(&shared, accumulators, nk_cross_epilogue_i32_to_f32_k, 0.015625f,
                                    nk_cross_norm_f32_k, metric, band, first_row, first_column, squares,
                                    nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, integer_norm, 0, 0.015625f),
                                    arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_e2m1_hopper_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                               nk_cross_tile_arguments_t const *arguments) {
    // A 512-byte boundary starts every stage, as the 64-byte swizzle's zero base offset requires.
    __shared__ __align__(1024) nk_cross_shared_ampere_t shared;

    int const copies = threadIdx.x < nk_cross_threads_ampere_k;
    int const squares = metric != nk_cross_metric_dot_k && (copies || nk_cross_symmetric_simt_(band));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;

        nk_fui32_t accumulators[64];
        nk_cross_accumulators_clear_hopper_(accumulators);
        nk_u32_t integer_norm = 0;
        nk_cross_prologue_hopper_(&shared, arguments, first_row, first_column, copies);

        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            nk_fence_proxy_async_hopper_();
            // This stage's copies are in, and every warp is done with the stage refilled below.
            __syncthreads();
            unsigned char *stage = shared.stages[read_stage][0];
            if (squares) nk_cross_stage_norm_e2m1_ampere_(stage, &integer_norm);
            nk_dots_e2m1_multiply_hopper_(accumulators, stage);
            nk_cross_prefetch_hopper_(&shared, arguments, first_row, first_column, slab, write_stage, copies);
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_i32_to_f32_k);
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_hopper_(&shared, accumulators, nk_cross_epilogue_i32_to_f32_k, 0.25f, nk_cross_norm_f32_k,
                                    metric, band, first_row, first_column, squares,
                                    nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, integer_norm, 0, 0.25f),
                                    arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_i8_hopper_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                             nk_cross_tile_arguments_t const *arguments) {
    // A 512-byte boundary starts every stage, as the 64-byte swizzle's zero base offset requires.
    __shared__ __align__(1024) nk_cross_shared_ampere_t shared;

    int const copies = threadIdx.x < nk_cross_threads_ampere_k;
    int const squares = metric != nk_cross_metric_dot_k && (copies || nk_cross_symmetric_simt_(band));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;

        nk_fui32_t accumulators[64];
        nk_cross_accumulators_clear_hopper_(accumulators);
        nk_u32_t integer_norm = 0;
        nk_cross_prologue_hopper_(&shared, arguments, first_row, first_column, copies);

        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            nk_fence_proxy_async_hopper_();
            // This stage's copies are in, and every warp is done with the stage refilled below.
            __syncthreads();
            unsigned char *stage = shared.stages[read_stage][0];
            if (squares) nk_cross_stage_norm_i8_ampere_(stage, &integer_norm);
            nk_dots_i8_multiply_hopper_(accumulators, stage);
            nk_cross_prefetch_hopper_(&shared, arguments, first_row, first_column, slab, write_stage, copies);
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_i32_k);
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_hopper_(
            &shared, accumulators, nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_i32_k, metric, band, first_row,
            first_column, squares, nk_cross_norm_finalize_simt_(nk_cross_norm_i32_k, integer_norm, 0, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_i4_hopper_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                             nk_cross_tile_arguments_t const *arguments) {
    // A 512-byte boundary starts every stage, as the 64-byte swizzle's zero base offset requires.
    __shared__ __align__(1024) nk_cross_shared_ampere_t shared;

    int const copies = threadIdx.x < nk_cross_threads_ampere_k;
    int const squares = metric != nk_cross_metric_dot_k && (copies || nk_cross_symmetric_simt_(band));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;

        nk_fui32_t accumulators[64];
        nk_cross_accumulators_clear_hopper_(accumulators);
        nk_u32_t integer_norm = 0;
        nk_cross_prologue_hopper_(&shared, arguments, first_row, first_column, copies);

        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            nk_fence_proxy_async_hopper_();
            // This stage's copies are in, and every warp is done with the stage refilled below.
            __syncthreads();
            unsigned char *stage = shared.stages[read_stage][0];
            if (squares) nk_cross_stage_norm_i4_ampere_(stage, &integer_norm);
            nk_dots_i4_multiply_hopper_(accumulators, stage);
            nk_cross_prefetch_hopper_(&shared, arguments, first_row, first_column, slab, write_stage, copies);
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_i32_k);
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_hopper_(
            &shared, accumulators, nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_i32_k, metric, band, first_row,
            first_column, squares, nk_cross_norm_finalize_simt_(nk_cross_norm_i32_k, integer_norm, 0, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_u8_hopper_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                             nk_cross_tile_arguments_t const *arguments) {
    // A 512-byte boundary starts every stage, as the 64-byte swizzle's zero base offset requires.
    __shared__ __align__(1024) nk_cross_shared_ampere_t shared;

    int const copies = threadIdx.x < nk_cross_threads_ampere_k;
    int const squares = metric != nk_cross_metric_dot_k && (copies || nk_cross_symmetric_simt_(band));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;

        nk_fui32_t accumulators[64];
        nk_cross_accumulators_clear_hopper_(accumulators);
        nk_u32_t integer_norm = 0;
        nk_cross_prologue_hopper_(&shared, arguments, first_row, first_column, copies);

        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            nk_fence_proxy_async_hopper_();
            // This stage's copies are in, and every warp is done with the stage refilled below.
            __syncthreads();
            unsigned char *stage = shared.stages[read_stage][0];
            if (squares) nk_cross_stage_norm_u8_ampere_(stage, &integer_norm);
            nk_dots_u8_multiply_hopper_(accumulators, stage);
            nk_cross_prefetch_hopper_(&shared, arguments, first_row, first_column, slab, write_stage, copies);
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_i32_k);
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_hopper_(
            &shared, accumulators, nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_u32_k, metric, band, first_row,
            first_column, squares, nk_cross_norm_finalize_simt_(nk_cross_norm_u32_k, integer_norm, 0, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_u4_hopper_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                             nk_cross_tile_arguments_t const *arguments) {
    // A 512-byte boundary starts every stage, as the 64-byte swizzle's zero base offset requires.
    __shared__ __align__(1024) nk_cross_shared_ampere_t shared;

    int const copies = threadIdx.x < nk_cross_threads_ampere_k;
    int const squares = metric != nk_cross_metric_dot_k && (copies || nk_cross_symmetric_simt_(band));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;

        nk_fui32_t accumulators[64];
        nk_cross_accumulators_clear_hopper_(accumulators);
        nk_u32_t integer_norm = 0;
        nk_cross_prologue_hopper_(&shared, arguments, first_row, first_column, copies);

        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            nk_fence_proxy_async_hopper_();
            // This stage's copies are in, and every warp is done with the stage refilled below.
            __syncthreads();
            unsigned char *stage = shared.stages[read_stage][0];
            if (squares) nk_cross_stage_norm_u4_ampere_(stage, &integer_norm);
            nk_dots_u4_multiply_hopper_(accumulators, stage);
            nk_cross_prefetch_hopper_(&shared, arguments, first_row, first_column, slab, write_stage, copies);
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(accumulators, 64, nk_cross_epilogue_i32_k);
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_hopper_(
            &shared, accumulators, nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_u32_k, metric, band, first_row,
            first_column, squares, nk_cross_norm_finalize_simt_(nk_cross_norm_u32_k, integer_norm, 0, 1.0f), arguments);
    }
}

#pragma endregion Tile

#pragma region BF16

nk_define_cross_pack_cuda_(bf16, hopper, bf16, bf16, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, bf16, hopper, bf16_hopper, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_cuda_(f16, hopper, f16, f16, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, f16, hopper, f16_hopper, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region E2M3

nk_define_cross_pack_cuda_(e2m3, hopper, e2m3, e2m3, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, e2m3, hopper, e2m3_hopper, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_cuda_(e2m1, hopper, e2m1x2, e2m1x2, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_cuda_(dot, e2m1, hopper, e2m1_hopper, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_pack_cuda_(i8, hopper, i8, i8, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, i8, hopper, i8_hopper, i8, i8, i32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_cuda_(i4, hopper, i4x2, i4x2, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_cuda_(dot, i4, hopper, i4_hopper, i4x2, i4x2, i32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_cuda_(u8, hopper, u8, u8, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, u8, hopper, u8_hopper, u8, u8, u32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_cuda_(u4, hopper, u4x2, u4x2, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_cuda_(dot, u4, hopper, u4_hopper, u4x2, u4x2, u32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion U4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_HOPPER
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_DOTS_HOPPER_CUH
