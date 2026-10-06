/**
 *  @file include/numkong/dots/sme.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *  @brief SIMD-accelerated Batched Dot Products for SME.
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses ARM Scalable Matrix Extension, SME, with @c ZA32 tiles for every narrow input type:
 *
 *  - @c svmopa_za32_f16_m: f16, e4m3, e5m2, e3m2 and NVFP4 decoded to f16, accumulated in f32
 *  - @c svmopa_za32_bf16_m: bf16 and the MX formats decoded to bf16, accumulated in f32
 *  - @c svmopa_za32_s8_m: i8, i4, e2m3 and e2m1 decoded to i8, accumulated in i32
 *  - @c svmopa_za32_u8_m: u8 and u4, accumulated in u32
 *
 *  SME tile dimensions, for SVL=512, i.e., Apple M4 and M5:
 *
 *  - @c ZA32 tile: @b [16,16] @c f32 or @c i32 elements, 1 KB
 *  - @c f16 and @c bf16 vectors: 32 elements per SVE vector
 *  - @c i8 and @c u8 vectors: 64 elements per SVE vector
 *  - @c f32 and @c i32 vectors: 16 elements per SVE vector
 *
 *  Every kernel shares one shape. A operand rows are decoded once per 16-row tile and depth chunk,
 *  transposed through ZA0 into a stack panel in MOPA operand order, and then all four ZA tiles
 *  accumulate: two panels against two column tiles for f16 and 8-bit inputs, one panel against
 *  four column tiles for bf16 inputs, whose 2×2 form measured slower. Symmetric kernels decode
 *  windows of eight column tiles once per depth chunk and reuse them for every row tile.
 *
 *  Block-scaled formats fold their scales into the operands. An NVFP4 element times its UE4M3 scale
 *  has at most 6 significant bits within 2⁻¹⁰ … 2688, an exact f16, so NVFP4 needs no guard and its
 *  tensor scales multiply the result once. MX elements have at most 4 significant bits, so an
 *  element times 2^(e_block − base) is an exact bf16 while a row spans at most 32 binades. A rows
 *  rebase to their smallest block exponent, MXFP4 rows and packed B columns to their largest, so
 *  the E2M1 fold is one saturating decrement, and the epilogue applies 2^(base_row + base_column)
 *  with one rounding. Rows or columns spanning more are recomputed by an exact scalar loop after
 *  streaming ends. FP4 packs keep one byte per code and decode B with @c TBL inside the loop, which
 *  matches @c LUTI4 without requiring SME2.
 *
 *  Every kernel is vector-length agnostic: tables of up to 16 halfwords or 32 bytes load as
 *  register pairs for the SVE2 two-register @c TBL, which holds them at SVL 128, and vectors wider
 *  than 512 bits decode block-scaled rows 32 dims at a time.
 *
 *  Rules measured on the M5 Pro, which every kernel here follows:
 *
 *  - Four rotating widening @c FMOPA take 1.87 ns, 1.1 TMAC/s. Integer, @c TBL and @c LD1
 *    instructions overlap them at about one per cycle, two per @c FMOPA, while streaming @c FMUL
 *    and @c FMLA sustain ~1.07 G/s, so hot loops spend at most half an FP instruction per @c FMOPA.
 *  - A function owning ZA calls only @c always_inline or `__arm_inout("za")` helpers; any other
 *    call emits `smstart za` and @c __arm_tpidr2_restore around it.
 *  - Operands are decoded once per 16 rows and depth chunk, never once per column tile.
 *
 *  @section dots_sme_instructions ARM SME Instructions
 *
 *  @verbatim
 *  Intrinsic                       Instruction                         Latency     Throughput
 *  svmopa_za32_f16_m               FMOPA   (ZA.S, P/M, Z.H, Z.H)       16cy        amortized
 *  svmopa_za32_bf16_m              BFMOPA   (ZA.S, P/M, Z.H, Z.H)      16cy        amortized
 *  svmopa_za32_s8_m                SMOPA   (ZA.S, P/M, Z.B, Z.B)       16cy        amortized
 *  svmopa_za32_u8_m                UMOPA   (ZA.S, P/M, Z.B, Z.B)       16cy        amortized
 *  svzero_za                       ZERO   (ZA)                         2cy         1/cy
 *  svst1_ver_za32                  ST1W   (ZA.S[Ws, #imm], P)          4cy         1/cy
 *  svwrite_hor_za32_u32_m          MOVA   (ZA.S[Ws, #imm], P/M, Z.S)   2cy         1/cy
 *  svread_hor_za32_f32_m           MOVA   (Z.S, P/M, ZA.S[Ws, #imm])   2cy         1/cy
 *  svtbl_u16                       TBL    (Z.H, {Z.H}, Z.H)            2cy         1/cy
 *  __arm_streaming                 SMSTART                             ~50-100cy
 *  __arm_streaming   (exit)        SMSTOP                              ~50-100cy
 *  __arm_new("za")                 ZA tile allocation                  0cy
 *  @endverbatim
 */
#ifndef NUMKONG_DOTS_SME_H
#define NUMKONG_DOTS_SME_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_SME_

#include "numkong/types.h"
#include "numkong/dots/serial.h" // `nk_dots_reduce_sumsq_f16_`, `nk_cross_operand_t`

#if defined(__cplusplus)
extern "C" {
#endif

/*  Safe SVE vector-length queries usable from non-streaming context. On Apple M4 (and other
 *  SME-only-SVE cores), SVE instructions like CNTW/CNTH/CNTB trap with SIGILL outside streaming
 *  mode. These helpers bracket the query with SMSTART SM / SMSTOP SM so the calling function's ABI
 *  is unchanged. Inside @c __arm_locally_streaming functions the plain `svcntXX()` intrinsics are
 *  fine. The transitions zero every Z and P register, so the asm declares the V and P registers
 *  clobbered — the "v" spelling is the one Clang reliably honors for values it keeps in FP/SIMD
 *  registers; with bare "z" names Clang kept a live @c s0 argument in place across the bracket and
 *  the first SMSTART silently zeroed it. */

/** Streaming SVL byte-element count (SVL/8) via SMSTART SM bracket. */
NUMKONG_INLINE nk_size_t nk_sme_cntb_(void) {
    nk_u64_t r;
    __asm__ __volatile__("smstart sm\n\tcntb %0\n\tsmstop sm"
                         : "=r"(r)
                         :
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                           "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                           "v27", "v28", "v29", "v30", "v31", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
                           "p9", "p10", "p11", "p12", "p13", "p14", "p15");
    return (nk_size_t)r;
}

/** Streaming SVL half-element count (SVL/16) via SMSTART SM bracket. */
NUMKONG_INLINE nk_size_t nk_sme_cnth_(void) {
    nk_u64_t r;
    __asm__ __volatile__("smstart sm\n\tcnth %0\n\tsmstop sm"
                         : "=r"(r)
                         :
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                           "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                           "v27", "v28", "v29", "v30", "v31", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
                           "p9", "p10", "p11", "p12", "p13", "p14", "p15");
    return (nk_size_t)r;
}

/** Streaming SVL word-element count (SVL/32) via SMSTART SM bracket. */
NUMKONG_INLINE nk_size_t nk_sme_cntw_(void) {
    nk_u64_t r;
    __asm__ __volatile__("smstart sm\n\tcntw %0\n\tsmstop sm"
                         : "=r"(r)
                         :
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                           "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                           "v27", "v28", "v29", "v30", "v31", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
                           "p9", "p10", "p11", "p12", "p13", "p14", "p15");
    return (nk_size_t)r;
}

/** Streaming SVL double-element count (SVL/64) via SMSTART SM bracket. */
NUMKONG_INLINE nk_size_t nk_sme_cntd_(void) {
    nk_u64_t r;
    __asm__ __volatile__("smstart sm\n\tcntd %0\n\tsmstop sm"
                         : "=r"(r)
                         :
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                           "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                           "v27", "v28", "v29", "v30", "v31", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
                           "p9", "p10", "p11", "p12", "p13", "p14", "p15");
    return (nk_size_t)r;
}

/** Enter streaming SVE mode (PSTATE.SM = 1). Caller is responsible for smstop. The transition
 *  zeroes every Z and P register, so they are declared clobbered — otherwise values the compiler
 *  caches in the callee-saved V8-V15 are silently lost. */
NUMKONG_INLINE void nk_sme_start_streaming_(void) {
    __asm__ __volatile__("smstart sm"
                         :
                         :
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                           "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                           "v27", "v28", "v29", "v30", "v31", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
                           "p9", "p10", "p11", "p12", "p13", "p14", "p15", "memory");
}

/** Exit streaming SVE mode (PSTATE.SM = 0). Must pair with nk_sme_start_streaming_. */
NUMKONG_INLINE void nk_sme_stop_streaming_(void) {
    __asm__ __volatile__("smstop sm"
                         :
                         :
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                           "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                           "v27", "v28", "v29", "v30", "v31", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
                           "p9", "p10", "p11", "p12", "p13", "p14", "p15", "memory");
}

/**
 *  @brief SME runtime stubs — weak definitions for symbols the compiler may reference from
 *      @c __arm_streaming or `__arm_new("za")` functions.
 *
 *  Every TU that includes this header emits a weak copy; the linker deduplicates to one.
 *
 *  @c __arm_tpidr2_save and @c __arm_tpidr2_restore implement the lazy ZA save/restore protocol
 *  used in `__arm_new("za")` prologues. They are always no-ops in NumKong because no
 *  @c NUMKONG_API function carries ZA state, so TPIDR2_EL0 is always null at entry. The
 *  @c used attribute is load-bearing under LTO: the prologue call is synthesized by the backend,
 *  long after IPA would drop these as unreferenced.
 *
 *  @c __arm_sc_memset, @c __arm_sc_memcpy, and @c __arm_sc_memmove are streaming-compatible memory
 *  routines the compiler may emit inside @c __arm_streaming functions. Apple Clang provides these
 *  in its runtime; upstream LLVM does not. GCC needs no stub because it never emits calls to them —
 *  and must not get one, as its `arm_sme.h` declares them with C++ linkage.
 */
__attribute__((weak, used)) void __arm_tpidr2_save(void) {}
__attribute__((weak, used)) void __arm_tpidr2_restore(void *blk) { nk_unused_(blk); }
#if defined(__clang__) // GCC's `arm_sme.h` declares these with C++ linkage, which collides here
/* Prevent memory-loop recognition from introducing calls back into these stubs. */
#pragma clang attribute push(__attribute__((no_builtin("memset", "memcpy", "memmove"))), apply_to = function)

__attribute__((weak, target("+sme"))) void *__arm_sc_memset(void *d, int c,
                                                            __SIZE_TYPE__ n) __arm_streaming_compatible {
    unsigned char *p = (unsigned char *)d;
    for (__SIZE_TYPE__ i = 0; i < n; i++) p[i] = (unsigned char)c;
    return d;
}
__attribute__((weak, target("+sme"))) void *__arm_sc_memcpy(void *d, void const *s,
                                                            __SIZE_TYPE__ n) __arm_streaming_compatible {
    unsigned char *dp = (unsigned char *)d;
    unsigned char const *sp = (unsigned char const *)s;
    for (__SIZE_TYPE__ i = 0; i < n; i++) dp[i] = sp[i];
    return d;
}
__attribute__((weak, target("+sme"))) void *__arm_sc_memmove(void *d, void const *s,
                                                             __SIZE_TYPE__ n) __arm_streaming_compatible {
    unsigned char *dp = (unsigned char *)d;
    unsigned char const *sp = (unsigned char const *)s;
    if (dp < sp) {
        for (__SIZE_TYPE__ i = 0; i < n; i++) dp[i] = sp[i];
    }
    else {
        for (__SIZE_TYPE__ i = n; i > 0; i--) dp[i - 1] = sp[i - 1];
    }
    return d;
}

#pragma clang attribute pop
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

/** SME-specific packed buffer header (64-byte aligned).
 *  Layout optimized for SME outer product access patterns with predicate-based edge handling. */
typedef struct {

    /** Column tiles, ⌈columns / tile dimension⌉. */
    nk_u32_t column_tile_count;

    /** Depth steps per column tile, one MOPA operand vector each. */
    nk_u32_t depth_tile_count;

    /** Columns, not padded, for the predicates. */
    nk_u32_t columns;

    /** Depth, not padded, for the predicates. */
    nk_u32_t depth;

    /** Streaming vector length in bytes at pack time, which every consumer validates. */
    nk_u32_t svl_bytes;

    /** Byte offset from the buffer start to the per-column norms. */
    nk_u32_t norms_offset;

    /** The packed operand's tensor scale, 1 when it has none, which every product multiplies in. */
    nk_f32_t tensor_scale;

    /** Zeroed; pads the header to 64 bytes. */
    nk_u32_t reserved[7];

    /** The capability that packed the buffer, which every consumer checks. */
    nk_capability_t capability;
} nk_dots_sme_packed_header_t;

/*  Selective ZA tile zeroing masks for `svzero_mask_za(mask)`, which zero individual tiles without
 *  destroying other accumulators. ZA.S tile t is bits t and t + 4; ZA.D tile t is bit t. */
enum {

    /** ZA0.S alone. */
    nk_sme_zero_za32_tile_0_k = 0x11,

    /** ZA1.S alone. */
    nk_sme_zero_za32_tile_1_k = 0x22,

    /** ZA2.S alone. */
    nk_sme_zero_za32_tile_2_k = 0x44,

    /** ZA3.S alone. */
    nk_sme_zero_za32_tile_3_k = 0x88,

    /** ZA1.S to ZA3.S, the accumulators, keeping the ZA0.S staging tile. */
    nk_sme_zero_za32_tiles_123_k = 0xEE,

    /** ZA0.D alone. */
    nk_sme_zero_za64_tile_0_k = 0x01,

    /** ZA1.D alone. */
    nk_sme_zero_za64_tile_1_k = 0x02,

    /** ZA2.D alone. */
    nk_sme_zero_za64_tile_2_k = 0x04,

    /** ZA1.D to ZA5.D and ZA7.D, the Ozaki products, keeping the staging tiles. */
    nk_sme_zero_za64_tiles_1_5_7_k = 0xBE,

    /** ZA1.D to ZA7.D, the accumulators, keeping the ZA0.D staging tile. */
    nk_sme_zero_za64_tiles_1_7_k = 0xFE,
};

/*  Stack panel geometry of the GEMM kernels, sized in bytes so every streaming vector length keeps
 *  the same frame. */
enum {

    /** Bytes of one tile's operand panel per depth chunk in packed kernels, 2048 f16 or 4096 byte
     *  dims at SVL 512; every chunk drains ZA once per tile group, and halving it cost skinny FP4
     *  GEMMs 15% of their throughput. */
    nk_sme_panel_bytes_k = 65536,

    /** Bytes of one tile's panel per depth chunk in symmetric kernels, 256 f16 dims at SVL 512. */
    nk_sme_window_panel_bytes_k = 8192,

    /** Column tiles a symmetric window decodes once per depth chunk. */
    nk_sme_window_tiles_k = 8,

    /** Widest UE8M0 exponent spread a row or column may span and still fold into bf16 exactly. */
    nk_sme_exponent_spread_k = 32,

    /** Rebasing exponent that leaves MX elements unscaled, for columns wider than the spread. */
    nk_sme_unfolded_base_k = -1024,

    /** Largest tile dimension the stack scratch serves, SVL 2048. */
    nk_sme_max_tile_k = 64,
};

/** Clears the lanes of @p bound left of the diagonal on row @p row_index, for a tile starting at
 *  @p column_start. */
NUMKONG_INLINE svbool_t nk_sme_diagonal_cut_b32x_(svbool_t bound, nk_size_t column_start,
                                                  nk_size_t row_index) NUMKONG_STREAMING_ {
    return svbic_b_z(bound, bound, svwhilelt_b32_u64(column_start, row_index));
}

/** Clears the lanes of @p bound left of the diagonal on row @p row_index, for a tile starting at
 *  @p column_start. */
NUMKONG_INLINE svbool_t nk_sme_diagonal_cut_b64x_(svbool_t bound, nk_size_t column_start,
                                                  nk_size_t row_index) NUMKONG_STREAMING_ {
    return svbic_b_z(bound, bound, svwhilelt_b64_u64(column_start, row_index));
}

#pragma region Decoders

/** Up to 16 u16 entries of @p data as the vector pair @c svtbl2_u16 reads as entries 0 … 15 at
 *  every vector length: the first vector holds what fits and the second the rest, which only SVL
 *  128 needs. Indices past the entries read zero. */
NUMKONG_INLINE svuint16x2_t nk_sme_table16_u16_(nk_u16_t const *data) NUMKONG_STREAMING_ {
    nk_size_t const lanes = svcnth(), split = lanes < 16 ? lanes : 16;
    return svcreate2_u16(svld1_u16(svwhilelt_b16_u64(0, 16), data),
                         svld1_u16(svwhilelt_b16_u64(split, 16), data + split));
}

/** Up to 32 byte entries of @p data as the vector pair @c svtbl2_u8 reads as entries 0 … 31 at
 *  every vector length, like @c nk_sme_table16_u16_. */
NUMKONG_INLINE svuint8x2_t nk_sme_table32_u8_(nk_u8_t const *data) NUMKONG_STREAMING_ {
    nk_size_t const lanes = svcntb(), split = lanes < 32 ? lanes : 32;
    return svcreate2_u8(svld1_u8(svwhilelt_b8_u64(0, 32), data), svld1_u8(svwhilelt_b8_u64(split, 32), data + split));
}

/** E2M1 codes as F16 bits, for NVFP4. */
static nk_align_(64) nk_u16_t const nk_sme_nvfp4_table_data_[16] = {0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200,
                                                                    0x4400, 0x4600, 0x8000, 0xB800, 0xBC00, 0xBE00,
                                                                    0xC000, 0xC200, 0xC400, 0xC600};

/** E2M1 codes as BF16 bits with both zeros at +0, so a saturating exponent decrement keeps zeros
 *  at zero. */
static nk_align_(64) nk_u16_t const nk_sme_mxfp4_table_data_[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040,
                                                                    0x4080, 0x40C0, 0x0000, 0xBF00, 0xBF80, 0xBFC0,
                                                                    0xC000, 0xC040, 0xC080, 0xC0C0};

/** @p table_u16x with @p shift subtracted, lane for lane, from the entries whose @p fixed_data flag
 *  is set. */
NUMKONG_INLINE svuint16x2_t nk_sme_table16_shift_(svuint16x2_t table_u16x, nk_u16_t const *fixed_data,
                                                  nk_u16_t shift) NUMKONG_STREAMING_ {
    svbool_t const all_b16x = svptrue_b16();
    svuint16x2_t const fixed_u16x = nk_sme_table16_u16_(fixed_data);
    return svcreate2_u16(
        svsub_n_u16_m(svcmpne_n_u16(all_b16x, svget2_u16(fixed_u16x, 0), 0), svget2_u16(table_u16x, 0), shift),
        svsub_n_u16_m(svcmpne_n_u16(all_b16x, svget2_u16(fixed_u16x, 1), 0), svget2_u16(table_u16x, 1), shift));
}

/** Bits of the mini-float magnitudes @p magnitude_u16x as @p base plus magnitude << @p shift,
 *  corrected by @p deltas_u16x at index (magnitude + @p rotation) mod 128, which puts the Inf and
 *  NaN magnitudes just below 128 first and the subnormals from zero after them. Indices past the
 *  table read zero, so no compare sits on the decode path; streaming integer instructions issue
 *  about once per cycle, which makes the instruction count the cost. */
NUMKONG_INLINE svuint16_t nk_minifloat_magnitudes_ssve_(svuint16_t magnitude_u16x, nk_size_t shift, nk_u16_t base,
                                                        nk_u16_t rotation,
                                                        svuint16x2_t deltas_u16x) NUMKONG_STREAMING_ {
    svbool_t const all_b16x = svptrue_b16();
    svuint16_t const index_u16x = rotation
                                      ? svand_n_u16_x(all_b16x, svadd_n_u16_x(all_b16x, magnitude_u16x, rotation), 0x7F)
                                      : magnitude_u16x;
    svuint16_t const bits_u16x = svadd_n_u16_x(all_b16x, svlsl_n_u16_x(all_b16x, magnitude_u16x, shift), base);
    return svadd_u16_x(all_b16x, bits_u16x, svtbl2_u16(deltas_u16x, index_u16x));
}

/**
 *  @brief Inline e4m3 → f16 conversion for streaming SVE.
 *
 *  @verbatim
 *  E4M3FN:  S EEEE MMM          bias = 7, no ∞, NaN = magnitude 0x7F
 *  F16:     S EEEEE MMMMMMMMMM  bias = 15
 *  Normal:     magnitude 8..126 → f16 = sign | ((magnitude << 7) + 0x2000)
 *  Subnormal:  magnitude 0..7   → exact f16 from a correction table
 *  NaN:        magnitude 127    → f16 quiet NaN 0x7E00 | sign
 *  @endverbatim
 *
 *  For normal values, the +0x2000 encodes the bias difference 15 − 7 = 8 and left-aligns the
 *  mantissa; @c nk_minifloat_magnitudes_ssve_ patches subnormals and NaN without compares.
 *
 *  @param[in] predicate_b16x Active-lane predicate.
 *  @param[in] bytes_u8x Pre-loaded e4m3 bytes from @c svld1_u8.
 *  @return @c svfloat16_t with converted values, zero for inactive lanes.
 */
NUMKONG_INLINE svfloat16_t nk_e4m3x_to_f16x_ssve_(svbool_t predicate_b16x, svuint8_t bytes_u8x) NUMKONG_STREAMING_ {
    static nk_align_(64)
        nk_u16_t const deltas_data[16] = {0x1E80, 0xE000, 0xF780, 0xFB00, 0xFC80, 0xFE00, 0xFE80, 0xFF00, 0xFF80};
    svbool_t const all_b16x = svptrue_b16();
    svuint16_t const codes_u16x = svunpklo_u16(bytes_u8x);
    svuint16_t const bits_u16x = nk_minifloat_magnitudes_ssve_(svand_n_u16_x(all_b16x, codes_u16x, 0x7F), 7, 0x2000, 1,
                                                               nk_sme_table16_u16_(deltas_data));
    svuint16_t const signed_u16x = svbsl_n_u16(svlsl_n_u16_x(all_b16x, codes_u16x, 8), bits_u16x, 0x8000);
    return svreinterpret_f16_u16(svsel_u16(predicate_b16x, signed_u16x, svdup_n_u16(0)));
}

/**
 *  @brief Inline e5m2 → f16 conversion returning @c svfloat16_t for direct use in GEMM.
 *
 *  @verbatim
 *  E5M2:  S EEEEE MM          1+5+2 bits, bias = 15, range [-57344, 57344]
 *  F16:   S EEEEE MMMMMMMMMM  1+5+10 bits, bias = 15
 *  @endverbatim
 *
 *  Since E5M2 and F16 share the same exponent bias of 15, normal values convert by simply shifting
 *  the magnitude left by 8 bits.
 *
 *  @param[in] predicate_b16x Predicate for 16-bit elements, as from @c svptrue_b16.
 *  @param[in] bytes_u8x Pre-loaded 64 bytes, an @c svuint8_t from @c svld1_u8.
 *  @return 32 F16 values as @c svfloat16_t, from the lower 32 bytes.
 */
NUMKONG_INLINE svfloat16_t nk_e5m2x_to_f16x_ssve_(svbool_t predicate_b16x, svuint8_t bytes_u8x) NUMKONG_STREAMING_ {
    // E5M2 and F16 share the same exponent bias (15), sign position, exponent width,
    // and mantissa field alignment. The conversion f16 = byte << 8 is exact for all
    // 256 values but NaN, whose codes first become the canonical quiet 0x7E.
    svuint16_t e5m2_u16x = svunpklo_u16(bytes_u8x);
    svuint16_t sign_u16x = svand_n_u16_x(predicate_b16x, e5m2_u16x, 0x80);
    svuint16_t lower7_u16x = svand_n_u16_x(predicate_b16x, e5m2_u16x, 0x7F);
    svbool_t is_nan_b16x = svcmpgt_n_u16(predicate_b16x, lower7_u16x, 0x7C);
    lower7_u16x = svdup_n_u16_m(lower7_u16x, is_nan_b16x, 0x7E);
    return svreinterpret_f16_u16(svlsl_n_u16_x(predicate_b16x, svorr_u16_x(predicate_b16x, lower7_u16x, sign_u16x), 8));
}

/**
 *  @brief Inline e2m3 → signed i8 conversion returning @c svint8_t for direct use in GEMM.
 *
 *  Uses a 32-entry magnitude LUT via the SVE2 two-register TBL, which holds it at every vector
 *  length. The e2m3 encoding keeps the sign in bit 5 and a magnitude index in bits 4:0. The LUT
 *  maps the magnitude index → sixteen times its value, an exact i8, then the sign is applied.
 *
 *  @param[in] predicate_b8x Predicate for 8-bit elements.
 *  @param[in] raw_bytes_u8x Pre-loaded e2m3 bytes as @c svuint8_t.
 *  @return Signed i8 values as @c svint8_t.
 */
NUMKONG_INLINE svint8_t nk_e2m3x_to_i8x_ssve_(svbool_t predicate_b8x, svuint8_t raw_bytes_u8x) NUMKONG_STREAMING_ {
    static nk_align_(64) nk_u8_t const lut_data[32] = {
        0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,  28,  30,  //
        32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, //
    };
    svuint8_t magnitude_u8x = svand_n_u8_x(predicate_b8x, raw_bytes_u8x, 0x1F);
    svuint8_t unsigned_value_u8x = svtbl2_u8(nk_sme_table32_u8_(lut_data), magnitude_u8x);
    svuint8_t sign_bits_u8x = svand_n_u8_x(predicate_b8x, raw_bytes_u8x, 0x20);
    svbool_t negate_mask_b8x = svcmpne_n_u8(predicate_b8x, sign_bits_u8x, 0);
    svint8_t positive_value_i8x = svreinterpret_s8_u8(unsigned_value_u8x);
    svint8_t negated_value_i8x = svneg_s8_x(predicate_b8x, positive_value_i8x);
    return svsel_s8(negate_mask_b8x, negated_value_i8x, positive_value_i8x);
}

/** Widens up to `svcntb()` E2M1 dimensions into doubled signed i8 lanes, zeroing lanes past
 *  @p dimensions. Even dimensions live in high nibbles. */
NUMKONG_INLINE svint8_t nk_e2m1x_to_i8x_ssve_(nk_e2m1x2_t const *pairs, nk_size_t dimensions) NUMKONG_STREAMING_ {
    static nk_align_(64) nk_i8_t const lut_data[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    nk_size_t const vector_dimensions = svcntb();
    if (dimensions > vector_dimensions) dimensions = vector_dimensions;
    svbool_t const predicate_all_b8x = svptrue_b8();
    svbool_t const pairs_predicate_b8x = svwhilelt_b8_u64(0u, dimensions / NUMKONG_NIBBLES_PER_BYTE);
    svuint8_t pairs_u8x = svld1_u8(pairs_predicate_b8x, (nk_u8_t const *)pairs);
    svuint8_t high_u8x = svlsr_n_u8_x(predicate_all_b8x, pairs_u8x, 4);
    svuint8_t low_u8x = svand_n_u8_x(predicate_all_b8x, pairs_u8x, 0x0F);
    svuint8_t codes_u8x = svzip1_u8(high_u8x, low_u8x);
    svint8_t lut_i8x = svld1_s8(svwhilelt_b8_u64(0u, 16u), lut_data);
    svint8_t doubled_i8x = svtbl_s8(lut_i8x, codes_u8x);
    return svsel_s8(svwhilelt_b8_u64(0u, dimensions), doubled_i8x, svdup_n_s8(0));
}

/**
 *  @brief Inline e3m2 → f16 conversion returning @c svfloat16_t for direct use in GEMM.
 *
 *  Every e3m2 value is an f16 with a zero low byte, so one byte lookup of the 5-bit magnitude in a
 *  32-entry table, held at every vector length by the two-register TBL, yields the high byte; the
 *  sign moves from bit 5.
 *
 *  @param[in] predicate_b16x Predicate for 16-bit elements.
 *  @param[in] bytes_u8x Pre-loaded bytes, an @c svuint8_t from @c svld1_u8.
 *  @return F16 values as @c svfloat16_t, zero for inactive lanes.
 */
NUMKONG_INLINE svfloat16_t nk_e3m2x_to_f16x_ssve_(svbool_t predicate_b16x, svuint8_t bytes_u8x) NUMKONG_STREAMING_ {
    static nk_align_(64) nk_u8_t const magnitude_high_lut[32] = {
        0x00, 0x2C, 0x30, 0x32, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
        0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
    };
    svuint8_t const magnitude_high_u8x = svtbl2_u8(nk_sme_table32_u8_(magnitude_high_lut),
                                                   svand_n_u8_x(svptrue_b8(), bytes_u8x, 0x1F));
    svuint16_t const bits_u16x = svreinterpret_u16_u8(svzip1_u8(svdup_n_u8(0), magnitude_high_u8x));
    svuint16_t const signed_u16x = svbsl_n_u16(svlsl_n_u16_x(predicate_b16x, svunpklo_u16(bytes_u8x), 10), bits_u16x,
                                               0x8000);
    return svreinterpret_f16_u16(svsel_u16(predicate_b16x, signed_u16x, svdup_n_u16(0)));
}

/** BF16 bits of mini-float codes, one per 16-bit lane, with @p magnitude_mask magnitudes and the
 *  sign in bit @p sign_bit, decoded by @c nk_minifloat_magnitudes_ssve_ from @p shift_bits,
 *  @p base, @p rotation and @p deltas_data. Every lane moves by @p shift, a multiple of 128 that
 *  scales by a power of two, except zero, Inf and NaN, whose @p fixed_entries corrections lead the
 *  table and so always sit in the first vector. Zero codes decode to zero, so lanes past a
 *  predicated load stay zero. */
NUMKONG_INLINE svuint16_t nk_minifloat_to_bf16x_ssve_(svuint16_t codes_u16x, nk_u16_t magnitude_mask,
                                                      nk_size_t sign_bit, nk_size_t shift_bits, nk_u16_t base,
                                                      nk_u16_t rotation, nk_u16_t const *deltas_data,
                                                      nk_size_t fixed_entries, nk_u16_t shift) NUMKONG_STREAMING_ {
    svbool_t const all_b16x = svptrue_b16();
    svuint16x2_t const table_u16x = nk_sme_table16_u16_(deltas_data);
    svuint16x2_t const deltas_u16x = svset2_u16(
        table_u16x, 0, svsub_n_u16_m(svwhilelt_b16_u64(0, fixed_entries), svget2_u16(table_u16x, 0), shift));
    svuint16_t const bits_u16x = nk_minifloat_magnitudes_ssve_(svand_n_u16_x(all_b16x, codes_u16x, magnitude_mask),
                                                               shift_bits, (nk_u16_t)(base + shift), rotation,
                                                               deltas_u16x);
    return svbsl_n_u16(svlsl_n_u16_x(all_b16x, codes_u16x, 15 - sign_bit), bits_u16x, 0x8000);
}

/** BF16 bits of E4M3 bytes, every finite nonzero lane moved by @p shift. Exact: E4M3 values have
 *  3-bit mantissas within BF16's range. */
NUMKONG_INLINE svuint16_t nk_e4m3x_to_bf16x_ssve_(svuint16_t codes_u16x, nk_u16_t shift) NUMKONG_STREAMING_ {
    // NaN at magnitude 127, zero, then the seven subnormals
    static nk_align_(64)
        nk_u16_t const deltas_data[16] = {0x3BD0, 0xC400, 0xFEF0, 0xFF60, 0xFF90, 0xFFC0, 0xFFD0, 0xFFE0, 0xFFF0};
    return nk_minifloat_to_bf16x_ssve_(codes_u16x, 0x7F, 7, 4, 0x3C00, 1, deltas_data, 2, shift);
}

/** BF16 bits of E5M2 bytes, Inf and NaN kept, every finite nonzero lane moved by @p shift. */
NUMKONG_INLINE svuint16_t nk_e5m2x_to_bf16x_ssve_(svuint16_t codes_u16x, nk_u16_t shift) NUMKONG_STREAMING_ {
    // Inf and three NaNs at magnitudes 124 … 127, zero, then the three subnormals
    static nk_align_(64)
        nk_u16_t const deltas_data[16] = {0x3800, 0x3820, 0x3800, 0x37E0, 0xC800, 0xFF60, 0xFFC0, 0xFFE0};
    return nk_minifloat_to_bf16x_ssve_(codes_u16x, 0x7F, 7, 5, 0x3800, 4, deltas_data, 5, shift);
}

/** BF16 bits of E2M3 bytes, sign in bit 5, every nonzero lane moved by @p shift. */
NUMKONG_INLINE svuint16_t nk_e2m3x_to_bf16x_ssve_(svuint16_t codes_u16x, nk_u16_t shift) NUMKONG_STREAMING_ {
    static nk_align_(64)
        nk_u16_t const deltas_data[16] = {0xC100, 0xFEF0, 0xFF60, 0xFF90, 0xFFC0, 0xFFD0, 0xFFE0, 0xFFF0};
    return nk_minifloat_to_bf16x_ssve_(codes_u16x, 0x1F, 5, 4, 0x3F00, 0, deltas_data, 1, shift);
}

/** BF16 bits of E3M2 bytes, sign in bit 5, every nonzero lane moved by @p shift. */
NUMKONG_INLINE svuint16_t nk_e3m2x_to_bf16x_ssve_(svuint16_t codes_u16x, nk_u16_t shift) NUMKONG_STREAMING_ {
    static nk_align_(64) nk_u16_t const deltas_data[16] = {0xC200, 0xFF60, 0xFFC0, 0xFFE0};
    return nk_minifloat_to_bf16x_ssve_(codes_u16x, 0x1F, 5, 5, 0x3E00, 0, deltas_data, 1, shift);
}

/** BF16 bits of @p count MX elements of @p element_dtype from dim @p first of one row of @p codes,
 *  at most one 32-dim block, times 2^(e_block − @p base) for the UE8M0 block scale @p scale. Scale
 *  code 0 yields zeros and 255 NaNs; a @p base of @c nk_sme_unfolded_base_k keeps the elements
 *  unscaled. E2M1 folds fastest with @p base at the largest block exponent, a saturating decrement
 *  that leaves the +0 entries alone. */
NUMKONG_INLINE svuint16_t nk_mx_to_bf16x_ssve_(nk_dtype_t element_dtype, nk_u8_t const *codes, nk_size_t first,
                                               nk_size_t count, nk_u8_t scale, nk_i32_t base) NUMKONG_STREAMING_ {
    // Every entry but the two zeros moves under an increasing fold
    static nk_align_(64) nk_u16_t const e2m1_scaled_data[16] = {0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1};
    svbool_t const all_b8x = svptrue_b8();
    if (scale == 0) return svdup_n_u16(0);
    if (scale == 255) return svsel_u16(svwhilelt_b16_u64(0, count), svdup_n_u16(0x7FC0), svdup_n_u16(0));
    // The fold moves table entries or decoder constants, off the path of the loaded codes
    nk_i32_t const exponent_shift = base == nk_sme_unfolded_base_k ? 0 : (nk_i32_t)scale - 127 - base;
    nk_u16_t const shift = (nk_u16_t)(exponent_shift * 128);
    if (element_dtype == nk_e2m1_k) {
        svuint8_t const pairs_u8x = svld1_u8(svwhilelt_b8_u64(0, count / 2), codes + first / 2);
        svuint8_t const nibbles_u8x = svzip1_u8(svlsr_n_u8_x(all_b8x, pairs_u8x, 4),
                                                svand_n_u8_x(all_b8x, pairs_u8x, 0x0F));
        svuint16x2_t const table_u16x = nk_sme_table16_u16_(nk_sme_mxfp4_table_data_);
        if (exponent_shift <= 0)
            return svqsub_n_u16(svtbl2_u16(table_u16x, svunpklo_u16(nibbles_u8x)), (nk_u16_t)(-exponent_shift * 128));
        return svtbl2_u16(nk_sme_table16_shift_(table_u16x, e2m1_scaled_data, (nk_u16_t)(0 - shift)),
                          svunpklo_u16(nibbles_u8x));
    }
    svuint16_t const codes_u16x = svld1ub_u16(svwhilelt_b16_u64(0, count), codes + first);
    if (element_dtype == nk_e2m3_k) return nk_e2m3x_to_bf16x_ssve_(codes_u16x, shift);
    if (element_dtype == nk_e3m2_k) return nk_e3m2x_to_bf16x_ssve_(codes_u16x, shift);
    if (element_dtype == nk_e4m3_k) return nk_e4m3x_to_bf16x_ssve_(codes_u16x, shift);
    return nk_e5m2x_to_bf16x_ssve_(codes_u16x, shift);
}

/** F16 products of @p count NVFP4 elements from dim @p first of one row, at most 32, and their
 *  UE4M3 block scales, exact; zero for lanes past @p count. */
NUMKONG_INLINE svuint16_t nk_nvfp4x_to_f16x_ssve_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t first,
                                                  nk_size_t count) NUMKONG_STREAMING_ {
    // F16 bits of every UE4M3 code, a NaN for 127, read by broadcast loads
    static nk_align_(64) nk_u16_t const scale_data[128] = {
        0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300, 0x2400, 0x2480, 0x2500, 0x2580, 0x2600,
        0x2680, 0x2700, 0x2780, 0x2800, 0x2880, 0x2900, 0x2980, 0x2A00, 0x2A80, 0x2B00, 0x2B80, 0x2C00, 0x2C80,
        0x2D00, 0x2D80, 0x2E00, 0x2E80, 0x2F00, 0x2F80, 0x3000, 0x3080, 0x3100, 0x3180, 0x3200, 0x3280, 0x3300,
        0x3380, 0x3400, 0x3480, 0x3500, 0x3580, 0x3600, 0x3680, 0x3700, 0x3780, 0x3800, 0x3880, 0x3900, 0x3980,
        0x3A00, 0x3A80, 0x3B00, 0x3B80, 0x3C00, 0x3C80, 0x3D00, 0x3D80, 0x3E00, 0x3E80, 0x3F00, 0x3F80, 0x4000,
        0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380, 0x4400, 0x4480, 0x4500, 0x4580, 0x4600, 0x4680,
        0x4700, 0x4780, 0x4800, 0x4880, 0x4900, 0x4980, 0x4A00, 0x4A80, 0x4B00, 0x4B80, 0x4C00, 0x4C80, 0x4D00,
        0x4D80, 0x4E00, 0x4E80, 0x4F00, 0x4F80, 0x5000, 0x5080, 0x5100, 0x5180, 0x5200, 0x5280, 0x5300, 0x5380,
        0x5400, 0x5480, 0x5500, 0x5580, 0x5600, 0x5680, 0x5700, 0x5780, 0x5800, 0x5880, 0x5900, 0x5980, 0x5A00,
        0x5A80, 0x5B00, 0x5B80, 0x5C00, 0x5C80, 0x5D00, 0x5D80, 0x5E00, 0x5E80, 0x5F00, 0x7E00,
    };
    svbool_t const predicate_b8x = svptrue_b8(), predicate_b16x = svptrue_b16();
    svuint8_t const pairs_u8x = svld1_u8(svwhilelt_b8_u64(0, count / 2), codes + first / 2);
    svuint8_t const nibbles_u8x = svzip1_u8(svlsr_n_u8_x(predicate_b8x, pairs_u8x, 4),
                                            svand_n_u8_x(predicate_b8x, pairs_u8x, 0x0F));
    svuint16_t const values_u16x = svtbl2_u16(nk_sme_table16_u16_(nk_sme_nvfp4_table_data_), svunpklo_u16(nibbles_u8x));
    nk_u8_t const *block_scales = scales + first / 16;
    svuint16_t const scales_u16x = svsel_u16(svwhilelt_b16_u64(0, 16), svdup_n_u16(scale_data[block_scales[0] & 0x7F]),
                                             svdup_n_u16(count > 16 ? scale_data[block_scales[1] & 0x7F] : 0));
    return svreinterpret_u16_f16(
        svmul_f16_x(predicate_b16x, svreinterpret_f16_u16(values_u16x), svreinterpret_f16_u16(scales_u16x)));
}

/** Smallest and largest UE8M0 exponent among the finite nonzero codes of @p blocks scales, both
 *  zero when there are none. */
NUMKONG_INLINE void nk_sme_exponent_range_(nk_u8_t const *scales, nk_size_t blocks, nk_i32_t *minimum,
                                           nk_i32_t *maximum) NUMKONG_STREAMABLE_ {
    nk_i32_t low = 255, high = 0;
    for (nk_size_t block = 0; block < blocks; ++block) {
        nk_i32_t const code = scales[block];
        if (code == 0 || code == 255) continue;
        if (code < low) low = code;
        if (code > high) high = code;
    }
    if (low > high) low = high = 127;
    *minimum = low - 127, *maximum = high - 127;
}

/** Up to 32 dims of block-scaled row @p row of @p operand from dim @p first, a multiple of 32: one
 *  MX block or two NVFP4 blocks, as the 16-bit MOPA operand. Zero past @p count. */
NUMKONG_INLINE svuint16_t nk_sme_decode_blocks_b16_(nk_dtype_t dtype, nk_cross_operand_t operand,
                                                    nk_u8_t const *row_bytes, nk_size_t row, nk_size_t first,
                                                    nk_size_t count, nk_i32_t base) NUMKONG_STREAMING_ {
    nk_u8_t const scale = operand.scales[row * operand.scales_stride + first / 32];
    switch (dtype) {
    case nk_nvfp4_k:
        return nk_nvfp4x_to_f16x_ssve_(row_bytes, operand.scales + row * operand.scales_stride, first, count);
    case nk_mxfp4_k: return nk_mx_to_bf16x_ssve_(nk_e2m1_k, row_bytes, first, count, scale, base);
    case nk_mxfp6e2m3_k: return nk_mx_to_bf16x_ssve_(nk_e2m3_k, row_bytes, first, count, scale, base);
    case nk_mxfp6e3m2_k: return nk_mx_to_bf16x_ssve_(nk_e3m2_k, row_bytes, first, count, scale, base);
    case nk_mxfp8e4m3_k: return nk_mx_to_bf16x_ssve_(nk_e4m3_k, row_bytes, first, count, scale, base);
    default: return nk_mx_to_bf16x_ssve_(nk_e5m2_k, row_bytes, first, count, scale, base);
    }
}

/** Up to one vector of dims of row @p row of @p operand of @p dtype from dim @p first, as the
 *  16-bit MOPA operand, which is f16 for f16, e4m3, e5m2, e3m2 and NVFP4, bf16 for bf16 and the MX
 *  formats, and whichever @p panel_dtype asks for e4m3. Zero past @p count. */
NUMKONG_INLINE svuint16_t nk_sme_decode_row_b16_(nk_dtype_t dtype, nk_dtype_t panel_dtype, nk_cross_operand_t operand,
                                                 nk_size_t row_stride, nk_size_t row, nk_size_t first, nk_size_t count,
                                                 nk_i32_t base) NUMKONG_STREAMING_ {
    nk_u8_t const *row_bytes = (nk_u8_t const *)operand.elements + row * row_stride;
    svbool_t const predicate_b16x = svwhilelt_b16_u64(0, count);
    switch (dtype) {
    case nk_f16_k:
    case nk_bf16_k: return svld1_u16(predicate_b16x, (nk_u16_t const *)row_bytes + first);
    case nk_e4m3_k:
        if (panel_dtype == nk_bf16_k) return nk_e4m3x_to_bf16x_ssve_(svld1ub_u16(predicate_b16x, row_bytes + first), 0);
        return svreinterpret_u16_f16(
            nk_e4m3x_to_f16x_ssve_(predicate_b16x, svld1_u8(svwhilelt_b8_u64(0, count), row_bytes + first)));
    case nk_e5m2_k:
        return svreinterpret_u16_f16(
            nk_e5m2x_to_f16x_ssve_(predicate_b16x, svld1_u8(svwhilelt_b8_u64(0, count), row_bytes + first)));
    case nk_e3m2_k:
        return svreinterpret_u16_f16(
            nk_e3m2x_to_f16x_ssve_(predicate_b16x, svld1_u8(svwhilelt_b8_u64(0, count), row_bytes + first)));
    default: {
        // Vectors past SVL 512 span several scale blocks, decoded 32 dims at a time and spliced
        svuint16_t row_u16x = nk_sme_decode_blocks_b16_(dtype, operand, row_bytes, row, first, count < 32 ? count : 32,
                                                        base);
        for (nk_size_t piece = 32; piece < count; piece += 32)
            row_u16x = svsplice_u16(svwhilelt_b16_u64(0, piece), row_u16x,
                                    nk_sme_decode_blocks_b16_(dtype, operand, row_bytes, row, first + piece,
                                                              count - piece < 32 ? count - piece : 32, base));
        return row_u16x;
    }
    }
}

/** Up to 64 bytes of row @p row of @p dtype from dim @p first as the 8-bit MOPA operand: raw i8 and
 *  u8, sixteen times e2m3, twice e2m1, and raw nibble pairs for i4 and u4. Zero past @p count. */
NUMKONG_INLINE svuint8_t nk_sme_decode_row_b8_(nk_dtype_t dtype, void const *rows_data, nk_size_t row_stride,
                                               nk_size_t row, nk_size_t first, nk_size_t count) NUMKONG_STREAMING_ {
    nk_u8_t const *row_bytes = (nk_u8_t const *)rows_data + row * row_stride;
    switch (dtype) {
    case nk_e2m3_k: {
        svbool_t const predicate_b8x = svwhilelt_b8_u64(0, count);
        return svreinterpret_u8_s8(nk_e2m3x_to_i8x_ssve_(predicate_b8x, svld1_u8(predicate_b8x, row_bytes + first)));
    }
    case nk_e2m1_k:
        return svreinterpret_u8_s8(nk_e2m1x_to_i8x_ssve_((nk_e2m1x2_t const *)(row_bytes + first / 2), count));
    case nk_u4_k:
    case nk_i4_k: return svld1_u8(svwhilelt_b8_u64(0, nk_size_divide_round_up_(count, 2)), row_bytes + first / 2);
    default: return svld1_u8(svwhilelt_b8_u64(0, count), row_bytes + first);
    }
}

#pragma endregion Decoders

#pragma region Panels

/** The body of @c nk_sme_stage_panel_b16_, inlined once per constant @p dtype so that the decoder
 *  switch and its table loads leave the row loop. */
NUMKONG_INLINE void nk_sme_stage_rows_b16_(nk_dtype_t dtype, nk_dtype_t panel_dtype, nk_cross_operand_t operand,
                                           nk_size_t row_stride, nk_size_t row_first, nk_size_t rows,
                                           nk_size_t depth_first, nk_size_t depth, nk_i32_t const *base_exponents,
                                           nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    nk_size_t const steps = nk_size_divide_round_up_(depth, 2);
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t batch = 0; batch < steps; batch += tile_dimension) {
        nk_size_t const remaining = depth - batch * 2;
        nk_size_t const count = remaining < vector_elements ? remaining : vector_elements;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            svuint16_t const row_u16x = nk_sme_decode_row_b16_(dtype, panel_dtype, operand, row_stride, row_first + row,
                                                               depth_first + batch * 2, count,
                                                               base_exponents ? base_exponents[row] : 0);
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_u16(row_u16x));
        }
        nk_size_t const batch_steps = steps - batch < tile_dimension ? steps - batch : tile_dimension;
        for (nk_size_t step = 0; step < batch_steps; ++step)
            svst1_ver_za32(0, (uint32_t)step, predicate_all_b32x, panel + (batch + step) * vector_elements);
    }
}

/** Decodes @p rows rows of @p operand from @p row_first over @p depth dims from @p depth_first into
 *  @p panel in MOPA order, word i of step s holding row i's dims 2s and 2s + 1, through the ZA0
 *  transpose. MX rows rebase by @p base_exponents, one per row; rows past @p rows are zero. */
NUMKONG_OUTLINED_ void nk_sme_stage_panel_b16_(nk_dtype_t dtype, nk_dtype_t panel_dtype, nk_cross_operand_t operand,
                                               nk_size_t row_stride, nk_size_t row_first, nk_size_t rows,
                                               nk_size_t depth_first, nk_size_t depth, nk_i32_t const *base_exponents,
                                               nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    switch (dtype) {
    case nk_f16_k:
    case nk_bf16_k:
        nk_sme_stage_rows_b16_(nk_f16_k, nk_f16_k, operand, row_stride, row_first, rows, depth_first, depth,
                               base_exponents, panel);
        break;
    case nk_e4m3_k:
        if (panel_dtype == nk_bf16_k)
            nk_sme_stage_rows_b16_(nk_e4m3_k, nk_bf16_k, operand, row_stride, row_first, rows, depth_first, depth,
                                   base_exponents, panel);
        else
            nk_sme_stage_rows_b16_(nk_e4m3_k, nk_f16_k, operand, row_stride, row_first, rows, depth_first, depth,
                                   base_exponents, panel);
        break;
    case nk_e5m2_k:
        nk_sme_stage_rows_b16_(nk_e5m2_k, nk_f16_k, operand, row_stride, row_first, rows, depth_first, depth,
                               base_exponents, panel);
        break;
    case nk_e3m2_k:
        nk_sme_stage_rows_b16_(nk_e3m2_k, nk_f16_k, operand, row_stride, row_first, rows, depth_first, depth,
                               base_exponents, panel);
        break;
    case nk_nvfp4_k:
        nk_sme_stage_rows_b16_(nk_nvfp4_k, nk_f16_k, operand, row_stride, row_first, rows, depth_first, depth,
                               base_exponents, panel);
        break;
    case nk_mxfp4_k:
        nk_sme_stage_rows_b16_(nk_mxfp4_k, nk_bf16_k, operand, row_stride, row_first, rows, depth_first, depth,
                               base_exponents, panel);
        break;
    case nk_mxfp6e2m3_k:
        nk_sme_stage_rows_b16_(nk_mxfp6e2m3_k, nk_bf16_k, operand, row_stride, row_first, rows, depth_first, depth,
                               base_exponents, panel);
        break;
    case nk_mxfp6e3m2_k:
        nk_sme_stage_rows_b16_(nk_mxfp6e3m2_k, nk_bf16_k, operand, row_stride, row_first, rows, depth_first, depth,
                               base_exponents, panel);
        break;
    case nk_mxfp8e4m3_k:
        nk_sme_stage_rows_b16_(nk_mxfp8e4m3_k, nk_bf16_k, operand, row_stride, row_first, rows, depth_first, depth,
                               base_exponents, panel);
        break;
    default:
        nk_sme_stage_rows_b16_(nk_mxfp8e5m2_k, nk_bf16_k, operand, row_stride, row_first, rows, depth_first, depth,
                               base_exponents, panel);
        break;
    }
}

/** Steps of 4 dims an 8-bit panel spans over @p depth dims: i4 and u4 split each 8-dim word into a
 *  low-nibble step and a high-nibble step, as their packs do. */
NUMKONG_INLINE nk_size_t nk_sme_panel_steps_b8_(nk_dtype_t dtype, nk_size_t depth) NUMKONG_STREAMABLE_ {
    if (dtype == nk_u4_k || dtype == nk_i4_k) return nk_size_divide_round_up_(depth, 8) * 2;
    return nk_size_divide_round_up_(depth, 4);
}

/** Decodes @p rows rows of @p dtype over @p depth dims from @p depth_first into the 8-bit
 *  @p panel, word i of step s holding row i's four bytes of that step; i4 and u4 steps alternate
 *  low and high nibbles, sign-extended for i4. */
NUMKONG_INLINE void nk_sme_stage_panel_b8_(nk_dtype_t dtype, void const *rows_data, nk_size_t row_stride,
                                           nk_size_t row_first, nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                           nk_u8_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    int const nibbles = dtype == nk_u4_k || dtype == nk_i4_k;
    nk_size_t const dims_per_word = nibbles ? 8 : 4;
    nk_size_t const words = nk_size_divide_round_up_(depth, dims_per_word);
    svbool_t const predicate_all_b8x = svptrue_b8(), predicate_all_b32x = svptrue_b32();
    for (nk_size_t batch = 0; batch < words; batch += tile_dimension) {
        nk_size_t const remaining = depth - batch * dims_per_word;
        nk_size_t const batch_dims = tile_dimension * dims_per_word;
        nk_size_t const count = remaining < batch_dims ? remaining : batch_dims;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row)
            svwrite_hor_za32_u32_m(
                0, (uint32_t)row, predicate_all_b32x,
                svreinterpret_u32_u8(nk_sme_decode_row_b8_(dtype, rows_data, row_stride, row_first + row,
                                                           depth_first + batch * dims_per_word, count)));
        nk_size_t const batch_words = words - batch < tile_dimension ? words - batch : tile_dimension;
        for (nk_size_t word = 0; word < batch_words; ++word) {
            if (!nibbles) {
                svst1_ver_za32(0, (uint32_t)word, predicate_all_b32x, panel + (batch + word) * vector_bytes);
                continue;
            }
            svuint8_t const packed_u8x = svreinterpret_u8_u32(
                svread_ver_za32_u32_m(svdup_n_u32(0), predicate_all_b32x, 0, (uint32_t)word));
            svuint8_t low_u8x, high_u8x;
            if (dtype == nk_i4_k) {
                svint8_t const packed_i8x = svreinterpret_s8_u8(packed_u8x);
                low_u8x = svreinterpret_u8_s8(
                    svasr_n_s8_x(predicate_all_b8x, svlsl_n_s8_x(predicate_all_b8x, packed_i8x, 4), 4));
                high_u8x = svreinterpret_u8_s8(svasr_n_s8_x(predicate_all_b8x, packed_i8x, 4));
            }
            else {
                low_u8x = svand_n_u8_x(predicate_all_b8x, packed_u8x, 0x0F);
                high_u8x = svlsr_n_u8_x(predicate_all_b8x, packed_u8x, 4);
            }
            svst1_u8(predicate_all_b8x, panel + (batch + word) * 2 * vector_bytes, low_u8x);
            svst1_u8(predicate_all_b8x, panel + ((batch + word) * 2 + 1) * vector_bytes, high_u8x);
        }
    }
}

/** Rebasing exponents of @p rows MX rows from @p row_first: each row's smallest finite block
 *  exponent, or its largest when @p largest is set. Returns whether some row spans more than
 *  @c nk_sme_exponent_spread_k binades. */
NUMKONG_INLINE int nk_sme_row_exponents_(nk_cross_operand_t operand, nk_size_t row_first, nk_size_t rows,
                                         nk_size_t blocks, int largest, nk_i32_t *exponents) NUMKONG_STREAMABLE_ {
    int wide = 0;
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t minimum, maximum;
        nk_sme_exponent_range_(operand.scales + (row_first + row) * operand.scales_stride, blocks, &minimum, &maximum);
        exponents[row] = largest ? maximum : minimum;
        wide |= maximum - minimum > nk_sme_exponent_spread_k;
    }
    return wide;
}

#pragma endregion Panels

#pragma region Outer Products

/** Accumulates two f16 panels against two column tiles over @p steps: ZA0 = first × first,
 *  ZA1 = first × second, ZA2 = second × first, ZA3 = second × second. */
NUMKONG_INLINE void nk_sme_mopa_2x2_f16_(nk_u16_t const *a_first, nk_u16_t const *a_second, nk_u16_t const *b_first,
                                         nk_u16_t const *b_second, nk_size_t steps) NUMKONG_STREAMING_
    __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    for (nk_size_t step = 0; step < steps; ++step) {
        nk_size_t const offset = step * vector_elements;
        svfloat16_t const a_first_f16x = svld1_f16(predicate_all_b16x, (float16_t const *)a_first + offset);
        svfloat16_t const a_second_f16x = svld1_f16(predicate_all_b16x, (float16_t const *)a_second + offset);
        svfloat16_t const b_first_f16x = svld1_f16(predicate_all_b16x, (float16_t const *)b_first + offset);
        svfloat16_t const b_second_f16x = svld1_f16(predicate_all_b16x, (float16_t const *)b_second + offset);
        svmopa_za32_f16_m(0, predicate_all_b16x, predicate_all_b16x, a_first_f16x, b_first_f16x);
        svmopa_za32_f16_m(1, predicate_all_b16x, predicate_all_b16x, a_first_f16x, b_second_f16x);
        svmopa_za32_f16_m(2, predicate_all_b16x, predicate_all_b16x, a_second_f16x, b_first_f16x);
        svmopa_za32_f16_m(3, predicate_all_b16x, predicate_all_b16x, a_second_f16x, b_second_f16x);
    }
}

/** Accumulates one f16 panel against four column tiles over @p steps into ZA0 to ZA3. */
NUMKONG_INLINE void nk_sme_mopa_1x4_f16_(nk_u16_t const *a, nk_u16_t const *b_first, nk_u16_t const *b_second,
                                         nk_u16_t const *b_third, nk_u16_t const *b_fourth,
                                         nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    for (nk_size_t step = 0; step < steps; ++step) {
        nk_size_t const offset = step * vector_elements;
        svfloat16_t const a_f16x = svld1_f16(predicate_all_b16x, (float16_t const *)a + offset);
        svmopa_za32_f16_m(0, predicate_all_b16x, predicate_all_b16x, a_f16x,
                          svld1_f16(predicate_all_b16x, (float16_t const *)b_first + offset));
        svmopa_za32_f16_m(1, predicate_all_b16x, predicate_all_b16x, a_f16x,
                          svld1_f16(predicate_all_b16x, (float16_t const *)b_second + offset));
        svmopa_za32_f16_m(2, predicate_all_b16x, predicate_all_b16x, a_f16x,
                          svld1_f16(predicate_all_b16x, (float16_t const *)b_third + offset));
        svmopa_za32_f16_m(3, predicate_all_b16x, predicate_all_b16x, a_f16x,
                          svld1_f16(predicate_all_b16x, (float16_t const *)b_fourth + offset));
    }
}

/** Accumulates one bf16 panel against four column tiles over @p steps into ZA0 to ZA3. */
NUMKONG_INLINE void nk_sme_mopa_1x4_bf16_(nk_u16_t const *a, nk_u16_t const *b_first, nk_u16_t const *b_second,
                                          nk_u16_t const *b_third, nk_u16_t const *b_fourth,
                                          nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    for (nk_size_t step = 0; step < steps; ++step) {
        nk_size_t const offset = step * vector_elements;
        svbfloat16_t const a_bf16x = svld1_bf16(predicate_all_b16x, (bfloat16_t const *)a + offset);
        svmopa_za32_bf16_m(0, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                           svld1_bf16(predicate_all_b16x, (bfloat16_t const *)b_first + offset));
        svmopa_za32_bf16_m(1, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                           svld1_bf16(predicate_all_b16x, (bfloat16_t const *)b_second + offset));
        svmopa_za32_bf16_m(2, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                           svld1_bf16(predicate_all_b16x, (bfloat16_t const *)b_third + offset));
        svmopa_za32_bf16_m(3, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                           svld1_bf16(predicate_all_b16x, (bfloat16_t const *)b_fourth + offset));
    }
}

/** Accumulates two 8-bit panels against two column tiles over @p steps, signed or
 *  @p unsigned_inputs, in the 2×2 tile order of @c nk_sme_mopa_2x2_f16_. */
NUMKONG_INLINE void nk_sme_mopa_2x2_b8_(int unsigned_inputs, nk_u8_t const *a_first, nk_u8_t const *a_second,
                                        nk_u8_t const *b_first, nk_u8_t const *b_second,
                                        nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b8x = svptrue_b8();
    nk_size_t const vector_bytes = svcntb();
    if (unsigned_inputs)
        for (nk_size_t step = 0; step < steps; ++step) {
            nk_size_t const offset = step * vector_bytes;
            svuint8_t const a_first_u8x = svld1_u8(predicate_all_b8x, a_first + offset);
            svuint8_t const a_second_u8x = svld1_u8(predicate_all_b8x, a_second + offset);
            svuint8_t const b_first_u8x = svld1_u8(predicate_all_b8x, b_first + offset);
            svuint8_t const b_second_u8x = svld1_u8(predicate_all_b8x, b_second + offset);
            svmopa_za32_u8_m(0, predicate_all_b8x, predicate_all_b8x, a_first_u8x, b_first_u8x);
            svmopa_za32_u8_m(1, predicate_all_b8x, predicate_all_b8x, a_first_u8x, b_second_u8x);
            svmopa_za32_u8_m(2, predicate_all_b8x, predicate_all_b8x, a_second_u8x, b_first_u8x);
            svmopa_za32_u8_m(3, predicate_all_b8x, predicate_all_b8x, a_second_u8x, b_second_u8x);
        }
    else
        for (nk_size_t step = 0; step < steps; ++step) {
            nk_size_t const offset = step * vector_bytes;
            svint8_t const a_first_i8x = svld1_s8(predicate_all_b8x, (nk_i8_t const *)a_first + offset);
            svint8_t const a_second_i8x = svld1_s8(predicate_all_b8x, (nk_i8_t const *)a_second + offset);
            svint8_t const b_first_i8x = svld1_s8(predicate_all_b8x, (nk_i8_t const *)b_first + offset);
            svint8_t const b_second_i8x = svld1_s8(predicate_all_b8x, (nk_i8_t const *)b_second + offset);
            svmopa_za32_s8_m(0, predicate_all_b8x, predicate_all_b8x, a_first_i8x, b_first_i8x);
            svmopa_za32_s8_m(1, predicate_all_b8x, predicate_all_b8x, a_first_i8x, b_second_i8x);
            svmopa_za32_s8_m(2, predicate_all_b8x, predicate_all_b8x, a_second_i8x, b_first_i8x);
            svmopa_za32_s8_m(3, predicate_all_b8x, predicate_all_b8x, a_second_i8x, b_second_i8x);
        }
}

/** Accumulates one 8-bit panel against four column tiles over @p steps, signed unless
 *  @p unsigned_inputs is set. */
NUMKONG_INLINE void nk_sme_mopa_1x4_b8_(int unsigned_inputs, nk_u8_t const *a, nk_u8_t const *b_first,
                                        nk_u8_t const *b_second, nk_u8_t const *b_third, nk_u8_t const *b_fourth,
                                        nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b8x = svptrue_b8();
    nk_size_t const vector_bytes = svcntb();
    if (unsigned_inputs)
        for (nk_size_t step = 0; step < steps; ++step) {
            nk_size_t const offset = step * vector_bytes;
            svuint8_t const a_u8x = svld1_u8(predicate_all_b8x, a + offset);
            svmopa_za32_u8_m(0, predicate_all_b8x, predicate_all_b8x, a_u8x,
                             svld1_u8(predicate_all_b8x, b_first + offset));
            svmopa_za32_u8_m(1, predicate_all_b8x, predicate_all_b8x, a_u8x,
                             svld1_u8(predicate_all_b8x, b_second + offset));
            svmopa_za32_u8_m(2, predicate_all_b8x, predicate_all_b8x, a_u8x,
                             svld1_u8(predicate_all_b8x, b_third + offset));
            svmopa_za32_u8_m(3, predicate_all_b8x, predicate_all_b8x, a_u8x,
                             svld1_u8(predicate_all_b8x, b_fourth + offset));
        }
    else
        for (nk_size_t step = 0; step < steps; ++step) {
            nk_size_t const offset = step * vector_bytes;
            svint8_t const a_i8x = svld1_s8(predicate_all_b8x, (nk_i8_t const *)a + offset);
            svmopa_za32_s8_m(0, predicate_all_b8x, predicate_all_b8x, a_i8x,
                             svld1_s8(predicate_all_b8x, (nk_i8_t const *)b_first + offset));
            svmopa_za32_s8_m(1, predicate_all_b8x, predicate_all_b8x, a_i8x,
                             svld1_s8(predicate_all_b8x, (nk_i8_t const *)b_second + offset));
            svmopa_za32_s8_m(2, predicate_all_b8x, predicate_all_b8x, a_i8x,
                             svld1_s8(predicate_all_b8x, (nk_i8_t const *)b_third + offset));
            svmopa_za32_s8_m(3, predicate_all_b8x, predicate_all_b8x, a_i8x,
                             svld1_s8(predicate_all_b8x, (nk_i8_t const *)b_fourth + offset));
        }
}

/** One NVFP4 B vector: byte codes through @p table_u16x, times the per-column block scales. */
NUMKONG_INLINE svfloat16_t nk_sme_nvfp4_b_vector_(svuint16x2_t table_u16x, nk_u8_t const *codes,
                                                  svfloat16_t scales_f16x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b16x = svptrue_b16();
    return svmul_f16_x(predicate_all_b16x,
                       svreinterpret_f16_u16(svtbl2_u16(table_u16x, svld1ub_u16(predicate_all_b16x, codes))),
                       scales_f16x);
}

/** One MXFP4 B vector: the byte codes mapped through @p table_u16x to BF16 bits, less the
 *  per-column block decrements. */
NUMKONG_INLINE svbfloat16_t nk_sme_mxfp4_b_vector_(svuint16x2_t table_u16x, nk_u8_t const *codes,
                                                   svuint16_t decrements_u16x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b16x = svptrue_b16();
    return svreinterpret_bf16_u16(
        svqsub_u16(svtbl2_u16(table_u16x, svld1ub_u16(predicate_all_b16x, codes)), decrements_u16x));
}

/** Accumulates two f16 NVFP4 A panels against two NVFP4 column tiles of byte codes and per-block
 *  f16 scale vectors, eight steps per block, in the 2×2 tile order of @c nk_sme_mopa_2x2_f16_. */
NUMKONG_INLINE void nk_sme_mopa_2x2_nvfp4_(nk_u16_t const *a_first, nk_u16_t const *a_second,
                                           nk_u8_t const *b_first_codes, nk_u8_t const *b_second_codes,
                                           nk_u16_t const *b_first_scales, nk_u16_t const *b_second_scales,
                                           nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    svuint16x2_t const table_u16x = nk_sme_table16_u16_(nk_sme_nvfp4_table_data_);
    for (nk_size_t block_first = 0; block_first < steps; block_first += 8) {
        nk_size_t const block_offset = block_first / 8 * vector_elements;
        svfloat16_t const first_scales_f16x = svld1_f16(predicate_all_b16x,
                                                        (float16_t const *)b_first_scales + block_offset);
        svfloat16_t const second_scales_f16x = svld1_f16(predicate_all_b16x,
                                                         (float16_t const *)b_second_scales + block_offset);
        nk_size_t const block_end = steps - block_first < 8 ? steps : block_first + 8;
        for (nk_size_t step = block_first; step < block_end; ++step) {
            nk_size_t const offset = step * vector_elements;
            svfloat16_t const a_first_f16x = svld1_f16(predicate_all_b16x, (float16_t const *)a_first + offset);
            svfloat16_t const a_second_f16x = svld1_f16(predicate_all_b16x, (float16_t const *)a_second + offset);
            svfloat16_t const b_first_f16x = nk_sme_nvfp4_b_vector_(table_u16x, b_first_codes + offset,
                                                                    first_scales_f16x);
            svfloat16_t const b_second_f16x = nk_sme_nvfp4_b_vector_(table_u16x, b_second_codes + offset,
                                                                     second_scales_f16x);
            svmopa_za32_f16_m(0, predicate_all_b16x, predicate_all_b16x, a_first_f16x, b_first_f16x);
            svmopa_za32_f16_m(1, predicate_all_b16x, predicate_all_b16x, a_first_f16x, b_second_f16x);
            svmopa_za32_f16_m(2, predicate_all_b16x, predicate_all_b16x, a_second_f16x, b_first_f16x);
            svmopa_za32_f16_m(3, predicate_all_b16x, predicate_all_b16x, a_second_f16x, b_second_f16x);
        }
    }
}

/** Accumulates two rebased bf16 MXFP4 A panels against two MXFP4 column tiles of byte codes and
 *  per-block decrement vectors, sixteen steps per block, in the 2×2 tile order of
 *  @c nk_sme_mopa_2x2_f16_. */
NUMKONG_INLINE void nk_sme_mopa_2x2_mxfp4_(nk_u16_t const *a_first, nk_u16_t const *a_second,
                                           nk_u8_t const *b_first_codes, nk_u8_t const *b_second_codes,
                                           nk_u16_t const *b_first_decrements, nk_u16_t const *b_second_decrements,
                                           nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    svuint16x2_t const table_u16x = nk_sme_table16_u16_(nk_sme_mxfp4_table_data_);
    for (nk_size_t block_first = 0; block_first < steps; block_first += 16) {
        nk_size_t const block_offset = block_first / 16 * vector_elements;
        svuint16_t const first_decrements_u16x = svld1_u16(predicate_all_b16x, b_first_decrements + block_offset);
        svuint16_t const second_decrements_u16x = svld1_u16(predicate_all_b16x, b_second_decrements + block_offset);
        nk_size_t const block_end = steps - block_first < 16 ? steps : block_first + 16;
        for (nk_size_t step = block_first; step < block_end; ++step) {
            nk_size_t const offset = step * vector_elements;
            svbfloat16_t const a_first_bf16x = svld1_bf16(predicate_all_b16x, (bfloat16_t const *)a_first + offset);
            svbfloat16_t const a_second_bf16x = svld1_bf16(predicate_all_b16x, (bfloat16_t const *)a_second + offset);
            svbfloat16_t const b_first_bf16x = nk_sme_mxfp4_b_vector_(table_u16x, b_first_codes + offset,
                                                                      first_decrements_u16x);
            svbfloat16_t const b_second_bf16x = nk_sme_mxfp4_b_vector_(table_u16x, b_second_codes + offset,
                                                                       second_decrements_u16x);
            svmopa_za32_bf16_m(0, predicate_all_b16x, predicate_all_b16x, a_first_bf16x, b_first_bf16x);
            svmopa_za32_bf16_m(1, predicate_all_b16x, predicate_all_b16x, a_first_bf16x, b_second_bf16x);
            svmopa_za32_bf16_m(2, predicate_all_b16x, predicate_all_b16x, a_second_bf16x, b_first_bf16x);
            svmopa_za32_bf16_m(3, predicate_all_b16x, predicate_all_b16x, a_second_bf16x, b_second_bf16x);
        }
    }
}

/** Accumulates one f16 NVFP4 A panel against four NVFP4 column tiles into ZA0 to ZA3, for the final
 *  16-row strip. */
NUMKONG_INLINE void nk_sme_mopa_1x4_nvfp4_(nk_u16_t const *a, nk_u8_t const *const *b_codes,
                                           nk_u16_t const *const *b_scales, nk_size_t steps) NUMKONG_STREAMING_
    __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    svuint16x2_t const table_u16x = nk_sme_table16_u16_(nk_sme_nvfp4_table_data_);
    for (nk_size_t block_first = 0; block_first < steps; block_first += 8) {
        nk_size_t const block_offset = block_first / 8 * vector_elements;
        svfloat16_t const first_scales_f16x = svld1_f16(predicate_all_b16x,
                                                        (float16_t const *)b_scales[0] + block_offset);
        svfloat16_t const second_scales_f16x = svld1_f16(predicate_all_b16x,
                                                         (float16_t const *)b_scales[1] + block_offset);
        svfloat16_t const third_scales_f16x = svld1_f16(predicate_all_b16x,
                                                        (float16_t const *)b_scales[2] + block_offset);
        svfloat16_t const fourth_scales_f16x = svld1_f16(predicate_all_b16x,
                                                         (float16_t const *)b_scales[3] + block_offset);
        nk_size_t const block_end = steps - block_first < 8 ? steps : block_first + 8;
        for (nk_size_t step = block_first; step < block_end; ++step) {
            nk_size_t const offset = step * vector_elements;
            svfloat16_t const a_f16x = svld1_f16(predicate_all_b16x, (float16_t const *)a + offset);
            svmopa_za32_f16_m(0, predicate_all_b16x, predicate_all_b16x, a_f16x,
                              nk_sme_nvfp4_b_vector_(table_u16x, b_codes[0] + offset, first_scales_f16x));
            svmopa_za32_f16_m(1, predicate_all_b16x, predicate_all_b16x, a_f16x,
                              nk_sme_nvfp4_b_vector_(table_u16x, b_codes[1] + offset, second_scales_f16x));
            svmopa_za32_f16_m(2, predicate_all_b16x, predicate_all_b16x, a_f16x,
                              nk_sme_nvfp4_b_vector_(table_u16x, b_codes[2] + offset, third_scales_f16x));
            svmopa_za32_f16_m(3, predicate_all_b16x, predicate_all_b16x, a_f16x,
                              nk_sme_nvfp4_b_vector_(table_u16x, b_codes[3] + offset, fourth_scales_f16x));
        }
    }
}

/** Accumulates one rebased bf16 MXFP4 A panel against four MXFP4 column tiles into ZA0 to ZA3, for
 *  the final 16-row strip. */
NUMKONG_INLINE void nk_sme_mopa_1x4_mxfp4_(nk_u16_t const *a, nk_u8_t const *const *b_codes,
                                           nk_u16_t const *const *b_decrements, nk_size_t steps) NUMKONG_STREAMING_
    __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    svuint16x2_t const table_u16x = nk_sme_table16_u16_(nk_sme_mxfp4_table_data_);
    for (nk_size_t block_first = 0; block_first < steps; block_first += 16) {
        nk_size_t const block_offset = block_first / 16 * vector_elements;
        svuint16_t const first_decrements_u16x = svld1_u16(predicate_all_b16x, b_decrements[0] + block_offset);
        svuint16_t const second_decrements_u16x = svld1_u16(predicate_all_b16x, b_decrements[1] + block_offset);
        svuint16_t const third_decrements_u16x = svld1_u16(predicate_all_b16x, b_decrements[2] + block_offset);
        svuint16_t const fourth_decrements_u16x = svld1_u16(predicate_all_b16x, b_decrements[3] + block_offset);
        nk_size_t const block_end = steps - block_first < 16 ? steps : block_first + 16;
        for (nk_size_t step = block_first; step < block_end; ++step) {
            nk_size_t const offset = step * vector_elements;
            svbfloat16_t const a_bf16x = svld1_bf16(predicate_all_b16x, (bfloat16_t const *)a + offset);
            svmopa_za32_bf16_m(0, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                               nk_sme_mxfp4_b_vector_(table_u16x, b_codes[0] + offset, first_decrements_u16x));
            svmopa_za32_bf16_m(1, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                               nk_sme_mxfp4_b_vector_(table_u16x, b_codes[1] + offset, second_decrements_u16x));
            svmopa_za32_bf16_m(2, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                               nk_sme_mxfp4_b_vector_(table_u16x, b_codes[2] + offset, third_decrements_u16x));
            svmopa_za32_bf16_m(3, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                               nk_sme_mxfp4_b_vector_(table_u16x, b_codes[3] + offset, fourth_decrements_u16x));
        }
    }
}

#pragma endregion Outer Products

#pragma region Epilogues

/** How a tile's accumulators become results. */
typedef struct {

    /** NVFP4 product of tensor scales, multiplied in after the last chunk, or null. */
    nk_f32_t const *tensor_factor;

    /** MX row exponents of the tile, applied with the column ones after the last chunk, or null. */
    nk_i32_t const *row_exponents;

    /** MX rebasing exponents of columns from @c column_origin on, or null. */
    nk_i8_t const *column_exponents;

    /** The column @c column_exponents starts at. */
    nk_size_t column_origin;

    /** Factor of an i32 → f32 conversion after the last chunk, or 0 to keep raw 32-bit integers. */
    nk_f32_t integer_factor;

    /** Whether lanes left of the diagonal stay untouched, as symmetric kernels store them. */
    int symmetric;
} nk_sme_epilogue_t;

/** One horizontal slice of ZA tile @p tile, which must fold to a constant after inlining. */
NUMKONG_INLINE svuint32_t nk_sme_read_tile_row_(nk_size_t tile, nk_size_t row) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b32x = svptrue_b32();
    switch (tile) {
    case 0: return svread_hor_za32_u32_m(svdup_n_u32(0), predicate_all_b32x, 0, (uint32_t)row);
    case 1: return svread_hor_za32_u32_m(svdup_n_u32(0), predicate_all_b32x, 1, (uint32_t)row);
    case 2: return svread_hor_za32_u32_m(svdup_n_u32(0), predicate_all_b32x, 2, (uint32_t)row);
    default: return svread_hor_za32_u32_m(svdup_n_u32(0), predicate_all_b32x, 3, (uint32_t)row);
    }
}

/** Stores one horizontal slice of ZA tile @p tile, which must fold to a constant after inlining. */
NUMKONG_INLINE void nk_sme_store_tile_row_(nk_size_t tile, nk_size_t row, svbool_t predicate_b32x,
                                           void *target) NUMKONG_STREAMING_ __arm_inout("za") {
    switch (tile) {
    case 0: svst1_hor_za32(0, (uint32_t)row, predicate_b32x, target); break;
    case 1: svst1_hor_za32(1, (uint32_t)row, predicate_b32x, target); break;
    case 2: svst1_hor_za32(2, (uint32_t)row, predicate_b32x, target); break;
    default: svst1_hor_za32(3, (uint32_t)row, predicate_b32x, target); break;
    }
}

/** Loads a horizontal ZA slice of @p tile, zeroing inactive lanes; @p tile must be a constant. */
NUMKONG_INLINE void nk_sme_load_tile_row_(nk_size_t tile, nk_size_t row, svbool_t predicate_b32x,
                                          void const *source) NUMKONG_STREAMING_ __arm_inout("za") {
    switch (tile) {
    case 0: svld1_hor_za32(0, (uint32_t)row, predicate_b32x, source); break;
    case 1: svld1_hor_za32(1, (uint32_t)row, predicate_b32x, source); break;
    case 2: svld1_hor_za32(2, (uint32_t)row, predicate_b32x, source); break;
    default: svld1_hor_za32(3, (uint32_t)row, predicate_b32x, source); break;
    }
}

/** Loads into ZA tile @p tile the raw partial sums an earlier chunk stored with
 *  @c nk_sme_store_tile_, the same rows and lanes, so that this chunk accumulates on top of them
 *  without vector arithmetic. */
NUMKONG_INLINE void nk_sme_load_tile_(nk_size_t tile, void const *c, nk_size_t c_stride, nk_size_t rows,
                                      nk_size_t row_first, nk_size_t column_first, nk_size_t columns,
                                      nk_sme_epilogue_t const *epilogue) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const columns_b32x = svwhilelt_b32_u64(column_first, columns);
    for (nk_size_t row = 0; row < rows; ++row) {
        svbool_t const load_b32x = epilogue->symmetric
                                       ? nk_sme_diagonal_cut_b32x_(columns_b32x, column_first, row_first + row)
                                       : columns_b32x;
        nk_sme_load_tile_row_(tile, row, load_b32x,
                              (nk_u32_t const *)((char const *)c + (row_first + row) * c_stride) + column_first);
    }
}

/** Stores rows [0, @p rows) of ZA tile @p tile at output rows from @p row_first and columns from
 *  @p column_first, lanes limited to @p columns: raw sums straight from ZA before the @p last
 *  chunk, which applies @p epilogue. F32 tiles hold floats, integer tiles 32-bit integers converted
 *  only by a nonzero integer factor. */
NUMKONG_INLINE void nk_sme_store_tile_(nk_size_t tile, int float_tile, void *c, nk_size_t c_stride, nk_size_t rows,
                                       nk_size_t row_first, nk_size_t column_first, nk_size_t columns, int last,
                                       nk_sme_epilogue_t const *epilogue) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const columns_b32x = svwhilelt_b32_u64(column_first, columns);
    int const raw = !last || (float_tile ? !epilogue->column_exponents && !epilogue->tensor_factor
                                         : epilogue->integer_factor == 0);
    svint32_t const column_exponents_i32x =
        epilogue->column_exponents
            ? svld1sb_s32(columns_b32x, epilogue->column_exponents + (column_first - epilogue->column_origin))
            : svdup_n_s32(0);
    for (nk_size_t row = 0; row < rows; ++row) {
        svbool_t const store_b32x = epilogue->symmetric
                                        ? nk_sme_diagonal_cut_b32x_(columns_b32x, column_first, row_first + row)
                                        : columns_b32x;
        nk_u32_t *target = (nk_u32_t *)((char *)c + (row_first + row) * c_stride) + column_first;
        if (raw) {
            nk_sme_store_tile_row_(tile, row, store_b32x, target);
            continue;
        }
        svuint32_t const row_u32x = nk_sme_read_tile_row_(tile, row);
        if (float_tile) {
            svfloat32_t row_f32x = svreinterpret_f32_u32(row_u32x);
            if (epilogue->column_exponents)
                row_f32x = svscale_f32_x(
                    store_b32x, row_f32x,
                    svadd_n_s32_x(store_b32x, column_exponents_i32x, epilogue->row_exponents[row]));
            if (epilogue->tensor_factor) row_f32x = svmul_n_f32_x(store_b32x, row_f32x, *epilogue->tensor_factor);
            svst1_f32(store_b32x, (nk_f32_t *)target, row_f32x);
        }
        else
            svst1_f32(store_b32x, (nk_f32_t *)target,
                      svmul_n_f32_x(store_b32x, svcvt_f32_s32_x(store_b32x, svreinterpret_s32_u32(row_u32x)),
                                    epilogue->integer_factor));
    }
}

#pragma endregion Epilogues

#pragma region Packs

/** Bytes of a pack of @p columns 16-bit columns @p depth deep, with their F32 squared norms. */
NUMKONG_INLINE nk_size_t nk_dots_pack_size_b16_sme_(nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = nk_sme_cntw_(), vector_elements = nk_sme_cnth_();
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, 2);
    return sizeof(nk_dots_sme_packed_header_t) + column_tile_count * depth_step_count * vector_elements * 2 +
           columns * sizeof(nk_f32_t);
}

/** Bytes of a pack of @p columns 8-bit columns of @p dtype @p depth deep, with their squared norms;
 *  i4 and u4 keep a low-nibble and a high-nibble byte per 4-bit dim. */
NUMKONG_INLINE nk_size_t nk_dots_pack_size_b8_sme_(nk_dtype_t dtype, nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = nk_sme_cntw_(), vector_bytes = nk_sme_cntb_();
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    return sizeof(nk_dots_sme_packed_header_t) +
           column_tile_count * nk_sme_panel_steps_b8_(dtype, depth) * vector_bytes + columns * sizeof(nk_u32_t);
}

/** Byte offsets of a block-scaled SME pack: header, values, FP4 block factors, norms, raw block
 *  scales and per-column rebasing exponents. */
typedef struct {
    nk_size_t values, factors, norms, scales, exponents, total;
} nk_dots_scaled_sme_layout_t;

/** The block-scaled pack layout of @p columns × @p depth for @p dtype at @p tile_dimension lanes:
 *  FP4 keeps one byte per code and one factor vector per column tile and block, MXFP6 and MXFP8
 *  keep folded bf16 values. */
NUMKONG_INLINE nk_dots_scaled_sme_layout_t nk_dots_scaled_sme_layout_(nk_dtype_t dtype, nk_size_t columns,
                                                                      nk_size_t depth,
                                                                      nk_size_t tile_dimension) NUMKONG_STREAMABLE_ {
    int const fp4 = dtype == nk_nvfp4_k || dtype == nk_mxfp4_k;
    nk_size_t const block_size = dtype == nk_nvfp4_k ? 16 : 32;
    nk_size_t const column_tiles = nk_size_divide_round_up_(columns, tile_dimension),
                    steps = nk_size_divide_round_up_(depth, 2);
    nk_size_t const blocks = depth / block_size, vector_elements = tile_dimension * 2;
    nk_dots_scaled_sme_layout_t layout;
    layout.values = sizeof(nk_dots_sme_packed_header_t);
    layout.factors = layout.values + column_tiles * steps * vector_elements * (fp4 ? 1 : 2);
    nk_size_t const factors_end = layout.factors + (fp4 ? column_tiles * blocks * vector_elements * 2 : 0);
    layout.norms = nk_size_round_up_to_multiple_(factors_end, 64);
    layout.scales = layout.norms + columns * sizeof(nk_f32_t);
    layout.exponents = layout.scales + columns * blocks;
    layout.total = layout.exponents + columns;
    return layout;
}

/** Writes the header of an SME pack once, from the window that packs column 0. */
NUMKONG_INLINE void nk_dots_sme_header_(void *b_packed, nk_size_t columns, nk_size_t depth, nk_size_t depth_steps,
                                        nk_size_t norms_offset, nk_f32_t tensor_scale, nk_size_t tile_dimension) {
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    for (nk_size_t word = 0; word < sizeof(*header) / sizeof(nk_u32_t); ++word) ((nk_u32_t *)header)[word] = 0;
    header->column_tile_count = (nk_u32_t)nk_size_divide_round_up_(columns, tile_dimension);
    header->depth_tile_count = (nk_u32_t)depth_steps;
    header->columns = (nk_u32_t)columns;
    header->depth = (nk_u32_t)depth;
    header->svl_bytes = (nk_u32_t)(tile_dimension * sizeof(nk_f32_t));
    header->norms_offset = (nk_u32_t)norms_offset;
    header->tensor_scale = tensor_scale;
    header->capability = nk_cap_sme_k;
}

/** The column tiles from @p tile_begin up to @p tile_end that a pack window over columns from
 *  @p columns_begin up to @p columns_end owns, every tile it touches short of tiles an earlier
 *  window already started. */
NUMKONG_INLINE void nk_dots_sme_window_tiles_(nk_size_t columns, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_size_t tile_dimension, nk_size_t *tile_begin, nk_size_t *tile_end) {
    nk_size_t const column_tiles = nk_size_divide_round_up_(columns, tile_dimension);
    *tile_begin = nk_size_divide_round_up_(columns_begin, tile_dimension);
    *tile_end = nk_size_divide_round_up_(columns_end, tile_dimension);
    if (*tile_end > column_tiles) *tile_end = column_tiles;
}

/** Packs 16-bit-panel columns of @p operand into whole SME column tiles from @p tile_begin up to
 *  @p tile_end, staged over the full depth. MX columns rebase by @p column_exponents, indexed from
 *  the first column of @p tile_begin, or stay unfolded where they say @c nk_sme_unfolded_base_k. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_pack_b16_sme_streaming_( //
    nk_dtype_t dtype, nk_cross_operand_t operand, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
    nk_u16_t *tiles, nk_size_t tile_begin, nk_size_t tile_end, nk_i32_t const *column_exponents) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    nk_size_t const steps = nk_size_divide_round_up_(depth, 2);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_i32_t const *bases = column_exponents ? column_exponents + (tile - tile_begin) * tile_dimension
                                                 : NUMKONG_NULL;
        nk_sme_stage_panel_b16_(dtype, dtype == nk_bf16_k ? nk_bf16_k : nk_f16_k, operand, b_stride, column_first,
                                tile_columns, 0, depth, bases, tiles + tile * steps * vector_elements);
    }
}

/** Packs 8-bit-panel columns of @p dtype into SME tiles from @p tile_begin up to @p tile_end. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_pack_b8_sme_streaming_( //
    nk_dtype_t dtype, void const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth, nk_u8_t *tiles,
    nk_size_t tile_begin, nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    nk_size_t const steps = nk_sme_panel_steps_b8_(dtype, depth);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_sme_stage_panel_b8_(dtype, b, b_stride, column_first, tile_columns, 0, depth,
                               tiles + tile * steps * vector_bytes);
    }
}

/** Squared norm of one 16-bit-panel column of @p dtype. */
NUMKONG_INLINE nk_f32_t nk_dots_sumsq_b16_sme_(nk_dtype_t dtype, void const *column, nk_size_t depth) {
    switch (dtype) {
    case nk_f16_k: return nk_dots_reduce_sumsq_f16_((nk_f16_t const *)column, depth, nk_cap_sme_k);
    case nk_bf16_k: return nk_dots_reduce_sumsq_bf16_((nk_bf16_t const *)column, depth, nk_cap_sme_k);
    case nk_e4m3_k: return nk_dots_reduce_sumsq_e4m3_((nk_e4m3_t const *)column, depth, nk_cap_sme_k);
    case nk_e5m2_k: return nk_dots_reduce_sumsq_e5m2_((nk_e5m2_t const *)column, depth, nk_cap_sme_k);
    default: return nk_dots_reduce_sumsq_e3m2_((nk_e3m2_t const *)column, depth, nk_cap_sme_k);
    }
}

/** Squared norm of one 8-bit-panel column of @p dtype, F32 bits for e2m3 and e2m1, else U32. */
NUMKONG_INLINE nk_u32_t nk_dots_sumsq_b8_sme_(nk_dtype_t dtype, void const *column, nk_size_t depth) {
    nk_fui32_t norm;
    switch (dtype) {
    case nk_i8_k: return nk_dots_reduce_sumsq_i8_((nk_i8_t const *)column, depth, nk_cap_sme_k);
    case nk_u8_k: return nk_dots_reduce_sumsq_u8_((nk_u8_t const *)column, depth, nk_cap_sme_k);
    case nk_i4_k: return nk_dots_reduce_sumsq_i4_((nk_i4x2_t const *)column, depth, nk_cap_sme_k);
    case nk_u4_k: return nk_dots_reduce_sumsq_u4_((nk_u4x2_t const *)column, depth, nk_cap_sme_k);
    case nk_e2m3_k: norm.f = nk_dots_reduce_sumsq_e2m3_((nk_e2m3_t const *)column, depth, nk_cap_sme_k); return norm.u;
    default: norm.f = nk_dots_reduce_sumsq_e2m1_((nk_e2m1x2_t const *)column, depth, nk_cap_sme_k); return norm.u;
    }
}

/** Packs 16-bit-panel columns of @p dtype from @p columns_begin up to @p columns_end with norms. */
NUMKONG_INLINE void nk_dots_pack_b16_sme_(nk_dtype_t dtype, void const *b, nk_size_t columns, nk_size_t depth,
                                          nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                          nk_size_t columns_end) {
    nk_size_t const tile_dimension = nk_sme_cntw_(), vector_elements = nk_sme_cnth_();
    nk_size_t const steps = nk_size_divide_round_up_(depth, 2);
    nk_size_t const norms_offset = sizeof(nk_dots_sme_packed_header_t) +
                                   nk_size_divide_round_up_(columns, tile_dimension) * steps * vector_elements * 2;
    if (columns_begin == 0) nk_dots_sme_header_(b_packed, columns, depth, steps, norms_offset, 1, tile_dimension);
    nk_size_t tile_begin, tile_end;
    nk_dots_sme_window_tiles_(columns, columns_begin, columns_end, tile_dimension, &tile_begin, &tile_end);
    nk_cross_operand_t const operand = {b, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_pack_b16_sme_streaming_(dtype, operand, b_stride, columns, depth,
                                    (nk_u16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t)), tile_begin,
                                    tile_end, NUMKONG_NULL);
    nk_sme_stop_streaming_();
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t column = columns_begin; column < columns_end; ++column)
        norms[column] = nk_dots_sumsq_b16_sme_(dtype, (char const *)b + column * b_stride, depth);
}

/** Packs 8-bit-panel columns of @p dtype from @p columns_begin up to @p columns_end with norms. */
NUMKONG_INLINE void nk_dots_pack_b8_sme_(nk_dtype_t dtype, void const *b, nk_size_t columns, nk_size_t depth,
                                         nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                         nk_size_t columns_end) {
    nk_size_t const tile_dimension = nk_sme_cntw_(), vector_bytes = nk_sme_cntb_();
    nk_size_t const steps = nk_sme_panel_steps_b8_(dtype, depth);
    nk_size_t const norms_offset = sizeof(nk_dots_sme_packed_header_t) +
                                   nk_size_divide_round_up_(columns, tile_dimension) * steps * vector_bytes;
    if (columns_begin == 0) nk_dots_sme_header_(b_packed, columns, depth, steps, norms_offset, 1, tile_dimension);
    nk_size_t tile_begin, tile_end;
    nk_dots_sme_window_tiles_(columns, columns_begin, columns_end, tile_dimension, &tile_begin, &tile_end);
    nk_sme_start_streaming_();
    nk_dots_pack_b8_sme_streaming_(dtype, b, b_stride, columns, depth,
                                   (nk_u8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t)), tile_begin,
                                   tile_end);
    nk_sme_stop_streaming_();
    nk_u32_t *norms = (nk_u32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t column = columns_begin; column < columns_end; ++column)
        norms[column] = nk_dots_sumsq_b8_sme_(dtype, (char const *)b + column * b_stride, depth);
}

/** Packs F16 @p b columns @p columns_begin to @p columns_end into SME tiles with squared norms. */
NUMKONG_INLINE void nk_dots_pack_f16_tiles_sme_(           //
    nk_f16_t const *b, nk_size_t columns, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end) {
    nk_dots_pack_b16_sme_(nk_f16_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
}

/** Packs BF16 @p b columns @p columns_begin to @p columns_end into SME tiles with squared norms. */
NUMKONG_INLINE void nk_dots_pack_bf16_tiles_sme_(           //
    nk_bf16_t const *b, nk_size_t columns, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end) {
    nk_dots_pack_b16_sme_(nk_bf16_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
}

/** Packs block-scaled columns of @p dtype from @p columns_begin up to @p columns_end. FP4 stores
 *  one byte per code, f16 block scales for NVFP4 and saturating exponent decrements for MXFP4 as
 *  factor vectors; MXFP6 and MXFP8 store folded bf16 values. MX columns rebase to their largest
 *  block exponent, and columns spanning more than @c nk_sme_exponent_spread_k binades, or holding a
 *  NaN scale in MXFP4, are marked -128 for the exact fallback. */
NUMKONG_INLINE void nk_dots_pack_scaled_sme_(nk_dtype_t dtype, nk_cross_operand_t b, nk_size_t columns, nk_size_t depth,
                                             nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                             nk_size_t columns_end) {
    nk_size_t const tile_dimension = nk_sme_cntw_(), vector_elements = tile_dimension * 2;
    int const nvfp4 = dtype == nk_nvfp4_k, fp4 = nvfp4 || dtype == nk_mxfp4_k;
    nk_size_t const block_size = nvfp4 ? 16 : 32, blocks = depth / block_size,
                    steps = nk_size_divide_round_up_(depth, 2);
    nk_assert_(depth % block_size == 0);
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_sme_layout_(dtype, columns, depth, tile_dimension);
    nk_f32_t const tensor_scale = nk_cross_tensor_scale_(b.tensor_scale);
    if (columns_begin == 0)
        nk_dots_sme_header_(b_packed, columns, depth, steps, layout.norms, tensor_scale, tile_dimension);
    nk_size_t tile_begin, tile_end;
    nk_dots_sme_window_tiles_(columns, columns_begin, columns_end, tile_dimension, &tile_begin, &tile_end);
    nk_u8_t *pack = (nk_u8_t *)b_packed;
    nk_u8_t *scales = pack + layout.scales;
    nk_i8_t *exponents = (nk_i8_t *)(pack + layout.exponents);
    nk_size_t const column_end = tile_end * tile_dimension < columns ? tile_end * tile_dimension : columns;
    for (nk_size_t column = tile_begin * tile_dimension; column < column_end; ++column) {
        nk_u8_t const *source_scales = b.scales + column * b.scales_stride;
        for (nk_size_t block = 0; block < blocks; ++block) scales[column * blocks + block] = source_scales[block];
        nk_i32_t minimum = 0, maximum = 0, spread = 0, nan_scale = 0;
        if (!nvfp4) {
            nk_sme_exponent_range_(source_scales, blocks, &minimum, &maximum);
            spread = maximum - minimum;
            for (nk_size_t block = 0; block < blocks; ++block) nan_scale |= source_scales[block] == 255;
        }
        exponents[column] = (nk_i8_t)(spread > nk_sme_exponent_spread_k || (fp4 && nan_scale) ? -128 : maximum);
    }
    if (fp4) {
        nk_u8_t *codes = pack + layout.values;
        nk_u16_t *factors = (nk_u16_t *)(pack + layout.factors);
        for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
            for (nk_size_t lane = 0; lane < tile_dimension; ++lane) {
                nk_size_t const column = tile * tile_dimension + lane;
                nk_u8_t const *source = (nk_u8_t const *)b.elements + column * b_stride;
                for (nk_size_t index = 0; index < steps * 2; ++index) {
                    nk_u8_t code = 0;
                    if (column < columns && index < depth) {
                        nk_u8_t const pair = source[index / 2];
                        code = (nk_u8_t)(index & 1 ? pair & 0x0F : pair >> 4);
                        if (!nvfp4 && scales[column * blocks + index / block_size] == 0) code = 0;
                    }
                    codes[(tile * steps + index / 2) * vector_elements + lane * 2 + (index & 1)] = code;
                }
                for (nk_size_t block = 0; block < blocks; ++block) {
                    nk_u16_t factor = 0;
                    if (column < columns && nvfp4) {
                        nk_f32_t scale_f32;
                        nk_f16_t scale_f16;
                        nk_ue4m3_to_f32_((nk_ue4m3_t const *)&scales[column * blocks + block], &scale_f32);
                        nk_f32_to_f16_(&scale_f32, &scale_f16);
                        nk_copy_bytes_(&factor, &scale_f16, sizeof(factor));
                    }
                    else if (column < columns) {
                        nk_i32_t const code = scales[column * blocks + block];
                        if (code != 0 && code != 255 && exponents[column] != -128)
                            factor = (nk_u16_t)((exponents[column] - (code - 127)) * 128);
                    }
                    factors[(tile * blocks + block) * vector_elements + lane * 2 + 0] = factor;
                    factors[(tile * blocks + block) * vector_elements + lane * 2 + 1] = factor;
                }
            }
        }
    }
    else {
        nk_i32_t bases[nk_sme_max_tile_k];
        for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
            for (nk_size_t lane = 0; lane < tile_dimension; ++lane) {
                nk_size_t const column = tile * tile_dimension + lane;
                bases[lane] = column >= columns || exponents[column] == -128 ? nk_sme_unfolded_base_k
                                                                             : exponents[column];
            }
            nk_sme_start_streaming_();
            nk_dots_pack_b16_sme_streaming_(dtype, b, b_stride, columns, depth, (nk_u16_t *)(pack + layout.values),
                                            tile, tile + 1, bases);
            nk_sme_stop_streaming_();
        }
    }
    nk_f32_t *norms = (nk_f32_t *)(pack + layout.norms);
    for (nk_size_t column = columns_begin; column < columns_end; ++column)
        norms[column] = nk_cross_scaled_sumsq_serial_(dtype, (nk_u8_t const *)b.elements + column * b_stride,
                                                      b.scales + column * b.scales_stride, tensor_scale, depth);
}

#pragma endregion Packs

#pragma region Packed Kernels

/** The tile count of the row block starting at @p row_first: two tiles while 2×2 bodies apply and
 *  more than one tile of rows remains, else one. */
NUMKONG_INLINE nk_size_t nk_sme_block_rows_(int pairs, nk_size_t rows, nk_size_t row_first,
                                            nk_size_t tile_dimension) NUMKONG_STREAMABLE_ {
    nk_size_t const remaining = rows - row_first;
    if (pairs && remaining > tile_dimension) return remaining < 2 * tile_dimension ? remaining : 2 * tile_dimension;
    return remaining < tile_dimension ? remaining : tile_dimension;
}

/** A × packed B for 16-bit-panel inputs: f16, bf16, e4m3, e5m2, e3m2, and the folded MXFP6 and
 *  MXFP8 packs. Rows are decoded once per tile and depth chunk; f16 inputs run 2×2 bodies with a
 *  1×4 final strip, bf16 inputs 1×4 bodies. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_packed_b16_sme_streaming_( //
    nk_dtype_t dtype, nk_cross_operand_t a, nk_size_t a_stride, void const *b_packed, nk_f32_t *c, nk_size_t rows,
    nk_size_t columns, nk_size_t depth, nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    nk_size_t const steps_total = header->depth_tile_count, column_tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    int const mx = dtype == nk_mxfp6e2m3_k || dtype == nk_mxfp6e3m2_k || dtype == nk_mxfp8e4m3_k ||
                   dtype == nk_mxfp8e5m2_k;
    int const bf16_inputs = dtype == nk_bf16_k || mx;
    nk_size_t const blocks = mx ? depth / 32 : 0;
    nk_align_(64) nk_u16_t panels[2][nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t exponents[2][nk_sme_max_tile_k];
    nk_sme_epilogue_t epilogue = {NUMKONG_NULL, NUMKONG_NULL, NUMKONG_NULL, 0, 0, 0};
    if (mx) {
        nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_sme_layout_(dtype, columns, depth, tile_dimension);
        b_tiles = (nk_u16_t const *)((nk_u8_t const *)b_packed + layout.values);
        epilogue.column_exponents = (nk_i8_t const *)b_packed + layout.exponents;
    }
    for (nk_size_t row_first = 0; row_first < rows;) {
        nk_size_t const block_rows = nk_sme_block_rows_(!bf16_inputs, rows, row_first, tile_dimension);
        nk_size_t const first_rows = block_rows < tile_dimension ? block_rows : tile_dimension;
        nk_size_t const second_rows = block_rows - first_rows;
        if (mx) {
            nk_sme_row_exponents_(a, row_first, first_rows, blocks, 0, exponents[0]);
            nk_sme_row_exponents_(a, row_first + first_rows, second_rows, blocks, 0, exponents[1]);
        }
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2), step_first = depth_first / 2;
            int const first = depth_first == 0, last = depth_first + chunk_depth >= depth;
            nk_dtype_t const panel_dtype = bf16_inputs ? nk_bf16_k : nk_f16_k;
            nk_sme_stage_panel_b16_(dtype, panel_dtype, a, a_stride, row_first, first_rows, depth_first, chunk_depth,
                                    mx ? exponents[0] : NUMKONG_NULL, panels[0]);
            if (second_rows)
                nk_sme_stage_panel_b16_(dtype, panel_dtype, a, a_stride, row_first + first_rows, second_rows,
                                        depth_first, chunk_depth, mx ? exponents[1] : NUMKONG_NULL, panels[1]);
            if (second_rows) {
                for (nk_size_t tile = 0; tile < column_tiles; tile += 2) {
                    nk_u16_t const *b_first = b_tiles + (tile * steps_total + step_first) * vector_elements;
                    int const has_second = tile + 1 < column_tiles;
                    nk_u16_t const *b_second = has_second ? b_first + steps_total * vector_elements : b_first;
                    nk_size_t const column_first = tile * tile_dimension;
                    svzero_za();
                    if (!first) {
                        nk_sme_load_tile_(0, c, c_stride, first_rows, row_first, column_first, columns, &epilogue);
                        if (has_second)
                            nk_sme_load_tile_(1, c, c_stride, first_rows, row_first, column_first + tile_dimension,
                                              columns, &epilogue);
                        nk_sme_load_tile_(2, c, c_stride, second_rows, row_first + first_rows, column_first, columns,
                                          &epilogue);
                        if (has_second)
                            nk_sme_load_tile_(3, c, c_stride, second_rows, row_first + first_rows,
                                              column_first + tile_dimension, columns, &epilogue);
                    }
                    nk_sme_mopa_2x2_f16_(panels[0], panels[1], b_first, b_second, steps);
                    epilogue.row_exponents = exponents[0];
                    nk_sme_store_tile_(0, 1, c, c_stride, first_rows, row_first, column_first, columns, last,
                                       &epilogue);
                    if (has_second)
                        nk_sme_store_tile_(1, 1, c, c_stride, first_rows, row_first, column_first + tile_dimension,
                                           columns, last, &epilogue);
                    epilogue.row_exponents = exponents[1];
                    nk_sme_store_tile_(2, 1, c, c_stride, second_rows, row_first + first_rows, column_first, columns,
                                       last, &epilogue);
                    if (has_second)
                        nk_sme_store_tile_(3, 1, c, c_stride, second_rows, row_first + first_rows,
                                           column_first + tile_dimension, columns, last, &epilogue);
                }
            }
            else {
                epilogue.row_exponents = exponents[0];
                for (nk_size_t tile = 0; tile < column_tiles; tile += 4) {
                    nk_u16_t const *b_quad[4];
                    nk_size_t const quad_tiles = column_tiles - tile < 4 ? column_tiles - tile : 4;
                    for (nk_size_t index = 0; index < 4; ++index)
                        b_quad[index] = b_tiles +
                                        ((tile + (index < quad_tiles ? index : 0)) * steps_total + step_first) *
                                            vector_elements;
                    svzero_za();
                    for (nk_size_t index = 0; index < quad_tiles && !first; ++index)
                        nk_sme_load_tile_(index, c, c_stride, first_rows, row_first, (tile + index) * tile_dimension,
                                          columns, &epilogue);
                    if (bf16_inputs)
                        nk_sme_mopa_1x4_bf16_(panels[0], b_quad[0], b_quad[1], b_quad[2], b_quad[3], steps);
                    else nk_sme_mopa_1x4_f16_(panels[0], b_quad[0], b_quad[1], b_quad[2], b_quad[3], steps);
                    for (nk_size_t index = 0; index < quad_tiles; ++index)
                        nk_sme_store_tile_(index, 1, c, c_stride, first_rows, row_first,
                                           (tile + index) * tile_dimension, columns, last, &epilogue);
                }
            }
        }
        row_first += block_rows;
    }
}

/** A × packed B for 8-bit-panel inputs: i8, u8, e2m3 and e2m1 as i8, i4 and u4 as split nibbles,
 *  2×2 bodies with a 1×4 final strip. Integer results stay raw; e2m3 and e2m1 results convert by
 *  1/256 and 1/4 respectively. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_packed_b8_sme_streaming_( //
    nk_dtype_t dtype, void const *a, nk_size_t a_stride, void const *b_packed, void *c, nk_size_t rows,
    nk_size_t columns, nk_size_t depth, nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u8_t const *b_tiles = (nk_u8_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    nk_size_t const steps_total = header->depth_tile_count, column_tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / vector_bytes * 4;
    int const unsigned_inputs = dtype == nk_u8_k || dtype == nk_u4_k;
    nk_align_(64) nk_u8_t panels[2][nk_sme_panel_bytes_k];
    nk_sme_epilogue_t epilogue = {NUMKONG_NULL, NUMKONG_NULL, NUMKONG_NULL, 0, 0, 0};
    epilogue.integer_factor = dtype == nk_e2m3_k ? 1.0f / 256 : dtype == nk_e2m1_k ? 0.25f : 0;
    for (nk_size_t row_first = 0; row_first < rows;) {
        nk_size_t const block_rows = nk_sme_block_rows_(1, rows, row_first, tile_dimension);
        nk_size_t const first_rows = block_rows < tile_dimension ? block_rows : tile_dimension;
        nk_size_t const second_rows = block_rows - first_rows;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_sme_panel_steps_b8_(dtype, chunk_depth), step_first = depth_first / 4;
            int const first = depth_first == 0, last = depth_first + chunk_depth >= depth;
            nk_sme_stage_panel_b8_(dtype, a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (second_rows)
                nk_sme_stage_panel_b8_(dtype, a, a_stride, row_first + first_rows, second_rows, depth_first,
                                       chunk_depth, panels[1]);
            if (second_rows) {
                for (nk_size_t tile = 0; tile < column_tiles; tile += 2) {
                    nk_u8_t const *b_first = b_tiles + (tile * steps_total + step_first) * vector_bytes;
                    int const has_second = tile + 1 < column_tiles;
                    nk_u8_t const *b_second = has_second ? b_first + steps_total * vector_bytes : b_first;
                    nk_size_t const column_first = tile * tile_dimension;
                    svzero_za();
                    if (!first) {
                        nk_sme_load_tile_(0, c, c_stride, first_rows, row_first, column_first, columns, &epilogue);
                        if (has_second)
                            nk_sme_load_tile_(1, c, c_stride, first_rows, row_first, column_first + tile_dimension,
                                              columns, &epilogue);
                        nk_sme_load_tile_(2, c, c_stride, second_rows, row_first + first_rows, column_first, columns,
                                          &epilogue);
                        if (has_second)
                            nk_sme_load_tile_(3, c, c_stride, second_rows, row_first + first_rows,
                                              column_first + tile_dimension, columns, &epilogue);
                    }
                    nk_sme_mopa_2x2_b8_(unsigned_inputs, panels[0], panels[1], b_first, b_second, steps);
                    nk_sme_store_tile_(0, 0, c, c_stride, first_rows, row_first, column_first, columns, last,
                                       &epilogue);
                    if (has_second)
                        nk_sme_store_tile_(1, 0, c, c_stride, first_rows, row_first, column_first + tile_dimension,
                                           columns, last, &epilogue);
                    nk_sme_store_tile_(2, 0, c, c_stride, second_rows, row_first + first_rows, column_first, columns,
                                       last, &epilogue);
                    if (has_second)
                        nk_sme_store_tile_(3, 0, c, c_stride, second_rows, row_first + first_rows,
                                           column_first + tile_dimension, columns, last, &epilogue);
                }
            }
            else {
                for (nk_size_t tile = 0; tile < column_tiles; tile += 4) {
                    nk_u8_t const *b_quad[4];
                    nk_size_t const quad_tiles = column_tiles - tile < 4 ? column_tiles - tile : 4;
                    for (nk_size_t index = 0; index < 4; ++index)
                        b_quad[index] = b_tiles +
                                        ((tile + (index < quad_tiles ? index : 0)) * steps_total + step_first) *
                                            vector_bytes;
                    svzero_za();
                    for (nk_size_t index = 0; index < quad_tiles && !first; ++index)
                        nk_sme_load_tile_(index, c, c_stride, first_rows, row_first, (tile + index) * tile_dimension,
                                          columns, &epilogue);
                    nk_sme_mopa_1x4_b8_(unsigned_inputs, panels[0], b_quad[0], b_quad[1], b_quad[2], b_quad[3], steps);
                    for (nk_size_t index = 0; index < quad_tiles; ++index)
                        nk_sme_store_tile_(index, 0, c, c_stride, first_rows, row_first,
                                           (tile + index) * tile_dimension, columns, last, &epilogue);
                }
            }
        }
        row_first += block_rows;
    }
}

/** A × packed B for NVFP4 and MXFP4: A rows decode to exact f16 or rebased bf16 panels, B decodes
 *  per step from one byte per code and per-block factor vectors; 2×2 bodies, 1×4 final strip. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_packed_fp4_sme_streaming_( //
    nk_dtype_t dtype, nk_cross_operand_t a, nk_size_t a_stride, void const *b_packed, nk_f32_t *c, nk_size_t rows,
    nk_size_t columns, nk_size_t depth, nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    int const mx = dtype == nk_mxfp4_k;
    nk_size_t const block_size = mx ? 32 : 16, blocks_total = depth / block_size;
    nk_size_t const steps_total = header->depth_tile_count, column_tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_sme_layout_(dtype, columns, depth, tile_dimension);
    nk_u8_t const *codes = (nk_u8_t const *)b_packed + layout.values;
    nk_u16_t const *factors = (nk_u16_t const *)((nk_u8_t const *)b_packed + layout.factors);
    nk_f32_t const tensor_factor = (a.tensor_scale ? *a.tensor_scale : 1) * header->tensor_scale;
    nk_align_(64) nk_u16_t panels[2][nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t exponents[2][nk_sme_max_tile_k];
    nk_sme_epilogue_t epilogue = {NUMKONG_NULL, NUMKONG_NULL, NUMKONG_NULL, 0, 0, 0};
    if (mx) epilogue.column_exponents = (nk_i8_t const *)b_packed + layout.exponents;
    else epilogue.tensor_factor = &tensor_factor;
    nk_dtype_t const panel_dtype = mx ? nk_bf16_k : nk_f16_k;
    for (nk_size_t row_first = 0; row_first < rows;) {
        nk_size_t const block_rows = nk_sme_block_rows_(1, rows, row_first, tile_dimension);
        nk_size_t const first_rows = block_rows < tile_dimension ? block_rows : tile_dimension;
        nk_size_t const second_rows = block_rows - first_rows;
        if (mx) {
            // The largest exponents make the E2M1 fold a single saturating decrement
            nk_sme_row_exponents_(a, row_first, first_rows, blocks_total, 1, exponents[0]);
            nk_sme_row_exponents_(a, row_first + first_rows, second_rows, blocks_total, 1, exponents[1]);
        }
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2), step_first = depth_first / 2;
            nk_size_t const block_first = depth_first / block_size;
            int const first = depth_first == 0, last = depth_first + chunk_depth >= depth;
            nk_sme_stage_panel_b16_(dtype, panel_dtype, a, a_stride, row_first, first_rows, depth_first, chunk_depth,
                                    mx ? exponents[0] : NUMKONG_NULL, panels[0]);
            if (second_rows)
                nk_sme_stage_panel_b16_(dtype, panel_dtype, a, a_stride, row_first + first_rows, second_rows,
                                        depth_first, chunk_depth, mx ? exponents[1] : NUMKONG_NULL, panels[1]);
            if (second_rows) {
                for (nk_size_t tile = 0; tile < column_tiles; tile += 2) {
                    int const has_second = tile + 1 < column_tiles;
                    nk_size_t const second_tile = has_second ? tile + 1 : tile;
                    nk_u8_t const *b_first_codes = codes + (tile * steps_total + step_first) * vector_elements;
                    nk_u8_t const *b_second_codes = codes + (second_tile * steps_total + step_first) * vector_elements;
                    nk_u16_t const *b_first_factors = factors + (tile * blocks_total + block_first) * vector_elements;
                    nk_u16_t const *b_second_factors = factors +
                                                       (second_tile * blocks_total + block_first) * vector_elements;
                    nk_size_t const column_first = tile * tile_dimension;
                    svzero_za();
                    if (!first) {
                        nk_sme_load_tile_(0, c, c_stride, first_rows, row_first, column_first, columns, &epilogue);
                        if (has_second)
                            nk_sme_load_tile_(1, c, c_stride, first_rows, row_first, column_first + tile_dimension,
                                              columns, &epilogue);
                        nk_sme_load_tile_(2, c, c_stride, second_rows, row_first + first_rows, column_first, columns,
                                          &epilogue);
                        if (has_second)
                            nk_sme_load_tile_(3, c, c_stride, second_rows, row_first + first_rows,
                                              column_first + tile_dimension, columns, &epilogue);
                    }
                    if (mx)
                        nk_sme_mopa_2x2_mxfp4_(panels[0], panels[1], b_first_codes, b_second_codes, b_first_factors,
                                               b_second_factors, steps);
                    else
                        nk_sme_mopa_2x2_nvfp4_(panels[0], panels[1], b_first_codes, b_second_codes, b_first_factors,
                                               b_second_factors, steps);
                    epilogue.row_exponents = exponents[0];
                    nk_sme_store_tile_(0, 1, c, c_stride, first_rows, row_first, column_first, columns, last,
                                       &epilogue);
                    if (has_second)
                        nk_sme_store_tile_(1, 1, c, c_stride, first_rows, row_first, column_first + tile_dimension,
                                           columns, last, &epilogue);
                    epilogue.row_exponents = exponents[1];
                    nk_sme_store_tile_(2, 1, c, c_stride, second_rows, row_first + first_rows, column_first, columns,
                                       last, &epilogue);
                    if (has_second)
                        nk_sme_store_tile_(3, 1, c, c_stride, second_rows, row_first + first_rows,
                                           column_first + tile_dimension, columns, last, &epilogue);
                }
            }
            else {
                epilogue.row_exponents = exponents[0];
                for (nk_size_t tile = 0; tile < column_tiles; tile += 4) {
                    nk_u8_t const *b_codes[4];
                    nk_u16_t const *b_factors[4];
                    nk_size_t const quad_tiles = column_tiles - tile < 4 ? column_tiles - tile : 4;
                    for (nk_size_t index = 0; index < 4; ++index) {
                        nk_size_t const quad_tile = tile + (index < quad_tiles ? index : 0);
                        b_codes[index] = codes + (quad_tile * steps_total + step_first) * vector_elements;
                        b_factors[index] = factors + (quad_tile * blocks_total + block_first) * vector_elements;
                    }
                    svzero_za();
                    for (nk_size_t index = 0; index < quad_tiles && !first; ++index)
                        nk_sme_load_tile_(index, c, c_stride, first_rows, row_first, (tile + index) * tile_dimension,
                                          columns, &epilogue);
                    if (mx) nk_sme_mopa_1x4_mxfp4_(panels[0], b_codes, b_factors, steps);
                    else nk_sme_mopa_1x4_nvfp4_(panels[0], b_codes, b_factors, steps);
                    for (nk_size_t index = 0; index < quad_tiles; ++index)
                        nk_sme_store_tile_(index, 1, c, c_stride, first_rows, row_first,
                                           (tile + index) * tile_dimension, columns, last, &epilogue);
                }
            }
        }
        row_first += block_rows;
    }
}

#pragma endregion Packed Kernels

#pragma region Symmetric Kernels

/** The upper triangle of V × Vᵀ over @p row_count rows from @p row_start for 16-bit-panel inputs
 *  and every block-scaled format. Windows of @c nk_sme_window_tiles_k column tiles decode once per
 *  depth chunk, each row tile decodes once per window, and 1×4 bodies sweep the window. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_symmetric_b16_sme_streaming_( //
    nk_dtype_t dtype, nk_cross_operand_t vectors, nk_size_t stride, nk_size_t count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const row_end = row_start + row_count < count ? row_start + row_count : count;
    if (row_start >= row_end) return;
    int const nvfp4 = dtype == nk_nvfp4_k;
    int const mx = dtype == nk_mxfp4_k || dtype == nk_mxfp6e2m3_k || dtype == nk_mxfp6e3m2_k ||
                   dtype == nk_mxfp8e4m3_k || dtype == nk_mxfp8e5m2_k;
    int const bf16_inputs = dtype == nk_bf16_k || mx;
    nk_dtype_t const panel_dtype = bf16_inputs ? nk_bf16_k : nk_f16_k;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2;
    nk_size_t const blocks = mx ? depth / 32 : 0;
    nk_size_t const window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_f32_t const tensor_scale = vectors.tensor_scale ? *vectors.tensor_scale : 1;
    nk_f32_t const tensor_factor = tensor_scale * tensor_scale;
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t row_exponents[nk_sme_max_tile_k], column_bases[nk_sme_window_tiles_k * nk_sme_max_tile_k];
    nk_i8_t column_exponents[nk_sme_window_tiles_k * nk_sme_max_tile_k];
    nk_sme_epilogue_t epilogue = {NUMKONG_NULL, row_exponents, NUMKONG_NULL, 0, 0, 1};
    if (nvfp4) epilogue.tensor_factor = &tensor_factor;
    for (nk_size_t window_first = row_start / tile_dimension * tile_dimension; window_first < count;
         window_first += window_width) {
        nk_size_t const window_columns = count - window_first < window_width ? count - window_first : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = row_end < window_first + window_columns ? row_end : window_first + window_columns;
        if (mx) {
            nk_sme_row_exponents_(vectors, window_first, window_columns, blocks, dtype == nk_mxfp4_k, column_bases);
            for (nk_size_t column = 0; column < window_columns; ++column)
                column_exponents[column] = (nk_i8_t)column_bases[column];
            epilogue.column_exponents = column_exponents, epilogue.column_origin = window_first;
        }
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            int const first = depth_first == 0, last = depth_first + chunk_depth >= depth;
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_sme_stage_panel_b16_(dtype, panel_dtype, vectors, stride, window_first + tile_first, tile_columns,
                                        depth_first, chunk_depth, mx ? column_bases + tile_first : NUMKONG_NULL,
                                        window[tile]);
            }
            for (nk_size_t row_first = row_start; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = row_end - row_first < tile_dimension ? row_end - row_first : tile_dimension;
                if (mx) nk_sme_row_exponents_(vectors, row_first, rows, blocks, dtype == nk_mxfp4_k, row_exponents);
                nk_sme_stage_panel_b16_(dtype, panel_dtype, vectors, stride, row_first, rows, depth_first, chunk_depth,
                                        mx ? row_exponents : NUMKONG_NULL, row_panel);
                for (nk_size_t tile = 0; tile < window_tiles; tile += 4) {
                    nk_size_t const column_first = window_first + tile * tile_dimension;
                    if (column_first + 4 * tile_dimension <= row_first) continue;
                    nk_size_t const quad_tiles = window_tiles - tile < 4 ? window_tiles - tile : 4;
                    nk_u16_t const *b_quad[4];
                    for (nk_size_t index = 0; index < 4; ++index)
                        b_quad[index] = window[tile + (index < quad_tiles ? index : 0)];
                    svzero_za();
                    for (nk_size_t index = 0; index < quad_tiles && !first; ++index)
                        nk_sme_load_tile_(index, result, result_stride, rows, row_first,
                                          column_first + index * tile_dimension, count, &epilogue);
                    if (bf16_inputs)
                        nk_sme_mopa_1x4_bf16_(row_panel, b_quad[0], b_quad[1], b_quad[2], b_quad[3], steps);
                    else nk_sme_mopa_1x4_f16_(row_panel, b_quad[0], b_quad[1], b_quad[2], b_quad[3], steps);
                    for (nk_size_t index = 0; index < quad_tiles; ++index)
                        nk_sme_store_tile_(index, 1, result, result_stride, rows, row_first,
                                           column_first + index * tile_dimension, count, last, &epilogue);
                }
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for 8-bit-panel inputs, with the windows of
 *  @c nk_dots_symmetric_b16_sme_streaming_. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_dots_symmetric_b8_sme_streaming_( //
    nk_dtype_t dtype, void const *vectors, nk_size_t stride, nk_size_t count, nk_size_t depth, void *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const row_end = row_start + row_count < count ? row_start + row_count : count;
    if (row_start >= row_end) return;
    int const unsigned_inputs = dtype == nk_u8_k || dtype == nk_u4_k;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 4;
    nk_size_t const window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_align_(64) nk_u8_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k];
    nk_align_(64) nk_u8_t row_panel[nk_sme_window_panel_bytes_k];
    nk_sme_epilogue_t epilogue = {NUMKONG_NULL, NUMKONG_NULL, NUMKONG_NULL, 0, 0, 1};
    epilogue.integer_factor = dtype == nk_e2m3_k ? 1.0f / 256 : dtype == nk_e2m1_k ? 0.25f : 0;
    for (nk_size_t window_first = row_start / tile_dimension * tile_dimension; window_first < count;
         window_first += window_width) {
        nk_size_t const window_columns = count - window_first < window_width ? count - window_first : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = row_end < window_first + window_columns ? row_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_sme_panel_steps_b8_(dtype, chunk_depth);
            int const first = depth_first == 0, last = depth_first + chunk_depth >= depth;
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_sme_stage_panel_b8_(dtype, vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                       chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = row_start; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = row_end - row_first < tile_dimension ? row_end - row_first : tile_dimension;
                nk_sme_stage_panel_b8_(dtype, vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                for (nk_size_t tile = 0; tile < window_tiles; tile += 4) {
                    nk_size_t const column_first = window_first + tile * tile_dimension;
                    if (column_first + 4 * tile_dimension <= row_first) continue;
                    nk_size_t const quad_tiles = window_tiles - tile < 4 ? window_tiles - tile : 4;
                    nk_u8_t const *b_quad[4];
                    for (nk_size_t index = 0; index < 4; ++index)
                        b_quad[index] = window[tile + (index < quad_tiles ? index : 0)];
                    svzero_za();
                    for (nk_size_t index = 0; index < quad_tiles && !first; ++index)
                        nk_sme_load_tile_(index, result, result_stride, rows, row_first,
                                          column_first + index * tile_dimension, count, &epilogue);
                    nk_sme_mopa_1x4_b8_(unsigned_inputs, row_panel, b_quad[0], b_quad[1], b_quad[2], b_quad[3], steps);
                    for (nk_size_t index = 0; index < quad_tiles; ++index)
                        nk_sme_store_tile_(index, 0, result, result_stride, rows, row_first,
                                           column_first + index * tile_dimension, count, last, &epilogue);
                }
            }
        }
    }
}

#pragma endregion Symmetric Kernels

#pragma region Block Scaled

/** The exact value of element @p index of packed column @p column, read back from a block-scaled
 *  SME pack as its FP4 code or folded bf16 value times the block scale or column exponent. */
NUMKONG_INLINE nk_f64_t nk_dots_scaled_sme_packed_value_(nk_dtype_t dtype, void const *b_packed, nk_size_t column,
                                                         nk_size_t index) NUMKONG_STREAMABLE_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const tile_dimension = header->svl_bytes / sizeof(nk_f32_t), vector_elements = tile_dimension * 2;
    nk_size_t const depth = header->depth, steps = header->depth_tile_count;
    nk_block_scaled_format_t const format = nk_block_scaled_format_of_dtype(dtype);
    nk_size_t const blocks = depth / format.block_size;
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_sme_layout_(dtype, header->columns, depth,
                                                                          tile_dimension);
    nk_u8_t const *pack = (nk_u8_t const *)b_packed;
    nk_size_t const slot = ((column / tile_dimension) * steps + index / 2) * vector_elements +
                           (column % tile_dimension) * 2 + (index & 1);
    nk_f64_t const scale = nk_block_scaled_decode_scale_serial_(
        pack[layout.scales + column * blocks + index / format.block_size], format.scale_dtype);
    if (dtype == nk_nvfp4_k || dtype == nk_mxfp4_k) {
        nk_u8_t const pair = (nk_u8_t)(pack[layout.values + slot] << 4); // element 0 reads the high nibble
        return nk_cross_scaled_element_serial_(nk_e2m1_k, &pair, 0) * scale;
    }
    nk_u16_t bits;
    nk_copy_bytes_(&bits, pack + layout.values + slot * 2, sizeof(bits));
    nk_fui32_t value;
    value.u = (nk_u32_t)bits << 16;
    nk_i8_t const exponent = ((nk_i8_t const *)(pack + layout.exponents))[column];
    if (exponent == -128) return (nk_f64_t)value.f * scale;
    nk_f64_t factor = 1;
    for (nk_i32_t step = 0; step < (exponent < 0 ? -exponent : exponent); ++step) factor *= 2;
    return exponent < 0 ? (nk_f64_t)value.f / factor : (nk_f64_t)value.f * factor;
}

/** Exact F64 dot of block-scaled row @p row of @p a with an element source over @p depth dims. */
NUMKONG_INLINE nk_f64_t nk_dots_scaled_sme_exact_row_(nk_dtype_t dtype, nk_cross_operand_t a, nk_size_t a_stride,
                                                      nk_size_t row, void const *b_packed, nk_cross_operand_t b,
                                                      nk_size_t b_stride, nk_size_t column,
                                                      nk_size_t depth) NUMKONG_STREAMABLE_ {
    nk_block_scaled_format_t const format = nk_block_scaled_format_of_dtype(dtype);
    nk_u8_t const *a_row = (nk_u8_t const *)a.elements + row * a_stride;
    nk_u8_t const *b_row = (nk_u8_t const *)b.elements + column * b_stride;
    nk_f64_t sum = 0;
    for (nk_size_t block = 0; block * format.block_size < depth; ++block) {
        nk_f64_t const a_scale = nk_block_scaled_decode_scale_serial_(a.scales[row * a.scales_stride + block],
                                                                      format.scale_dtype);
        nk_f64_t const b_scale = b_packed ? 1
                                          : nk_block_scaled_decode_scale_serial_(
                                                b.scales[column * b.scales_stride + block], format.scale_dtype);
        nk_f64_t block_sum = 0;
        for (nk_size_t index = block * format.block_size; index != (block + 1) * format.block_size; ++index) {
            nk_f64_t const b_value = b_packed ? nk_dots_scaled_sme_packed_value_(dtype, b_packed, column, index)
                                              : nk_cross_scaled_element_serial_(format.element_dtype, b_row, index);
            block_sum += nk_cross_scaled_element_serial_(format.element_dtype, a_row, index) * b_value;
        }
        sum += block_sum * a_scale * b_scale;
    }
    return sum;
}

/** Whether MX row @p row of @p operand spans more than @c nk_sme_exponent_spread_k binades. */
NUMKONG_INLINE int nk_dots_scaled_sme_wide_row_(nk_cross_operand_t operand, nk_size_t row, nk_size_t blocks) {
    nk_i32_t minimum, maximum;
    nk_sme_exponent_range_(operand.scales + row * operand.scales_stride, blocks, &minimum, &maximum);
    return maximum - minimum > nk_sme_exponent_spread_k;
}

/** Packed block-scaled dots on SME: the streaming kernel, then the exact scalar fallback over MX
 *  rows and packed columns wider than the fold covers. */
NUMKONG_INLINE nk_status_t nk_dots_scaled_packed_sme_(nk_dtype_t dtype, void const *a, void const *packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)packed;
    if (header->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_assert_(header->columns == columns && header->depth == depth);
    nk_cross_operand_t const operand = nk_cross_operand_(dtype, a, a_stride);
    nk_sme_start_streaming_();
    if (dtype == nk_nvfp4_k || dtype == nk_mxfp4_k)
        nk_dots_packed_fp4_sme_streaming_(dtype, operand, a_stride, packed, c, rows, columns, depth, c_stride);
    else nk_dots_packed_b16_sme_streaming_(dtype, operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    if (dtype == nk_nvfp4_k) return nk_success_k;
    nk_size_t const blocks = depth / 32;
    nk_size_t const tile_dimension = header->svl_bytes / sizeof(nk_f32_t);
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_sme_layout_(dtype, columns, depth, tile_dimension);
    nk_i8_t const *column_exponents = (nk_i8_t const *)packed + layout.exponents;
    nk_cross_operand_t const unpacked = {NUMKONG_NULL, NUMKONG_NULL, 0, NUMKONG_NULL};
    int wide_columns = 0;
    for (nk_size_t column = 0; column < columns; ++column) wide_columns |= column_exponents[column] == -128;
    for (nk_size_t row = 0; row < rows; ++row) {
        int const wide_row = nk_dots_scaled_sme_wide_row_(operand, row, blocks);
        if (!wide_row && !wide_columns) continue;
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        for (nk_size_t column = 0; column < columns; ++column)
            if (wide_row || column_exponents[column] == -128)
                c_row[column] = (nk_f32_t)nk_dots_scaled_sme_exact_row_(dtype, operand, a_stride, row, packed, unpacked,
                                                                        0, column, depth);
    }
    return nk_success_k;
}

/** Symmetric block-scaled dots on SME: the streaming kernel, then the exact scalar fallback over MX
 *  vectors wider than the fold covers. */
NUMKONG_INLINE nk_status_t nk_dots_scaled_symmetric_sme_(nk_dtype_t dtype, void const *vectors, nk_size_t count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count) {
    if (row_start >= count) return nk_success_k;
    row_count = nk_min_of_two(row_count, count - row_start);
    nk_cross_operand_t const operand = nk_cross_operand_(dtype, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(dtype, operand, stride, count, depth, result, result_stride, row_start,
                                         row_count);
    nk_sme_stop_streaming_();
    if (dtype == nk_nvfp4_k) return nk_success_k;
    nk_size_t const blocks = depth / 32;
    int wide_vectors = 0;
    for (nk_size_t vector = row_start; vector < count && !wide_vectors; ++vector)
        wide_vectors |= nk_dots_scaled_sme_wide_row_(operand, vector, blocks);
    if (!wide_vectors) return nk_success_k;
    for (nk_size_t row = row_start; row < row_start + row_count; ++row) {
        int const wide_row = nk_dots_scaled_sme_wide_row_(operand, row, blocks);
        nk_f32_t *result_row = (nk_f32_t *)((char *)result + row * result_stride);
        for (nk_size_t column = row; column < count; ++column)
            if (wide_row || nk_dots_scaled_sme_wide_row_(operand, column, blocks))
                result_row[column] = (nk_f32_t)nk_dots_scaled_sme_exact_row_(dtype, operand, stride, row, NUMKONG_NULL,
                                                                             operand, stride, column, depth);
    }
    return nk_success_k;
}

#pragma endregion Block Scaled

#if NUMKONG_TARGET_SME

/** Rejects packs another capability produced. */
NUMKONG_INLINE nk_status_t nk_dots_sme_check_(void const *b_packed) {
    return ((nk_dots_sme_packed_header_t const *)b_packed)->capability == nk_cap_sme_k ? nk_success_k
                                                                                       : nk_pack_mismatch_k;
}

/** Reads back the columns and depth an SME pack was made for. */
NUMKONG_INLINE nk_status_t nk_dots_sme_shape_(void const *b_packed, nk_size_t *columns, nk_size_t *depth) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

/** Packed dots of a 16-bit-panel @p dtype through the streaming kernel. */
NUMKONG_INLINE nk_status_t nk_dots_packed_b16_sme_(nk_dtype_t dtype, void const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride) {
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(dtype, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

/** Packed dots of an 8-bit-panel @p dtype through the streaming kernel. */
NUMKONG_INLINE nk_status_t nk_dots_packed_b8_sme_(nk_dtype_t dtype, void const *a, void const *b_packed, void *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride) {
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(dtype, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

/** Symmetric dots of a 16-bit-panel @p dtype through the streaming kernel. */
NUMKONG_INLINE nk_status_t nk_dots_symmetric_b16_sme_(nk_dtype_t dtype, void const *vectors, nk_size_t count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count) {
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(dtype, operand, stride, count, depth, result, result_stride, row_start,
                                         row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

/** Symmetric dots of an 8-bit-panel @p dtype through the streaming kernel. */
NUMKONG_INLINE nk_status_t nk_dots_symmetric_b8_sme_(nk_dtype_t dtype, void const *vectors, nk_size_t count,
                                                     nk_size_t depth, nk_size_t stride, void *result,
                                                     nk_size_t result_stride, nk_size_t row_start,
                                                     nk_size_t row_count) {
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(dtype, vectors, stride, count, depth, result, result_stride, row_start,
                                        row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma region F16 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_f16_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_f16_sme(nk_f16_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b16_sme_(nk_f16_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_f16_sme(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b16_sme_(nk_f16_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    return nk_dots_symmetric_b16_sme_(nk_f16_k, vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
}

NUMKONG_API nk_status_t nk_dots_pack_size_bf16_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_bf16_sme(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b16_sme_(nk_bf16_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_bf16_sme(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b16_sme_(nk_bf16_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    return nk_dots_symmetric_b16_sme_(nk_bf16_k, vectors, vectors_count, depth, stride, result, result_stride,
                                      row_start, row_count);
}

#pragma endregion F16 Floats

#pragma region I8 Integers

NUMKONG_API nk_status_t nk_dots_pack_size_i8_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(nk_i8_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_i8_sme(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b8_sme_(nk_i8_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_i8_sme(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b8_sme_(nk_i8_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    return nk_dots_symmetric_b8_sme_(nk_i8_k, vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
}

#pragma endregion I8 Integers

#pragma region E4M3 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e4m3_sme(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b16_sme_(nk_e4m3_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e4m3_sme(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b16_sme_(nk_e4m3_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    return nk_dots_symmetric_b16_sme_(nk_e4m3_k, vectors, vectors_count, depth, stride, result, result_stride,
                                      row_start, row_count);
}

#pragma endregion E4M3 Floats

#pragma region E5M2 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e5m2_sme(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b16_sme_(nk_e5m2_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e5m2_sme(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b16_sme_(nk_e5m2_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    return nk_dots_symmetric_b16_sme_(nk_e5m2_k, vectors, vectors_count, depth, stride, result, result_stride,
                                      row_start, row_count);
}

#pragma endregion E5M2 Floats

#pragma region E2M3 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(nk_e2m3_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e2m3_sme(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b8_sme_(nk_e2m3_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e2m3_sme(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b8_sme_(nk_e2m3_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    return nk_dots_symmetric_b8_sme_(nk_e2m3_k, vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
}

#pragma endregion E2M3 Floats

#pragma region E2M1 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(nk_e2m1_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e2m1_sme(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b8_sme_(nk_e2m1_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e2m1_sme(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b8_sme_(nk_e2m1_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_sme(nk_e2m1x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_symmetric_b8_sme_(nk_e2m1_k, vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
}

#pragma endregion E2M1 Floats

#pragma region E3M2 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e3m2_sme(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b16_sme_(nk_e3m2_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e3m2_sme(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b16_sme_(nk_e3m2_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    return nk_dots_symmetric_b16_sme_(nk_e3m2_k, vectors, vectors_count, depth, stride, result, result_stride,
                                      row_start, row_count);
}

#pragma endregion E3M2 Floats

#pragma region U8 Integers

NUMKONG_API nk_status_t nk_dots_pack_size_u8_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(nk_u8_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_u8_sme(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b8_sme_(nk_u8_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_u8_sme(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b8_sme_(nk_u8_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    return nk_dots_symmetric_b8_sme_(nk_u8_k, vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
}

#pragma endregion U8 Integers

/*  u4 and i4 packs keep each 4-bit dim as a byte, split per 8-dim word into a low-nibble vector and
 *  a high-nibble vector, so their panels and outer products match u8 and i8 at twice the steps. */

#pragma region U4 Integers

/** Packs 4-bit columns of @p dtype into split low and high nibble bytes, sign-extended for i4. */
NUMKONG_INLINE void nk_dots_pack_b4_sme_(nk_dtype_t dtype, void const *b, nk_size_t columns, nk_size_t depth,
                                         nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                         nk_size_t columns_end) {
    nk_size_t const tile_dimension = nk_sme_cntw_(), vector_bytes = nk_sme_cntb_();
    nk_size_t const words = nk_size_divide_round_up_(depth, 8), packed_depth = depth / NUMKONG_NIBBLES_PER_BYTE;
    nk_size_t const steps = words * 2;
    nk_size_t const norms_offset = sizeof(nk_dots_sme_packed_header_t) +
                                   nk_size_divide_round_up_(columns, tile_dimension) * steps * vector_bytes;
    if (columns_begin == 0) nk_dots_sme_header_(b_packed, columns, depth, steps, norms_offset, 1, tile_dimension);
    nk_size_t tile_begin, tile_end;
    nk_dots_sme_window_tiles_(columns, columns_begin, columns_end, tile_dimension, &tile_begin, &tile_end);
    nk_u8_t *tiles = (nk_u8_t *)b_packed + sizeof(nk_dots_sme_packed_header_t);
    for (nk_size_t index = tile_begin * steps * vector_bytes; index < tile_end * steps * vector_bytes; ++index)
        tiles[index] = 0;
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile)
        for (nk_size_t lane = 0; lane < tile_dimension && tile * tile_dimension + lane < columns; ++lane) {
            nk_u8_t const *source = (nk_u8_t const *)b + (tile * tile_dimension + lane) * b_stride;
            for (nk_size_t byte = 0; byte < packed_depth; ++byte) {
                nk_u8_t low = source[byte] & 0x0F, high = source[byte] >> 4;
                if (dtype == nk_i4_k) low = (nk_u8_t)((low ^ 8) - 8), high = (nk_u8_t)((high ^ 8) - 8);
                nk_u8_t *word = tiles + (tile * steps + byte / 4 * 2) * vector_bytes + lane * 4 + byte % 4;
                word[0] = low, word[vector_bytes] = high;
            }
        }
    nk_u32_t *norms = (nk_u32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t column = columns_begin; column < columns_end; ++column)
        norms[column] = nk_dots_sumsq_b8_sme_(dtype, (char const *)b + column * b_stride, depth);
}

NUMKONG_API nk_status_t nk_dots_pack_size_u4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(nk_u4_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_u4_sme(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b4_sme_(nk_u4_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_u4_sme(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b8_sme_(nk_u4_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}

#pragma endregion U4 Integers

#pragma region I4 Integers

NUMKONG_API nk_status_t nk_dots_pack_size_i4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(nk_i4_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_i4_sme(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_b4_sme_(nk_i4_k, b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_i4_sme(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_packed_b8_sme_(nk_i4_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    return nk_dots_symmetric_b8_sme_(nk_u4_k, vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
}
NUMKONG_API nk_status_t nk_dots_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    return nk_dots_symmetric_b8_sme_(nk_i4_k, vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
}

#pragma endregion I4 Integers

#pragma region Block Scaled Formats

/** Bytes of a block-scaled SME pack of @p columns × @p depth for @p dtype. */
NUMKONG_INLINE nk_size_t nk_dots_pack_size_scaled_sme_(nk_dtype_t dtype, nk_size_t columns, nk_size_t depth) {
    return nk_dots_scaled_sme_layout_(dtype, columns, depth, nk_sme_cntw_()).total;
}

/** Packs block-scaled columns of @p dtype from the API's reference @p b, rows @p b_stride apart. */
NUMKONG_INLINE void nk_dots_pack_scaled_from_reference_sme_(nk_dtype_t dtype, void const *b, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                            nk_size_t columns_begin, nk_size_t columns_end) {
    nk_dots_pack_scaled_sme_(dtype, nk_cross_operand_(dtype, b, b_stride), columns, depth, b_stride, b_packed,
                             columns_begin, columns_end);
}

NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_scaled_sme_(nk_nvfp4_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_sme(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_scaled_from_reference_sme_(nk_nvfp4_k, b, columns, depth, b_stride, b_packed, columns_begin,
                                            columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_packed_sme_(nk_nvfp4_k, a, packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_symmetric_sme_(nk_nvfp4_k, vectors, count, depth, stride, result, result_stride, row_start,
                                         row_count);
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_scaled_sme_(nk_mxfp4_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_sme(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_scaled_from_reference_sme_(nk_mxfp4_k, b, columns, depth, b_stride, b_packed, columns_begin,
                                            columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_packed_sme_(nk_mxfp4_k, a, packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_symmetric_sme_(nk_mxfp4_k, vectors, count, depth, stride, result, result_stride, row_start,
                                         row_count);
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_scaled_sme_(nk_mxfp6e2m3_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_scaled_from_reference_sme_(nk_mxfp6e2m3_k, b, columns, depth, b_stride, b_packed, columns_begin,
                                            columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_packed_sme_(nk_mxfp6e2m3_k, a, packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_symmetric_sme_(nk_mxfp6e2m3_k, vectors, count, depth, stride, result, result_stride,
                                         row_start, row_count);
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_scaled_sme_(nk_mxfp6e3m2_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_scaled_from_reference_sme_(nk_mxfp6e3m2_k, b, columns, depth, b_stride, b_packed, columns_begin,
                                            columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_packed_sme_(nk_mxfp6e3m2_k, a, packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_symmetric_sme_(nk_mxfp6e3m2_k, vectors, count, depth, stride, result, result_stride,
                                         row_start, row_count);
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_scaled_sme_(nk_mxfp8e4m3_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_scaled_from_reference_sme_(nk_mxfp8e4m3_k, b, columns, depth, b_stride, b_packed, columns_begin,
                                            columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_packed_sme_(nk_mxfp8e4m3_k, a, packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_symmetric_sme_(nk_mxfp8e4m3_k, vectors, count, depth, stride, result, result_stride,
                                         row_start, row_count);
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_scaled_sme_(nk_mxfp8e5m2_k, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_sme_shape_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_scaled_from_reference_sme_(nk_mxfp8e5m2_k, b, columns, depth, b_stride, b_packed, columns_begin,
                                            columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_packed_sme_(nk_mxfp8e5m2_k, a, packed, c, rows, columns, depth, a_stride, c_stride);
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_scaled_symmetric_sme_(nk_mxfp8e5m2_k, vectors, count, depth, stride, result, result_stride,
                                         row_start, row_count);
}

#pragma endregion Block Scaled Formats

#endif // NUMKONG_TARGET_SME

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_SME_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_DOTS_SME_H
