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
 *  Every kernel shares one shape and is spelled out per input type. A operand rows are decoded
 *  once per 16-row tile and depth chunk, transposed through ZA0 into a stack panel in MOPA operand
 *  order, and then all four ZA tiles accumulate: two panels against two column tiles for f16 and
 *  8-bit inputs, one panel against four column tiles for bf16 inputs, whose 2×2 form measured
 *  slower. Symmetric kernels decode windows of eight column tiles once per depth chunk and reuse
 *  them for every row tile. Partial sums leave ZA raw, and the conversions run over finished rows.
 *
 *  Block-scaled formats fold their scales into the operands. An NVFP4 element times its UE4M3 scale
 *  has at most 6 significant bits within 2⁻¹⁰ … 2688, an exact f16, so NVFP4 needs no guard. MX
 *  elements have at most 4 significant bits, so an element times 2^(e_block − base) is an exact
 *  bf16 while a row spans at most 32 binades; every row and column rebases to its largest block
 *  exponent, so the E2M1 fold is one saturating decrement. Kernels leave sums relative to the
 *  tensor scales and to 2^(E_row + E_column), and finishers apply them next to norms kept the same
 *  way, a squared norm being its relative sum times 4^E, so extreme scales neither overflow nor
 *  flush before the result does. Rows or columns spanning more take exact wide sums, rescaled into
 *  the same relative form.
 *  FP4 packs keep one byte per code and decode B with @c TBL inside the loop, which matches
 *  @c LUTI4 without requiring SME2, while decoding packed nibbles in the loop measured 15–20%
 *  slower; MXFP6 and MXFP8 packs store the folded BF16 values. Packs store each 16-column tile in
 *  MOPA operand order, 2 depth steps per 32-bit lane for 16-bit operands and 4 for 8-bit ones, as
 *  the A panels are. Depth chunks of 2048 dimensions for 16-bit operands and 4096 for 8-bit ones
 *  keep two panels within 128 KB of stack; longer rows spill their partial sums from ZA and reload
 *  them before the next chunk, and each chunk pays one pipeline drain per tile group.
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
#include "numkong/dots/serial.h" // `nk_cross_operand_t`, `nk_cross_scaled_exact_wide_mxfp4_serial_`
#include "numkong/reduce/sve.h"  // `nk_svaddv_f32_`

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
NUMKONG_INLINE nk_size_t nk_cntb_sme_(void) {
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
NUMKONG_INLINE nk_size_t nk_cnth_sme_(void) {
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
NUMKONG_INLINE nk_size_t nk_cntw_sme_(void) {
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
NUMKONG_INLINE nk_size_t nk_cntd_sme_(void) {
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
NUMKONG_INLINE void nk_start_sme_streaming_(void) {
    __asm__ __volatile__("smstart sm"
                         :
                         :
                         : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                           "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                           "v27", "v28", "v29", "v30", "v31", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
                           "p9", "p10", "p11", "p12", "p13", "p14", "p15", "memory");
}

/** Exit streaming SVE mode (PSTATE.SM = 0). Must pair with nk_start_sme_streaming_. */
NUMKONG_INLINE void nk_stop_sme_streaming_(void) {
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

    /** ZA0.H alone, which spans ZA0.S and ZA2.S. */
    nk_sme_zero_za16_tile_0_k = 0x55,

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

    /** Columns whose norms and exponents a symmetric block-scaled finisher keeps at a time. */
    nk_sme_finish_columns_k = 1024,
};

/** Clears the lanes of @p bound left of the diagonal on row @p row_index, for a tile starting at
 *  @p column_start. */
NUMKONG_INLINE svbool_t nk_diagonal_cut_b32x_sme_(svbool_t bound, nk_size_t column_start,
                                                  nk_size_t row_index) NUMKONG_STREAMING_ {
    return svbic_b_z(bound, bound, svwhilelt_b32_u64(column_start, row_index));
}

/** Clears the lanes of @p bound left of the diagonal on row @p row_index, for a tile starting at
 *  @p column_start. */
NUMKONG_INLINE svbool_t nk_diagonal_cut_b64x_sme_(svbool_t bound, nk_size_t column_start,
                                                  nk_size_t row_index) NUMKONG_STREAMING_ {
    return svbic_b_z(bound, bound, svwhilelt_b64_u64(column_start, row_index));
}

#pragma region Decoders

/** Up to 16 u16 entries of @p data as the vector pair @c svtbl2_u16 reads as entries 0 … 15 at
 *  every vector length: the first vector holds what fits and the second the rest, which only SVL
 *  128 needs. Indices past the entries read zero. */
NUMKONG_INLINE svuint16x2_t nk_table16_u16_sme_(nk_u16_t const *data) NUMKONG_STREAMING_ {
    nk_size_t const lanes = svcnth(), split = lanes < 16 ? lanes : 16;
    return svcreate2_u16(svld1_u16(svwhilelt_b16_u64(0, 16), data),
                         svld1_u16(svwhilelt_b16_u64(split, 16), data + split));
}

/** Up to 32 byte entries of @p data as the vector pair @c svtbl2_u8 reads as entries 0 … 31 at
 *  every vector length, like @c nk_table16_u16_sme_. */
NUMKONG_INLINE svuint8x2_t nk_table32_u8_sme_(nk_u8_t const *data) NUMKONG_STREAMING_ {
    nk_size_t const lanes = svcntb(), split = lanes < 32 ? lanes : 32;
    return svcreate2_u8(svld1_u8(svwhilelt_b8_u64(0, 32), data), svld1_u8(svwhilelt_b8_u64(split, 32), data + split));
}

/** E2M1 codes as F16 bits, for NVFP4. */
static nk_align_(64) nk_u16_t const nk_nvfp4_table_data_sme_[16] = {0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200,
                                                                    0x4400, 0x4600, 0x8000, 0xB800, 0xBC00, 0xBE00,
                                                                    0xC000, 0xC200, 0xC400, 0xC600};

/** E2M1 codes as BF16 bits with both zeros at +0, so a saturating exponent decrement keeps zeros
 *  at zero. */
static nk_align_(64) nk_u16_t const nk_mxfp4_table_data_sme_[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040,
                                                                    0x4080, 0x40C0, 0x0000, 0xBF00, 0xBF80, 0xBFC0,
                                                                    0xC000, 0xC040, 0xC080, 0xC0C0};

/** Bits of the mini-float magnitudes @p magnitude_u16x as @p base plus magnitude << @p shift,
 *  corrected by @p deltas_u16x at index (magnitude + @p rotation) mod 128, which puts the Inf and
 *  NaN magnitudes just below 128 first and the subnormals from zero after them. Indices past the
 *  table read zero, so no compare sits on the decode path; streaming integer instructions issue
 *  about once per cycle, which makes the instruction count the cost. */
NUMKONG_INLINE svuint16_t nk_minifloat_magnitudes_sme_streaming_(svuint16_t magnitude_u16x, nk_size_t shift,
                                                                 nk_u16_t base, nk_u16_t rotation,
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
 *  mantissa; @c nk_minifloat_magnitudes_sme_streaming_ patches subnormals and NaN without compares.
 *
 *  @param[in] predicate_b16x Active-lane predicate.
 *  @param[in] bytes_u8x Pre-loaded e4m3 bytes from @c svld1_u8.
 *  @return @c svfloat16_t with converted values, zero for inactive lanes.
 */
NUMKONG_INLINE svfloat16_t nk_e4m3x_to_f16x_sme_streaming_(svbool_t predicate_b16x,
                                                           svuint8_t bytes_u8x) NUMKONG_STREAMING_ {
    static nk_align_(64)
        nk_u16_t const deltas_data[16] = {0x1E80, 0xE000, 0xF780, 0xFB00, 0xFC80, 0xFE00, 0xFE80, 0xFF00, 0xFF80};
    svbool_t const all_b16x = svptrue_b16();
    svuint16_t const codes_u16x = svunpklo_u16(bytes_u8x);
    svuint16_t const bits_u16x = nk_minifloat_magnitudes_sme_streaming_(svand_n_u16_x(all_b16x, codes_u16x, 0x7F), 7,
                                                                        0x2000, 1, nk_table16_u16_sme_(deltas_data));
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
NUMKONG_INLINE svfloat16_t nk_e5m2x_to_f16x_sme_streaming_(svbool_t predicate_b16x,
                                                           svuint8_t bytes_u8x) NUMKONG_STREAMING_ {
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
NUMKONG_INLINE svint8_t nk_e2m3x_to_i8x_sme_streaming_(svbool_t predicate_b8x,
                                                       svuint8_t raw_bytes_u8x) NUMKONG_STREAMING_ {
    static nk_align_(64) nk_u8_t const lut_data[32] = {
        0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,  28,  30,  //
        32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, //
    };
    svuint8_t magnitude_u8x = svand_n_u8_x(predicate_b8x, raw_bytes_u8x, 0x1F);
    svuint8_t unsigned_value_u8x = svtbl2_u8(nk_table32_u8_sme_(lut_data), magnitude_u8x);
    svuint8_t sign_bits_u8x = svand_n_u8_x(predicate_b8x, raw_bytes_u8x, 0x20);
    svbool_t negate_mask_b8x = svcmpne_n_u8(predicate_b8x, sign_bits_u8x, 0);
    svint8_t positive_value_i8x = svreinterpret_s8_u8(unsigned_value_u8x);
    svint8_t negated_value_i8x = svneg_s8_x(predicate_b8x, positive_value_i8x);
    return svsel_s8(negate_mask_b8x, negated_value_i8x, positive_value_i8x);
}

/** Widens up to `svcntb()` E2M1 dimensions into doubled signed i8 lanes, zeroing lanes past
 *  @p dimensions. Even dimensions live in high nibbles. */
NUMKONG_INLINE svint8_t nk_e2m1x_to_i8x_sme_streaming_(nk_e2m1x2_t const *pairs,
                                                       nk_size_t dimensions) NUMKONG_STREAMING_ {
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
NUMKONG_INLINE svfloat16_t nk_e3m2x_to_f16x_sme_streaming_(svbool_t predicate_b16x,
                                                           svuint8_t bytes_u8x) NUMKONG_STREAMING_ {
    static nk_align_(64) nk_u8_t const magnitude_high_lut[32] = {
        0x00, 0x2C, 0x30, 0x32, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
        0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
    };
    svuint8_t const magnitude_high_u8x = svtbl2_u8(nk_table32_u8_sme_(magnitude_high_lut),
                                                   svand_n_u8_x(svptrue_b8(), bytes_u8x, 0x1F));
    svuint16_t const bits_u16x = svreinterpret_u16_u8(svzip1_u8(svdup_n_u8(0), magnitude_high_u8x));
    svuint16_t const signed_u16x = svbsl_n_u16(svlsl_n_u16_x(predicate_b16x, svunpklo_u16(bytes_u8x), 10), bits_u16x,
                                               0x8000);
    return svreinterpret_f16_u16(svsel_u16(predicate_b16x, signed_u16x, svdup_n_u16(0)));
}

/** BF16 bits of mini-float codes, one per 16-bit lane, with @p magnitude_mask magnitudes and the
 *  sign in bit @p sign_bit, decoded by @c nk_minifloat_magnitudes_sme_streaming_ from
 *  @p shift_bits, @p base, @p rotation and @p deltas_data. Every lane moves by @p shift, a multiple
 *  of 128 that scales by a power of two, except zero, Inf and NaN, whose @p fixed_entries
 *  corrections lead the table and so always sit in the first vector. Zero codes decode to zero, so
 *  lanes past a predicated load stay zero. */
NUMKONG_INLINE svuint16_t nk_minifloat_to_bf16x_sme_streaming_(svuint16_t codes_u16x, nk_u16_t magnitude_mask,
                                                               nk_size_t sign_bit, nk_size_t shift_bits, nk_u16_t base,
                                                               nk_u16_t rotation, nk_u16_t const *deltas_data,
                                                               nk_size_t fixed_entries,
                                                               nk_u16_t shift) NUMKONG_STREAMING_ {
    svbool_t const all_b16x = svptrue_b16();
    svuint16x2_t const table_u16x = nk_table16_u16_sme_(deltas_data);
    svuint16x2_t const deltas_u16x = svset2_u16(
        table_u16x, 0, svsub_n_u16_m(svwhilelt_b16_u64(0, fixed_entries), svget2_u16(table_u16x, 0), shift));
    svuint16_t const bits_u16x = nk_minifloat_magnitudes_sme_streaming_(
        svand_n_u16_x(all_b16x, codes_u16x, magnitude_mask), shift_bits, (nk_u16_t)(base + shift), rotation,
        deltas_u16x);
    return svbsl_n_u16(svlsl_n_u16_x(all_b16x, codes_u16x, 15 - sign_bit), bits_u16x, 0x8000);
}

/** BF16 bits of E4M3 bytes, every finite nonzero lane moved by @p shift. Exact: E4M3 values have
 *  3-bit mantissas within BF16's range. */
NUMKONG_INLINE svuint16_t nk_e4m3x_to_bf16x_sme_streaming_(svuint16_t codes_u16x, nk_u16_t shift) NUMKONG_STREAMING_ {
    // NaN at magnitude 127, zero, then the seven subnormals
    static nk_align_(64)
        nk_u16_t const deltas_data[16] = {0x3BD0, 0xC400, 0xFEF0, 0xFF60, 0xFF90, 0xFFC0, 0xFFD0, 0xFFE0, 0xFFF0};
    return nk_minifloat_to_bf16x_sme_streaming_(codes_u16x, 0x7F, 7, 4, 0x3C00, 1, deltas_data, 2, shift);
}

/** BF16 bits of E5M2 bytes, Inf and NaN kept, every finite nonzero lane moved by @p shift. */
NUMKONG_INLINE svuint16_t nk_e5m2x_to_bf16x_sme_streaming_(svuint16_t codes_u16x, nk_u16_t shift) NUMKONG_STREAMING_ {
    // Inf and three NaNs at magnitudes 124 … 127, zero, then the three subnormals
    static nk_align_(64)
        nk_u16_t const deltas_data[16] = {0x3800, 0x3820, 0x3800, 0x37E0, 0xC800, 0xFF60, 0xFFC0, 0xFFE0};
    return nk_minifloat_to_bf16x_sme_streaming_(codes_u16x, 0x7F, 7, 5, 0x3800, 4, deltas_data, 5, shift);
}

/** BF16 bits of E2M3 bytes, sign in bit 5, every nonzero lane moved by @p shift. */
NUMKONG_INLINE svuint16_t nk_e2m3x_to_bf16x_sme_streaming_(svuint16_t codes_u16x, nk_u16_t shift) NUMKONG_STREAMING_ {
    static nk_align_(64)
        nk_u16_t const deltas_data[16] = {0xC100, 0xFEF0, 0xFF60, 0xFF90, 0xFFC0, 0xFFD0, 0xFFE0, 0xFFF0};
    return nk_minifloat_to_bf16x_sme_streaming_(codes_u16x, 0x1F, 5, 4, 0x3F00, 0, deltas_data, 1, shift);
}

/** BF16 bits of E3M2 bytes, sign in bit 5, every nonzero lane moved by @p shift. */
NUMKONG_INLINE svuint16_t nk_e3m2x_to_bf16x_sme_streaming_(svuint16_t codes_u16x, nk_u16_t shift) NUMKONG_STREAMING_ {
    static nk_align_(64) nk_u16_t const deltas_data[16] = {0xC200, 0xFF60, 0xFFC0, 0xFFE0};
    return nk_minifloat_to_bf16x_sme_streaming_(codes_u16x, 0x1F, 5, 5, 0x3E00, 0, deltas_data, 1, shift);
}

/** BF16 bits of @p count E2M1 elements from dim @p first of one row of @p codes, moved by
 *  @p exponent_shift binades, never positive under largest-exponent rebasing: a saturating
 *  decrement that leaves the +0 entries alone. */
NUMKONG_INLINE svuint16_t nk_load_mx_e2m1_sme_(nk_u8_t const *codes, nk_size_t first, nk_size_t count,
                                               nk_i32_t exponent_shift) NUMKONG_STREAMING_ {
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t const pairs_u8x = svld1_u8(svwhilelt_b8_u64(0, count / 2), codes + first / 2);
    svuint8_t const nibbles_u8x = svzip1_u8(svlsr_n_u8_x(all_b8x, pairs_u8x, 4),
                                            svand_n_u8_x(all_b8x, pairs_u8x, 0x0F));
    return svqsub_n_u16(svtbl2_u16(nk_table16_u16_sme_(nk_mxfp4_table_data_sme_), svunpklo_u16(nibbles_u8x)),
                        (nk_u16_t)(-exponent_shift * 128));
}

/** BF16 bits of @p count E2M3 bytes from dim @p first of @p codes, moved by @p exponent_shift. */
NUMKONG_INLINE svuint16_t nk_load_mx_e2m3_sme_(nk_u8_t const *codes, nk_size_t first, nk_size_t count,
                                               nk_i32_t exponent_shift) NUMKONG_STREAMING_ {
    return nk_e2m3x_to_bf16x_sme_streaming_(svld1ub_u16(svwhilelt_b16_u64(0, count), codes + first),
                                            (nk_u16_t)(exponent_shift * 128));
}

/** BF16 bits of @p count E3M2 bytes from dim @p first of @p codes, moved by @p exponent_shift. */
NUMKONG_INLINE svuint16_t nk_load_mx_e3m2_sme_(nk_u8_t const *codes, nk_size_t first, nk_size_t count,
                                               nk_i32_t exponent_shift) NUMKONG_STREAMING_ {
    return nk_e3m2x_to_bf16x_sme_streaming_(svld1ub_u16(svwhilelt_b16_u64(0, count), codes + first),
                                            (nk_u16_t)(exponent_shift * 128));
}

/** BF16 bits of @p count E4M3 bytes from dim @p first of @p codes, moved by @p exponent_shift. */
NUMKONG_INLINE svuint16_t nk_load_mx_e4m3_sme_(nk_u8_t const *codes, nk_size_t first, nk_size_t count,
                                               nk_i32_t exponent_shift) NUMKONG_STREAMING_ {
    return nk_e4m3x_to_bf16x_sme_streaming_(svld1ub_u16(svwhilelt_b16_u64(0, count), codes + first),
                                            (nk_u16_t)(exponent_shift * 128));
}

/** BF16 bits of @p count E5M2 bytes from dim @p first of @p codes, moved by @p exponent_shift. */
NUMKONG_INLINE svuint16_t nk_load_mx_e5m2_sme_(nk_u8_t const *codes, nk_size_t first, nk_size_t count,
                                               nk_i32_t exponent_shift) NUMKONG_STREAMING_ {
    return nk_e5m2x_to_bf16x_sme_streaming_(svld1ub_u16(svwhilelt_b16_u64(0, count), codes + first),
                                            (nk_u16_t)(exponent_shift * 128));
}

/** The shift of an MX block with UE8M0 scale @p scale against the rebasing exponent @p base, none
 *  for @c nk_sme_unfolded_base_k. */
NUMKONG_INLINE nk_i32_t nk_mx_shift_sme_(nk_u8_t scale, nk_i32_t base) NUMKONG_STREAMABLE_ {
    return base == nk_sme_unfolded_base_k ? 0 : (nk_i32_t)scale - 127 - base;
}

/** @p values_u16x of one MX block of @p count dims with UE8M0 scale @p scale: zeros for code 0
 *  and NaNs for 255, which no shift reaches. */
NUMKONG_INLINE svuint16_t nk_mx_block_sme_(svuint16_t values_u16x, nk_u8_t scale, nk_size_t count) NUMKONG_STREAMING_ {
    if (scale == 0) return svdup_n_u16(0);
    if (scale == 255) return svsel_u16(svwhilelt_b16_u64(0, count), svdup_n_u16(0x7FC0), svdup_n_u16(0));
    return values_u16x;
}

/** The rebasing exponent of @p blocks UE8M0 scales, their largest finite exponent or 0 when none
 *  is, with in @p wide whether they span more than @c nk_sme_exponent_spread_k binades. */
NUMKONG_INLINE nk_i32_t nk_mx_base_sme_(nk_u8_t const *scales, nk_size_t blocks, int *wide) NUMKONG_STREAMABLE_ {
    nk_i32_t low = 255, high = 0;
    for (nk_size_t block = 0; block < blocks; ++block) {
        nk_i32_t const code = scales[block];
        if (code == 0 || code == 255) continue;
        low = code < low ? code : low, high = code > high ? code : high;
    }
    *wide = low <= high && high - low > nk_sme_exponent_spread_k;
    return low <= high ? high - 127 : 0;
}

/** The rebasing exponents of @p rows MX rows of @p operand from @p row_first. */
NUMKONG_INLINE void nk_row_bases_sme_(nk_cross_operand_t operand, nk_size_t row_first, nk_size_t rows, nk_size_t blocks,
                                      nk_i32_t *bases) NUMKONG_STREAMABLE_ {
    int wide;
    for (nk_size_t row = 0; row < rows; ++row)
        bases[row] = nk_mx_base_sme_(operand.scales + (row_first + row) * operand.scales_stride, blocks, &wide);
}

/** UE4M3 codes as F16 bits, a NaN for 127, read by broadcast loads. */
static nk_align_(64) nk_u16_t const nk_ue4m3_table_data_sme_[128] = {
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

/** F16 products of @p count NVFP4 elements from dim @p first of one row, at most 32, and their
 *  UE4M3 block scales, exact; zero for lanes past @p count. */
NUMKONG_INLINE svuint16_t nk_nvfp4x_to_f16x_sme_streaming_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t first,
                                                           nk_size_t count) NUMKONG_STREAMING_ {
    nk_u16_t const *scale_data = nk_ue4m3_table_data_sme_;
    svbool_t const predicate_b8x = svptrue_b8(), predicate_b16x = svptrue_b16();
    svuint8_t const pairs_u8x = svld1_u8(svwhilelt_b8_u64(0, count / 2), codes + first / 2);
    svuint8_t const nibbles_u8x = svzip1_u8(svlsr_n_u8_x(predicate_b8x, pairs_u8x, 4),
                                            svand_n_u8_x(predicate_b8x, pairs_u8x, 0x0F));
    svuint16_t const values_u16x = svtbl2_u16(nk_table16_u16_sme_(nk_nvfp4_table_data_sme_), svunpklo_u16(nibbles_u8x));
    nk_u8_t const *block_scales = scales + first / 16;
    svuint16_t const scales_u16x = svsel_u16(svwhilelt_b16_u64(0, 16), svdup_n_u16(scale_data[block_scales[0] & 0x7F]),
                                             svdup_n_u16(count > 16 ? scale_data[block_scales[1] & 0x7F] : 0));
    return svreinterpret_u16_f16(
        svmul_f16_x(predicate_b16x, svreinterpret_f16_u16(values_u16x), svreinterpret_f16_u16(scales_u16x)));
}

/*  Block-scaled row decoders: up to one vector of dims of the row of @p codes and @p scales from
 *  dim @p first, a multiple of 32, zero past @p count. Vectors past SVL 512 span several scale
 *  blocks, decoded 32 dims at a time and spliced. */

/** NVFP4 dims times their block scales as exact f16. */
NUMKONG_INLINE svuint16_t nk_decode_row_nvfp4_sme_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t first,
                                                   nk_size_t count) NUMKONG_STREAMING_ {
    svuint16_t row_u16x = svdup_n_u16(0);
    for (nk_size_t piece = 0; piece < count; piece += 32)
        row_u16x = svsplice_u16(
            svwhilelt_b16_u64(0, piece), row_u16x,
            nk_nvfp4x_to_f16x_sme_streaming_(codes, scales, first + piece, count - piece < 32 ? count - piece : 32));
    return row_u16x;
}

/** MXFP4 dims as bf16 rebased by @p base. */
NUMKONG_INLINE svuint16_t nk_decode_row_mxfp4_sme_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t first,
                                                   nk_size_t count, nk_i32_t base) NUMKONG_STREAMING_ {
    svuint16_t row_u16x = svdup_n_u16(0);
    for (nk_size_t piece = 0; piece < count; piece += 32) {
        nk_size_t const piece_count = count - piece < 32 ? count - piece : 32;
        nk_u8_t const scale = scales[(first + piece) / 32];
        svuint16_t const piece_u16x = nk_mx_block_sme_(
            nk_load_mx_e2m1_sme_(codes, first + piece, piece_count, nk_mx_shift_sme_(scale, base)), scale, piece_count);
        row_u16x = svsplice_u16(svwhilelt_b16_u64(0, piece), row_u16x, piece_u16x);
    }
    return row_u16x;
}

/** MXFP6 E2M3 dims as bf16 rebased by @p base. */
NUMKONG_INLINE svuint16_t nk_decode_row_mxfp6e2m3_sme_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t first,
                                                       nk_size_t count, nk_i32_t base) NUMKONG_STREAMING_ {
    svuint16_t row_u16x = svdup_n_u16(0);
    for (nk_size_t piece = 0; piece < count; piece += 32) {
        nk_size_t const piece_count = count - piece < 32 ? count - piece : 32;
        nk_u8_t const scale = scales[(first + piece) / 32];
        svuint16_t const piece_u16x = nk_mx_block_sme_(
            nk_load_mx_e2m3_sme_(codes, first + piece, piece_count, nk_mx_shift_sme_(scale, base)), scale, piece_count);
        row_u16x = svsplice_u16(svwhilelt_b16_u64(0, piece), row_u16x, piece_u16x);
    }
    return row_u16x;
}

/** MXFP6 E3M2 dims as bf16 rebased by @p base. */
NUMKONG_INLINE svuint16_t nk_decode_row_mxfp6e3m2_sme_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t first,
                                                       nk_size_t count, nk_i32_t base) NUMKONG_STREAMING_ {
    svuint16_t row_u16x = svdup_n_u16(0);
    for (nk_size_t piece = 0; piece < count; piece += 32) {
        nk_size_t const piece_count = count - piece < 32 ? count - piece : 32;
        nk_u8_t const scale = scales[(first + piece) / 32];
        svuint16_t const piece_u16x = nk_mx_block_sme_(
            nk_load_mx_e3m2_sme_(codes, first + piece, piece_count, nk_mx_shift_sme_(scale, base)), scale, piece_count);
        row_u16x = svsplice_u16(svwhilelt_b16_u64(0, piece), row_u16x, piece_u16x);
    }
    return row_u16x;
}

/** MXFP8 E4M3 dims as bf16 rebased by @p base. */
NUMKONG_INLINE svuint16_t nk_decode_row_mxfp8e4m3_sme_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t first,
                                                       nk_size_t count, nk_i32_t base) NUMKONG_STREAMING_ {
    svuint16_t row_u16x = svdup_n_u16(0);
    for (nk_size_t piece = 0; piece < count; piece += 32) {
        nk_size_t const piece_count = count - piece < 32 ? count - piece : 32;
        nk_u8_t const scale = scales[(first + piece) / 32];
        svuint16_t const piece_u16x = nk_mx_block_sme_(
            nk_load_mx_e4m3_sme_(codes, first + piece, piece_count, nk_mx_shift_sme_(scale, base)), scale, piece_count);
        row_u16x = svsplice_u16(svwhilelt_b16_u64(0, piece), row_u16x, piece_u16x);
    }
    return row_u16x;
}

/** MXFP8 E5M2 dims as bf16 rebased by @p base. */
NUMKONG_INLINE svuint16_t nk_decode_row_mxfp8e5m2_sme_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t first,
                                                       nk_size_t count, nk_i32_t base) NUMKONG_STREAMING_ {
    svuint16_t row_u16x = svdup_n_u16(0);
    for (nk_size_t piece = 0; piece < count; piece += 32) {
        nk_size_t const piece_count = count - piece < 32 ? count - piece : 32;
        nk_u8_t const scale = scales[(first + piece) / 32];
        svuint16_t const piece_u16x = nk_mx_block_sme_(
            nk_load_mx_e5m2_sme_(codes, first + piece, piece_count, nk_mx_shift_sme_(scale, base)), scale, piece_count);
        row_u16x = svsplice_u16(svwhilelt_b16_u64(0, piece), row_u16x, piece_u16x);
    }
    return row_u16x;
}

#pragma endregion Decoders

#pragma region Panels

/** Stores vertical slices [0, @p steps) of ZA0.S into @p panel from step @p step_first, one MOPA
 *  operand vector each: the transpose of the rows a stager wrote horizontally. */
NUMKONG_INLINE void nk_store_panel_steps_sme_(void *panel, nk_size_t step_first, nk_size_t steps) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const vector_bytes = svcntb();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t step = 0; step < steps; ++step)
        svst1_ver_za32(0, (uint32_t)step, predicate_all_b32x, (nk_u8_t *)panel + (step_first + step) * vector_bytes);
}

/*  16-bit stagers decode @p rows rows from @p row_first over @p depth dims from @p depth_first into
 *  @p panel in MOPA order, word i of step s holding row i's dims 2s and 2s + 1, through the ZA0
 *  transpose; rows past @p rows stay zero. MX rows rebase by @p bases, one per row. */

/** Stages raw f16 or bf16 rows. */
NUMKONG_OUTLINED_ void nk_stage_panel_b16_sme_(nk_u16_t const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                               nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                               nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svbool_t const predicate_b16x = svwhilelt_b16_u64(0, count);
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_u16_t const *row_values =
                (nk_u16_t const *)((nk_u8_t const *)rows_data + (row_first + row) * row_stride) + depth_first + dims;
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x,
                                   svreinterpret_u32_u16(svld1_u16(predicate_b16x, row_values)));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Stages E4M3 rows as f16. */
NUMKONG_OUTLINED_ void nk_stage_panel_e4m3_sme_(nk_e4m3_t const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                                nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                                nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svbool_t const predicate_b8x = svwhilelt_b8_u64(0, count), predicate_b16x = svwhilelt_b16_u64(0, count);
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_u8_t const *row_bytes = (nk_u8_t const *)rows_data + (row_first + row) * row_stride + depth_first + dims;
            svfloat16_t const row_f16x = nk_e4m3x_to_f16x_sme_streaming_(predicate_b16x,
                                                                         svld1_u8(predicate_b8x, row_bytes));
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_f16(row_f16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Stages E5M2 rows as f16. */
NUMKONG_OUTLINED_ void nk_stage_panel_e5m2_sme_(nk_e5m2_t const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                                nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                                nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svbool_t const predicate_b8x = svwhilelt_b8_u64(0, count), predicate_b16x = svwhilelt_b16_u64(0, count);
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_u8_t const *row_bytes = (nk_u8_t const *)rows_data + (row_first + row) * row_stride + depth_first + dims;
            svfloat16_t const row_f16x = nk_e5m2x_to_f16x_sme_streaming_(predicate_b16x,
                                                                         svld1_u8(predicate_b8x, row_bytes));
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_f16(row_f16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Stages E3M2 rows as f16. */
NUMKONG_OUTLINED_ void nk_stage_panel_e3m2_sme_(nk_e3m2_t const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                                nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                                nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svbool_t const predicate_b8x = svwhilelt_b8_u64(0, count), predicate_b16x = svwhilelt_b16_u64(0, count);
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_u8_t const *row_bytes = (nk_u8_t const *)rows_data + (row_first + row) * row_stride + depth_first + dims;
            svfloat16_t const row_f16x = nk_e3m2x_to_f16x_sme_streaming_(predicate_b16x,
                                                                         svld1_u8(predicate_b8x, row_bytes));
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_f16(row_f16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Stages NVFP4 rows times their block scales as f16. */
NUMKONG_OUTLINED_ void nk_stage_panel_nvfp4_sme_(nk_cross_operand_t operand, nk_size_t row_stride, nk_size_t row_first,
                                                 nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                                 nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_size_t const row_index = row_first + row;
            svuint16_t const row_u16x = nk_decode_row_nvfp4_sme_(
                (nk_u8_t const *)operand.elements + row_index * row_stride,
                operand.scales + row_index * operand.scales_stride, depth_first + dims, count);
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_u16(row_u16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Stages MXFP4 rows as rebased bf16. */
NUMKONG_OUTLINED_ void nk_stage_panel_mxfp4_sme_(nk_cross_operand_t operand, nk_size_t row_stride, nk_size_t row_first,
                                                 nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                                 nk_i32_t const *bases, nk_u16_t *panel) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_size_t const row_index = row_first + row;
            svuint16_t const row_u16x = nk_decode_row_mxfp4_sme_(
                (nk_u8_t const *)operand.elements + row_index * row_stride,
                operand.scales + row_index * operand.scales_stride, depth_first + dims, count, bases[row]);
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_u16(row_u16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Stages MXFP6 E2M3 rows as rebased bf16. */
NUMKONG_OUTLINED_ void nk_stage_panel_mxfp6e2m3_sme_(nk_cross_operand_t operand, nk_size_t row_stride,
                                                     nk_size_t row_first, nk_size_t rows, nk_size_t depth_first,
                                                     nk_size_t depth, nk_i32_t const *bases,
                                                     nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_size_t const row_index = row_first + row;
            svuint16_t const row_u16x = nk_decode_row_mxfp6e2m3_sme_(
                (nk_u8_t const *)operand.elements + row_index * row_stride,
                operand.scales + row_index * operand.scales_stride, depth_first + dims, count, bases[row]);
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_u16(row_u16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Stages MXFP6 E3M2 rows as rebased bf16. */
NUMKONG_OUTLINED_ void nk_stage_panel_mxfp6e3m2_sme_(nk_cross_operand_t operand, nk_size_t row_stride,
                                                     nk_size_t row_first, nk_size_t rows, nk_size_t depth_first,
                                                     nk_size_t depth, nk_i32_t const *bases,
                                                     nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_size_t const row_index = row_first + row;
            svuint16_t const row_u16x = nk_decode_row_mxfp6e3m2_sme_(
                (nk_u8_t const *)operand.elements + row_index * row_stride,
                operand.scales + row_index * operand.scales_stride, depth_first + dims, count, bases[row]);
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_u16(row_u16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Stages MXFP8 E4M3 rows as rebased bf16. */
NUMKONG_OUTLINED_ void nk_stage_panel_mxfp8e4m3_sme_(nk_cross_operand_t operand, nk_size_t row_stride,
                                                     nk_size_t row_first, nk_size_t rows, nk_size_t depth_first,
                                                     nk_size_t depth, nk_i32_t const *bases,
                                                     nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_size_t const row_index = row_first + row;
            svuint16_t const row_u16x = nk_decode_row_mxfp8e4m3_sme_(
                (nk_u8_t const *)operand.elements + row_index * row_stride,
                operand.scales + row_index * operand.scales_stride, depth_first + dims, count, bases[row]);
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_u16(row_u16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Stages MXFP8 E5M2 rows as rebased bf16. */
NUMKONG_OUTLINED_ void nk_stage_panel_mxfp8e5m2_sme_(nk_cross_operand_t operand, nk_size_t row_stride,
                                                     nk_size_t row_first, nk_size_t rows, nk_size_t depth_first,
                                                     nk_size_t depth, nk_i32_t const *bases,
                                                     nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_size_t const row_index = row_first + row;
            svuint16_t const row_u16x = nk_decode_row_mxfp8e5m2_sme_(
                (nk_u8_t const *)operand.elements + row_index * row_stride,
                operand.scales + row_index * operand.scales_stride, depth_first + dims, count, bases[row]);
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_u16(row_u16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Steps of 4 dims an 8-bit panel spans over @p depth dims in words of @p dims_per_word: 4 for
 *  bytes, 8 for i4 and u4, which split each word into a low-nibble step and a high-nibble step. */
NUMKONG_INLINE nk_size_t nk_panel_steps_b8_sme_(nk_size_t dims_per_word, nk_size_t depth) NUMKONG_STREAMABLE_ {
    return nk_size_divide_round_up_(depth, dims_per_word) * (dims_per_word / 4);
}

/*  8-bit stagers decode rows like the 16-bit ones into the 8-bit @p panel, word i of step s holding
 *  row i's four bytes of that step. */

/** Stages raw i8 or u8 rows. */
NUMKONG_OUTLINED_ void nk_stage_panel_b8_sme_(void const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                              nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                              nk_u8_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_bytes = svcntb();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_bytes) {
        nk_size_t const count = depth - dims < vector_bytes ? depth - dims : vector_bytes;
        svbool_t const predicate_b8x = svwhilelt_b8_u64(0, count);
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_u8_t const *row_bytes = (nk_u8_t const *)rows_data + (row_first + row) * row_stride + depth_first + dims;
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x,
                                   svreinterpret_u32_u8(svld1_u8(predicate_b8x, row_bytes)));
        }
        nk_store_panel_steps_sme_(panel, dims / 4, nk_size_divide_round_up_(count, 4));
    }
}

/** Stages E2M3 rows as sixteen times their values in i8. */
NUMKONG_OUTLINED_ void nk_stage_panel_e2m3_sme_(nk_e2m3_t const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                                nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                                nk_u8_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_bytes = svcntb();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_bytes) {
        nk_size_t const count = depth - dims < vector_bytes ? depth - dims : vector_bytes;
        svbool_t const predicate_b8x = svwhilelt_b8_u64(0, count);
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_u8_t const *row_bytes = (nk_u8_t const *)rows_data + (row_first + row) * row_stride + depth_first + dims;
            svint8_t const row_i8x = nk_e2m3x_to_i8x_sme_streaming_(predicate_b8x, svld1_u8(predicate_b8x, row_bytes));
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_s8(row_i8x));
        }
        nk_store_panel_steps_sme_(panel, dims / 4, nk_size_divide_round_up_(count, 4));
    }
}

/** Stages E2M1 rows as twice their values in i8. */
NUMKONG_OUTLINED_ void nk_stage_panel_e2m1_sme_(nk_e2m1x2_t const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                                nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                                nk_u8_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_bytes = svcntb();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_bytes) {
        nk_size_t const count = depth - dims < vector_bytes ? depth - dims : vector_bytes;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_u8_t const *row_bytes = (nk_u8_t const *)rows_data + (row_first + row) * row_stride;
            svint8_t const row_i8x = nk_e2m1x_to_i8x_sme_streaming_(
                (nk_e2m1x2_t const *)(row_bytes + (depth_first + dims) / 2), count);
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_s8(row_i8x));
        }
        nk_store_panel_steps_sme_(panel, dims / 4, nk_size_divide_round_up_(count, 4));
    }
}

/** Writes into ZA0.S rows of @p rows rows of nibble pairs from @p row_first, @p count dims from
 *  dim @p first, ready for a nibble stager to read 8-dim words vertically. */
NUMKONG_INLINE void nk_write_nibble_rows_sme_(void const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                              nk_size_t rows, nk_size_t first, nk_size_t count) NUMKONG_STREAMING_
    __arm_inout("za") {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svbool_t const predicate_b8x = svwhilelt_b8_u64(0, nk_size_divide_round_up_(count, 2));
    svzero_mask_za(nk_sme_zero_za32_tile_0_k);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_u8_t const *row_bytes = (nk_u8_t const *)rows_data + (row_first + row) * row_stride + first / 2;
        svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x,
                               svreinterpret_u32_u8(svld1_u8(predicate_b8x, row_bytes)));
    }
}

/** Stages i4 rows as sign-extended nibble steps, each 8-dim word a low step and a high step. */
NUMKONG_OUTLINED_ void nk_stage_panel_i4_sme_(nk_i4x2_t const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                              nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                              nk_u8_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_bytes = svcntb(), batch_dims = 2 * vector_bytes;
    svbool_t const predicate_all_b8x = svptrue_b8(), predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += batch_dims) {
        nk_size_t const count = depth - dims < batch_dims ? depth - dims : batch_dims;
        nk_write_nibble_rows_sme_(rows_data, row_stride, row_first, rows, depth_first + dims, count);
        for (nk_size_t word = 0; word < nk_size_divide_round_up_(count, 8); ++word) {
            svint8_t const packed_i8x = svreinterpret_s8_u32(
                svread_ver_za32_u32_m(svdup_n_u32(0), predicate_all_b32x, 0, (uint32_t)word));
            nk_u8_t *steps = panel + (dims / 4 + 2 * word) * vector_bytes;
            svst1_s8(predicate_all_b8x, (nk_i8_t *)steps,
                     svasr_n_s8_x(predicate_all_b8x, svlsl_n_s8_x(predicate_all_b8x, packed_i8x, 4), 4));
            svst1_s8(predicate_all_b8x, (nk_i8_t *)steps + vector_bytes,
                     svasr_n_s8_x(predicate_all_b8x, packed_i8x, 4));
        }
    }
}

/** Stages u4 rows as zero-extended nibble steps, each 8-dim word a low step and a high step. */
NUMKONG_OUTLINED_ void nk_stage_panel_u4_sme_(nk_u4x2_t const *rows_data, nk_size_t row_stride, nk_size_t row_first,
                                              nk_size_t rows, nk_size_t depth_first, nk_size_t depth,
                                              nk_u8_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_bytes = svcntb(), batch_dims = 2 * vector_bytes;
    svbool_t const predicate_all_b8x = svptrue_b8(), predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += batch_dims) {
        nk_size_t const count = depth - dims < batch_dims ? depth - dims : batch_dims;
        nk_write_nibble_rows_sme_(rows_data, row_stride, row_first, rows, depth_first + dims, count);
        for (nk_size_t word = 0; word < nk_size_divide_round_up_(count, 8); ++word) {
            svuint8_t const packed_u8x = svreinterpret_u8_u32(
                svread_ver_za32_u32_m(svdup_n_u32(0), predicate_all_b32x, 0, (uint32_t)word));
            nk_u8_t *steps = panel + (dims / 4 + 2 * word) * vector_bytes;
            svst1_u8(predicate_all_b8x, steps, svand_n_u8_x(predicate_all_b8x, packed_u8x, 0x0F));
            svst1_u8(predicate_all_b8x, steps + vector_bytes, svlsr_n_u8_x(predicate_all_b8x, packed_u8x, 4));
        }
    }
}

#pragma endregion Panels

#pragma region Outer Products

/** Accumulates two f16 panels against two column tiles over @p steps: ZA0 = first × first,
 *  ZA1 = first × second, ZA2 = second × first, ZA3 = second × second. */
NUMKONG_INLINE void nk_mopa_2x2_f16_sme_(nk_u16_t const *a_first, nk_u16_t const *a_second, nk_u16_t const *b_first,
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
NUMKONG_INLINE void nk_mopa_1x4_f16_sme_(nk_u16_t const *a, nk_u16_t const *b_first, nk_u16_t const *b_second,
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
NUMKONG_INLINE void nk_mopa_1x4_bf16_sme_(nk_u16_t const *a, nk_u16_t const *b_first, nk_u16_t const *b_second,
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

/** Accumulates two i8 panels against two column tiles over @p steps, in the 2×2 tile order of
 *  @c nk_mopa_2x2_f16_sme_. */
NUMKONG_INLINE void nk_mopa_2x2_i8_sme_(nk_u8_t const *a_first, nk_u8_t const *a_second, nk_u8_t const *b_first,
                                        nk_u8_t const *b_second, nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b8x = svptrue_b8();
    nk_size_t const vector_bytes = svcntb();
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

/** Accumulates two u8 panels against two column tiles over @p steps, in the 2×2 tile order of
 *  @c nk_mopa_2x2_f16_sme_. */
NUMKONG_INLINE void nk_mopa_2x2_u8_sme_(nk_u8_t const *a_first, nk_u8_t const *a_second, nk_u8_t const *b_first,
                                        nk_u8_t const *b_second, nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b8x = svptrue_b8();
    nk_size_t const vector_bytes = svcntb();
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
}

/** Accumulates one i8 panel against four column tiles over @p steps into ZA0 to ZA3. */
NUMKONG_INLINE void nk_mopa_1x4_i8_sme_(nk_u8_t const *a, nk_u8_t const *b_first, nk_u8_t const *b_second,
                                        nk_u8_t const *b_third, nk_u8_t const *b_fourth,
                                        nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b8x = svptrue_b8();
    nk_size_t const vector_bytes = svcntb();
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

/** Accumulates one u8 panel against four column tiles over @p steps into ZA0 to ZA3. */
NUMKONG_INLINE void nk_mopa_1x4_u8_sme_(nk_u8_t const *a, nk_u8_t const *b_first, nk_u8_t const *b_second,
                                        nk_u8_t const *b_third, nk_u8_t const *b_fourth,
                                        nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b8x = svptrue_b8();
    nk_size_t const vector_bytes = svcntb();
    for (nk_size_t step = 0; step < steps; ++step) {
        nk_size_t const offset = step * vector_bytes;
        svuint8_t const a_u8x = svld1_u8(predicate_all_b8x, a + offset);
        svmopa_za32_u8_m(0, predicate_all_b8x, predicate_all_b8x, a_u8x, svld1_u8(predicate_all_b8x, b_first + offset));
        svmopa_za32_u8_m(1, predicate_all_b8x, predicate_all_b8x, a_u8x,
                         svld1_u8(predicate_all_b8x, b_second + offset));
        svmopa_za32_u8_m(2, predicate_all_b8x, predicate_all_b8x, a_u8x, svld1_u8(predicate_all_b8x, b_third + offset));
        svmopa_za32_u8_m(3, predicate_all_b8x, predicate_all_b8x, a_u8x,
                         svld1_u8(predicate_all_b8x, b_fourth + offset));
    }
}

/** One NVFP4 B vector: byte codes through @p table_u16x, times the per-column block scales. */
NUMKONG_INLINE svfloat16_t nk_nvfp4_b_vector_sme_(svuint16x2_t table_u16x, nk_u8_t const *codes,
                                                  svfloat16_t scales_f16x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b16x = svptrue_b16();
    return svmul_f16_x(predicate_all_b16x,
                       svreinterpret_f16_u16(svtbl2_u16(table_u16x, svld1ub_u16(predicate_all_b16x, codes))),
                       scales_f16x);
}

/** One MXFP4 B vector: the byte codes mapped through @p table_u16x to BF16 bits, less the
 *  per-column block decrements, which saturate zero blocks to zero. */
NUMKONG_INLINE svbfloat16_t nk_mxfp4_b_vector_sme_(svuint16x2_t table_u16x, nk_u8_t const *codes,
                                                   svuint16_t decrements_u16x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b16x = svptrue_b16();
    return svreinterpret_bf16_u16(
        svqsub_u16(svtbl2_u16(table_u16x, svld1ub_u16(predicate_all_b16x, codes)), decrements_u16x));
}

/** Accumulates two f16 NVFP4 A panels against two NVFP4 column tiles of byte codes and per-block
 *  f16 scale vectors, eight steps per block, in the 2×2 tile order of @c nk_mopa_2x2_f16_sme_. */
NUMKONG_INLINE void nk_mopa_2x2_nvfp4_sme_(nk_u16_t const *a_first, nk_u16_t const *a_second,
                                           nk_u8_t const *b_first_codes, nk_u8_t const *b_second_codes,
                                           nk_u16_t const *b_first_scales, nk_u16_t const *b_second_scales,
                                           nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    svuint16x2_t const table_u16x = nk_table16_u16_sme_(nk_nvfp4_table_data_sme_);
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
            svfloat16_t const b_first_f16x = nk_nvfp4_b_vector_sme_(table_u16x, b_first_codes + offset,
                                                                    first_scales_f16x);
            svfloat16_t const b_second_f16x = nk_nvfp4_b_vector_sme_(table_u16x, b_second_codes + offset,
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
 *  @c nk_mopa_2x2_f16_sme_. */
NUMKONG_INLINE void nk_mopa_2x2_mxfp4_sme_(nk_u16_t const *a_first, nk_u16_t const *a_second,
                                           nk_u8_t const *b_first_codes, nk_u8_t const *b_second_codes,
                                           nk_u16_t const *b_first_decrements, nk_u16_t const *b_second_decrements,
                                           nk_size_t steps) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    svuint16x2_t const table_u16x = nk_table16_u16_sme_(nk_mxfp4_table_data_sme_);
    for (nk_size_t block_first = 0; block_first < steps; block_first += 16) {
        nk_size_t const block_offset = block_first / 16 * vector_elements;
        svuint16_t const first_decrements_u16x = svld1_u16(predicate_all_b16x, b_first_decrements + block_offset);
        svuint16_t const second_decrements_u16x = svld1_u16(predicate_all_b16x, b_second_decrements + block_offset);
        nk_size_t const block_end = steps - block_first < 16 ? steps : block_first + 16;
        for (nk_size_t step = block_first; step < block_end; ++step) {
            nk_size_t const offset = step * vector_elements;
            svbfloat16_t const a_first_bf16x = svld1_bf16(predicate_all_b16x, (bfloat16_t const *)a_first + offset);
            svbfloat16_t const a_second_bf16x = svld1_bf16(predicate_all_b16x, (bfloat16_t const *)a_second + offset);
            svbfloat16_t const b_first_bf16x = nk_mxfp4_b_vector_sme_(table_u16x, b_first_codes + offset,
                                                                      first_decrements_u16x);
            svbfloat16_t const b_second_bf16x = nk_mxfp4_b_vector_sme_(table_u16x, b_second_codes + offset,
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
NUMKONG_INLINE void nk_mopa_1x4_nvfp4_sme_(nk_u16_t const *a, nk_u8_t const *const *b_codes,
                                           nk_u16_t const *const *b_scales, nk_size_t steps) NUMKONG_STREAMING_
    __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    svuint16x2_t const table_u16x = nk_table16_u16_sme_(nk_nvfp4_table_data_sme_);
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
                              nk_nvfp4_b_vector_sme_(table_u16x, b_codes[0] + offset, first_scales_f16x));
            svmopa_za32_f16_m(1, predicate_all_b16x, predicate_all_b16x, a_f16x,
                              nk_nvfp4_b_vector_sme_(table_u16x, b_codes[1] + offset, second_scales_f16x));
            svmopa_za32_f16_m(2, predicate_all_b16x, predicate_all_b16x, a_f16x,
                              nk_nvfp4_b_vector_sme_(table_u16x, b_codes[2] + offset, third_scales_f16x));
            svmopa_za32_f16_m(3, predicate_all_b16x, predicate_all_b16x, a_f16x,
                              nk_nvfp4_b_vector_sme_(table_u16x, b_codes[3] + offset, fourth_scales_f16x));
        }
    }
}

/** Accumulates one rebased bf16 MXFP4 A panel against four MXFP4 column tiles into ZA0 to ZA3, for
 *  the final 16-row strip. */
NUMKONG_INLINE void nk_mopa_1x4_mxfp4_sme_(nk_u16_t const *a, nk_u8_t const *const *b_codes,
                                           nk_u16_t const *const *b_decrements, nk_size_t steps) NUMKONG_STREAMING_
    __arm_inout("za") {
    svbool_t const predicate_all_b16x = svptrue_b16();
    nk_size_t const vector_elements = svcnth();
    svuint16x2_t const table_u16x = nk_table16_u16_sme_(nk_mxfp4_table_data_sme_);
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
                               nk_mxfp4_b_vector_sme_(table_u16x, b_codes[0] + offset, first_decrements_u16x));
            svmopa_za32_bf16_m(1, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                               nk_mxfp4_b_vector_sme_(table_u16x, b_codes[1] + offset, second_decrements_u16x));
            svmopa_za32_bf16_m(2, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                               nk_mxfp4_b_vector_sme_(table_u16x, b_codes[2] + offset, third_decrements_u16x));
            svmopa_za32_bf16_m(3, predicate_all_b16x, predicate_all_b16x, a_bf16x,
                               nk_mxfp4_b_vector_sme_(table_u16x, b_codes[3] + offset, fourth_decrements_u16x));
        }
    }
}

#pragma endregion Outer Products

#pragma region Tile Sweeps

/** Stores one horizontal slice of ZA tile @p tile, which must fold to a constant after inlining. */
NUMKONG_INLINE void nk_store_tile_row_sme_(nk_size_t tile, nk_size_t row, svbool_t predicate_b32x,
                                           void *target) NUMKONG_STREAMING_ __arm_inout("za") {
    switch (tile) {
    case 0: svst1_hor_za32(0, (uint32_t)row, predicate_b32x, target); break;
    case 1: svst1_hor_za32(1, (uint32_t)row, predicate_b32x, target); break;
    case 2: svst1_hor_za32(2, (uint32_t)row, predicate_b32x, target); break;
    default: svst1_hor_za32(3, (uint32_t)row, predicate_b32x, target); break;
    }
}

/** Loads a horizontal ZA slice of @p tile, zeroing inactive lanes; @p tile must be a constant. */
NUMKONG_INLINE void nk_load_tile_row_sme_(nk_size_t tile, nk_size_t row, svbool_t predicate_b32x,
                                          void const *source) NUMKONG_STREAMING_ __arm_inout("za") {
    switch (tile) {
    case 0: svld1_hor_za32(0, (uint32_t)row, predicate_b32x, source); break;
    case 1: svld1_hor_za32(1, (uint32_t)row, predicate_b32x, source); break;
    case 2: svld1_hor_za32(2, (uint32_t)row, predicate_b32x, source); break;
    default: svld1_hor_za32(3, (uint32_t)row, predicate_b32x, source); break;
    }
}

/** Loads into ZA tile @p tile the raw 32-bit partials an earlier chunk stored at rows from
 *  @p row_first and columns from @p column_first, lanes limited to @p columns and, when @p upper,
 *  to the diagonal on, so that this chunk accumulates on top of them without vector arithmetic. */
NUMKONG_INLINE void nk_load_tile_sme_(nk_size_t tile, void const *c, nk_size_t c_stride, nk_size_t rows,
                                      nk_size_t row_first, nk_size_t column_first, nk_size_t columns,
                                      int upper) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const columns_b32x = svwhilelt_b32_u64(column_first, columns);
    for (nk_size_t row = 0; row < rows; ++row) {
        svbool_t const load_b32x = upper ? nk_diagonal_cut_b32x_sme_(columns_b32x, column_first, row_first + row)
                                         : columns_b32x;
        nk_load_tile_row_sme_(tile, row, load_b32x,
                              (nk_u32_t const *)((char const *)c + (row_first + row) * c_stride) + column_first);
    }
}

/** Stores rows [0, @p rows) of ZA tile @p tile raw, as @c nk_load_tile_sme_ reads them back. */
NUMKONG_INLINE void nk_store_tile_sme_(nk_size_t tile, void *c, nk_size_t c_stride, nk_size_t rows, nk_size_t row_first,
                                       nk_size_t column_first, nk_size_t columns, int upper) NUMKONG_STREAMING_
    __arm_inout("za") {
    svbool_t const columns_b32x = svwhilelt_b32_u64(column_first, columns);
    for (nk_size_t row = 0; row < rows; ++row) {
        svbool_t const store_b32x = upper ? nk_diagonal_cut_b32x_sme_(columns_b32x, column_first, row_first + row)
                                          : columns_b32x;
        nk_store_tile_row_sme_(tile, row, store_b32x,
                               (nk_u32_t *)((char *)c + (row_first + row) * c_stride) + column_first);
    }
}

/** Loads the partials of the 2×2 group at @p column_first: ZA0 and ZA1 over @p first_rows rows from
 *  @p row_first, ZA2 and ZA3 over the @p second_rows after them. */
NUMKONG_INLINE void nk_load_tiles_2x2_sme_(void const *c, nk_size_t c_stride, nk_size_t row_first, nk_size_t first_rows,
                                           nk_size_t second_rows, nk_size_t column_first,
                                           nk_size_t columns) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const second_column = column_first + svcntw();
    nk_load_tile_sme_(0, c, c_stride, first_rows, row_first, column_first, columns, 0);
    nk_load_tile_sme_(1, c, c_stride, first_rows, row_first, second_column, columns, 0);
    nk_load_tile_sme_(2, c, c_stride, second_rows, row_first + first_rows, column_first, columns, 0);
    nk_load_tile_sme_(3, c, c_stride, second_rows, row_first + first_rows, second_column, columns, 0);
}

/** Stores the 2×2 group @c nk_load_tiles_2x2_sme_ loads. */
NUMKONG_INLINE void nk_store_tiles_2x2_sme_(void *c, nk_size_t c_stride, nk_size_t row_first, nk_size_t first_rows,
                                            nk_size_t second_rows, nk_size_t column_first,
                                            nk_size_t columns) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const second_column = column_first + svcntw();
    nk_store_tile_sme_(0, c, c_stride, first_rows, row_first, column_first, columns, 0);
    nk_store_tile_sme_(1, c, c_stride, first_rows, row_first, second_column, columns, 0);
    nk_store_tile_sme_(2, c, c_stride, second_rows, row_first + first_rows, column_first, columns, 0);
    nk_store_tile_sme_(3, c, c_stride, second_rows, row_first + first_rows, second_column, columns, 0);
}

/** Loads the partials of the 1×4 group of @p rows rows from @p row_first at @p column_first into
 *  ZA0 to ZA3; tiles past @p columns load nothing. */
NUMKONG_INLINE void nk_load_tiles_1x4_sme_(void const *c, nk_size_t c_stride, nk_size_t rows, nk_size_t row_first,
                                           nk_size_t column_first, nk_size_t columns, int upper) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    nk_load_tile_sme_(0, c, c_stride, rows, row_first, column_first, columns, upper);
    nk_load_tile_sme_(1, c, c_stride, rows, row_first, column_first + tile_dimension, columns, upper);
    nk_load_tile_sme_(2, c, c_stride, rows, row_first, column_first + 2 * tile_dimension, columns, upper);
    nk_load_tile_sme_(3, c, c_stride, rows, row_first, column_first + 3 * tile_dimension, columns, upper);
}

/** Stores the 1×4 group @c nk_load_tiles_1x4_sme_ loads. */
NUMKONG_INLINE void nk_store_tiles_1x4_sme_(void *c, nk_size_t c_stride, nk_size_t rows, nk_size_t row_first,
                                            nk_size_t column_first, nk_size_t columns, int upper) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    nk_store_tile_sme_(0, c, c_stride, rows, row_first, column_first, columns, upper);
    nk_store_tile_sme_(1, c, c_stride, rows, row_first, column_first + tile_dimension, columns, upper);
    nk_store_tile_sme_(2, c, c_stride, rows, row_first, column_first + 2 * tile_dimension, columns, upper);
    nk_store_tile_sme_(3, c, c_stride, rows, row_first, column_first + 3 * tile_dimension, columns, upper);
}

/*  Sweeps accumulate staged A panels against @p tiles column tiles of B, @p tile_stride elements
 *  apart from @p b_tiles, over @p steps steps, on top of the raw partials in @p c unless @p first,
 *  and store raw partials back. 2×2 sweeps take two panels of @p first_rows and @p second_rows rows
 *  from @p row_first; 1×4 sweeps take one panel of @p rows rows against columns from
 *  @p column_first, skipping groups left of the diagonal and cutting lanes there when @p upper.
 *  Groups past the last tile reuse its operands into tiles no store reaches. */

/** Sweeps two f16 panels in 2×2 groups. */
NUMKONG_INLINE void nk_sweep_2x2_f16_sme_(nk_u16_t const *first_panel, nk_u16_t const *second_panel,
                                          nk_size_t first_rows, nk_size_t second_rows, nk_size_t row_first,
                                          nk_u16_t const *b_tiles, nk_size_t tile_stride, nk_size_t tiles,
                                          nk_size_t steps, int first, void *c, nk_size_t c_stride,
                                          nk_size_t columns) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 2) {
        nk_u16_t const *b_first = b_tiles + tile * tile_stride;
        nk_u16_t const *b_second = tile + 1 < tiles ? b_first + tile_stride : b_first;
        svzero_za();
        if (!first)
            nk_load_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
        nk_mopa_2x2_f16_sme_(first_panel, second_panel, b_first, b_second, steps);
        nk_store_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
    }
}

/** Sweeps two i8 panels in 2×2 groups. */
NUMKONG_INLINE void nk_sweep_2x2_i8_sme_(nk_u8_t const *first_panel, nk_u8_t const *second_panel, nk_size_t first_rows,
                                         nk_size_t second_rows, nk_size_t row_first, nk_u8_t const *b_tiles,
                                         nk_size_t tile_stride, nk_size_t tiles, nk_size_t steps, int first, void *c,
                                         nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 2) {
        nk_u8_t const *b_first = b_tiles + tile * tile_stride;
        nk_u8_t const *b_second = tile + 1 < tiles ? b_first + tile_stride : b_first;
        svzero_za();
        if (!first)
            nk_load_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
        nk_mopa_2x2_i8_sme_(first_panel, second_panel, b_first, b_second, steps);
        nk_store_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
    }
}

/** Sweeps two u8 panels in 2×2 groups. */
NUMKONG_INLINE void nk_sweep_2x2_u8_sme_(nk_u8_t const *first_panel, nk_u8_t const *second_panel, nk_size_t first_rows,
                                         nk_size_t second_rows, nk_size_t row_first, nk_u8_t const *b_tiles,
                                         nk_size_t tile_stride, nk_size_t tiles, nk_size_t steps, int first, void *c,
                                         nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 2) {
        nk_u8_t const *b_first = b_tiles + tile * tile_stride;
        nk_u8_t const *b_second = tile + 1 < tiles ? b_first + tile_stride : b_first;
        svzero_za();
        if (!first)
            nk_load_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
        nk_mopa_2x2_u8_sme_(first_panel, second_panel, b_first, b_second, steps);
        nk_store_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
    }
}

/** Sweeps one f16 panel in 1×4 groups. */
NUMKONG_INLINE void nk_sweep_1x4_f16_sme_(nk_u16_t const *panel, nk_size_t rows, nk_size_t row_first,
                                          nk_u16_t const *b_tiles, nk_size_t tile_stride, nk_size_t tiles,
                                          nk_size_t column_first, nk_size_t steps, int first, int upper, void *c,
                                          nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 4) {
        nk_size_t const quad_first = column_first + tile * tile_dimension;
        if (upper && quad_first + 4 * tile_dimension <= row_first) continue;
        nk_u16_t const *b_quad = b_tiles + tile * tile_stride;
        svzero_za();
        if (!first) nk_load_tiles_1x4_sme_(c, c_stride, rows, row_first, quad_first, columns, upper);
        nk_mopa_1x4_f16_sme_(panel, b_quad, b_quad + (tile + 1 < tiles) * tile_stride,
                             b_quad + (tile + 2 < tiles) * 2 * tile_stride,
                             b_quad + (tile + 3 < tiles) * 3 * tile_stride, steps);
        nk_store_tiles_1x4_sme_(c, c_stride, rows, row_first, quad_first, columns, upper);
    }
}

/** Sweeps one bf16 panel in 1×4 groups. */
NUMKONG_INLINE void nk_sweep_1x4_bf16_sme_(nk_u16_t const *panel, nk_size_t rows, nk_size_t row_first,
                                           nk_u16_t const *b_tiles, nk_size_t tile_stride, nk_size_t tiles,
                                           nk_size_t column_first, nk_size_t steps, int first, int upper, void *c,
                                           nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 4) {
        nk_size_t const quad_first = column_first + tile * tile_dimension;
        if (upper && quad_first + 4 * tile_dimension <= row_first) continue;
        nk_u16_t const *b_quad = b_tiles + tile * tile_stride;
        svzero_za();
        if (!first) nk_load_tiles_1x4_sme_(c, c_stride, rows, row_first, quad_first, columns, upper);
        nk_mopa_1x4_bf16_sme_(panel, b_quad, b_quad + (tile + 1 < tiles) * tile_stride,
                              b_quad + (tile + 2 < tiles) * 2 * tile_stride,
                              b_quad + (tile + 3 < tiles) * 3 * tile_stride, steps);
        nk_store_tiles_1x4_sme_(c, c_stride, rows, row_first, quad_first, columns, upper);
    }
}

/** Sweeps one i8 panel in 1×4 groups. */
NUMKONG_INLINE void nk_sweep_1x4_i8_sme_(nk_u8_t const *panel, nk_size_t rows, nk_size_t row_first,
                                         nk_u8_t const *b_tiles, nk_size_t tile_stride, nk_size_t tiles,
                                         nk_size_t column_first, nk_size_t steps, int first, int upper, void *c,
                                         nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 4) {
        nk_size_t const quad_first = column_first + tile * tile_dimension;
        if (upper && quad_first + 4 * tile_dimension <= row_first) continue;
        nk_u8_t const *b_quad = b_tiles + tile * tile_stride;
        svzero_za();
        if (!first) nk_load_tiles_1x4_sme_(c, c_stride, rows, row_first, quad_first, columns, upper);
        nk_mopa_1x4_i8_sme_(panel, b_quad, b_quad + (tile + 1 < tiles) * tile_stride,
                            b_quad + (tile + 2 < tiles) * 2 * tile_stride,
                            b_quad + (tile + 3 < tiles) * 3 * tile_stride, steps);
        nk_store_tiles_1x4_sme_(c, c_stride, rows, row_first, quad_first, columns, upper);
    }
}

/** Sweeps one u8 panel in 1×4 groups. */
NUMKONG_INLINE void nk_sweep_1x4_u8_sme_(nk_u8_t const *panel, nk_size_t rows, nk_size_t row_first,
                                         nk_u8_t const *b_tiles, nk_size_t tile_stride, nk_size_t tiles,
                                         nk_size_t column_first, nk_size_t steps, int first, int upper, void *c,
                                         nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 4) {
        nk_size_t const quad_first = column_first + tile * tile_dimension;
        if (upper && quad_first + 4 * tile_dimension <= row_first) continue;
        nk_u8_t const *b_quad = b_tiles + tile * tile_stride;
        svzero_za();
        if (!first) nk_load_tiles_1x4_sme_(c, c_stride, rows, row_first, quad_first, columns, upper);
        nk_mopa_1x4_u8_sme_(panel, b_quad, b_quad + (tile + 1 < tiles) * tile_stride,
                            b_quad + (tile + 2 < tiles) * 2 * tile_stride,
                            b_quad + (tile + 3 < tiles) * 3 * tile_stride, steps);
        nk_store_tiles_1x4_sme_(c, c_stride, rows, row_first, quad_first, columns, upper);
    }
}

/** Sweeps two f16 NVFP4 panels in 2×2 groups against byte codes @p code_stride bytes apart and
 *  f16 block scale vectors @p factor_stride halfwords apart. */
NUMKONG_INLINE void nk_sweep_2x2_nvfp4_sme_(nk_u16_t const *first_panel, nk_u16_t const *second_panel,
                                            nk_size_t first_rows, nk_size_t second_rows, nk_size_t row_first,
                                            nk_u8_t const *b_codes, nk_size_t code_stride, nk_u16_t const *b_factors,
                                            nk_size_t factor_stride, nk_size_t tiles, nk_size_t steps, int first,
                                            void *c, nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 2) {
        nk_size_t const second_tile = tile + 1 < tiles ? tile + 1 : tile;
        svzero_za();
        if (!first)
            nk_load_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
        nk_mopa_2x2_nvfp4_sme_(first_panel, second_panel, b_codes + tile * code_stride,
                               b_codes + second_tile * code_stride, b_factors + tile * factor_stride,
                               b_factors + second_tile * factor_stride, steps);
        nk_store_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
    }
}

/** Sweeps two rebased bf16 MXFP4 panels in 2×2 groups against byte codes and decrement vectors. */
NUMKONG_INLINE void nk_sweep_2x2_mxfp4_sme_(nk_u16_t const *first_panel, nk_u16_t const *second_panel,
                                            nk_size_t first_rows, nk_size_t second_rows, nk_size_t row_first,
                                            nk_u8_t const *b_codes, nk_size_t code_stride, nk_u16_t const *b_factors,
                                            nk_size_t factor_stride, nk_size_t tiles, nk_size_t steps, int first,
                                            void *c, nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 2) {
        nk_size_t const second_tile = tile + 1 < tiles ? tile + 1 : tile;
        svzero_za();
        if (!first)
            nk_load_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
        nk_mopa_2x2_mxfp4_sme_(first_panel, second_panel, b_codes + tile * code_stride,
                               b_codes + second_tile * code_stride, b_factors + tile * factor_stride,
                               b_factors + second_tile * factor_stride, steps);
        nk_store_tiles_2x2_sme_(c, c_stride, row_first, first_rows, second_rows, tile * tile_dimension, columns);
    }
}

/** Sweeps one f16 NVFP4 panel in 1×4 groups against byte codes and block scale vectors. */
NUMKONG_INLINE void nk_sweep_1x4_nvfp4_sme_(nk_u16_t const *panel, nk_size_t rows, nk_size_t row_first,
                                            nk_u8_t const *b_codes, nk_size_t code_stride, nk_u16_t const *b_factors,
                                            nk_size_t factor_stride, nk_size_t tiles, nk_size_t steps, int first,
                                            void *c, nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 4) {
        nk_u8_t const *quad_codes[4];
        nk_u16_t const *quad_factors[4];
        for (nk_size_t index = 0; index < 4; ++index) {
            nk_size_t const quad_tile = tile + index < tiles ? tile + index : tile;
            quad_codes[index] = b_codes + quad_tile * code_stride;
            quad_factors[index] = b_factors + quad_tile * factor_stride;
        }
        svzero_za();
        if (!first) nk_load_tiles_1x4_sme_(c, c_stride, rows, row_first, tile * tile_dimension, columns, 0);
        nk_mopa_1x4_nvfp4_sme_(panel, quad_codes, quad_factors, steps);
        nk_store_tiles_1x4_sme_(c, c_stride, rows, row_first, tile * tile_dimension, columns, 0);
    }
}

/** Sweeps one rebased bf16 MXFP4 panel in 1×4 groups against byte codes and decrement vectors. */
NUMKONG_INLINE void nk_sweep_1x4_mxfp4_sme_(nk_u16_t const *panel, nk_size_t rows, nk_size_t row_first,
                                            nk_u8_t const *b_codes, nk_size_t code_stride, nk_u16_t const *b_factors,
                                            nk_size_t factor_stride, nk_size_t tiles, nk_size_t steps, int first,
                                            void *c, nk_size_t c_stride, nk_size_t columns) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw();
    for (nk_size_t tile = 0; tile < tiles; tile += 4) {
        nk_u8_t const *quad_codes[4];
        nk_u16_t const *quad_factors[4];
        for (nk_size_t index = 0; index < 4; ++index) {
            nk_size_t const quad_tile = tile + index < tiles ? tile + index : tile;
            quad_codes[index] = b_codes + quad_tile * code_stride;
            quad_factors[index] = b_factors + quad_tile * factor_stride;
        }
        svzero_za();
        if (!first) nk_load_tiles_1x4_sme_(c, c_stride, rows, row_first, tile * tile_dimension, columns, 0);
        nk_mopa_1x4_mxfp4_sme_(panel, quad_codes, quad_factors, steps);
        nk_store_tiles_1x4_sme_(c, c_stride, rows, row_first, tile * tile_dimension, columns, 0);
    }
}

/** Converts in place the i32 sums of rows from @p row_first up to @p row_end to F32 times
 *  @p factor, over the first @p columns columns, or from each row's diagonal on when @p upper. */
NUMKONG_INLINE void nk_scale_rows_i32_sme_(void *c, nk_size_t c_stride, nk_size_t row_first, nk_size_t row_end,
                                           nk_size_t columns, int upper, nk_f32_t factor) NUMKONG_STREAMING_ {
    for (nk_size_t row = row_first; row < row_end; ++row) {
        nk_f32_t *row_values = (nk_f32_t *)((char *)c + row * c_stride);
        for (nk_size_t column = upper ? row : 0; column < columns; column += svcntw()) {
            svbool_t const predicate_b32x = svwhilelt_b32_u64(column, columns);
            svint32_t const sums_i32x = svld1_s32(predicate_b32x, (nk_i32_t const *)row_values + column);
            svst1_f32(predicate_b32x, row_values + column,
                      svmul_n_f32_x(predicate_b32x, svcvt_f32_s32_x(predicate_b32x, sums_i32x), factor));
        }
    }
}

#pragma endregion Tile Sweeps

#pragma region Packs

/** Bytes of a pack of @p columns 16-bit columns @p depth deep, with their F32 squared norms. */
NUMKONG_INLINE nk_size_t nk_dots_pack_size_b16_sme_(nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = nk_cntw_sme_(), vector_elements = nk_cnth_sme_();
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, 2);
    return sizeof(nk_dots_sme_packed_header_t) + column_tile_count * depth_step_count * vector_elements * 2 +
           columns * sizeof(nk_f32_t);
}

/** Bytes of a pack of @p columns 8-bit columns @p depth deep in words of @p dims_per_word, with
 *  their squared norms; i4 and u4 keep a low-nibble and a high-nibble byte per 4-bit dim. */
NUMKONG_INLINE nk_size_t nk_dots_pack_size_b8_sme_(nk_size_t dims_per_word, nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = nk_cntw_sme_(), vector_bytes = nk_cntb_sme_();
    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    return sizeof(nk_dots_sme_packed_header_t) +
           column_tile_count * nk_panel_steps_b8_sme_(dims_per_word, depth) * vector_bytes + columns * sizeof(nk_u32_t);
}

/** Writes the header of an SME pack once, from the window that packs column 0. */
NUMKONG_INLINE void nk_dots_header_sme_(void *b_packed, nk_size_t columns, nk_size_t depth, nk_size_t depth_steps,
                                        nk_size_t norms_offset, nk_f32_t tensor_scale, nk_size_t tile_dimension) {
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    nk_u32_t *header_words = (nk_u32_t *)header;
    for (nk_size_t word = 0; word < sizeof(*header) / sizeof(nk_u32_t); ++word) header_words[word] = 0;
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
NUMKONG_INLINE void nk_dots_window_tiles_sme_(nk_size_t columns, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_size_t tile_dimension, nk_size_t *tile_begin, nk_size_t *tile_end) {
    nk_size_t const column_tiles = nk_size_divide_round_up_(columns, tile_dimension);
    *tile_begin = nk_size_divide_round_up_(columns_begin, tile_dimension);
    *tile_end = nk_size_divide_round_up_(columns_end, tile_dimension);
    if (*tile_end > column_tiles) *tile_end = column_tiles;
}

/** Writes the header of a dense SME pack of @p steps steps of @p step_bytes from the window that
 *  packs column 0, and returns the offset of its norms with the tiles the window owns. */
NUMKONG_INLINE nk_size_t nk_dots_pack_window_sme_(void *b_packed, nk_size_t columns, nk_size_t depth, nk_size_t steps,
                                                  nk_size_t step_bytes, nk_size_t columns_begin, nk_size_t columns_end,
                                                  nk_size_t *tile_begin, nk_size_t *tile_end) {
    nk_size_t const tile_dimension = nk_cntw_sme_();
    nk_size_t const norms_offset = sizeof(nk_dots_sme_packed_header_t) +
                                   nk_size_divide_round_up_(columns, tile_dimension) * steps * step_bytes;
    if (columns_begin == 0) nk_dots_header_sme_(b_packed, columns, depth, steps, norms_offset, 1, tile_dimension);
    nk_dots_window_tiles_sme_(columns, columns_begin, columns_end, tile_dimension, tile_begin, tile_end);
    return norms_offset;
}

/*  Pack norms accumulate in ZA, one row per column: depth step @c s of a packed vector, holding one
 *  pair or quad per column, lands in vertical slice `s % svcntw()`, the lane the matching
 *  `nk_dots_reduce_sumsq_*_ssve_` helper adds that step into, so a reduction of each row reproduces
 *  the A-side norm bit for bit. ZA0.S stays free for staging. */

/** Adds the squares of the F16 pairs of @p packed_f16x, the first values into slice @p slice of
 *  ZA1.S and the second ones into ZA2.S. */
NUMKONG_INLINE void nk_pack_sumsq_f16x_sme_(svfloat16_t packed_f16x, nk_size_t slice) NUMKONG_STREAMING_
    __arm_inout("za") {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svfloat32_t first_f32x = svcvt_f32_f16_x(predicate_all_b32x, packed_f16x);
    svfloat32_t second_f32x = svcvtlt_f32_f16_x(predicate_all_b32x, packed_f16x);
    svfloat32_t first_sumsq_f32x = svread_ver_za32_f32_m(svdup_f32(0), predicate_all_b32x, 1, slice);
    svfloat32_t second_sumsq_f32x = svread_ver_za32_f32_m(svdup_f32(0), predicate_all_b32x, 2, slice);
    svwrite_ver_za32_f32_m(1, slice, predicate_all_b32x,
                           svmla_f32_m(predicate_all_b32x, first_sumsq_f32x, first_f32x, first_f32x));
    svwrite_ver_za32_f32_m(2, slice, predicate_all_b32x,
                           svmla_f32_m(predicate_all_b32x, second_sumsq_f32x, second_f32x, second_f32x));
}

/** Adds the BFDOT of each BF16 pair of @p packed_bf16x with itself into slice @p slice of ZA1.S. */
NUMKONG_INLINE void nk_pack_sumsq_bf16x_sme_(svbfloat16_t packed_bf16x, nk_size_t slice) NUMKONG_STREAMING_
    __arm_inout("za") {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svfloat32_t sumsq_f32x = svread_ver_za32_f32_m(svdup_f32(0), predicate_all_b32x, 1, slice);
    svwrite_ver_za32_f32_m(1, slice, predicate_all_b32x, svbfdot_f32(sumsq_f32x, packed_bf16x, packed_bf16x));
}

/** Adds the SDOT of each I8 quad of @p packed_i8x with itself into slice @p slice of ZA1.S. */
NUMKONG_INLINE void nk_pack_sumsq_i8x_sme_(svint8_t packed_i8x, nk_size_t slice) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svint32_t sumsq_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 1, slice);
    svwrite_ver_za32_s32_m(1, slice, predicate_all_b32x, svdot_s32(sumsq_i32x, packed_i8x, packed_i8x));
}

/** Adds the UDOT of each U8 quad of @p packed_u8x with itself into slice @p slice of ZA1.S. */
NUMKONG_INLINE void nk_pack_sumsq_u8x_sme_(svuint8_t packed_u8x, nk_size_t slice) NUMKONG_STREAMING_ __arm_inout("za") {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svuint32_t sumsq_u32x = svread_ver_za32_u32_m(svdup_u32(0), predicate_all_b32x, 1, slice);
    svwrite_ver_za32_u32_m(1, slice, predicate_all_b32x, svdot_u32(sumsq_u32x, packed_u8x, packed_u8x));
}

/** Writes the squared norms of the @p columns columns of one staged f16 tile of @p steps steps;
 *  the staged steps are still in L1, so the norms reread them rather than the source. */
NUMKONG_INLINE void nk_pack_norms_f16_sme_(nk_u16_t const *tile_values, nk_size_t steps, nk_size_t columns,
                                           nk_f32_t *norms) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    svbool_t const predicate_all_b16x = svptrue_b16(), predicate_all_b32x = svptrue_b32();
    svzero_mask_za(nk_sme_zero_za32_tiles_123_k);
    for (nk_size_t step = 0; step < steps; ++step)
        nk_pack_sumsq_f16x_sme_(svld1_f16(predicate_all_b16x, (float16_t const *)tile_values + step * vector_elements),
                                step % tile_dimension);
    for (nk_size_t column = 0; column < columns; ++column)
        norms[column] =
            nk_svaddv_f32_(predicate_all_b32x, svread_hor_za32_f32_m(svdup_f32(0), predicate_all_b32x, 1, column)) +
            nk_svaddv_f32_(predicate_all_b32x, svread_hor_za32_f32_m(svdup_f32(0), predicate_all_b32x, 2, column));
}

/** Writes the squared norms of the columns of one staged bf16 tile, like the f16 helper. */
NUMKONG_INLINE void nk_pack_norms_bf16_sme_(nk_u16_t const *tile_values, nk_size_t steps, nk_size_t columns,
                                            nk_f32_t *norms) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    svbool_t const predicate_all_b16x = svptrue_b16(), predicate_all_b32x = svptrue_b32();
    svzero_mask_za(nk_sme_zero_za32_tiles_123_k);
    for (nk_size_t step = 0; step < steps; ++step)
        nk_pack_sumsq_bf16x_sme_(
            svld1_bf16(predicate_all_b16x, (bfloat16_t const *)tile_values + step * vector_elements),
            step % tile_dimension);
    for (nk_size_t column = 0; column < columns; ++column)
        norms[column] = nk_svaddv_f32_(predicate_all_b32x,
                                       svread_hor_za32_f32_m(svdup_f32(0), predicate_all_b32x, 1, column));
}

/** Writes the 32-bit squared norms of the columns of one staged i8 tile. */
NUMKONG_INLINE void nk_pack_norms_i8_sme_(nk_u8_t const *tile_values, nk_size_t steps, nk_size_t columns,
                                          nk_u32_t *norms) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    svbool_t const predicate_all_b8x = svptrue_b8(), predicate_all_b32x = svptrue_b32();
    svzero_mask_za(nk_sme_zero_za32_tiles_123_k);
    for (nk_size_t step = 0; step < steps; ++step)
        nk_pack_sumsq_i8x_sme_(svld1_s8(predicate_all_b8x, (nk_i8_t const *)tile_values + step * vector_bytes),
                               step % tile_dimension);
    for (nk_size_t column = 0; column < columns; ++column)
        norms[column] = (nk_u32_t)nk_svaddv_s32_(predicate_all_b32x,
                                                 svread_hor_za32_s32_m(svdup_s32(0), predicate_all_b32x, 1, column));
}

/** Writes the 32-bit squared norms of the columns of one staged u8 tile. */
NUMKONG_INLINE void nk_pack_norms_u8_sme_(nk_u8_t const *tile_values, nk_size_t steps, nk_size_t columns,
                                          nk_u32_t *norms) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    svbool_t const predicate_all_b8x = svptrue_b8(), predicate_all_b32x = svptrue_b32();
    svzero_mask_za(nk_sme_zero_za32_tiles_123_k);
    for (nk_size_t step = 0; step < steps; ++step)
        nk_pack_sumsq_u8x_sme_(svld1_u8(predicate_all_b8x, tile_values + step * vector_bytes), step % tile_dimension);
    for (nk_size_t column = 0; column < columns; ++column)
        norms[column] = (nk_u32_t)nk_svaddv_u32_(predicate_all_b32x,
                                                 svread_hor_za32_u32_m(svdup_u32(0), predicate_all_b32x, 1, column));
}

/** Turns @p columns 32-bit squared norms into F32 bits times @p factor in place. */
NUMKONG_INLINE void nk_pack_norms_scale_sme_(nk_u32_t *norms, nk_size_t columns, nk_f32_t factor) NUMKONG_STREAMING_ {
    svbool_t const predicate_b32x = svwhilelt_b32_u64(0, columns);
    svfloat32_t const norms_f32x = svcvt_f32_u32_x(predicate_b32x, svld1_u32(predicate_b32x, norms));
    svst1_f32(predicate_b32x, (nk_f32_t *)norms, svmul_n_f32_x(predicate_b32x, norms_f32x, factor));
}

/*  Dense packs stage whole column tiles from @p tile_begin up to @p tile_end over the full depth,
 *  then take the squared norms of their staged steps. */

/** Packs f16 columns. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_f16_sme_streaming_(nk_f16_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                         nk_u16_t *tiles, nk_f32_t *norms, nk_size_t tile_begin,
                                         nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_size_divide_round_up_(depth, 2);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u16_t *tile_values = tiles + tile * steps * svcnth();
        nk_stage_panel_b16_sme_((nk_u16_t const *)b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_f16_sme_(tile_values, steps, tile_columns, norms + column_first);
    }
}

/** Packs bf16 columns. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_bf16_sme_streaming_(nk_bf16_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                          nk_u16_t *tiles, nk_f32_t *norms, nk_size_t tile_begin,
                                          nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_size_divide_round_up_(depth, 2);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u16_t *tile_values = tiles + tile * steps * svcnth();
        nk_stage_panel_b16_sme_((nk_u16_t const *)b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_bf16_sme_(tile_values, steps, tile_columns, norms + column_first);
    }
}

/** Packs e4m3 columns as f16. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_e4m3_sme_streaming_(nk_e4m3_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                          nk_u16_t *tiles, nk_f32_t *norms, nk_size_t tile_begin,
                                          nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_size_divide_round_up_(depth, 2);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u16_t *tile_values = tiles + tile * steps * svcnth();
        nk_stage_panel_e4m3_sme_(b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_f16_sme_(tile_values, steps, tile_columns, norms + column_first);
    }
}

/** Packs e5m2 columns as f16. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_e5m2_sme_streaming_(nk_e5m2_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                          nk_u16_t *tiles, nk_f32_t *norms, nk_size_t tile_begin,
                                          nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_size_divide_round_up_(depth, 2);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u16_t *tile_values = tiles + tile * steps * svcnth();
        nk_stage_panel_e5m2_sme_(b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_f16_sme_(tile_values, steps, tile_columns, norms + column_first);
    }
}

/** Packs e3m2 columns as f16. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_e3m2_sme_streaming_(nk_e3m2_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                          nk_u16_t *tiles, nk_f32_t *norms, nk_size_t tile_begin,
                                          nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_size_divide_round_up_(depth, 2);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u16_t *tile_values = tiles + tile * steps * svcnth();
        nk_stage_panel_e3m2_sme_(b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_f16_sme_(tile_values, steps, tile_columns, norms + column_first);
    }
}

/** Packs i8 columns. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_i8_sme_streaming_(nk_i8_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                        nk_u8_t *tiles, nk_u32_t *norms, nk_size_t tile_begin,
                                        nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_panel_steps_b8_sme_(4, depth);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u8_t *tile_values = tiles + tile * steps * svcntb();
        nk_stage_panel_b8_sme_(b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_i8_sme_(tile_values, steps, tile_columns, norms + column_first);
    }
}

/** Packs u8 columns. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_u8_sme_streaming_(nk_u8_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                        nk_u8_t *tiles, nk_u32_t *norms, nk_size_t tile_begin,
                                        nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_panel_steps_b8_sme_(4, depth);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u8_t *tile_values = tiles + tile * steps * svcntb();
        nk_stage_panel_b8_sme_(b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_u8_sme_(tile_values, steps, tile_columns, norms + column_first);
    }
}

/** Packs e2m3 columns as i8 sixteenths, with F32 norms. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_e2m3_sme_streaming_(nk_e2m3_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                          nk_u8_t *tiles, nk_u32_t *norms, nk_size_t tile_begin,
                                          nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_panel_steps_b8_sme_(4, depth);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u8_t *tile_values = tiles + tile * steps * svcntb();
        nk_stage_panel_e2m3_sme_(b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_i8_sme_(tile_values, steps, tile_columns, norms + column_first);
        nk_pack_norms_scale_sme_(norms + column_first, tile_columns, 1.0f / 256);
    }
}

/** Packs e2m1 columns as doubled i8, with F32 norms. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_e2m1_sme_streaming_(nk_e2m1x2_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                          nk_u8_t *tiles, nk_u32_t *norms, nk_size_t tile_begin,
                                          nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_panel_steps_b8_sme_(4, depth);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u8_t *tile_values = tiles + tile * steps * svcntb();
        nk_stage_panel_e2m1_sme_(b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_i8_sme_(tile_values, steps, tile_columns, norms + column_first);
        nk_pack_norms_scale_sme_(norms + column_first, tile_columns, 0.25f);
    }
}

/** Packs i4 columns as sign-extended nibble steps. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_i4_sme_streaming_(nk_i4x2_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                        nk_u8_t *tiles, nk_u32_t *norms, nk_size_t tile_begin,
                                        nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_panel_steps_b8_sme_(8, depth);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u8_t *tile_values = tiles + tile * steps * svcntb();
        nk_stage_panel_i4_sme_(b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_i8_sme_(tile_values, steps, tile_columns, norms + column_first);
    }
}

/** Packs u4 columns as zero-extended nibble steps. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_u4_sme_streaming_(nk_u4x2_t const *b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                        nk_u8_t *tiles, nk_u32_t *norms, nk_size_t tile_begin,
                                        nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), steps = nk_panel_steps_b8_sme_(8, depth);
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_u8_t *tile_values = tiles + tile * steps * svcntb();
        nk_stage_panel_u4_sme_(b, b_stride, column_first, tile_columns, 0, depth, tile_values);
        nk_pack_norms_u8_sme_(tile_values, steps, tile_columns, norms + column_first);
    }
}

/** Packs F16 @p b columns @p columns_begin to @p columns_end into SME tiles with squared norms. */
NUMKONG_INLINE void nk_dots_pack_f16_tiles_sme_(           //
    nk_f16_t const *b, nk_size_t columns, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end) {
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth,
                                                            nk_size_divide_round_up_(depth, 2), nk_cntb_sme_(),
                                                            columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_f16_sme_streaming_(b, b_stride, columns, depth,
                                    (nk_u16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t)),
                                    (nk_f32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
}

/** Packs BF16 @p b columns @p columns_begin to @p columns_end into SME tiles with squared norms. */
NUMKONG_INLINE void nk_dots_pack_bf16_tiles_sme_(           //
    nk_bf16_t const *b, nk_size_t columns, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end) {
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth,
                                                            nk_size_divide_round_up_(depth, 2), nk_cntb_sme_(),
                                                            columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_bf16_sme_streaming_(b, b_stride, columns, depth,
                                     (nk_u16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t)),
                                     (nk_f32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
}

/** Byte offsets of a block-scaled SME pack: header, values, FP4 block factors, relative norms and
 *  their exponents, raw block scales, fold exponents and the raw code rows the exact path reads. */
typedef struct {
    nk_size_t values, factors, norms, exponents, scales, bases, codes, total;
} nk_dots_scaled_sme_layout_t;

/** The block-scaled pack layout of @p columns × @p depth in blocks of @p block_size at
 *  @p tile_dimension lanes: @p code_bits 4 keeps one byte per FP4 code and one factor vector per
 *  column tile and block, 8 keeps folded bf16 values. */
NUMKONG_INLINE nk_dots_scaled_sme_layout_t nk_dots_scaled_layout_sme_(nk_size_t columns, nk_size_t depth,
                                                                      nk_size_t tile_dimension, nk_size_t block_size,
                                                                      nk_size_t code_bits) NUMKONG_STREAMABLE_ {
    nk_size_t const column_tiles = nk_size_divide_round_up_(columns, tile_dimension),
                    steps = nk_size_divide_round_up_(depth, 2);
    nk_size_t const blocks = depth / block_size, vector_elements = tile_dimension * 2;
    nk_size_t const factor_bytes = code_bits == 4 ? column_tiles * blocks * vector_elements * 2 : 0;
    nk_dots_scaled_sme_layout_t layout;
    layout.values = sizeof(nk_dots_sme_packed_header_t);
    layout.factors = layout.values + column_tiles * steps * vector_elements * (code_bits == 4 ? 1 : 2);
    layout.norms = nk_size_round_up_to_multiple_(layout.factors + factor_bytes, 64);
    layout.exponents = layout.norms + columns * sizeof(nk_f32_t);
    layout.scales = layout.exponents + columns * sizeof(nk_i32_t);
    layout.bases = layout.scales + columns * blocks;
    layout.codes = layout.bases + columns;
    layout.total = layout.codes + columns * depth * code_bits / 8;
    return layout;
}

/** The packed columns of a block-scaled SME pack, as the finishers and the exact path read them. */
typedef struct {

    /** Relative squared norms, each column's norm being it times 4^E. */
    nk_f32_t const *norms;

    /** The exponents E of the norms, which the relative dots carry too. */
    nk_i32_t const *exponents;

    /** Fold exponents, −128 for columns the exact path takes. */
    nk_i8_t const *bases;

    /** The raw code and scale rows. */
    nk_cross_operand_t raw;

    /** Bytes between raw code rows. */
    nk_size_t raw_stride;

    /** The mantissa of the packed tensor scale, its exponent being in @c exponents. */
    nk_f32_t mantissa;
} nk_dots_scaled_sme_columns_t;

/** The columns of a block-scaled SME pack in blocks of @p block_size of @p code_bits codes. */
NUMKONG_INLINE nk_dots_scaled_sme_columns_t nk_dots_scaled_columns_sme_(void const *b_packed, nk_size_t block_size,
                                                                        nk_size_t code_bits) NUMKONG_STREAMABLE_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const depth = header->depth;
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_layout_sme_(
        header->columns, depth, header->svl_bytes / sizeof(nk_f32_t), block_size, code_bits);
    nk_u8_t const *pack = (nk_u8_t const *)b_packed;
    nk_dots_scaled_sme_columns_t columns;
    nk_i32_t tensor_exponent;
    columns.norms = (nk_f32_t const *)(pack + layout.norms);
    columns.exponents = (nk_i32_t const *)(pack + layout.exponents);
    columns.bases = (nk_i8_t const *)(pack + layout.bases);
    columns.raw.elements = pack + layout.codes, columns.raw.scales = pack + layout.scales;
    columns.raw.scales_stride = depth / block_size, columns.raw.tensor_scale = NUMKONG_NULL;
    columns.raw_stride = depth * code_bits / 8;
    columns.mantissa = nk_split_f32_serial_(header->tensor_scale, &tensor_exponent);
    return columns;
}

/** Copies @p row_bytes bytes of rows from @p column_first up to @p column_end of @p source,
 *  @p source_stride apart, into the rows of @p target. */
NUMKONG_INLINE void nk_copy_rows_sme_(nk_u8_t const *source, nk_size_t source_stride, nk_size_t row_bytes,
                                      nk_size_t column_first, nk_size_t column_end,
                                      nk_u8_t *target) NUMKONG_STREAMING_ {
    nk_size_t const vector_bytes = svcntb();
    for (nk_size_t column = column_first; column < column_end; ++column)
        for (nk_size_t byte = 0; byte < row_bytes; byte += vector_bytes) {
            svbool_t const predicate_b8x = svwhilelt_b8_u64(byte, row_bytes);
            svst1_u8(predicate_b8x, target + column * row_bytes + byte,
                     svld1_u8(predicate_b8x, source + column * source_stride + byte));
        }
}

/** A wide squared norm as a mantissa in [1, 4), or zero, or NaN, times 4 to the @p exponent. */
NUMKONG_INLINE nk_f32_t nk_wide_norm_sme_(nk_cross_wide_sum_t sumsq, nk_i32_t *exponent) NUMKONG_STREAMABLE_ {
    nk_i32_t split;
    nk_f32_t mantissa = nk_split_f32_serial_(sumsq.sum, &split);
    nk_i32_t power = sumsq.sum == 0 || sumsq.sum != sumsq.sum ? 0 : split + sumsq.exponent;
    if (power & 1) mantissa *= 2, power -= 1;
    *exponent = power / 2;
    return mantissa;
}

/** Adds the squares of the BF16 pairs @p bits_u16x into the even and odd sums of @p sums_f32x2. */
NUMKONG_INLINE svfloat32x2_t nk_sumsq_bf16x_sme_(svfloat32x2_t sums_f32x2, svuint16_t bits_u16x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svuint32_t const words_u32x = svreinterpret_u32_u16(bits_u16x);
    svfloat32_t const even_f32x = svreinterpret_f32_u32(svlsl_n_u32_x(predicate_all_b32x, words_u32x, 16));
    svfloat32_t const odd_f32x = svreinterpret_f32_u32(svand_n_u32_x(predicate_all_b32x, words_u32x, 0xFFFF0000u));
    return svcreate2_f32(svmla_f32_x(predicate_all_b32x, svget2_f32(sums_f32x2, 0), even_f32x, even_f32x),
                         svmla_f32_x(predicate_all_b32x, svget2_f32(sums_f32x2, 1), odd_f32x, odd_f32x));
}

/** Adds the squares of the F16 pairs @p bits_u16x into the even and odd sums of @p sums_f32x2. */
NUMKONG_INLINE svfloat32x2_t nk_sumsq_f16x_sme_(svfloat32x2_t sums_f32x2, svuint16_t bits_u16x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svfloat16_t const halves_f16x = svreinterpret_f16_u16(bits_u16x);
    svfloat32_t const even_f32x = svcvt_f32_f16_x(predicate_all_b32x, halves_f16x);
    svfloat32_t const odd_f32x = svcvtlt_f32_f16_x(predicate_all_b32x, halves_f16x);
    return svcreate2_f32(svmla_f32_x(predicate_all_b32x, svget2_f32(sums_f32x2, 0), even_f32x, even_f32x),
                         svmla_f32_x(predicate_all_b32x, svget2_f32(sums_f32x2, 1), odd_f32x, odd_f32x));
}

/** The sum of the even and odd sums of @p sums_f32x2. */
NUMKONG_INLINE nk_f32_t nk_sumsq_reduce_sme_(svfloat32x2_t sums_f32x2) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    return nk_svaddv_f32_(predicate_all_b32x,
                          svadd_f32_x(predicate_all_b32x, svget2_f32(sums_f32x2, 0), svget2_f32(sums_f32x2, 1)));
}

/*  Relative squared norms of one block-scaled row @p row of @p operand, the norm being the result
 *  times 4^E for the E left in @p exponent, which the relative dots of the row carry too: NVFP4
 *  rows sum their exact f16 values times the tensor scale's squared mantissa, its exponent being
 *  E; MX rows the fold covers sum their values rebased to their largest block exponent E; wider MX
 *  rows take their exact norm. */

/** Relative squared norm of an NVFP4 row. */
NUMKONG_INLINE nk_f32_t nk_dots_scaled_row_nvfp4_sme_(nk_cross_operand_t operand, nk_size_t row_stride, nk_size_t row,
                                                      nk_size_t depth, nk_i32_t *exponent) NUMKONG_STREAMING_ {
    nk_u8_t const *codes = (nk_u8_t const *)operand.elements + row * row_stride;
    nk_u8_t const *scales = operand.scales + row * operand.scales_stride;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(operand.tensor_scale), exponent);
    nk_size_t const vector_elements = svcnth();
    svfloat32x2_t sums_f32x2 = svcreate2_f32(svdup_n_f32(0), svdup_n_f32(0));
    for (nk_size_t first = 0; first < depth; first += vector_elements)
        sums_f32x2 = nk_sumsq_f16x_sme_(
            sums_f32x2, nk_decode_row_nvfp4_sme_(codes, scales, first,
                                                 depth - first < vector_elements ? depth - first : vector_elements));
    return nk_sumsq_reduce_sme_(sums_f32x2) * (mantissa * mantissa);
}

/** Relative squared norm of an MXFP4 row. */
NUMKONG_INLINE nk_f32_t nk_dots_scaled_row_mxfp4_sme_(nk_cross_operand_t operand, nk_size_t row_stride, nk_size_t row,
                                                      nk_size_t depth, nk_i32_t *exponent) NUMKONG_STREAMING_ {
    nk_u8_t const *codes = (nk_u8_t const *)operand.elements + row * row_stride;
    nk_u8_t const *scales = operand.scales + row * operand.scales_stride;
    int wide;
    nk_i32_t const base = nk_mx_base_sme_(scales, depth / 32, &wide);
    if (wide) {
        nk_cross_wide_sum_t sumsq, twin_sumsq;
        nk_cross_scaled_exact_wide_mxfp4_serial_(codes, scales, codes, scales, depth, &sumsq, &twin_sumsq);
        return nk_wide_norm_sme_(sumsq, exponent);
    }
    nk_size_t const vector_elements = svcnth();
    svfloat32x2_t sums_f32x2 = svcreate2_f32(svdup_n_f32(0), svdup_n_f32(0));
    for (nk_size_t first = 0; first < depth; first += vector_elements)
        sums_f32x2 = nk_sumsq_bf16x_sme_(
            sums_f32x2,
            nk_decode_row_mxfp4_sme_(codes, scales, first,
                                     depth - first < vector_elements ? depth - first : vector_elements, base));
    *exponent = base;
    return nk_sumsq_reduce_sme_(sums_f32x2);
}

/** Relative squared norm of an MXFP6 E2M3 row. */
NUMKONG_INLINE nk_f32_t nk_dots_scaled_row_mxfp6e2m3_sme_(nk_cross_operand_t operand, nk_size_t row_stride,
                                                          nk_size_t row, nk_size_t depth,
                                                          nk_i32_t *exponent) NUMKONG_STREAMING_ {
    nk_u8_t const *codes = (nk_u8_t const *)operand.elements + row * row_stride;
    nk_u8_t const *scales = operand.scales + row * operand.scales_stride;
    int wide;
    nk_i32_t const base = nk_mx_base_sme_(scales, depth / 32, &wide);
    if (wide) {
        nk_cross_wide_sum_t sumsq, twin_sumsq;
        nk_cross_scaled_exact_wide_mxfp6e2m3_serial_(codes, scales, codes, scales, depth, &sumsq, &twin_sumsq);
        return nk_wide_norm_sme_(sumsq, exponent);
    }
    nk_size_t const vector_elements = svcnth();
    svfloat32x2_t sums_f32x2 = svcreate2_f32(svdup_n_f32(0), svdup_n_f32(0));
    for (nk_size_t first = 0; first < depth; first += vector_elements)
        sums_f32x2 = nk_sumsq_bf16x_sme_(
            sums_f32x2,
            nk_decode_row_mxfp6e2m3_sme_(codes, scales, first,
                                         depth - first < vector_elements ? depth - first : vector_elements, base));
    *exponent = base;
    return nk_sumsq_reduce_sme_(sums_f32x2);
}

/** Relative squared norm of an MXFP6 E3M2 row. */
NUMKONG_INLINE nk_f32_t nk_dots_scaled_row_mxfp6e3m2_sme_(nk_cross_operand_t operand, nk_size_t row_stride,
                                                          nk_size_t row, nk_size_t depth,
                                                          nk_i32_t *exponent) NUMKONG_STREAMING_ {
    nk_u8_t const *codes = (nk_u8_t const *)operand.elements + row * row_stride;
    nk_u8_t const *scales = operand.scales + row * operand.scales_stride;
    int wide;
    nk_i32_t const base = nk_mx_base_sme_(scales, depth / 32, &wide);
    if (wide) {
        nk_cross_wide_sum_t sumsq, twin_sumsq;
        nk_cross_scaled_exact_wide_mxfp6e3m2_serial_(codes, scales, codes, scales, depth, &sumsq, &twin_sumsq);
        return nk_wide_norm_sme_(sumsq, exponent);
    }
    nk_size_t const vector_elements = svcnth();
    svfloat32x2_t sums_f32x2 = svcreate2_f32(svdup_n_f32(0), svdup_n_f32(0));
    for (nk_size_t first = 0; first < depth; first += vector_elements)
        sums_f32x2 = nk_sumsq_bf16x_sme_(
            sums_f32x2,
            nk_decode_row_mxfp6e3m2_sme_(codes, scales, first,
                                         depth - first < vector_elements ? depth - first : vector_elements, base));
    *exponent = base;
    return nk_sumsq_reduce_sme_(sums_f32x2);
}

/** Relative squared norm of an MXFP8 E4M3 row. */
NUMKONG_INLINE nk_f32_t nk_dots_scaled_row_mxfp8e4m3_sme_(nk_cross_operand_t operand, nk_size_t row_stride,
                                                          nk_size_t row, nk_size_t depth,
                                                          nk_i32_t *exponent) NUMKONG_STREAMING_ {
    nk_u8_t const *codes = (nk_u8_t const *)operand.elements + row * row_stride;
    nk_u8_t const *scales = operand.scales + row * operand.scales_stride;
    int wide;
    nk_i32_t const base = nk_mx_base_sme_(scales, depth / 32, &wide);
    if (wide) {
        nk_cross_wide_sum_t sumsq, twin_sumsq;
        nk_cross_scaled_exact_wide_mxfp8e4m3_serial_(codes, scales, codes, scales, depth, &sumsq, &twin_sumsq);
        return nk_wide_norm_sme_(sumsq, exponent);
    }
    nk_size_t const vector_elements = svcnth();
    svfloat32x2_t sums_f32x2 = svcreate2_f32(svdup_n_f32(0), svdup_n_f32(0));
    for (nk_size_t first = 0; first < depth; first += vector_elements)
        sums_f32x2 = nk_sumsq_bf16x_sme_(
            sums_f32x2,
            nk_decode_row_mxfp8e4m3_sme_(codes, scales, first,
                                         depth - first < vector_elements ? depth - first : vector_elements, base));
    *exponent = base;
    return nk_sumsq_reduce_sme_(sums_f32x2);
}

/** Relative squared norm of an MXFP8 E5M2 row. */
NUMKONG_INLINE nk_f32_t nk_dots_scaled_row_mxfp8e5m2_sme_(nk_cross_operand_t operand, nk_size_t row_stride,
                                                          nk_size_t row, nk_size_t depth,
                                                          nk_i32_t *exponent) NUMKONG_STREAMING_ {
    nk_u8_t const *codes = (nk_u8_t const *)operand.elements + row * row_stride;
    nk_u8_t const *scales = operand.scales + row * operand.scales_stride;
    int wide;
    nk_i32_t const base = nk_mx_base_sme_(scales, depth / 32, &wide);
    if (wide) {
        nk_cross_wide_sum_t sumsq, twin_sumsq;
        nk_cross_scaled_exact_wide_mxfp8e5m2_serial_(codes, scales, codes, scales, depth, &sumsq, &twin_sumsq);
        return nk_wide_norm_sme_(sumsq, exponent);
    }
    nk_size_t const vector_elements = svcnth();
    svfloat32x2_t sums_f32x2 = svcreate2_f32(svdup_n_f32(0), svdup_n_f32(0));
    for (nk_size_t first = 0; first < depth; first += vector_elements)
        sums_f32x2 = nk_sumsq_bf16x_sme_(
            sums_f32x2,
            nk_decode_row_mxfp8e5m2_sme_(codes, scales, first,
                                         depth - first < vector_elements ? depth - first : vector_elements, base));
    *exponent = base;
    return nk_sumsq_reduce_sme_(sums_f32x2);
}

/** Writes the FP4 codes of @p tile_columns columns of @p b from @p column_first into one column
 *  tile of byte panels, halfword i of step s holding column i's codes of dims 2s and 2s + 1,
 *  through the ZA0.H transpose; columns past @p tile_columns stay zero. */
NUMKONG_INLINE void nk_pack_fp4_codes_sme_(nk_u8_t const *b, nk_size_t b_stride, nk_size_t column_first,
                                           nk_size_t tile_columns, nk_size_t depth, nk_u8_t *codes) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const vector_bytes = svcntb(), vector_elements = svcnth();
    svbool_t const predicate_all_b8x = svptrue_b8(), predicate_all_b16x = svptrue_b16();
    svbool_t const tile_b16x = svwhilelt_b16_u64(0, svcntw());
    for (nk_size_t dims = 0; dims < depth; dims += vector_bytes) {
        nk_size_t const count = depth - dims < vector_bytes ? depth - dims : vector_bytes;
        svbool_t const pairs_b8x = svwhilelt_b8_u64(0, count / 2);
        svzero_mask_za(nk_sme_zero_za16_tile_0_k);
        for (nk_size_t column = 0; column < tile_columns; ++column) {
            svuint8_t const pairs_u8x = svld1_u8(pairs_b8x, b + (column_first + column) * b_stride + dims / 2);
            svuint8_t const codes_u8x = svzip1_u8(svlsr_n_u8_x(predicate_all_b8x, pairs_u8x, 4),
                                                  svand_n_u8_x(predicate_all_b8x, pairs_u8x, 0x0F));
            svwrite_hor_za16_u16_m(0, (uint32_t)column, predicate_all_b16x, svreinterpret_u16_u8(codes_u8x));
        }
        for (nk_size_t step = 0; step < count / 2; ++step)
            svst1_ver_za16(0, (uint32_t)step, tile_b16x, codes + (dims / 2 + step) * vector_elements);
    }
}

/** Writes the per-block factor vectors of one FP4 column tile from the @p factors_u16x the caller
 *  builds per column, through the ZA0.S transpose: both halfwords of word i of block b hold column
 *  i's factor. Columns past @p tile_columns stay zero. */
NUMKONG_INLINE void nk_store_factor_slices_sme_(nk_size_t block_first, nk_size_t blocks,
                                                nk_u16_t *factors) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t block = 0; block < blocks; ++block)
        svst1_ver_za32(0, (uint32_t)block, predicate_all_b32x, factors + (block_first + block) * vector_elements);
}

/** Packs NVFP4 columns of whole column tiles from @p tile_begin up to @p tile_end: byte codes, f16
 *  block scale vectors, raw code and scale rows, and relative norms with their exponents. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_nvfp4_sme_streaming_(nk_cross_operand_t b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                           nk_u8_t *pack, nk_size_t tile_begin, nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth(), blocks = depth / 16;
    nk_size_t const steps = nk_size_divide_round_up_(depth, 2);
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_layout_sme_(columns, depth, tile_dimension, 16, 4);
    nk_f32_t *norms = (nk_f32_t *)(pack + layout.norms);
    nk_i32_t *exponents = (nk_i32_t *)(pack + layout.exponents);
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_pack_fp4_codes_sme_((nk_u8_t const *)b.elements, b_stride, column_first, tile_columns, depth,
                               pack + layout.values + tile * steps * vector_elements);
        for (nk_size_t block_first = 0; block_first < blocks; block_first += tile_dimension) {
            nk_size_t const batch = blocks - block_first < tile_dimension ? blocks - block_first : tile_dimension;
            svbool_t const batch_b8x = svwhilelt_b8_u64(0, batch), batch_b16x = svwhilelt_b16_u64(0, batch);
            svzero_mask_za(nk_sme_zero_za32_tile_0_k);
            for (nk_size_t column = 0; column < tile_columns; ++column) {
                nk_u8_t const *scales = b.scales + (column_first + column) * b.scales_stride + block_first;
                svuint8_t const codes_u8x = svand_n_u8_x(svptrue_b8(), svld1_u8(batch_b8x, scales), 0x7F);
                svuint16_t const factors_u16x = svreinterpret_u16_f16(
                    nk_e4m3x_to_f16x_sme_streaming_(batch_b16x, codes_u8x));
                svwrite_hor_za32_u32_m(0, (uint32_t)column, predicate_all_b32x,
                                       svreinterpret_u32_u16(svzip1_u16(factors_u16x, factors_u16x)));
            }
            nk_store_factor_slices_sme_(block_first, batch,
                                        (nk_u16_t *)(pack + layout.factors) + tile * blocks * vector_elements);
        }
        nk_size_t const column_end = column_first + tile_columns;
        for (nk_size_t column = column_first; column < column_end; ++column)
            norms[column] = nk_dots_scaled_row_nvfp4_sme_(b, b_stride, column, depth, &exponents[column]);
        nk_copy_rows_sme_((nk_u8_t const *)b.elements, b_stride, depth / 2, column_first, column_end,
                          pack + layout.codes);
        nk_copy_rows_sme_(b.scales, b.scales_stride, blocks, column_first, column_end, pack + layout.scales);
        nk_i8_t *column_bases = (nk_i8_t *)(pack + layout.bases);
        for (nk_size_t column = column_first; column < column_end; ++column) column_bases[column] = 0;
    }
}

/** Packs MXFP4 columns of whole column tiles: byte codes, decrement vectors against each column's
 *  largest block exponent, raw rows, and relative norms with their exponents. Columns past the
 *  fold or holding a NaN scale are marked for the exact path. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_mxfp4_sme_streaming_(nk_cross_operand_t b, nk_size_t b_stride, nk_size_t columns, nk_size_t depth,
                                           nk_u8_t *pack, nk_size_t tile_begin, nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth(), blocks = depth / 32;
    nk_size_t const steps = nk_size_divide_round_up_(depth, 2);
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_layout_sme_(columns, depth, tile_dimension, 32, 4);
    nk_f32_t *norms = (nk_f32_t *)(pack + layout.norms);
    nk_i32_t *exponents = (nk_i32_t *)(pack + layout.exponents);
    nk_i8_t *bases = (nk_i8_t *)(pack + layout.bases);
    svbool_t const predicate_all_b16x = svptrue_b16(), predicate_all_b32x = svptrue_b32();
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_size_t const column_end = column_first + tile_columns;
        for (nk_size_t column = column_first; column < column_end; ++column) {
            nk_u8_t const *scales = b.scales + column * b.scales_stride;
            int wide, nan_scale = 0;
            nk_i32_t const base = nk_mx_base_sme_(scales, blocks, &wide);
            for (nk_size_t block = 0; block < blocks; ++block) nan_scale |= scales[block] == 255;
            bases[column] = (nk_i8_t)(wide || nan_scale ? -128 : base);
            norms[column] = nk_dots_scaled_row_mxfp4_sme_(b, b_stride, column, depth, &exponents[column]);
        }
        nk_pack_fp4_codes_sme_((nk_u8_t const *)b.elements, b_stride, column_first, tile_columns, depth,
                               pack + layout.values + tile * steps * vector_elements);
        for (nk_size_t block_first = 0; block_first < blocks; block_first += tile_dimension) {
            nk_size_t const batch = blocks - block_first < tile_dimension ? blocks - block_first : tile_dimension;
            svbool_t const batch_b16x = svwhilelt_b16_u64(0, batch);
            svzero_mask_za(nk_sme_zero_za32_tile_0_k);
            for (nk_size_t column = 0; column < tile_columns; ++column) {
                nk_i32_t const base = bases[column_first + column];
                svuint16_t const codes_u16x = svld1ub_u16(
                    batch_b16x, b.scales + (column_first + column) * b.scales_stride + block_first);
                // A decrement past every table entry saturates a zero block to zero
                svuint16_t factors_u16x = svlsl_n_u16_x(
                    predicate_all_b16x, svsubr_n_u16_x(predicate_all_b16x, codes_u16x, (nk_u16_t)(base + 127)), 7);
                factors_u16x = svsel_u16(svcmpeq_n_u16(predicate_all_b16x, codes_u16x, 0), svdup_n_u16(0xFFFF),
                                         factors_u16x);
                if (base == -128) factors_u16x = svdup_n_u16(0);
                svwrite_hor_za32_u32_m(0, (uint32_t)column, predicate_all_b32x,
                                       svreinterpret_u32_u16(svzip1_u16(factors_u16x, factors_u16x)));
            }
            nk_store_factor_slices_sme_(block_first, batch,
                                        (nk_u16_t *)(pack + layout.factors) + tile * blocks * vector_elements);
        }
        nk_copy_rows_sme_((nk_u8_t const *)b.elements, b_stride, depth / 2, column_first, column_end,
                          pack + layout.codes);
        nk_copy_rows_sme_(b.scales, b.scales_stride, blocks, column_first, column_end, pack + layout.scales);
    }
}

/** Fold exponents of one tile of MX columns from @p column_first: each column's largest block
 *  exponent, or none for columns past the fold, which @p bases_i8 marks −128. */
NUMKONG_INLINE void nk_pack_mx_bases_sme_(nk_cross_operand_t b, nk_size_t column_first, nk_size_t tile_columns,
                                          nk_size_t blocks, nk_i8_t *bases_i8, nk_i32_t *bases) NUMKONG_STREAMABLE_ {
    for (nk_size_t column = 0; column < tile_columns; ++column) {
        int wide;
        nk_i32_t const base = nk_mx_base_sme_(b.scales + (column_first + column) * b.scales_stride, blocks, &wide);
        bases_i8[column_first + column] = (nk_i8_t)(wide ? -128 : base);
        bases[column] = wide ? nk_sme_unfolded_base_k : base;
    }
}

/*  MXFP6 and MXFP8 packs fold whole column tiles into bf16 values rebased per column, with raw rows
 *  and relative norms with their exponents. */

/** Packs MXFP6 E2M3 columns. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_mxfp6e2m3_sme_streaming_(nk_cross_operand_t b, nk_size_t b_stride, nk_size_t columns,
                                               nk_size_t depth, nk_u8_t *pack, nk_size_t tile_begin,
                                               nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), blocks = depth / 32, steps = nk_size_divide_round_up_(depth, 2);
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_layout_sme_(columns, depth, tile_dimension, 32, 8);
    nk_i32_t bases[nk_sme_max_tile_k];
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_size_t const column_end = column_first + tile_columns;
        nk_pack_mx_bases_sme_(b, column_first, tile_columns, blocks, (nk_i8_t *)(pack + layout.bases), bases);
        nk_stage_panel_mxfp6e2m3_sme_(b, b_stride, column_first, tile_columns, 0, depth, bases,
                                      (nk_u16_t *)(pack + layout.values) + tile * steps * svcnth());
        nk_f32_t *column_norms = (nk_f32_t *)(pack + layout.norms);
        for (nk_size_t column = column_first; column < column_end; ++column)
            column_norms[column] = nk_dots_scaled_row_mxfp6e2m3_sme_(b, b_stride, column, depth,
                                                                     (nk_i32_t *)(pack + layout.exponents) + column);
        nk_copy_rows_sme_((nk_u8_t const *)b.elements, b_stride, depth, column_first, column_end, pack + layout.codes);
        nk_copy_rows_sme_(b.scales, b.scales_stride, blocks, column_first, column_end, pack + layout.scales);
    }
}

/** Packs MXFP6 E3M2 columns. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_mxfp6e3m2_sme_streaming_(nk_cross_operand_t b, nk_size_t b_stride, nk_size_t columns,
                                               nk_size_t depth, nk_u8_t *pack, nk_size_t tile_begin,
                                               nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), blocks = depth / 32, steps = nk_size_divide_round_up_(depth, 2);
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_layout_sme_(columns, depth, tile_dimension, 32, 8);
    nk_i32_t bases[nk_sme_max_tile_k];
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_size_t const column_end = column_first + tile_columns;
        nk_pack_mx_bases_sme_(b, column_first, tile_columns, blocks, (nk_i8_t *)(pack + layout.bases), bases);
        nk_stage_panel_mxfp6e3m2_sme_(b, b_stride, column_first, tile_columns, 0, depth, bases,
                                      (nk_u16_t *)(pack + layout.values) + tile * steps * svcnth());
        nk_f32_t *column_norms = (nk_f32_t *)(pack + layout.norms);
        for (nk_size_t column = column_first; column < column_end; ++column)
            column_norms[column] = nk_dots_scaled_row_mxfp6e3m2_sme_(b, b_stride, column, depth,
                                                                     (nk_i32_t *)(pack + layout.exponents) + column);
        nk_copy_rows_sme_((nk_u8_t const *)b.elements, b_stride, depth, column_first, column_end, pack + layout.codes);
        nk_copy_rows_sme_(b.scales, b.scales_stride, blocks, column_first, column_end, pack + layout.scales);
    }
}

/** Packs MXFP8 E4M3 columns. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_mxfp8e4m3_sme_streaming_(nk_cross_operand_t b, nk_size_t b_stride, nk_size_t columns,
                                               nk_size_t depth, nk_u8_t *pack, nk_size_t tile_begin,
                                               nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), blocks = depth / 32, steps = nk_size_divide_round_up_(depth, 2);
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_layout_sme_(columns, depth, tile_dimension, 32, 8);
    nk_i32_t bases[nk_sme_max_tile_k];
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_size_t const column_end = column_first + tile_columns;
        nk_pack_mx_bases_sme_(b, column_first, tile_columns, blocks, (nk_i8_t *)(pack + layout.bases), bases);
        nk_stage_panel_mxfp8e4m3_sme_(b, b_stride, column_first, tile_columns, 0, depth, bases,
                                      (nk_u16_t *)(pack + layout.values) + tile * steps * svcnth());
        nk_f32_t *column_norms = (nk_f32_t *)(pack + layout.norms);
        for (nk_size_t column = column_first; column < column_end; ++column)
            column_norms[column] = nk_dots_scaled_row_mxfp8e4m3_sme_(b, b_stride, column, depth,
                                                                     (nk_i32_t *)(pack + layout.exponents) + column);
        nk_copy_rows_sme_((nk_u8_t const *)b.elements, b_stride, depth, column_first, column_end, pack + layout.codes);
        nk_copy_rows_sme_(b.scales, b.scales_stride, blocks, column_first, column_end, pack + layout.scales);
    }
}

/** Packs MXFP8 E5M2 columns. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_pack_mxfp8e5m2_sme_streaming_(nk_cross_operand_t b, nk_size_t b_stride, nk_size_t columns,
                                               nk_size_t depth, nk_u8_t *pack, nk_size_t tile_begin,
                                               nk_size_t tile_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), blocks = depth / 32, steps = nk_size_divide_round_up_(depth, 2);
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_layout_sme_(columns, depth, tile_dimension, 32, 8);
    nk_i32_t bases[nk_sme_max_tile_k];
    for (nk_size_t tile = tile_begin; tile < tile_end; ++tile) {
        nk_size_t const column_first = tile * tile_dimension;
        nk_size_t const tile_columns = columns - column_first < tile_dimension ? columns - column_first
                                                                               : tile_dimension;
        nk_size_t const column_end = column_first + tile_columns;
        nk_pack_mx_bases_sme_(b, column_first, tile_columns, blocks, (nk_i8_t *)(pack + layout.bases), bases);
        nk_stage_panel_mxfp8e5m2_sme_(b, b_stride, column_first, tile_columns, 0, depth, bases,
                                      (nk_u16_t *)(pack + layout.values) + tile * steps * svcnth());
        nk_f32_t *column_norms = (nk_f32_t *)(pack + layout.norms);
        for (nk_size_t column = column_first; column < column_end; ++column)
            column_norms[column] = nk_dots_scaled_row_mxfp8e5m2_sme_(b, b_stride, column, depth,
                                                                     (nk_i32_t *)(pack + layout.exponents) + column);
        nk_copy_rows_sme_((nk_u8_t const *)b.elements, b_stride, depth, column_first, column_end, pack + layout.codes);
        nk_copy_rows_sme_(b.scales, b.scales_stride, blocks, column_first, column_end, pack + layout.scales);
    }
}

/** Writes the header of a block-scaled pack of @p b with norms at @p norms_offset from the window
 *  that packs column 0, and returns the tiles the window owns. */
NUMKONG_INLINE void nk_dots_pack_scaled_window_sme_(nk_cross_operand_t b, nk_size_t columns, nk_size_t depth,
                                                    void *b_packed, nk_size_t norms_offset, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_size_t *tile_begin, nk_size_t *tile_end) {
    nk_size_t const tile_dimension = nk_cntw_sme_();
    if (columns_begin == 0)
        nk_dots_header_sme_(b_packed, columns, depth, nk_size_divide_round_up_(depth, 2), norms_offset,
                            nk_cross_tensor_scale_serial_(b.tensor_scale), tile_dimension);
    nk_dots_window_tiles_sme_(columns, columns_begin, columns_end, tile_dimension, tile_begin, tile_end);
}

#pragma endregion Packs

/*  Packed kernels multiply rows of A, staged per tile and depth chunk, by a pack of B, leaving raw
 *  32-bit sums in C: f16-panel inputs in 2×2 bodies with a 1×4 final strip, bf16-panel inputs in
 *  1×4 bodies; e2m3 and e2m1 rows convert to F32 once finished, and block-scaled sums stay
 *  relative to their exponents for the finishers. */

#pragma region Packed Kernels

/** A × packed B for f16 rows. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_f16_sme_streaming_(nk_f16_t const *a, nk_size_t a_stride, void const *b_packed, nk_f32_t *c,
                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                           nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    nk_size_t const tile_stride = header->depth_tile_count * vector_elements, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_align_(64) nk_u16_t panels[2][nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            nk_u16_t const *b_chunk = b_tiles + depth_first / 2 * vector_elements;
            nk_stage_panel_b16_sme_((nk_u16_t const *)a, a_stride, row_first, first_rows, depth_first, chunk_depth,
                                    panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_f16_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                      depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_b16_sme_((nk_u16_t const *)a, a_stride, row_first + first_rows, second_rows, depth_first,
                                    chunk_depth, panels[1]);
            nk_sweep_2x2_f16_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                  steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/** A × packed B for bf16 rows. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_bf16_sme_streaming_(nk_bf16_t const *a, nk_size_t a_stride, void const *b_packed, nk_f32_t *c,
                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    nk_size_t const tile_stride = header->depth_tile_count * vector_elements, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_align_(64) nk_u16_t panel[nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t row_first = 0; row_first < rows; row_first += tile_dimension) {
        nk_size_t const block_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_stage_panel_b16_sme_((nk_u16_t const *)a, a_stride, row_first, block_rows, depth_first, chunk_depth,
                                    panel);
            nk_sweep_1x4_bf16_sme_(panel, block_rows, row_first, b_tiles + depth_first / 2 * vector_elements,
                                   tile_stride, tiles, 0, nk_size_divide_round_up_(chunk_depth, 2), depth_first == 0, 0,
                                   c, c_stride, columns);
        }
    }
}

/** A × packed B for e4m3 rows decoded to f16. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_e4m3_sme_streaming_(nk_e4m3_t const *a, nk_size_t a_stride, void const *b_packed, nk_f32_t *c,
                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    nk_size_t const tile_stride = header->depth_tile_count * vector_elements, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_align_(64) nk_u16_t panels[2][nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            nk_u16_t const *b_chunk = b_tiles + depth_first / 2 * vector_elements;
            nk_stage_panel_e4m3_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_f16_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                      depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_e4m3_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                     panels[1]);
            nk_sweep_2x2_f16_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                  steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/** A × packed B for e5m2 rows decoded to f16. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_e5m2_sme_streaming_(nk_e5m2_t const *a, nk_size_t a_stride, void const *b_packed, nk_f32_t *c,
                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    nk_size_t const tile_stride = header->depth_tile_count * vector_elements, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_align_(64) nk_u16_t panels[2][nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            nk_u16_t const *b_chunk = b_tiles + depth_first / 2 * vector_elements;
            nk_stage_panel_e5m2_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_f16_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                      depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_e5m2_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                     panels[1]);
            nk_sweep_2x2_f16_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                  steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/** A × packed B for e3m2 rows decoded to f16. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_e3m2_sme_streaming_(nk_e3m2_t const *a, nk_size_t a_stride, void const *b_packed, nk_f32_t *c,
                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth();
    nk_size_t const tile_stride = header->depth_tile_count * vector_elements, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_align_(64) nk_u16_t panels[2][nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            nk_u16_t const *b_chunk = b_tiles + depth_first / 2 * vector_elements;
            nk_stage_panel_e3m2_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_f16_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                      depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_e3m2_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                     panels[1]);
            nk_sweep_2x2_f16_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                  steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/** A × packed B for i8 rows. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_i8_sme_streaming_(nk_i8_t const *a, nk_size_t a_stride, void const *b_packed, nk_i32_t *c,
                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                          nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u8_t const *b_tiles = (nk_u8_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    nk_size_t const tile_stride = header->depth_tile_count * vector_bytes, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / vector_bytes * 4;
    nk_align_(64) nk_u8_t panels[2][nk_sme_panel_bytes_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(4, chunk_depth);
            nk_u8_t const *b_chunk = b_tiles + depth_first / 4 * vector_bytes;
            nk_stage_panel_b8_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_i8_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                     depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_b8_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                   panels[1]);
            nk_sweep_2x2_i8_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                 steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/** A × packed B for u8 rows. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_u8_sme_streaming_(nk_u8_t const *a, nk_size_t a_stride, void const *b_packed, nk_u32_t *c,
                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                          nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u8_t const *b_tiles = (nk_u8_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    nk_size_t const tile_stride = header->depth_tile_count * vector_bytes, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / vector_bytes * 4;
    nk_align_(64) nk_u8_t panels[2][nk_sme_panel_bytes_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(4, chunk_depth);
            nk_u8_t const *b_chunk = b_tiles + depth_first / 4 * vector_bytes;
            nk_stage_panel_b8_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_u8_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                     depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_b8_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                   panels[1]);
            nk_sweep_2x2_u8_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                 steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/** A × packed B for e2m3 rows as i8 sixteenths, converted to F32 per finished row block. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_e2m3_sme_streaming_(nk_e2m3_t const *a, nk_size_t a_stride, void const *b_packed, nk_f32_t *c,
                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u8_t const *b_tiles = (nk_u8_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    nk_size_t const tile_stride = header->depth_tile_count * vector_bytes, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / vector_bytes * 4;
    nk_align_(64) nk_u8_t panels[2][nk_sme_panel_bytes_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(4, chunk_depth);
            nk_u8_t const *b_chunk = b_tiles + depth_first / 4 * vector_bytes;
            nk_stage_panel_e2m3_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_i8_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                     depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_e2m3_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                     panels[1]);
            nk_sweep_2x2_i8_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                 steps, depth_first == 0, c, c_stride, columns);
        }
        nk_scale_rows_i32_sme_(c, c_stride, row_first, row_first + first_rows + second_rows, columns, 0, 1.0f / 256);
    }
}

/** A × packed B for e2m1 rows as doubled i8, converted to F32 per finished row block. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_e2m1_sme_streaming_(nk_e2m1x2_t const *a, nk_size_t a_stride, void const *b_packed, nk_f32_t *c,
                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u8_t const *b_tiles = (nk_u8_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    nk_size_t const tile_stride = header->depth_tile_count * vector_bytes, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / vector_bytes * 4;
    nk_align_(64) nk_u8_t panels[2][nk_sme_panel_bytes_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(4, chunk_depth);
            nk_u8_t const *b_chunk = b_tiles + depth_first / 4 * vector_bytes;
            nk_stage_panel_e2m1_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_i8_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                     depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_e2m1_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                     panels[1]);
            nk_sweep_2x2_i8_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                 steps, depth_first == 0, c, c_stride, columns);
        }
        nk_scale_rows_i32_sme_(c, c_stride, row_first, row_first + first_rows + second_rows, columns, 0, 0.25f);
    }
}

/** A × packed B for i4 rows as sign-extended nibble steps. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_i4_sme_streaming_(nk_i4x2_t const *a, nk_size_t a_stride, void const *b_packed, nk_i32_t *c,
                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                          nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u8_t const *b_tiles = (nk_u8_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    nk_size_t const tile_stride = header->depth_tile_count * vector_bytes, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / vector_bytes * 4;
    nk_align_(64) nk_u8_t panels[2][nk_sme_panel_bytes_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(8, chunk_depth);
            nk_u8_t const *b_chunk = b_tiles + depth_first / 4 * vector_bytes;
            nk_stage_panel_i4_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_i8_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                     depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_i4_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                   panels[1]);
            nk_sweep_2x2_i8_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                 steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/** A × packed B for u4 rows as zero-extended nibble steps. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_u4_sme_streaming_(nk_u4x2_t const *a, nk_size_t a_stride, void const *b_packed, nk_u32_t *c,
                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                          nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u8_t const *b_tiles = (nk_u8_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_bytes = svcntb();
    nk_size_t const tile_stride = header->depth_tile_count * vector_bytes, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / vector_bytes * 4;
    nk_align_(64) nk_u8_t panels[2][nk_sme_panel_bytes_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(8, chunk_depth);
            nk_u8_t const *b_chunk = b_tiles + depth_first / 4 * vector_bytes;
            nk_stage_panel_u4_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_u8_sme_(panels[0], first_rows, row_first, b_chunk, tile_stride, tiles, 0, steps,
                                     depth_first == 0, 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_u4_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                   panels[1]);
            nk_sweep_2x2_u8_sme_(panels[0], panels[1], first_rows, second_rows, row_first, b_chunk, tile_stride, tiles,
                                 steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/** A × packed B for NVFP4 rows, exact f16 panels against byte codes and block scale vectors,
 *  relative to the tensor scales. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_nvfp4_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride, void const *b_packed,
                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                             nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth(), blocks = depth / 16;
    nk_size_t const code_stride = header->depth_tile_count * vector_elements, factor_stride = blocks * vector_elements;
    nk_size_t const tiles = header->column_tile_count, chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_layout_sme_(columns, depth, tile_dimension, 16, 4);
    nk_u8_t const *b_codes = (nk_u8_t const *)b_packed + layout.values;
    nk_u16_t const *b_factors = (nk_u16_t const *)((nk_u8_t const *)b_packed + layout.factors);
    nk_align_(64) nk_u16_t panels[2][nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            nk_u8_t const *codes = b_codes + depth_first / 2 * vector_elements;
            nk_u16_t const *factors = b_factors + depth_first / 16 * vector_elements;
            nk_stage_panel_nvfp4_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_nvfp4_sme_(panels[0], first_rows, row_first, codes, code_stride, factors, factor_stride,
                                        tiles, steps, depth_first == 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_nvfp4_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                      panels[1]);
            nk_sweep_2x2_nvfp4_sme_(panels[0], panels[1], first_rows, second_rows, row_first, codes, code_stride,
                                    factors, factor_stride, tiles, steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/** A × packed B for MXFP4 rows, panels rebased to each row's largest block exponent against byte
 *  codes and decrement vectors, relative to the row and column exponents. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_mxfp4_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride, void const *b_packed,
                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                             nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth(), blocks = depth / 32;
    nk_size_t const code_stride = header->depth_tile_count * vector_elements, factor_stride = blocks * vector_elements;
    nk_size_t const tiles = header->column_tile_count, chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_dots_scaled_sme_layout_t const layout = nk_dots_scaled_layout_sme_(columns, depth, tile_dimension, 32, 4);
    nk_u8_t const *b_codes = (nk_u8_t const *)b_packed + layout.values;
    nk_u16_t const *b_factors = (nk_u16_t const *)((nk_u8_t const *)b_packed + layout.factors);
    nk_align_(64) nk_u16_t panels[2][nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t bases[2][nk_sme_max_tile_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 2 * tile_dimension) {
        nk_size_t const first_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_size_t const remaining = rows - row_first - first_rows;
        nk_size_t const second_rows = remaining < tile_dimension ? remaining : tile_dimension;
        nk_row_bases_sme_(a, row_first, first_rows, blocks, bases[0]);
        nk_row_bases_sme_(a, row_first + first_rows, second_rows, blocks, bases[1]);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            nk_u8_t const *codes = b_codes + depth_first / 2 * vector_elements;
            nk_u16_t const *factors = b_factors + depth_first / 32 * vector_elements;
            nk_stage_panel_mxfp4_sme_(a, a_stride, row_first, first_rows, depth_first, chunk_depth, bases[0],
                                      panels[0]);
            if (!second_rows) {
                nk_sweep_1x4_mxfp4_sme_(panels[0], first_rows, row_first, codes, code_stride, factors, factor_stride,
                                        tiles, steps, depth_first == 0, c, c_stride, columns);
                continue;
            }
            nk_stage_panel_mxfp4_sme_(a, a_stride, row_first + first_rows, second_rows, depth_first, chunk_depth,
                                      bases[1], panels[1]);
            nk_sweep_2x2_mxfp4_sme_(panels[0], panels[1], first_rows, second_rows, row_first, codes, code_stride,
                                    factors, factor_stride, tiles, steps, depth_first == 0, c, c_stride, columns);
        }
    }
}

/*  MXFP6 and MXFP8 packed kernels stage rows rebased to their largest block exponent against
 *  packs of folded bf16 values in 1×4 bodies, relative to the row and column exponents. */

/** A × packed B for MXFP6 E2M3 rows. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_mxfp6e2m3_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride, void const *b_packed,
                                                 nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth(), blocks = depth / 32;
    nk_size_t const tile_stride = header->depth_tile_count * vector_elements, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_align_(64) nk_u16_t panel[nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t bases[nk_sme_max_tile_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += tile_dimension) {
        nk_size_t const block_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_row_bases_sme_(a, row_first, block_rows, blocks, bases);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_stage_panel_mxfp6e2m3_sme_(a, a_stride, row_first, block_rows, depth_first, chunk_depth, bases, panel);
            nk_sweep_1x4_bf16_sme_(panel, block_rows, row_first, b_tiles + depth_first / 2 * vector_elements,
                                   tile_stride, tiles, 0, nk_size_divide_round_up_(chunk_depth, 2), depth_first == 0, 0,
                                   c, c_stride, columns);
        }
    }
}

/** A × packed B for MXFP6 E3M2 rows. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_mxfp6e3m2_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride, void const *b_packed,
                                                 nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth(), blocks = depth / 32;
    nk_size_t const tile_stride = header->depth_tile_count * vector_elements, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_align_(64) nk_u16_t panel[nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t bases[nk_sme_max_tile_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += tile_dimension) {
        nk_size_t const block_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_row_bases_sme_(a, row_first, block_rows, blocks, bases);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_stage_panel_mxfp6e3m2_sme_(a, a_stride, row_first, block_rows, depth_first, chunk_depth, bases, panel);
            nk_sweep_1x4_bf16_sme_(panel, block_rows, row_first, b_tiles + depth_first / 2 * vector_elements,
                                   tile_stride, tiles, 0, nk_size_divide_round_up_(chunk_depth, 2), depth_first == 0, 0,
                                   c, c_stride, columns);
        }
    }
}

/** A × packed B for MXFP8 E4M3 rows. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_mxfp8e4m3_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride, void const *b_packed,
                                                 nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth(), blocks = depth / 32;
    nk_size_t const tile_stride = header->depth_tile_count * vector_elements, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_align_(64) nk_u16_t panel[nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t bases[nk_sme_max_tile_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += tile_dimension) {
        nk_size_t const block_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_row_bases_sme_(a, row_first, block_rows, blocks, bases);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_stage_panel_mxfp8e4m3_sme_(a, a_stride, row_first, block_rows, depth_first, chunk_depth, bases, panel);
            nk_sweep_1x4_bf16_sme_(panel, block_rows, row_first, b_tiles + depth_first / 2 * vector_elements,
                                   tile_stride, tiles, 0, nk_size_divide_round_up_(chunk_depth, 2), depth_first == 0, 0,
                                   c, c_stride, columns);
        }
    }
}

/** A × packed B for MXFP8 E5M2 rows. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_packed_mxfp8e5m2_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride, void const *b_packed,
                                                 nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u16_t const *b_tiles = (nk_u16_t const *)(header + 1);
    nk_size_t const tile_dimension = svcntw(), vector_elements = svcnth(), blocks = depth / 32;
    nk_size_t const tile_stride = header->depth_tile_count * vector_elements, tiles = header->column_tile_count;
    nk_size_t const chunk_dims = nk_sme_panel_bytes_k / svcntb() * 2;
    nk_align_(64) nk_u16_t panel[nk_sme_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t bases[nk_sme_max_tile_k];
    for (nk_size_t row_first = 0; row_first < rows; row_first += tile_dimension) {
        nk_size_t const block_rows = rows - row_first < tile_dimension ? rows - row_first : tile_dimension;
        nk_row_bases_sme_(a, row_first, block_rows, blocks, bases);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_stage_panel_mxfp8e5m2_sme_(a, a_stride, row_first, block_rows, depth_first, chunk_depth, bases, panel);
            nk_sweep_1x4_bf16_sme_(panel, block_rows, row_first, b_tiles + depth_first / 2 * vector_elements,
                                   tile_stride, tiles, 0, nk_size_divide_round_up_(chunk_depth, 2), depth_first == 0, 0,
                                   c, c_stride, columns);
        }
    }
}

#pragma endregion Packed Kernels

/*  Symmetric kernels write the upper triangle of V × Vᵀ over the requested rows: windows of
 *  @c nk_sme_window_tiles_k column tiles decode once per depth chunk, each row tile decodes once
 *  per window, and 1×4 bodies sweep the window; conversions and relative sums follow the packed
 *  kernels, from e2m3 and e2m1 to the block-scaled formats. */

#pragma region Symmetric Kernels

/** The upper triangle of V × Vᵀ for f16 vectors. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_f16_sme_streaming_(nk_f16_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                              nk_size_t depth, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_b16_sme_((nk_u16_t const *)vectors, stride, window_first + tile_first, tile_columns,
                                        depth_first, chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_b16_sme_((nk_u16_t const *)vectors, stride, row_first, rows, depth_first, chunk_depth,
                                        row_panel);
                nk_sweep_1x4_f16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles, window_first,
                                      steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for bf16 vectors. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_bf16_sme_streaming_(nk_bf16_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                               nk_size_t depth, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_b16_sme_((nk_u16_t const *)vectors, stride, window_first + tile_first, tile_columns,
                                        depth_first, chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_b16_sme_((nk_u16_t const *)vectors, stride, row_first, rows, depth_first, chunk_depth,
                                        row_panel);
                nk_sweep_1x4_bf16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles,
                                       window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for e4m3 vectors decoded to f16. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_e4m3_sme_streaming_(nk_e4m3_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                               nk_size_t depth, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_e4m3_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                         chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_e4m3_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_f16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles, window_first,
                                      steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for e5m2 vectors decoded to f16. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_e5m2_sme_streaming_(nk_e5m2_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                               nk_size_t depth, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_e5m2_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                         chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_e5m2_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_f16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles, window_first,
                                      steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for e3m2 vectors decoded to f16. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_e3m2_sme_streaming_(nk_e3m2_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                               nk_size_t depth, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_e3m2_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                         chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_e3m2_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_f16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles, window_first,
                                      steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for NVFP4 vectors as exact f16, relative to the tensor scale. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_nvfp4_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count,
                                                nk_size_t depth, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_nvfp4_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                          chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_nvfp4_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_f16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles, window_first,
                                      steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for MXFP4 vectors rebased to their largest block exponents. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_mxfp4_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count,
                                                nk_size_t depth, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2, blocks = depth / 32;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t row_bases[nk_sme_max_tile_k], column_bases[nk_sme_window_tiles_k * nk_sme_max_tile_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        nk_row_bases_sme_(vectors, window_first, window_columns, blocks, column_bases);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_mxfp4_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                          chunk_depth, column_bases + tile_first, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_row_bases_sme_(vectors, row_first, rows, blocks, row_bases);
                nk_stage_panel_mxfp4_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_bases,
                                          row_panel);
                nk_sweep_1x4_bf16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles,
                                       window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for MXFP6 E2M3 vectors rebased to their largest block exponents. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_mxfp6e2m3_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                    nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
                                                    nk_size_t result_stride, nk_size_t rows_begin,
                                                    nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2, blocks = depth / 32;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t row_bases[nk_sme_max_tile_k], column_bases[nk_sme_window_tiles_k * nk_sme_max_tile_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        nk_row_bases_sme_(vectors, window_first, window_columns, blocks, column_bases);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_mxfp6e2m3_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                              chunk_depth, column_bases + tile_first, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_row_bases_sme_(vectors, row_first, rows, blocks, row_bases);
                nk_stage_panel_mxfp6e2m3_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_bases,
                                              row_panel);
                nk_sweep_1x4_bf16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles,
                                       window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for MXFP6 E3M2 vectors rebased to their largest block exponents. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_mxfp6e3m2_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                    nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
                                                    nk_size_t result_stride, nk_size_t rows_begin,
                                                    nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2, blocks = depth / 32;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t row_bases[nk_sme_max_tile_k], column_bases[nk_sme_window_tiles_k * nk_sme_max_tile_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        nk_row_bases_sme_(vectors, window_first, window_columns, blocks, column_bases);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_mxfp6e3m2_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                              chunk_depth, column_bases + tile_first, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_row_bases_sme_(vectors, row_first, rows, blocks, row_bases);
                nk_stage_panel_mxfp6e3m2_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_bases,
                                              row_panel);
                nk_sweep_1x4_bf16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles,
                                       window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for MXFP8 E4M3 vectors rebased to their largest block exponents. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_mxfp8e4m3_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                    nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
                                                    nk_size_t result_stride, nk_size_t rows_begin,
                                                    nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2, blocks = depth / 32;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t row_bases[nk_sme_max_tile_k], column_bases[nk_sme_window_tiles_k * nk_sme_max_tile_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        nk_row_bases_sme_(vectors, window_first, window_columns, blocks, column_bases);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_mxfp8e4m3_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                              chunk_depth, column_bases + tile_first, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_row_bases_sme_(vectors, row_first, rows, blocks, row_bases);
                nk_stage_panel_mxfp8e4m3_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_bases,
                                              row_panel);
                nk_sweep_1x4_bf16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles,
                                       window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for MXFP8 E5M2 vectors rebased to their largest block exponents. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_mxfp8e5m2_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                    nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
                                                    nk_size_t result_stride, nk_size_t rows_begin,
                                                    nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 2, blocks = depth / 32;
    nk_size_t const panel_elements = nk_sme_window_panel_bytes_k / sizeof(nk_u16_t);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u16_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_align_(64) nk_u16_t row_panel[nk_sme_window_panel_bytes_k / sizeof(nk_u16_t)];
    nk_i32_t row_bases[nk_sme_max_tile_k], column_bases[nk_sme_window_tiles_k * nk_sme_max_tile_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        nk_row_bases_sme_(vectors, window_first, window_columns, blocks, column_bases);
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_size_divide_round_up_(chunk_depth, 2);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_mxfp8e5m2_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                              chunk_depth, column_bases + tile_first, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_row_bases_sme_(vectors, row_first, rows, blocks, row_bases);
                nk_stage_panel_mxfp8e5m2_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_bases,
                                              row_panel);
                nk_sweep_1x4_bf16_sme_(row_panel, rows, row_first, window[0], panel_elements, window_tiles,
                                       window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for i8 vectors. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_i8_sme_streaming_(nk_i8_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                             nk_size_t depth, nk_i32_t *result, nk_size_t result_stride,
                                             nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 4;
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u8_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k];
    nk_align_(64) nk_u8_t row_panel[nk_sme_window_panel_bytes_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(4, chunk_depth);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_b8_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                       chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_b8_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_i8_sme_(row_panel, rows, row_first, window[0], nk_sme_window_panel_bytes_k, window_tiles,
                                     window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for u8 vectors. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_u8_sme_streaming_(nk_u8_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                             nk_size_t depth, nk_u32_t *result, nk_size_t result_stride,
                                             nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 4;
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u8_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k];
    nk_align_(64) nk_u8_t row_panel[nk_sme_window_panel_bytes_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(4, chunk_depth);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_b8_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                       chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_b8_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_u8_sme_(row_panel, rows, row_first, window[0], nk_sme_window_panel_bytes_k, window_tiles,
                                     window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for e2m3 vectors as i8 sixteenths, converted to F32 at the end. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_e2m3_sme_streaming_(nk_e2m3_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                               nk_size_t depth, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 4;
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u8_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k];
    nk_align_(64) nk_u8_t row_panel[nk_sme_window_panel_bytes_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(4, chunk_depth);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_e2m3_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                         chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_e2m3_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_i8_sme_(row_panel, rows, row_first, window[0], nk_sme_window_panel_bytes_k, window_tiles,
                                     window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
    nk_scale_rows_i32_sme_(result, result_stride, rows_begin, rows_end, vector_count, 1, 1.0f / 256);
}

/** The upper triangle of V × Vᵀ for e2m1 vectors as doubled i8, converted to F32 at the end. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_e2m1_sme_streaming_(nk_e2m1x2_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                               nk_size_t depth, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 4;
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u8_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k];
    nk_align_(64) nk_u8_t row_panel[nk_sme_window_panel_bytes_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(4, chunk_depth);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_e2m1_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                         chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_e2m1_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_i8_sme_(row_panel, rows, row_first, window[0], nk_sme_window_panel_bytes_k, window_tiles,
                                     window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
    nk_scale_rows_i32_sme_(result, result_stride, rows_begin, rows_end, vector_count, 1, 0.25f);
}

/** The upper triangle of V × Vᵀ for i4 vectors as sign-extended nibble steps. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_i4_sme_streaming_(nk_i4x2_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                             nk_size_t depth, nk_i32_t *result, nk_size_t result_stride,
                                             nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 4;
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u8_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k];
    nk_align_(64) nk_u8_t row_panel[nk_sme_window_panel_bytes_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(8, chunk_depth);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_i4_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                       chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_i4_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_i8_sme_(row_panel, rows, row_first, window[0], nk_sme_window_panel_bytes_k, window_tiles,
                                     window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

/** The upper triangle of V × Vᵀ for u4 vectors as zero-extended nibble steps. */
__arm_new("za") NUMKONG_OUTLINED_
    void nk_dots_symmetric_u4_sme_streaming_(nk_u4x2_t const *vectors, nk_size_t stride, nk_size_t vector_count,
                                             nk_size_t depth, nk_u32_t *result, nk_size_t result_stride,
                                             nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), window_width = nk_sme_window_tiles_k * tile_dimension;
    nk_size_t const chunk_dims = nk_sme_window_panel_bytes_k / svcntb() * 4;
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_align_(64) nk_u8_t window[nk_sme_window_tiles_k][nk_sme_window_panel_bytes_k];
    nk_align_(64) nk_u8_t row_panel[nk_sme_window_panel_bytes_k];
    for (nk_size_t window_first = rows_begin / tile_dimension * tile_dimension;
         window_first < vector_count && rows_begin < rows_end; window_first += window_width) {
        nk_size_t const window_remaining = vector_count - window_first;
        nk_size_t const window_columns = window_remaining < window_width ? window_remaining : window_width;
        nk_size_t const window_tiles = nk_size_divide_round_up_(window_columns, tile_dimension);
        nk_size_t const row_limit = rows_end < window_first + window_columns ? rows_end : window_first + window_columns;
        for (nk_size_t depth_first = 0; depth_first == 0 || depth_first < depth; depth_first += chunk_dims) {
            nk_size_t const chunk_depth = depth - depth_first < chunk_dims ? depth - depth_first : chunk_dims;
            nk_size_t const steps = nk_panel_steps_b8_sme_(8, chunk_depth);
            for (nk_size_t tile = 0; tile < window_tiles; ++tile) {
                nk_size_t const tile_first = tile * tile_dimension;
                nk_size_t const tile_columns = window_columns - tile_first < tile_dimension
                                                   ? window_columns - tile_first
                                                   : tile_dimension;
                nk_stage_panel_u4_sme_(vectors, stride, window_first + tile_first, tile_columns, depth_first,
                                       chunk_depth, window[tile]);
            }
            for (nk_size_t row_first = rows_begin; row_first < row_limit; row_first += tile_dimension) {
                nk_size_t const rows = rows_end - row_first < tile_dimension ? rows_end - row_first : tile_dimension;
                nk_stage_panel_u4_sme_(vectors, stride, row_first, rows, depth_first, chunk_depth, row_panel);
                nk_sweep_1x4_u8_sme_(row_panel, rows, row_first, window[0], nk_sme_window_panel_bytes_k, window_tiles,
                                     window_first, steps, depth_first == 0, 1, result, result_stride, vector_count);
            }
        }
    }
}

#pragma endregion Symmetric Kernels

/*  Exact passes overwrite the relative sums of rows from @p row_first up to @p row_end of @p a
 *  against columns from @p column_first up to @p column_end of @p b, from each row on when
 *  @p upper, where the row or the column spans more than the fold covers, with exact sums relative
 *  to 2^(E_row + E_column): wide rows take the exponent of their exact norm, @p column_bases marks
 *  wide columns −128, and @p column_exponents holds every column's E, both indexed from
 *  @p column_first. */

#pragma region Block Scaled Finishers

/** Exact pass over MXFP4 rows. */
NUMKONG_INLINE void nk_dots_scaled_exact_mxfp4_sme_(nk_cross_operand_t a, nk_size_t a_stride, nk_size_t row_first,
                                                    nk_size_t row_end, nk_cross_operand_t b, nk_size_t b_stride,
                                                    nk_size_t column_first, nk_size_t column_end,
                                                    nk_i8_t const *column_bases, nk_i32_t const *column_exponents,
                                                    int upper, nk_size_t depth, nk_f32_t *c,
                                                    nk_size_t c_stride) NUMKONG_STREAMABLE_ {
    nk_size_t const blocks = depth / 32;
    int wide_columns = 0;
    for (nk_size_t column = column_first; column < column_end; ++column)
        wide_columns |= column_bases[column - column_first] == -128;
    for (nk_size_t row = row_first; row < row_end; ++row) {
        nk_u8_t const *a_codes = (nk_u8_t const *)a.elements + row * a_stride;
        nk_u8_t const *a_scales = a.scales + row * a.scales_stride;
        int wide_row;
        nk_i32_t row_exponent = nk_mx_base_sme_(a_scales, blocks, &wide_row);
        if (!wide_row && !wide_columns) continue;
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        for (nk_size_t column = upper && row > column_first ? row : column_first; column < column_end; ++column) {
            if (!wide_row && column_bases[column - column_first] != -128) continue;
            nk_cross_wide_sum_t a_sumsq, b_sumsq;
            nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp4_serial_(
                a_codes, a_scales, (nk_u8_t const *)b.elements + column * b_stride, b.scales + column * b.scales_stride,
                depth, &a_sumsq, &b_sumsq);
            if (wide_row) nk_wide_norm_sme_(a_sumsq, &row_exponent);
            c_row[column] = nk_scale_f32_serial_(dot.sum,
                                                 dot.exponent - row_exponent - column_exponents[column - column_first]);
        }
    }
}

/** Exact pass over MXFP6 E2M3 rows. */
NUMKONG_INLINE void nk_dots_scaled_exact_mxfp6e2m3_sme_(nk_cross_operand_t a, nk_size_t a_stride, nk_size_t row_first,
                                                        nk_size_t row_end, nk_cross_operand_t b, nk_size_t b_stride,
                                                        nk_size_t column_first, nk_size_t column_end,
                                                        nk_i8_t const *column_bases, nk_i32_t const *column_exponents,
                                                        int upper, nk_size_t depth, nk_f32_t *c,
                                                        nk_size_t c_stride) NUMKONG_STREAMABLE_ {
    nk_size_t const blocks = depth / 32;
    int wide_columns = 0;
    for (nk_size_t column = column_first; column < column_end; ++column)
        wide_columns |= column_bases[column - column_first] == -128;
    for (nk_size_t row = row_first; row < row_end; ++row) {
        nk_u8_t const *a_codes = (nk_u8_t const *)a.elements + row * a_stride;
        nk_u8_t const *a_scales = a.scales + row * a.scales_stride;
        int wide_row;
        nk_i32_t row_exponent = nk_mx_base_sme_(a_scales, blocks, &wide_row);
        if (!wide_row && !wide_columns) continue;
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        for (nk_size_t column = upper && row > column_first ? row : column_first; column < column_end; ++column) {
            if (!wide_row && column_bases[column - column_first] != -128) continue;
            nk_cross_wide_sum_t a_sumsq, b_sumsq;
            nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp6e2m3_serial_(
                a_codes, a_scales, (nk_u8_t const *)b.elements + column * b_stride, b.scales + column * b.scales_stride,
                depth, &a_sumsq, &b_sumsq);
            if (wide_row) nk_wide_norm_sme_(a_sumsq, &row_exponent);
            c_row[column] = nk_scale_f32_serial_(dot.sum,
                                                 dot.exponent - row_exponent - column_exponents[column - column_first]);
        }
    }
}

/** Exact pass over MXFP6 E3M2 rows. */
NUMKONG_INLINE void nk_dots_scaled_exact_mxfp6e3m2_sme_(nk_cross_operand_t a, nk_size_t a_stride, nk_size_t row_first,
                                                        nk_size_t row_end, nk_cross_operand_t b, nk_size_t b_stride,
                                                        nk_size_t column_first, nk_size_t column_end,
                                                        nk_i8_t const *column_bases, nk_i32_t const *column_exponents,
                                                        int upper, nk_size_t depth, nk_f32_t *c,
                                                        nk_size_t c_stride) NUMKONG_STREAMABLE_ {
    nk_size_t const blocks = depth / 32;
    int wide_columns = 0;
    for (nk_size_t column = column_first; column < column_end; ++column)
        wide_columns |= column_bases[column - column_first] == -128;
    for (nk_size_t row = row_first; row < row_end; ++row) {
        nk_u8_t const *a_codes = (nk_u8_t const *)a.elements + row * a_stride;
        nk_u8_t const *a_scales = a.scales + row * a.scales_stride;
        int wide_row;
        nk_i32_t row_exponent = nk_mx_base_sme_(a_scales, blocks, &wide_row);
        if (!wide_row && !wide_columns) continue;
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        for (nk_size_t column = upper && row > column_first ? row : column_first; column < column_end; ++column) {
            if (!wide_row && column_bases[column - column_first] != -128) continue;
            nk_cross_wide_sum_t a_sumsq, b_sumsq;
            nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp6e3m2_serial_(
                a_codes, a_scales, (nk_u8_t const *)b.elements + column * b_stride, b.scales + column * b.scales_stride,
                depth, &a_sumsq, &b_sumsq);
            if (wide_row) nk_wide_norm_sme_(a_sumsq, &row_exponent);
            c_row[column] = nk_scale_f32_serial_(dot.sum,
                                                 dot.exponent - row_exponent - column_exponents[column - column_first]);
        }
    }
}

/** Exact pass over MXFP8 E4M3 rows. */
NUMKONG_INLINE void nk_dots_scaled_exact_mxfp8e4m3_sme_(nk_cross_operand_t a, nk_size_t a_stride, nk_size_t row_first,
                                                        nk_size_t row_end, nk_cross_operand_t b, nk_size_t b_stride,
                                                        nk_size_t column_first, nk_size_t column_end,
                                                        nk_i8_t const *column_bases, nk_i32_t const *column_exponents,
                                                        int upper, nk_size_t depth, nk_f32_t *c,
                                                        nk_size_t c_stride) NUMKONG_STREAMABLE_ {
    nk_size_t const blocks = depth / 32;
    int wide_columns = 0;
    for (nk_size_t column = column_first; column < column_end; ++column)
        wide_columns |= column_bases[column - column_first] == -128;
    for (nk_size_t row = row_first; row < row_end; ++row) {
        nk_u8_t const *a_codes = (nk_u8_t const *)a.elements + row * a_stride;
        nk_u8_t const *a_scales = a.scales + row * a.scales_stride;
        int wide_row;
        nk_i32_t row_exponent = nk_mx_base_sme_(a_scales, blocks, &wide_row);
        if (!wide_row && !wide_columns) continue;
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        for (nk_size_t column = upper && row > column_first ? row : column_first; column < column_end; ++column) {
            if (!wide_row && column_bases[column - column_first] != -128) continue;
            nk_cross_wide_sum_t a_sumsq, b_sumsq;
            nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp8e4m3_serial_(
                a_codes, a_scales, (nk_u8_t const *)b.elements + column * b_stride, b.scales + column * b.scales_stride,
                depth, &a_sumsq, &b_sumsq);
            if (wide_row) nk_wide_norm_sme_(a_sumsq, &row_exponent);
            c_row[column] = nk_scale_f32_serial_(dot.sum,
                                                 dot.exponent - row_exponent - column_exponents[column - column_first]);
        }
    }
}

/** Exact pass over MXFP8 E5M2 rows. */
NUMKONG_INLINE void nk_dots_scaled_exact_mxfp8e5m2_sme_(nk_cross_operand_t a, nk_size_t a_stride, nk_size_t row_first,
                                                        nk_size_t row_end, nk_cross_operand_t b, nk_size_t b_stride,
                                                        nk_size_t column_first, nk_size_t column_end,
                                                        nk_i8_t const *column_bases, nk_i32_t const *column_exponents,
                                                        int upper, nk_size_t depth, nk_f32_t *c,
                                                        nk_size_t c_stride) NUMKONG_STREAMABLE_ {
    nk_size_t const blocks = depth / 32;
    int wide_columns = 0;
    for (nk_size_t column = column_first; column < column_end; ++column)
        wide_columns |= column_bases[column - column_first] == -128;
    for (nk_size_t row = row_first; row < row_end; ++row) {
        nk_u8_t const *a_codes = (nk_u8_t const *)a.elements + row * a_stride;
        nk_u8_t const *a_scales = a.scales + row * a.scales_stride;
        int wide_row;
        nk_i32_t row_exponent = nk_mx_base_sme_(a_scales, blocks, &wide_row);
        if (!wide_row && !wide_columns) continue;
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        for (nk_size_t column = upper && row > column_first ? row : column_first; column < column_end; ++column) {
            if (!wide_row && column_bases[column - column_first] != -128) continue;
            nk_cross_wide_sum_t a_sumsq, b_sumsq;
            nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp8e5m2_serial_(
                a_codes, a_scales, (nk_u8_t const *)b.elements + column * b_stride, b.scales + column * b.scales_stride,
                depth, &a_sumsq, &b_sumsq);
            if (wide_row) nk_wide_norm_sme_(a_sumsq, &row_exponent);
            c_row[column] = nk_scale_f32_serial_(dot.sum,
                                                 dot.exponent - row_exponent - column_exponents[column - column_first]);
        }
    }
}

/** Finishes @p count relative dots of one row in place: times @p mantissa and two to the power of
 *  @p row_exponent plus each column's E in @p column_exponents, rounding once. */
NUMKONG_INLINE void nk_dots_from_relative_sme_streaming_(nk_f32_t *dots, nk_size_t count, nk_f32_t mantissa,
                                                         nk_i32_t row_exponent,
                                                         nk_i32_t const *column_exponents) NUMKONG_STREAMING_ {
    for (nk_size_t column = 0; column < count; column += svcntw()) {
        svbool_t const predicate_b32x = svwhilelt_b32_u64(column, count);
        svfloat32_t const dots_f32x = svmul_n_f32_x(predicate_b32x, svld1_f32(predicate_b32x, dots + column), mantissa);
        svint32_t const exponents_i32x = svadd_n_s32_x(
            predicate_b32x, svld1_s32(predicate_b32x, column_exponents + column), row_exponent);
        svst1_f32(predicate_b32x, dots + column, svscale_f32_x(predicate_b32x, dots_f32x, exponents_i32x));
    }
}

/*  Dots finalizers turn relative sums into dots in place. Packed ones run the exact pass against
 *  the pack's raw rows first; symmetric ones keep the exponents of @c nk_sme_finish_columns_k
 *  columns at a time. Rows the fold covers take their largest block exponent without a norm. */

/** Finishes packed NVFP4 dots: times both tensor scales. */
NUMKONG_OUTLINED_ void nk_dots_packed_finalize_nvfp4_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                    void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                                    nk_size_t columns, nk_size_t depth,
                                                                    nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 16, 4);
    nk_i32_t exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &exponent) *
                              b.mantissa;
    nk_unused_(a_stride), nk_unused_(depth);
    for (nk_size_t row = 0; row < rows; ++row)
        nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, exponent,
                                             b.exponents);
}

/** Finishes symmetric NVFP4 dots: times the tensor scale squared. */
NUMKONG_OUTLINED_ void nk_dots_symmetric_finalize_nvfp4_sme_streaming_(nk_cross_operand_t vectors, nk_size_t count,
                                                                       nk_f32_t *result, nk_size_t result_stride,
                                                                       nk_size_t rows_begin,
                                                                       nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_i32_t exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale), &exponent);
    nk_i32_t exponents[nk_sme_finish_columns_k];
    for (nk_size_t column = 0; column < nk_sme_finish_columns_k; ++column) exponents[column] = exponent;
    for (nk_size_t row = rows_begin; row < rows_end; ++row) {
        nk_f32_t *result_row = (nk_f32_t *)((char *)result + row * result_stride);
        for (nk_size_t chunk = row; chunk < count; chunk += nk_sme_finish_columns_k)
            nk_dots_from_relative_sme_streaming_(
                result_row + chunk, count - chunk < nk_sme_finish_columns_k ? count - chunk : nk_sme_finish_columns_k,
                mantissa * mantissa, exponent, exponents);
    }
}

/** Finishes packed MXFP4 dots. */
NUMKONG_OUTLINED_ void nk_dots_packed_finalize_mxfp4_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                    void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                                    nk_size_t columns, nk_size_t depth,
                                                                    nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 4);
    nk_dots_scaled_exact_mxfp4_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                    depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        int wide;
        nk_i32_t exponent = nk_mx_base_sme_(a.scales + row * a.scales_stride, depth / 32, &wide);
        if (wide) nk_dots_scaled_row_mxfp4_sme_(a, a_stride, row, depth, &exponent);
        nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), columns, 1, exponent,
                                             b.exponents);
    }
}

/** Finishes packed MXFP6 E2M3 dots. */
NUMKONG_OUTLINED_ void nk_dots_packed_finalize_mxfp6e2m3_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                        void const *b_packed, nk_f32_t *c,
                                                                        nk_size_t rows, nk_size_t columns,
                                                                        nk_size_t depth,
                                                                        nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_dots_scaled_exact_mxfp6e2m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        int wide;
        nk_i32_t exponent = nk_mx_base_sme_(a.scales + row * a.scales_stride, depth / 32, &wide);
        if (wide) nk_dots_scaled_row_mxfp6e2m3_sme_(a, a_stride, row, depth, &exponent);
        nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), columns, 1, exponent,
                                             b.exponents);
    }
}

/** Finishes packed MXFP6 E3M2 dots. */
NUMKONG_OUTLINED_ void nk_dots_packed_finalize_mxfp6e3m2_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                        void const *b_packed, nk_f32_t *c,
                                                                        nk_size_t rows, nk_size_t columns,
                                                                        nk_size_t depth,
                                                                        nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_dots_scaled_exact_mxfp6e3m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        int wide;
        nk_i32_t exponent = nk_mx_base_sme_(a.scales + row * a.scales_stride, depth / 32, &wide);
        if (wide) nk_dots_scaled_row_mxfp6e3m2_sme_(a, a_stride, row, depth, &exponent);
        nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), columns, 1, exponent,
                                             b.exponents);
    }
}

/** Finishes packed MXFP8 E4M3 dots. */
NUMKONG_OUTLINED_ void nk_dots_packed_finalize_mxfp8e4m3_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                        void const *b_packed, nk_f32_t *c,
                                                                        nk_size_t rows, nk_size_t columns,
                                                                        nk_size_t depth,
                                                                        nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_dots_scaled_exact_mxfp8e4m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        int wide;
        nk_i32_t exponent = nk_mx_base_sme_(a.scales + row * a.scales_stride, depth / 32, &wide);
        if (wide) nk_dots_scaled_row_mxfp8e4m3_sme_(a, a_stride, row, depth, &exponent);
        nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), columns, 1, exponent,
                                             b.exponents);
    }
}

/** Finishes packed MXFP8 E5M2 dots. */
NUMKONG_OUTLINED_ void nk_dots_packed_finalize_mxfp8e5m2_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                        void const *b_packed, nk_f32_t *c,
                                                                        nk_size_t rows, nk_size_t columns,
                                                                        nk_size_t depth,
                                                                        nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_dots_scaled_exact_mxfp8e5m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        int wide;
        nk_i32_t exponent = nk_mx_base_sme_(a.scales + row * a.scales_stride, depth / 32, &wide);
        if (wide) nk_dots_scaled_row_mxfp8e5m2_sme_(a, a_stride, row, depth, &exponent);
        nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), columns, 1, exponent,
                                             b.exponents);
    }
}

/*  Symmetric MX dots finalizers take the columns from @p rows_begin on in chunks: their exponents
 *  and wide marks, the exact pass over the chunk, then each row's dots in the chunk. */

/** Finishes symmetric MXFP4 dots. */
NUMKONG_OUTLINED_ void nk_dots_symmetric_finalize_mxfp4_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                       nk_size_t count, nk_size_t depth,
                                                                       nk_f32_t *result, nk_size_t result_stride,
                                                                       nk_size_t rows_begin,
                                                                       nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = count - chunk < nk_sme_finish_columns_k ? count : chunk + nk_sme_finish_columns_k;
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            exponents[column - chunk] = nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            if (wide) nk_dots_scaled_row_mxfp4_sme_(vectors, stride, column, depth, &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp4_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end, bases,
                                        exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            int wide;
            nk_i32_t row_exponent = nk_mx_base_sme_(vectors.scales + row * vectors.scales_stride, blocks, &wide);
            if (wide) nk_dots_scaled_row_mxfp4_sme_(vectors, stride, row, depth, &row_exponent);
            nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                                 chunk_end - first, 1, row_exponent, exponents + (first - chunk));
        }
    }
}

/** Finishes symmetric MXFP6 E2M3 dots. */
NUMKONG_OUTLINED_ void nk_dots_symmetric_finalize_mxfp6e2m3_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                           nk_size_t count, nk_size_t depth,
                                                                           nk_f32_t *result, nk_size_t result_stride,
                                                                           nk_size_t rows_begin,
                                                                           nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = count - chunk < nk_sme_finish_columns_k ? count : chunk + nk_sme_finish_columns_k;
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            exponents[column - chunk] = nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            if (wide) nk_dots_scaled_row_mxfp6e2m3_sme_(vectors, stride, column, depth, &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp6e2m3_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            int wide;
            nk_i32_t row_exponent = nk_mx_base_sme_(vectors.scales + row * vectors.scales_stride, blocks, &wide);
            if (wide) nk_dots_scaled_row_mxfp6e2m3_sme_(vectors, stride, row, depth, &row_exponent);
            nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                                 chunk_end - first, 1, row_exponent, exponents + (first - chunk));
        }
    }
}

/** Finishes symmetric MXFP6 E3M2 dots. */
NUMKONG_OUTLINED_ void nk_dots_symmetric_finalize_mxfp6e3m2_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                           nk_size_t count, nk_size_t depth,
                                                                           nk_f32_t *result, nk_size_t result_stride,
                                                                           nk_size_t rows_begin,
                                                                           nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = count - chunk < nk_sme_finish_columns_k ? count : chunk + nk_sme_finish_columns_k;
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            exponents[column - chunk] = nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            if (wide) nk_dots_scaled_row_mxfp6e3m2_sme_(vectors, stride, column, depth, &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp6e3m2_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            int wide;
            nk_i32_t row_exponent = nk_mx_base_sme_(vectors.scales + row * vectors.scales_stride, blocks, &wide);
            if (wide) nk_dots_scaled_row_mxfp6e3m2_sme_(vectors, stride, row, depth, &row_exponent);
            nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                                 chunk_end - first, 1, row_exponent, exponents + (first - chunk));
        }
    }
}

/** Finishes symmetric MXFP8 E4M3 dots. */
NUMKONG_OUTLINED_ void nk_dots_symmetric_finalize_mxfp8e4m3_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                           nk_size_t count, nk_size_t depth,
                                                                           nk_f32_t *result, nk_size_t result_stride,
                                                                           nk_size_t rows_begin,
                                                                           nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = count - chunk < nk_sme_finish_columns_k ? count : chunk + nk_sme_finish_columns_k;
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            exponents[column - chunk] = nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            if (wide) nk_dots_scaled_row_mxfp8e4m3_sme_(vectors, stride, column, depth, &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp8e4m3_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            int wide;
            nk_i32_t row_exponent = nk_mx_base_sme_(vectors.scales + row * vectors.scales_stride, blocks, &wide);
            if (wide) nk_dots_scaled_row_mxfp8e4m3_sme_(vectors, stride, row, depth, &row_exponent);
            nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                                 chunk_end - first, 1, row_exponent, exponents + (first - chunk));
        }
    }
}

/** Finishes symmetric MXFP8 E5M2 dots. */
NUMKONG_OUTLINED_ void nk_dots_symmetric_finalize_mxfp8e5m2_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                           nk_size_t count, nk_size_t depth,
                                                                           nk_f32_t *result, nk_size_t result_stride,
                                                                           nk_size_t rows_begin,
                                                                           nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = count - chunk < nk_sme_finish_columns_k ? count : chunk + nk_sme_finish_columns_k;
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            exponents[column - chunk] = nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            if (wide) nk_dots_scaled_row_mxfp8e5m2_sme_(vectors, stride, column, depth, &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp8e5m2_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            int wide;
            nk_i32_t row_exponent = nk_mx_base_sme_(vectors.scales + row * vectors.scales_stride, blocks, &wide);
            if (wide) nk_dots_scaled_row_mxfp8e5m2_sme_(vectors, stride, row, depth, &row_exponent);
            nk_dots_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                                 chunk_end - first, 1, row_exponent, exponents + (first - chunk));
        }
    }
}

#pragma endregion Block Scaled Finishers

#if NUMKONG_TARGET_SME

/** Rejects packs another capability produced. */
NUMKONG_INLINE nk_status_t nk_dots_check_sme_(void const *b_packed) {
    return ((nk_dots_sme_packed_header_t const *)b_packed)->capability == nk_cap_sme_k ? nk_success_k
                                                                                       : nk_pack_mismatch_k;
}

/** Reads back the columns and depth an SME pack was made for. */
NUMKONG_INLINE nk_status_t nk_dots_shape_sme_(void const *b_packed, nk_size_t *columns, nk_size_t *depth) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

#pragma region F16 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_f16_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_f16_sme(nk_f16_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_f16_tiles_sme_(b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_f16_sme(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_f16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_f16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                         rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_bf16_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_bf16_sme(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_pack_bf16_tiles_sme_(b, columns, depth, b_stride, b_packed, columns_begin, columns_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_bf16_sme(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_bf16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_bf16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion F16 Floats

#pragma region I8 Integers

NUMKONG_API nk_status_t nk_dots_pack_size_i8_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(/*dims_per_word=*/4, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_i8_sme(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth, nk_panel_steps_b8_sme_(4, depth),
                                                            nk_cntb_sme_(), columns_begin, columns_end, &tile_begin,
                                                            &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_i8_sme_streaming_(b, b_stride, columns, depth,
                                   (nk_u8_t *)b_packed + sizeof(nk_dots_sme_packed_header_t),
                                   (nk_u32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_i8_sme(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_i8_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_i8_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                        rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion I8 Integers

#pragma region E4M3 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e4m3_sme(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth,
                                                            nk_size_divide_round_up_(depth, 2), nk_cntb_sme_(),
                                                            columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_e4m3_sme_streaming_(b, b_stride, columns, depth,
                                     (nk_u16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t)),
                                     (nk_f32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e4m3_sme(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_e4m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E4M3 Floats

#pragma region E5M2 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e5m2_sme(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth,
                                                            nk_size_divide_round_up_(depth, 2), nk_cntb_sme_(),
                                                            columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_e5m2_sme_streaming_(b, b_stride, columns, depth,
                                     (nk_u16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t)),
                                     (nk_f32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e5m2_sme(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_e5m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E5M2 Floats

#pragma region E2M3 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(/*dims_per_word=*/4, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e2m3_sme(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth, nk_panel_steps_b8_sme_(4, depth),
                                                            nk_cntb_sme_(), columns_begin, columns_end, &tile_begin,
                                                            &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_e2m3_sme_streaming_(b, b_stride, columns, depth,
                                     (nk_u8_t *)b_packed + sizeof(nk_dots_sme_packed_header_t),
                                     (nk_u32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e2m3_sme(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_e2m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E2M3 Floats

#pragma region E2M1 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(/*dims_per_word=*/4, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e2m1_sme(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth, nk_panel_steps_b8_sme_(4, depth),
                                                            nk_cntb_sme_(), columns_begin, columns_end, &tile_begin,
                                                            &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_e2m1_sme_streaming_(b, b_stride, columns, depth,
                                     (nk_u8_t *)b_packed + sizeof(nk_dots_sme_packed_header_t),
                                     (nk_u32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e2m1_sme(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_e2m1_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_sme(nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_start_sme_streaming_();
    nk_dots_symmetric_e2m1_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E2M1 Floats

#pragma region E3M2 Floats

NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b16_sme_(columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_e3m2_sme(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth,
                                                            nk_size_divide_round_up_(depth, 2), nk_cntb_sme_(),
                                                            columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_e3m2_sme_streaming_(b, b_stride, columns, depth,
                                     (nk_u16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t)),
                                     (nk_f32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_e3m2_sme(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_e3m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E3M2 Floats

#pragma region U8 Integers

NUMKONG_API nk_status_t nk_dots_pack_size_u8_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(/*dims_per_word=*/4, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_u8_sme(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth, nk_panel_steps_b8_sme_(4, depth),
                                                            nk_cntb_sme_(), columns_begin, columns_end, &tile_begin,
                                                            &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_u8_sme_streaming_(b, b_stride, columns, depth,
                                   (nk_u8_t *)b_packed + sizeof(nk_dots_sme_packed_header_t),
                                   (nk_u32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_u8_sme(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_u8_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_u8_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                        rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion U8 Integers

/*  u4 and i4 packs keep each 4-bit dim as a byte, split per 8-dim word into a low-nibble vector and
 *  a high-nibble vector, so their panels and outer products match u8 and i8 at twice the steps. */

#pragma region U4 Integers

NUMKONG_API nk_status_t nk_dots_pack_size_u4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(/*dims_per_word=*/8, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_u4_sme(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth, nk_panel_steps_b8_sme_(8, depth),
                                                            nk_cntb_sme_(), columns_begin, columns_end, &tile_begin,
                                                            &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_u4_sme_streaming_(b, b_stride, columns, depth,
                                   (nk_u8_t *)b_packed + sizeof(nk_dots_sme_packed_header_t),
                                   (nk_u32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_u4_sme(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_u4_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_u4_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                        rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion U4 Integers

#pragma region I4 Integers

NUMKONG_API nk_status_t nk_dots_pack_size_i4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_pack_size_b8_sme_(/*dims_per_word=*/8, columns, depth);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_i4_sme(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t tile_begin, tile_end;
    nk_size_t const norms_offset = nk_dots_pack_window_sme_(b_packed, columns, depth, nk_panel_steps_b8_sme_(8, depth),
                                                            nk_cntb_sme_(), columns_begin, columns_end, &tile_begin,
                                                            &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_i4_sme_streaming_(b, b_stride, columns, depth,
                                   (nk_u8_t *)b_packed + sizeof(nk_dots_sme_packed_header_t),
                                   (nk_u32_t *)((char *)b_packed + norms_offset), tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_i4_sme(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_start_sme_streaming_();
    nk_dots_packed_i4_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    nk_start_sme_streaming_();
    nk_dots_symmetric_i4_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                        rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion I4 Integers

#pragma region Block Scaled Formats

NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 16, 4).total;
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_sme(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL && depth % 16 == 0);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_nvfp4_k, b, b_stride);
    nk_size_t tile_begin, tile_end;
    nk_dots_pack_scaled_window_sme_(operand, columns, depth, b_packed,
                                    nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 16, 4).norms,
                                    columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_nvfp4_sme_streaming_(operand, b_stride, columns, depth, (nk_u8_t *)b_packed, tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_nvfp4_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_nvfp4_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_dots_packed_finalize_nvfp4_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                    nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_nvfp4_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_nvfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_dots_symmetric_finalize_nvfp4_sme_streaming_(operand, vector_count, result, result_stride, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 4).total;
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_sme(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL && depth % 32 == 0);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp4_k, b, b_stride);
    nk_size_t tile_begin, tile_end;
    nk_dots_pack_scaled_window_sme_(operand, columns, depth, b_packed,
                                    nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 4).norms,
                                    columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_mxfp4_sme_streaming_(operand, b_stride, columns, depth, (nk_u8_t *)b_packed, tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp4_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp4_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_dots_packed_finalize_mxfp4_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                    nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp4_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_dots_symmetric_finalize_mxfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                    rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 8).total;
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL && depth % 32 == 0);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, b, b_stride);
    nk_size_t tile_begin, tile_end;
    nk_dots_pack_scaled_window_sme_(operand, columns, depth, b_packed,
                                    nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 8).norms,
                                    columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_mxfp6e2m3_sme_streaming_(operand, b_stride, columns, depth, (nk_u8_t *)b_packed, tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp6e2m3_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_dots_packed_finalize_mxfp6e2m3_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp6e2m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_dots_symmetric_finalize_mxfp6e2m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                        rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 8).total;
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL && depth % 32 == 0);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, b, b_stride);
    nk_size_t tile_begin, tile_end;
    nk_dots_pack_scaled_window_sme_(operand, columns, depth, b_packed,
                                    nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 8).norms,
                                    columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_mxfp6e3m2_sme_streaming_(operand, b_stride, columns, depth, (nk_u8_t *)b_packed, tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp6e3m2_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_dots_packed_finalize_mxfp6e3m2_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp6e3m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_dots_symmetric_finalize_mxfp6e3m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                        rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 8).total;
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL && depth % 32 == 0);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, b, b_stride);
    nk_size_t tile_begin, tile_end;
    nk_dots_pack_scaled_window_sme_(operand, columns, depth, b_packed,
                                    nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 8).norms,
                                    columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_mxfp8e4m3_sme_streaming_(operand, b_stride, columns, depth, (nk_u8_t *)b_packed, tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp8e4m3_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_dots_packed_finalize_mxfp8e4m3_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp8e4m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_dots_symmetric_finalize_mxfp8e4m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                        rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 8).total;
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_dots_shape_sme_(b_packed, columns, depth);
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL && depth % 32 == 0);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, b, b_stride);
    nk_size_t tile_begin, tile_end;
    nk_dots_pack_scaled_window_sme_(operand, columns, depth, b_packed,
                                    nk_dots_scaled_layout_sme_(columns, depth, nk_cntw_sme_(), 32, 8).norms,
                                    columns_begin, columns_end, &tile_begin, &tile_end);
    nk_start_sme_streaming_();
    nk_dots_pack_mxfp8e5m2_sme_streaming_(operand, b_stride, columns, depth, (nk_u8_t *)b_packed, tile_begin, tile_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp8e5m2_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_dots_packed_finalize_mxfp8e5m2_sme_streaming_(operand, a_stride, packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp8e5m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_dots_symmetric_finalize_mxfp8e5m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                        rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
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
