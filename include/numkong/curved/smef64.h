/**
 *  @brief SIMD-accelerated Curved Space Similarity for SME F64.
 *  @file include/numkong/curved/smef64.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements bilinear forms and Mahalanobis distance using ARM SME:
 *  - f32 inputs: GEMV via f64 FMOPA (widening load f32→f64, exact accumulation)
 *  - f64 inputs: row-by-row streaming SVE with Dot2 (Ogita-Rump-Oishi 2005)
 *  - f32c complex: 4-FMOPA complex GEMV with FMOPS for the cᵢₘ×bᵢₘ subtraction
 *  - f64c complex: interleaved Dot2 with permute + deferred XOR sign-flip
 *
 *  Complex number history — approaches tried and abandoned:
 *
 *  1. Ozaki 3-way FMOPA (f64c): Split each f64 into 19+17+17 bit mantissa parts,
 *     compute 7 ZA tiles of FMOPA/FMOPS per inner step (24 tile ops total).
 *     Abandoned: the 3× split + 4× complex cross-terms = 12× tile ops vs real,
 *     with staging overhead dominating at GEMV (not GEMM) granularity.
 *
 *  2. Deinterleaved 4-accumulator SVE Dot2 (f64c): Separate real/imaginary via
 *     UZP1/UZP2, run 4 independent Dot2 chains (rr, ii, ri, ir). Theoretically
 *     matches the serial kernel's arithmetic intensity, but UZP on SVE requires
 *     loading 2 vectors to produce 1 full-width deinterleaved vector, and the
 *     total ops/byte is identical to the interleaved approach (~28 SVE ops/iter).
 *
 *  3. Simple (non-Ozaki) f64 FMOPA for complex: Would give ~5-10 GFLOP/s but
 *     drops Dot2 compensation entirely (naive f64 accumulation, ~BLAS precision).
 *     Not implemented because precision is a core requirement for f64 kernels.
 *
 *  The current interleaved Dot2 approach (2 accumulators + svtbl swap + XOR sign
 *  flip) is the best balance found: ~15 SVE ops/iter vs ~28 for deinterleaved,
 *  with identical Dot2 precision. The ~1.5 GFLOP/s throughput is limited by the
 *  SME coprocessor's slow per-instruction pipeline — the serial version achieves
 *  ~2.2 GFLOP/s despite using software Dekker FMA (~20 ops/TwoProd vs SVE's 3)
 *  because it runs on the faster main core.
 *
 *  On Apple M4, SVE instructions are only available inside SME streaming mode.
 *  Functions using SVE intrinsics are marked `__arm_locally_streaming` in a
 *  `_streaming_` helper; the NK_PUBLIC entry point is a thin non-streaming
 *  wrapper. NEON intrinsics cannot be called from streaming mode, so Mahalanobis
 *  functions split into a streaming helper (SVE) and a non-streaming wrapper
 *  (NEON sqrt).
 *
 *  @see Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product"
 */
#ifndef NK_CURVED_SMEF64_H
#define NK_CURVED_SMEF64_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SMEF64

#include "numkong/types.h"
#include "numkong/spatial/neon.h"  // `nk_f64_sqrt_neon`
#include "numkong/dots/sme.h"      // nk_sme_zero_za64_tile_0_, etc. (for f32 FMOPA)
#include "numkong/curved/serial.h" // `nk_bilinear_f64_serial`, etc.

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sme-f64f64"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#endif

/**
 *  @brief SVE Dot2 accumulator: sum += a × b with error compensation.
 *  Uses TwoProd (svneg+svnmls) and TwoSum error-free transformations.
 */
NK_PUBLIC void nk_dot2_f64_sve_accumulate_(svbool_t predicate_b64x, svfloat64_t *sum, svfloat64_t *comp,
                                           svfloat64_t a_f64x, svfloat64_t b_f64x) NK_STREAMING_ {
    svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_f64x, b_f64x);
    svfloat64_t product_error_f64x = svneg_f64_x(predicate_b64x,
                                                 svnmls_f64_x(predicate_b64x, product_f64x, a_f64x, b_f64x));
    svfloat64_t running_sum_f64x = svadd_f64_m(predicate_b64x, *sum, product_f64x);
    svfloat64_t recovered_addend_f64x = svsub_f64_x(predicate_b64x, running_sum_f64x, *sum);
    svfloat64_t sum_error_f64x = svadd_f64_x(
        predicate_b64x,
        svsub_f64_x(predicate_b64x, *sum, svsub_f64_x(predicate_b64x, running_sum_f64x, recovered_addend_f64x)),
        svsub_f64_x(predicate_b64x, product_f64x, recovered_addend_f64x));
    *sum = running_sum_f64x;
    *comp = svadd_f64_m(predicate_b64x, *comp, svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
}

/**
 *  @brief f32 bilinear: GEMV via FMOPA (widening f32→f64, exact accumulation).
 *  ZA0.D = C staging, ZA1.D = GEMV accumulator.
 */
__arm_locally_streaming __arm_new("za") static void nk_bilinear_f32_smef64_streaming_(
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t dimensions, nk_f64_t *result) {
    svbool_t predicate_body_b64x = svptrue_b64();
    nk_size_t tile_dimension = svcntd();
    nk_f64_t outer_sum_f64 = 0.0;

    for (nk_size_t row = 0; row < dimensions; row += tile_dimension) {
        nk_size_t rows_remaining = (row + tile_dimension <= dimensions) ? tile_dimension : (dimensions - row);
        svbool_t row_predicate_b64x = svwhilelt_b64_u64(0u, rows_remaining);

        svzero_mask_za(nk_sme_zero_za64_tile_1_);

        for (nk_size_t j = 0; j < dimensions; j += tile_dimension) {
            nk_size_t batch_size = (j + tile_dimension <= dimensions) ? tile_dimension : (dimensions - j);
            svbool_t batch_predicate_b64x = svwhilelt_b64_u64(0u, batch_size);

            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                svfloat64_t c_row_f64x = svcvt_f64_f32_x(
                    batch_predicate_b64x,
                    svreinterpret_f32_u64(
                        svld1uw_u64(batch_predicate_b64x, (nk_u32_t const *)(c + (row + r) * dimensions + j))));
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64x, c_row_f64x);
            }

            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_col_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 0, k);
                svmopa_za64_f64_m(1, row_predicate_b64x, row_predicate_b64x, c_col_f64x, svdup_f64((nk_f64_t)b[j + k]));
            }
        }

        svfloat64_t v_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 1, 0);
        svfloat64_t a_f64x = svcvt_f64_f32_x(
            row_predicate_b64x, svreinterpret_f32_u64(svld1uw_u64(row_predicate_b64x, (nk_u32_t const *)(a + row))));
        outer_sum_f64 += svaddv_f64(predicate_body_b64x, svmul_f64_x(row_predicate_b64x, a_f64x, v_f64x));
        NK_UNPOISON(&outer_sum_f64, sizeof(outer_sum_f64));
    }

    *result = outer_sum_f64;
}

NK_PUBLIC void nk_bilinear_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t dimensions,
                                      nk_f64_t *result) {
    nk_bilinear_f32_smef64_streaming_(a, b, c, dimensions, result);
}

/**
 *  @brief f32 Mahalanobis: GEMV v = C×d via FMOPA, where d = a − b (exact in f64).
 *  ZA0.D = C staging, ZA1.D = GEMV accumulator.
 */
__arm_locally_streaming __arm_new("za") static nk_f64_t
    nk_mahalanobis_f32_smef64_streaming_(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c,
                                         nk_size_t dimensions) {

    svbool_t predicate_body_b64x = svptrue_b64();
    nk_size_t tile_dimension = svcntd();
    nk_f64_t outer_sum_f64 = 0.0;

    for (nk_size_t row = 0; row < dimensions; row += tile_dimension) {
        nk_size_t rows_remaining = (row + tile_dimension <= dimensions) ? tile_dimension : (dimensions - row);
        svbool_t row_predicate_b64x = svwhilelt_b64_u64(0u, rows_remaining);

        svzero_mask_za(nk_sme_zero_za64_tile_1_);

        for (nk_size_t j = 0; j < dimensions; j += tile_dimension) {
            nk_size_t batch_size = (j + tile_dimension <= dimensions) ? tile_dimension : (dimensions - j);
            svbool_t batch_predicate_b64x = svwhilelt_b64_u64(0u, batch_size);

            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                svfloat64_t c_row_f64x = svcvt_f64_f32_x(
                    batch_predicate_b64x,
                    svreinterpret_f32_u64(
                        svld1uw_u64(batch_predicate_b64x, (nk_u32_t const *)(c + (row + r) * dimensions + j))));
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64x, c_row_f64x);
            }

            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_col_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 0, k);
                nk_f64_t d_k = (nk_f64_t)a[j + k] - (nk_f64_t)b[j + k];
                svmopa_za64_f64_m(1, row_predicate_b64x, row_predicate_b64x, c_col_f64x, svdup_f64(d_k));
            }
        }

        svfloat64_t v_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 1, 0);
        svfloat64_t a_f64x = svcvt_f64_f32_x(
            row_predicate_b64x, svreinterpret_f32_u64(svld1uw_u64(row_predicate_b64x, (nk_u32_t const *)(a + row))));
        svfloat64_t b_f64x = svcvt_f64_f32_x(
            row_predicate_b64x, svreinterpret_f32_u64(svld1uw_u64(row_predicate_b64x, (nk_u32_t const *)(b + row))));
        svfloat64_t d_f64x = svsub_f64_x(row_predicate_b64x, a_f64x, b_f64x);
        outer_sum_f64 += svaddv_f64(predicate_body_b64x, svmul_f64_x(row_predicate_b64x, d_f64x, v_f64x));
        NK_UNPOISON(&outer_sum_f64, sizeof(outer_sum_f64));
    }

    return outer_sum_f64;
}

NK_PUBLIC void nk_mahalanobis_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t dimensions,
                                         nk_f64_t *result) {
    nk_f64_t quadratic = nk_mahalanobis_f32_smef64_streaming_(a, b, c, dimensions);
    *result = nk_f64_sqrt_neon(quadratic > 0 ? quadratic : 0);
}

/**
 *  @brief f64 bilinear: row-by-row streaming SVE with Dot2 compensation.
 *  4-row fast path shares b_f64x loads; 1-row tail for remainder.
 */
__arm_locally_streaming static void nk_bilinear_f64_smef64_streaming_(nk_f64_t const *a, nk_f64_t const *b,
                                                                      nk_f64_t const *c, nk_size_t dimensions,
                                                                      nk_f64_t *result) {
    svbool_t predicate_all_b64x = svptrue_b64();
    nk_f64_t outer_sum = 0.0, outer_comp = 0.0;
    nk_size_t row = 0;

    // 4-row fast path: share b_f64x load across 4 rows
    for (; row + 4 <= dimensions; row += 4) {
        nk_f64_t a0 = a[row + 0], a1 = a[row + 1], a2 = a[row + 2], a3 = a[row + 3];
        svfloat64_t sum_0_f64x = svdup_f64(0), compensation_0_f64x = svdup_f64(0);
        svfloat64_t sum_1_f64x = svdup_f64(0), compensation_1_f64x = svdup_f64(0);
        svfloat64_t sum_2_f64x = svdup_f64(0), compensation_2_f64x = svdup_f64(0);
        svfloat64_t sum_3_f64x = svdup_f64(0), compensation_3_f64x = svdup_f64(0);
        nk_size_t j = 0;
        svbool_t predicate_b64x = svwhilelt_b64(j, dimensions);

        while (svptest_first(predicate_all_b64x, predicate_b64x)) {
            svfloat64_t b_f64x = svld1_f64(predicate_b64x, b + j);
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_0_f64x, &compensation_0_f64x,
                                        svld1_f64(predicate_b64x, c + (row + 0) * dimensions + j), b_f64x);
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_1_f64x, &compensation_1_f64x,
                                        svld1_f64(predicate_b64x, c + (row + 1) * dimensions + j), b_f64x);
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_2_f64x, &compensation_2_f64x,
                                        svld1_f64(predicate_b64x, c + (row + 2) * dimensions + j), b_f64x);
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_3_f64x, &compensation_3_f64x,
                                        svld1_f64(predicate_b64x, c + (row + 3) * dimensions + j), b_f64x);
            j += svcntd();
            predicate_b64x = svwhilelt_b64(j, dimensions);
        }

        nk_f64_t s0 = svaddv_f64(predicate_all_b64x, sum_0_f64x);
        nk_f64_t c0 = svaddv_f64(predicate_all_b64x, compensation_0_f64x);
        nk_f64_t s1 = svaddv_f64(predicate_all_b64x, sum_1_f64x);
        nk_f64_t c1 = svaddv_f64(predicate_all_b64x, compensation_1_f64x);
        nk_f64_t s2 = svaddv_f64(predicate_all_b64x, sum_2_f64x);
        nk_f64_t c2 = svaddv_f64(predicate_all_b64x, compensation_2_f64x);
        nk_f64_t s3 = svaddv_f64(predicate_all_b64x, sum_3_f64x);
        nk_f64_t c3 = svaddv_f64(predicate_all_b64x, compensation_3_f64x);
        NK_UNPOISON(&s0, sizeof(s0));
        NK_UNPOISON(&c0, sizeof(c0));
        NK_UNPOISON(&s1, sizeof(s1));
        NK_UNPOISON(&c1, sizeof(c1));
        NK_UNPOISON(&s2, sizeof(s2));
        NK_UNPOISON(&c2, sizeof(c2));
        NK_UNPOISON(&s3, sizeof(s3));
        NK_UNPOISON(&c3, sizeof(c3));
        nk_f64_dot2_(&outer_sum, &outer_comp, a0, s0 + c0);
        nk_f64_dot2_(&outer_sum, &outer_comp, a1, s1 + c1);
        nk_f64_dot2_(&outer_sum, &outer_comp, a2, s2 + c2);
        nk_f64_dot2_(&outer_sum, &outer_comp, a3, s3 + c3);
    }

    // 1-row tail
    for (; row < dimensions; ++row) {
        svfloat64_t sum_f64x = svdup_f64(0.0), compensation_f64x = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t predicate_b64x = svwhilelt_b64(j, dimensions);

        while (svptest_first(predicate_all_b64x, predicate_b64x)) {
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_f64x, &compensation_f64x,
                                        svld1_f64(predicate_b64x, c + row * dimensions + j),
                                        svld1_f64(predicate_b64x, b + j));
            j += svcntd();
            predicate_b64x = svwhilelt_b64(j, dimensions);
        }

        nk_f64_t s = svaddv_f64(predicate_all_b64x, sum_f64x);
        nk_f64_t comp = svaddv_f64(predicate_all_b64x, compensation_f64x);
        NK_UNPOISON(&s, sizeof(s));
        NK_UNPOISON(&comp, sizeof(comp));
        nk_f64_t cb_j = s + comp;
        nk_f64_dot2_(&outer_sum, &outer_comp, a[row], cb_j);
    }

    *result = outer_sum + outer_comp;
}

NK_PUBLIC void nk_bilinear_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t dimensions,
                                      nk_f64_t *result) {
    nk_bilinear_f64_smef64_streaming_(a, b, c, dimensions, result);
}

/**
 *  @brief f64 Mahalanobis: row-by-row streaming SVE with Dot2 compensation.
 *  4-row fast path shares (a−b) column vector; 1-row tail for remainder.
 */
__arm_locally_streaming static nk_f64_t nk_mahalanobis_f64_smef64_streaming_(nk_f64_t const *a, nk_f64_t const *b,
                                                                             nk_f64_t const *c, nk_size_t dimensions) {
    svbool_t predicate_all_b64x = svptrue_b64();
    nk_f64_t outer_sum = 0.0, outer_comp = 0.0;
    nk_size_t row = 0;

    // 4-row fast path: share (a−b) column vector across 4 rows
    for (; row + 4 <= dimensions; row += 4) {
        nk_f64_t d0 = a[row + 0] - b[row + 0], d1 = a[row + 1] - b[row + 1];
        nk_f64_t d2 = a[row + 2] - b[row + 2], d3 = a[row + 3] - b[row + 3];
        svfloat64_t sum_0_f64x = svdup_f64(0), compensation_0_f64x = svdup_f64(0);
        svfloat64_t sum_1_f64x = svdup_f64(0), compensation_1_f64x = svdup_f64(0);
        svfloat64_t sum_2_f64x = svdup_f64(0), compensation_2_f64x = svdup_f64(0);
        svfloat64_t sum_3_f64x = svdup_f64(0), compensation_3_f64x = svdup_f64(0);
        nk_size_t j = 0;
        svbool_t predicate_b64x = svwhilelt_b64(j, dimensions);

        while (svptest_first(predicate_all_b64x, predicate_b64x)) {
            svfloat64_t diff_col_f64x = svsub_f64_x(predicate_b64x, svld1_f64(predicate_b64x, a + j),
                                                    svld1_f64(predicate_b64x, b + j));
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_0_f64x, &compensation_0_f64x,
                                        svld1_f64(predicate_b64x, c + (row + 0) * dimensions + j), diff_col_f64x);
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_1_f64x, &compensation_1_f64x,
                                        svld1_f64(predicate_b64x, c + (row + 1) * dimensions + j), diff_col_f64x);
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_2_f64x, &compensation_2_f64x,
                                        svld1_f64(predicate_b64x, c + (row + 2) * dimensions + j), diff_col_f64x);
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_3_f64x, &compensation_3_f64x,
                                        svld1_f64(predicate_b64x, c + (row + 3) * dimensions + j), diff_col_f64x);
            j += svcntd();
            predicate_b64x = svwhilelt_b64(j, dimensions);
        }

        nk_f64_t s0 = svaddv_f64(predicate_all_b64x, sum_0_f64x);
        nk_f64_t c0 = svaddv_f64(predicate_all_b64x, compensation_0_f64x);
        nk_f64_t s1 = svaddv_f64(predicate_all_b64x, sum_1_f64x);
        nk_f64_t c1 = svaddv_f64(predicate_all_b64x, compensation_1_f64x);
        nk_f64_t s2 = svaddv_f64(predicate_all_b64x, sum_2_f64x);
        nk_f64_t c2 = svaddv_f64(predicate_all_b64x, compensation_2_f64x);
        nk_f64_t s3 = svaddv_f64(predicate_all_b64x, sum_3_f64x);
        nk_f64_t c3 = svaddv_f64(predicate_all_b64x, compensation_3_f64x);
        NK_UNPOISON(&s0, sizeof(s0));
        NK_UNPOISON(&c0, sizeof(c0));
        NK_UNPOISON(&s1, sizeof(s1));
        NK_UNPOISON(&c1, sizeof(c1));
        NK_UNPOISON(&s2, sizeof(s2));
        NK_UNPOISON(&c2, sizeof(c2));
        NK_UNPOISON(&s3, sizeof(s3));
        NK_UNPOISON(&c3, sizeof(c3));
        nk_f64_dot2_(&outer_sum, &outer_comp, d0, s0 + c0);
        nk_f64_dot2_(&outer_sum, &outer_comp, d1, s1 + c1);
        nk_f64_dot2_(&outer_sum, &outer_comp, d2, s2 + c2);
        nk_f64_dot2_(&outer_sum, &outer_comp, d3, s3 + c3);
    }

    // 1-row tail
    for (; row < dimensions; ++row) {
        nk_f64_t diff_row = a[row] - b[row];
        svfloat64_t sum_f64x = svdup_f64(0.0), compensation_f64x = svdup_f64(0.0);
        nk_size_t j = 0;
        svbool_t predicate_b64x = svwhilelt_b64(j, dimensions);

        while (svptest_first(predicate_all_b64x, predicate_b64x)) {
            svfloat64_t diff_col_f64x = svsub_f64_x(predicate_b64x, svld1_f64(predicate_b64x, a + j),
                                                    svld1_f64(predicate_b64x, b + j));
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_f64x, &compensation_f64x,
                                        svld1_f64(predicate_b64x, c + row * dimensions + j), diff_col_f64x);
            j += svcntd();
            predicate_b64x = svwhilelt_b64(j, dimensions);
        }

        nk_f64_t s = svaddv_f64(predicate_all_b64x, sum_f64x);
        nk_f64_t comp = svaddv_f64(predicate_all_b64x, compensation_f64x);
        NK_UNPOISON(&s, sizeof(s));
        NK_UNPOISON(&comp, sizeof(comp));
        nk_f64_t cb_j = s + comp;
        nk_f64_dot2_(&outer_sum, &outer_comp, diff_row, cb_j);
    }

    return outer_sum + outer_comp;
}

NK_PUBLIC void nk_mahalanobis_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t dimensions,
                                         nk_f64_t *result) {
    nk_f64_t quadratic = nk_mahalanobis_f64_smef64_streaming_(a, b, c, dimensions);
    *result = nk_f64_sqrt_neon(quadratic > 0 ? quadratic : 0);
}

/**
 *  @brief f32c bilinear: complex GEMV via FMOPA (widening f32→f64).
 *  ZA0.D = C staging, ZA1.D = v_real accumulator, ZA2.D = v_imag accumulator.
 */
__arm_locally_streaming __arm_new("za") static void nk_bilinear_f32c_smef64_streaming_(nk_f32c_t const *a_pairs,
                                                                                       nk_f32c_t const *b_pairs,
                                                                                       nk_f32c_t const *c_pairs,
                                                                                       nk_size_t dimensions,
                                                                                       nk_f64c_t *results) {
    svbool_t predicate_body_b64x = svptrue_b64();
    nk_size_t tile_dimension = svcntd();
    nk_f64_t outer_sum_real_f64 = 0.0, outer_sum_imag_f64 = 0.0;

    for (nk_size_t row = 0; row < dimensions; row += tile_dimension) {
        nk_size_t rows_remaining = (row + tile_dimension <= dimensions) ? tile_dimension : (dimensions - row);
        svbool_t row_predicate_b64x = svwhilelt_b64_u64(0u, rows_remaining);

        svzero_mask_za(nk_sme_zero_za64_tile_1_);
        svzero_mask_za(nk_sme_zero_za64_tile_2_);

        for (nk_size_t j = 0; j < dimensions; j += tile_dimension) {
            nk_size_t batch_size = (j + tile_dimension <= dimensions) ? tile_dimension : (dimensions - j);
            svbool_t batch_predicate_b64x = svwhilelt_b64_u64(0u, batch_size);
            svbool_t batch_predicate_b32x = svwhilelt_b32_u64(0u, batch_size + batch_size);

            // Pass 1: Stage C_real into ZA0
            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                svfloat32_t c_f32x = svld1_f32(batch_predicate_b32x,
                                               (nk_f32_t const *)c_pairs + ((row + r) * dimensions + j) * 2);
                svfloat64_t c_real_f64x = svcvt_f64_f32_x(batch_predicate_b64x, svtrn1_f32(c_f32x, c_f32x));
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64x, c_real_f64x);
            }

            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_re_col_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 0, k);
                svmopa_za64_f64_m(1, row_predicate_b64x, row_predicate_b64x, c_re_col_f64x,
                                  svdup_f64((nk_f64_t)b_pairs[j + k].real)); // v_real += c_real × b_real
                svmopa_za64_f64_m(2, row_predicate_b64x, row_predicate_b64x, c_re_col_f64x,
                                  svdup_f64((nk_f64_t)b_pairs[j + k].imag)); // v_imag += c_real × b_imag
            }

            // Pass 2: Stage C_imag into ZA0
            svzero_mask_za(nk_sme_zero_za64_tile_0_);
            for (nk_size_t r = 0; r < rows_remaining; r++) {
                svfloat32_t c_f32x = svld1_f32(batch_predicate_b32x,
                                               (nk_f32_t const *)c_pairs + ((row + r) * dimensions + j) * 2);
                svfloat64_t c_imag_f64x = svcvt_f64_f32_x(batch_predicate_b64x, svtrn2_f32(c_f32x, c_f32x));
                svwrite_hor_za64_f64_m(0, r, batch_predicate_b64x, c_imag_f64x);
            }

            for (nk_size_t k = 0; k < batch_size; k++) {
                svfloat64_t c_im_col_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 0, k);
                svmopa_za64_f64_m(2, row_predicate_b64x, row_predicate_b64x, c_im_col_f64x,
                                  svdup_f64((nk_f64_t)b_pairs[j + k].real)); // v_imag += c_imag × b_real
                svmops_za64_f64_m(1, row_predicate_b64x, row_predicate_b64x, c_im_col_f64x,
                                  svdup_f64((nk_f64_t)b_pairs[j + k].imag)); // v_real -= c_imag × b_imag
            }
        }

        svfloat64_t v_re_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 1, 0);
        svfloat64_t v_im_f64x = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64x, 2, 0);

        // Deinterleave a[row:row+tile]
        svbool_t row_predicate_b32x = svwhilelt_b32_u64(0u, rows_remaining + rows_remaining);
        svfloat32_t a_f32x = svld1_f32(row_predicate_b32x, (nk_f32_t const *)a_pairs + row * 2);
        svfloat64_t a_re_f64x = svcvt_f64_f32_x(row_predicate_b64x, svtrn1_f32(a_f32x, a_f32x));
        svfloat64_t a_im_f64x = svcvt_f64_f32_x(row_predicate_b64x, svtrn2_f32(a_f32x, a_f32x));

        // Complex dot: a × v
        outer_sum_real_f64 += svaddv_f64(
            predicate_body_b64x, svsub_f64_x(row_predicate_b64x, svmul_f64_x(row_predicate_b64x, a_re_f64x, v_re_f64x),
                                             svmul_f64_x(row_predicate_b64x, a_im_f64x, v_im_f64x)));
        NK_UNPOISON(&outer_sum_real_f64, sizeof(outer_sum_real_f64));
        outer_sum_imag_f64 += svaddv_f64(
            predicate_body_b64x, svadd_f64_x(row_predicate_b64x, svmul_f64_x(row_predicate_b64x, a_re_f64x, v_im_f64x),
                                             svmul_f64_x(row_predicate_b64x, a_im_f64x, v_re_f64x)));
        NK_UNPOISON(&outer_sum_imag_f64, sizeof(outer_sum_imag_f64));
    }

    results->real = outer_sum_real_f64;
    results->imag = outer_sum_imag_f64;
}

NK_PUBLIC void nk_bilinear_f32c_smef64(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_f32c_t const *c_pairs,
                                       nk_size_t dimensions, nk_f64c_t *results) {
    nk_bilinear_f32c_smef64_streaming_(a_pairs, b_pairs, c_pairs, dimensions, results);
}

/**
 *  @brief f64c bilinear: interleaved Dot2 with permute + deferred XOR sign-flip.
 *  2 accumulators instead of 4, halving inner loop work (~15 vs ~28 SVE ops).
 */
__arm_locally_streaming static void nk_bilinear_f64c_smef64_streaming_(nk_f64c_t const *a_pairs,
                                                                       nk_f64c_t const *b_pairs,
                                                                       nk_f64c_t const *c_pairs, nk_size_t dimensions,
                                                                       nk_f64c_t *results) {
    svbool_t predicate_all_b64x = svptrue_b64();
    nk_f64_t outer_sum_real = 0.0, outer_comp_real = 0.0;
    nk_f64_t outer_sum_imag = 0.0, outer_comp_imag = 0.0;
    nk_size_t const n2 = dimensions * 2; // total f64 elements in interleaved layout

    // swap_idx_u64x = [1,0,3,2,5,4,...] — swap adjacent f64 lanes
    svuint64_t swap_idx_u64x = sveor_u64_x(predicate_all_b64x, svindex_u64(0, 1), svdup_u64(1));
    // sign_mask_u64x = [0, 0x8000..., 0, 0x8000..., ...] — sign bit in odd positions
    svuint64_t sign_mask_u64x = svlsl_u64_x(
        predicate_all_b64x, svand_u64_x(predicate_all_b64x, svindex_u64(0, 1), svdup_u64(1)), svdup_u64(63));

    for (nk_size_t row = 0; row < dimensions; ++row) {
        nk_f64_t a_real = a_pairs[row].real;
        nk_f64_t a_imag = a_pairs[row].imag;

        // 2 interleaved Dot2 accumulators (instead of 4 deinterleaved)
        svfloat64_t sum_real_f64x = svdup_f64(0), comp_real_f64x = svdup_f64(0);
        svfloat64_t sum_imag_f64x = svdup_f64(0), comp_imag_f64x = svdup_f64(0);
        nk_size_t j = 0;
        svbool_t predicate_b64x = svwhilelt_b64(j, n2);

        while (svptest_first(predicate_all_b64x, predicate_b64x)) {
            // Load interleaved [re₀, im₀, re₁, im₁, ...] — no deinterleave needed
            svfloat64_t b_f64x = svld1_f64(predicate_b64x, (nk_f64_t const *)b_pairs + j);
            svfloat64_t c_f64x = svld1_f64(predicate_b64x, (nk_f64_t const *)c_pairs + row * n2 + j);
            svfloat64_t c_swapped_f64x = svtbl_f64(c_f64x, swap_idx_u64x);

            // 2 Dot2 accumulators instead of 4:
            // sum_real_f64x accumulates [c_real×b_real, c_imag×b_imag, ...] (sign-flip deferred)
            // sum_imag_f64x accumulates [c_imag×b_real, c_real×b_imag, ...] (all positive)
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_real_f64x, &comp_real_f64x, c_f64x, b_f64x);
            nk_dot2_f64_sve_accumulate_(predicate_b64x, &sum_imag_f64x, &comp_imag_f64x, c_swapped_f64x, b_f64x);

            j += svcntd();
            predicate_b64x = svwhilelt_b64(j, n2);
        }

        // Flip sign of odd positions in sum_real_f64x: [c_real×b_real, -(c_imag×b_imag), ...]
        sum_real_f64x = svreinterpret_f64_u64(
            sveor_u64_x(predicate_all_b64x, svreinterpret_u64_f64(sum_real_f64x), sign_mask_u64x));
        comp_real_f64x = svreinterpret_f64_u64(
            sveor_u64_x(predicate_all_b64x, svreinterpret_u64_f64(comp_real_f64x), sign_mask_u64x));
        nk_f64_t inner_real = svaddv_f64(predicate_all_b64x,
                                         svadd_f64_x(predicate_all_b64x, sum_real_f64x, comp_real_f64x));
        nk_f64_t inner_imag = svaddv_f64(predicate_all_b64x,
                                         svadd_f64_x(predicate_all_b64x, sum_imag_f64x, comp_imag_f64x));
        NK_UNPOISON(&inner_real, sizeof(inner_real));
        NK_UNPOISON(&inner_imag, sizeof(inner_imag));

        // Outer Dot2 complex multiply: a × inner
        nk_f64_dot2_(&outer_sum_real, &outer_comp_real, a_real, inner_real);
        nk_f64_dot2_(&outer_sum_real, &outer_comp_real, -a_imag, inner_imag);
        nk_f64_dot2_(&outer_sum_imag, &outer_comp_imag, a_real, inner_imag);
        nk_f64_dot2_(&outer_sum_imag, &outer_comp_imag, a_imag, inner_real);
    }

    results->real = outer_sum_real + outer_comp_real;
    results->imag = outer_sum_imag + outer_comp_imag;
}

NK_PUBLIC void nk_bilinear_f64c_smef64(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_f64c_t const *c_pairs,
                                       nk_size_t dimensions, nk_f64c_t *results) {
    nk_bilinear_f64c_smef64_streaming_(a_pairs, b_pairs, c_pairs, dimensions, results);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SMEF64
#endif // NK_TARGET_ARM64_
#endif // NK_CURVED_SMEF64_H
