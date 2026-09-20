/**
 *  @brief SVE2-accelerated Sparse Vector Operations.
 *  @file include/numkong/sparse/sve2.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/sparse.h
 */
#ifndef NK_SPARSE_SVE2_H
#define NK_SPARSE_SVE2_H

#if NK_TARGET_ARM64_

#include "numkong/types.h"
#include "numkong/reduce/sve.h" // `nk_svaddv_f64_`

#if defined(__cplusplus)
extern "C" {
#endif

/*  SVE2 introduces many new integer-oriented instructions, extending some of the NEON functionality
 *  to variable-length SVE registers. Those include "compare multiple" intrinsics:
 *
 *  - `svmatch[_u16]` that matches each scalar in first vector against all members of a 128-bit lane in the second.
 *  - `svhistcnt[_s32]_z` does something similar, performing an inclusive prefix scan.
 *  - `svtbx[_u16]` does extended table lookup
 *
 *  Other notable instructions:
 *
 *  - `DUP`: Broadcast indexed predicate element
 *    https://developer.arm.com/documentation/ddi0602/2021-06/SVE-Instructions/DUP--predicate---Broadcast-indexed-predicate-element-?lang=en
 *  - `SCLAMP` and `UCLAMP`: clamp values, i.e. combined min+max
 *    https://developer.arm.com/documentation/ddi0602/2021-06/SVE-Instructions/SCLAMP--Signed-clamp-to-minimum-maximum-vector-?lang=en
 *    https://developer.arm.com/documentation/ddi0602/2021-06/SVE-Instructions/UCLAMP--Unsigned-clamp-to-minimum-maximum-vector-?lang=en
 *  - `TBLQ`: Table lookup quadword
 *    https://developer.arm.com/documentation/ddi0602/2022-12/SVE-Instructions/TBLQ--Programmable-table-lookup-within-each-quadword-vector-segment--zeroing--?lang=en
 *
 *  Great resources for SVE2 intrinsics:
 *
 *  > ARM's Scalable Vector Extensions: A Critical Look at SVE2 For Integer Workloads
 *    https://gist.github.com/zingaburga/805669eb891c820bd220418ee3f0d6bd
 */
#if NK_TARGET_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+sve2")
#endif

NK_PUBLIC void nk_sparse_intersect_u16_sve2( //
    nk_u16_t const *a, nk_u16_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u16_t *result, nk_size_t *count) {

    // A single SVE lane is 128 bits wide, so one lane fits 8 values.
    nk_size_t const register_size = svcnth();
    nk_size_t const lanes_count = register_size / 8;
    nk_size_t a_idx = 0, b_idx = 0;
    nk_size_t c = 0;

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress_b16x = svwhilelt_b16_u64(a_idx, a_length);
        svbool_t b_progress_b16x = svwhilelt_b16_u64(b_idx, b_length);
        svuint16_t a_u16x = svld1_u16(a_progress_b16x, a + a_idx);
        svuint16_t b_u16x = svld1_u16(b_progress_b16x, b + b_idx);

        // Intersecting registers with `svmatch_u16` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = svlastb(a_progress_b16x, a_u16x);
        nk_u16_t b_min = svlasta(svpfalse_b(), b_u16x);
        nk_u16_t b_max = svlastb(b_progress_b16x, b_u16x);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress_b16x = svwhilelt_b16_u64(a_idx, a_length);
            a_u16x = svld1_u16(a_progress_b16x, a + a_idx);
            a_max = svlastb(a_progress_b16x, a_u16x);
        }
        a_min = svlasta(svpfalse_b(), a_u16x);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress_b16x = svwhilelt_b16_u64(b_idx, b_length);
            b_u16x = svld1_u16(b_progress_b16x, b + b_idx);
            b_max = svlastb(b_progress_b16x, b_u16x);
        }
        b_min = svlasta(svpfalse_b(), b_u16x);

        // Before we evaluate the intersection size, obfurscating the order in `b_u16x`,
        // let's estimate how much we will need to advance the pointers afterwards.
        // For that, we don't even need to broadcast the values in SVE, as the whole
        // register can be compared against a scalar:
        //
        //      svuint16_t a_last_broadcasted =  svdup_n_u16(a_max);
        //      svuint16_t b_last_broadcasted =  svdup_n_u16(b_max);
        svbool_t a_mask_b16x = svcmple_n_u16(a_progress_b16x, a_u16x, b_max);
        svbool_t b_mask_b16x = svcmple_n_u16(b_progress_b16x, b_u16x, a_max);
        nk_u64_t a_step = svcntp_b16(a_progress_b16x, a_mask_b16x);
        nk_u64_t b_step = svcntp_b16(b_progress_b16x, b_mask_b16x);

        // `svmatch`/`svhistcnt` ignore the predicate on their second operand; replace the zero tail with a real member.
        b_u16x = svsel_u16(b_progress_b16x, b_u16x, svdup_n_u16(b_max));

        // Compare `a_u16x` with each lane of `b_u16x`
        svbool_t equal_mask_b16x = svmatch_u16(a_progress_b16x, a_u16x, b_u16x);
        for (nk_size_t i = 1; i < lanes_count; i++) {
            b_u16x = svext_u16(b_u16x, b_u16x, 8);
            equal_mask_b16x = svorr_z(svptrue_b16(), equal_mask_b16x, svmatch_u16(a_progress_b16x, a_u16x, b_u16x));
        }
        nk_size_t equal_count = svcntp_b16(svptrue_b16(), equal_mask_b16x);

        // `svcompact` has no 16-bit form before SVE2p2, so compact each half at 32 bits and store
        // back narrowed. This is a single output stream, so the halves simply append — none of the
        // merge that pairing two compacted streams would need.
        if (result) {
            svbool_t low_b32x = svunpklo_b(equal_mask_b16x);
            svbool_t high_b32x = svunpkhi_b(equal_mask_b16x);
            nk_u64_t low_count = svcntp_b32(svptrue_b32(), low_b32x);
            svst1h_u32(svwhilelt_b32_u64(0u, low_count), result + c, svcompact_u32(low_b32x, svunpklo_u32(a_u16x)));
            svst1h_u32(svwhilelt_b32_u64(0u, equal_count - low_count), result + c + low_count,
                       svcompact_u32(high_b32x, svunpkhi_u32(a_u16x)));
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
        c += equal_count;
    }
    *count = c;
}

NK_PUBLIC void nk_sparse_intersect_u32_sve2( //
    nk_u32_t const *a, nk_u32_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u32_t *result, nk_size_t *count) {

    // A single SVE lane is 128 bits wide, so one lane fits 4 values.
    nk_size_t const register_size = svcntw();
    nk_size_t const lanes_count = register_size / 4;
    nk_size_t a_idx = 0, b_idx = 0;
    nk_size_t c = 0;

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress_b32x = svwhilelt_b32_u64(a_idx, a_length);
        svbool_t b_progress_b32x = svwhilelt_b32_u64(b_idx, b_length);
        svuint32_t a_u32x = svld1_u32(a_progress_b32x, a + a_idx);
        svuint32_t b_u32x = svld1_u32(b_progress_b32x, b + b_idx);

        // Intersecting registers with `svmatch_u16` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u32_t a_min;
        nk_u32_t a_max = svlastb(a_progress_b32x, a_u32x);
        nk_u32_t b_min = svlasta(svpfalse_b(), b_u32x);
        nk_u32_t b_max = svlastb(b_progress_b32x, b_u32x);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress_b32x = svwhilelt_b32_u64(a_idx, a_length);
            a_u32x = svld1_u32(a_progress_b32x, a + a_idx);
            a_max = svlastb(a_progress_b32x, a_u32x);
        }
        a_min = svlasta(svpfalse_b(), a_u32x);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress_b32x = svwhilelt_b32_u64(b_idx, b_length);
            b_u32x = svld1_u32(b_progress_b32x, b + b_idx);
            b_max = svlastb(b_progress_b32x, b_u32x);
        }
        b_min = svlasta(svpfalse_b(), b_u32x);

        // Before we evaluate the intersection size, obfurscating the order in `b_u32x`,
        // let's estimate how much we will need to advance the pointers afterwards.
        // For that, we don't even need to broadcast the values in SVE, as the whole
        // register can be compared against a scalar:
        //
        //      svuint32_t a_last_broadcasted =  svdup_n_u32(a_max);
        //      svuint32_t b_last_broadcasted =  svdup_n_u32(b_max);
        svbool_t a_mask_b32x = svcmple_n_u32(a_progress_b32x, a_u32x, b_max);
        svbool_t b_mask_b32x = svcmple_n_u32(b_progress_b32x, b_u32x, a_max);
        nk_u64_t a_step = svcntp_b32(a_progress_b32x, a_mask_b32x);
        nk_u64_t b_step = svcntp_b32(b_progress_b32x, b_mask_b32x);

        // `svmatch`/`svhistcnt` ignore the predicate on their second operand; replace the zero tail with a real member.
        b_u32x = svsel_u32(b_progress_b32x, b_u32x, svdup_n_u32(b_max));

        // Comparing `a_u32x` with each lane of `b_u32x` can't be done with `svmatch`,
        // the same way as in `nk_sparse_intersect_u16_sve2`, as that instruction is only
        // available for 8-bit and 16-bit integers.
        //
        //      svbool_t equal_mask_b32x = svpfalse_b();
        //      for (nk_size_t i = 0; i < register_size; i++) {
        //          equal_mask_b32x = svorr_z(svptrue_b32(), equal_mask_b32x, svcmpeq_u32(a_progress, a_u32x, b_u32x));
        //          b_u32x = svext_u32(b_u32x, b_u32x, 1);
        //      }
        //      nk_size_t equal_count = svcntp_b32(a_progress, equal_mask_b32x);
        //
        // Alternatively, one can use histogram instructions, like `svhistcnt_u32_z`.
        // They practically compute the prefix-matching count, which is equivalent to
        // the lower triangle of the row-major intersection matrix.
        // To compute the upper triangle, we can reverse (with `svrev_b32`) the order of
        // elements and repeat the operation, accumulating the results for top and bottom.
        // Let's look at 4x element registers as an example:
        //
        //      ⊐ α = {A, B, C, D}, β = {X, Y, Z, W}:
        //
        //      hist(α, β):           hist(α_rev, β_rev):
        //
        //        X Y Z W               W Z Y X
        //      A 1 0 0 0             D 1 0 0 0
        //      B 1 1 0 0             C 1 1 0 0
        //      C 1 1 1 0             B 1 1 1 0
        //      D 1 1 1 1             A 1 1 1 1
        //
        svuint32_t hist_low_u32x = svhistcnt_u32_z(a_progress_b32x, a_u32x, b_u32x);
        svuint32_t a_rev_u32x = svrev_u32(a_u32x);
        svuint32_t b_rev_u32x = svrev_u32(b_u32x);
        svuint32_t hist_high_u32x = svrev_u32(svhistcnt_u32_z(svptrue_b32(), a_rev_u32x, b_rev_u32x));
        svuint32_t hist_u32x = svorr_u32_x(a_progress_b32x, hist_low_u32x, hist_high_u32x);
        svbool_t equal_mask_b32x = svcmpne_n_u32(a_progress_b32x, hist_u32x, 0);
        nk_size_t equal_count = svcntp_b32(a_progress_b32x, equal_mask_b32x);

        // Use SVE2 svcompact to compress matching elements and store to result buffer
        if (result) {
            svuint32_t compacted_u32x = svcompact_u32(equal_mask_b32x, a_u32x);
            svbool_t store_predicate_b32x = svwhilelt_b32_u64(0u, equal_count);
            svst1_u32(store_predicate_b32x, result + c, compacted_u32x);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
        c += equal_count;
    }
    *count = c;
}

NK_PUBLIC void nk_sparse_intersect_u64_sve2( //
    nk_u64_t const *a, nk_u64_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u64_t *result, nk_size_t *count) {

    // A single SVE lane is 128 bits wide, so one lane fits 2 values.
    nk_size_t const register_size = svcntd();
    nk_size_t const lanes_count = register_size / 2;
    nk_size_t a_idx = 0, b_idx = 0;
    nk_size_t c = 0;

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress_b64x = svwhilelt_b64_u64(a_idx, a_length);
        svbool_t b_progress_b64x = svwhilelt_b64_u64(b_idx, b_length);
        svuint64_t a_u64x = svld1_u64(a_progress_b64x, a + a_idx);
        svuint64_t b_u64x = svld1_u64(b_progress_b64x, b + b_idx);

        // Intersecting registers involves comparisons,
        // so we want to avoid it if the slices don't overlap at all.
        nk_u64_t a_min;
        nk_u64_t a_max = svlastb(a_progress_b64x, a_u64x);
        nk_u64_t b_min = svlasta(svpfalse_b(), b_u64x);
        nk_u64_t b_max = svlastb(b_progress_b64x, b_u64x);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress_b64x = svwhilelt_b64_u64(a_idx, a_length);
            a_u64x = svld1_u64(a_progress_b64x, a + a_idx);
            a_max = svlastb(a_progress_b64x, a_u64x);
        }
        a_min = svlasta(svpfalse_b(), a_u64x);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress_b64x = svwhilelt_b64_u64(b_idx, b_length);
            b_u64x = svld1_u64(b_progress_b64x, b + b_idx);
            b_max = svlastb(b_progress_b64x, b_u64x);
        }
        b_min = svlasta(svpfalse_b(), b_u64x);

        // Estimate how much we will need to advance the pointers afterwards.
        svbool_t a_mask_b64x = svcmple_n_u64(a_progress_b64x, a_u64x, b_max);
        svbool_t b_mask_b64x = svcmple_n_u64(b_progress_b64x, b_u64x, a_max);
        nk_u64_t a_step = svcntp_b64(a_progress_b64x, a_mask_b64x);
        nk_u64_t b_step = svcntp_b64(b_progress_b64x, b_mask_b64x);

        // `svmatch`/`svhistcnt` ignore the predicate on their second operand; replace the zero tail with a real member.
        b_u64x = svsel_u64(b_progress_b64x, b_u64x, svdup_n_u64(b_max));

        // Use histogram instructions like `svhistcnt_u64_z` to compute intersection.
        // They compute the prefix-matching count, equivalent to the lower triangle
        // of the row-major intersection matrix.
        svuint64_t hist_low_u64x = svhistcnt_u64_z(a_progress_b64x, a_u64x, b_u64x);
        svuint64_t a_rev_u64x = svrev_u64(a_u64x);
        svuint64_t b_rev_u64x = svrev_u64(b_u64x);
        svuint64_t hist_high_u64x = svrev_u64(svhistcnt_u64_z(svptrue_b64(), a_rev_u64x, b_rev_u64x));
        svuint64_t hist_u64x = svorr_u64_x(a_progress_b64x, hist_low_u64x, hist_high_u64x);
        svbool_t equal_mask_b64x = svcmpne_n_u64(a_progress_b64x, hist_u64x, 0);
        nk_size_t equal_count = svcntp_b64(a_progress_b64x, equal_mask_b64x);

        // Use SVE2 svcompact to compress matching elements and store to result buffer
        if (result) {
            svuint64_t compacted_u64x = svcompact_u64(equal_mask_b64x, a_u64x);
            svbool_t store_predicate_b64x = svwhilelt_b64_u64(0u, equal_count);
            svst1_u64(store_predicate_b64x, result + c, compacted_u64x);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
        c += equal_count;
    }
    *count = c;
}

NK_PUBLIC void nk_sparse_dot_u32f32_sve2(                 //
    nk_u32_t const *a, nk_u32_t const *b,                 //
    nk_f32_t const *a_weights, nk_f32_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,               //
    nk_f64_t *product) {

    nk_size_t const register_size = svcntw();
    nk_size_t const vector_length_f64 = svcntd();
    nk_size_t a_idx = 0, b_idx = 0;
    svbool_t const predicate_all_b32x = svptrue_b32();
    svbool_t const predicate_all_b64x = svptrue_b64();
    svfloat64_t product_f64x = svdup_f64(0.0);

    while (a_idx < a_length && b_idx < b_length) {
        // Load indices with progress predicates
        svbool_t a_progress_b32x = svwhilelt_b32_u64(a_idx, a_length);
        svbool_t b_progress_b32x = svwhilelt_b32_u64(b_idx, b_length);
        svuint32_t a_u32x = svld1_u32(a_progress_b32x, a + a_idx);
        svuint32_t b_u32x = svld1_u32(b_progress_b32x, b + b_idx);

        // Avoid expensive intersection if slices don't overlap at all
        nk_u32_t a_min;
        nk_u32_t a_max = svlastb(a_progress_b32x, a_u32x);
        nk_u32_t b_min = svlasta(svpfalse_b(), b_u32x);
        nk_u32_t b_max = svlastb(b_progress_b32x, b_u32x);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress_b32x = svwhilelt_b32_u64(a_idx, a_length);
            a_u32x = svld1_u32(a_progress_b32x, a + a_idx);
            a_max = svlastb(a_progress_b32x, a_u32x);
        }
        a_min = svlasta(svpfalse_b(), a_u32x);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress_b32x = svwhilelt_b32_u64(b_idx, b_length);
            b_u32x = svld1_u32(b_progress_b32x, b + b_idx);
            b_max = svlastb(b_progress_b32x, b_u32x);
        }
        b_min = svlasta(svpfalse_b(), b_u32x);

        // Calculate step sizes before modifying vectors
        svbool_t a_mask_b32x = svcmple_n_u32(a_progress_b32x, a_u32x, b_max);
        svbool_t b_mask_b32x = svcmple_n_u32(b_progress_b32x, b_u32x, a_max);
        nk_u64_t a_step = svcntp_b32(a_progress_b32x, a_mask_b32x);
        nk_u64_t b_step = svcntp_b32(b_progress_b32x, b_mask_b32x);

        // `svmatch`/`svhistcnt` ignore the predicate on their second operand; replace the zero tail with a real member.
        // Each vector is the other's ungoverned second operand here.
        a_u32x = svsel_u32(a_progress_b32x, a_u32x, svdup_n_u32(a_max));
        b_u32x = svsel_u32(b_progress_b32x, b_u32x, svdup_n_u32(b_max));

        // Use histogram-based intersection (svmatch_u32 doesn't exist)
        svuint32_t hist_low_u32x = svhistcnt_u32_z(a_progress_b32x, a_u32x, b_u32x);
        svuint32_t a_rev_u32x = svrev_u32(a_u32x);
        svuint32_t b_rev_u32x = svrev_u32(b_u32x);
        svuint32_t hist_high_u32x = svrev_u32(svhistcnt_u32_z(predicate_all_b32x, a_rev_u32x, b_rev_u32x));
        svuint32_t hist_u32x = svorr_u32_x(a_progress_b32x, hist_low_u32x, hist_high_u32x);
        svbool_t a_equal_mask_b32x = svcmpne_n_u32(a_progress_b32x, hist_u32x, 0);
        svbool_t a_overlap_mask_b32x = svand_b_z(predicate_all_b32x, a_progress_b32x, a_equal_mask_b32x);

        if (!svptest_any(a_progress_b32x, a_overlap_mask_b32x)) {
            a_idx += a_step;
            b_idx += b_step;
            continue;
        }

        // Compute b overlap mask (symmetric histogram: which b elements match something in a)
        svuint32_t b_hist_low_u32x = svhistcnt_u32_z(b_progress_b32x, b_u32x, a_u32x);
        svuint32_t b_hist_high_u32x = svrev_u32(svhistcnt_u32_z(predicate_all_b32x, b_rev_u32x, a_rev_u32x));
        svuint32_t b_hist_u32x = svorr_u32_x(b_progress_b32x, b_hist_low_u32x, b_hist_high_u32x);
        svbool_t b_overlap_mask_b32x = svand_b_z(predicate_all_b32x, b_progress_b32x,
                                                 svcmpne_n_u32(b_progress_b32x, b_hist_u32x, 0));

        // Compact matching weights — both arrays are sorted, so svcompact
        // preserves relative order and aligns corresponding intersection pairs.
        svfloat32_t a_matched_f32x = svcompact_f32(a_overlap_mask_b32x, svld1_f32(a_progress_b32x, a_weights + a_idx));
        svfloat32_t b_matched_f32x = svcompact_f32(b_overlap_mask_b32x, svld1_f32(b_progress_b32x, b_weights + b_idx));

        // Widen to f64 and accumulate. svcvt_f64_f32 converts even-indexed f32
        // elements; svcvtlt_f64_f32 converts odd-indexed f32 elements.
        nk_size_t match_count = svcntp_b32(a_progress_b32x, a_overlap_mask_b32x);
        svbool_t pred_even_b64x = svwhilelt_b64_u64(0u, (match_count + 1) / 2);
        svbool_t pred_odd_b64x = svwhilelt_b64_u64(0u, match_count / 2);
        product_f64x = svmla_f64_m(pred_even_b64x, product_f64x, svcvt_f64_f32_x(pred_even_b64x, a_matched_f32x),
                                   svcvt_f64_f32_x(pred_even_b64x, b_matched_f32x));
        product_f64x = svmla_f64_m(pred_odd_b64x, product_f64x, svcvtlt_f64_f32_x(pred_odd_b64x, a_matched_f32x),
                                   svcvtlt_f64_f32_x(pred_odd_b64x, b_matched_f32x));

        // Advance
        a_idx += a_step;
        b_idx += b_step;
    }
    *product = nk_svaddv_f64_(predicate_all_b64x, product_f64x);
}

NK_PUBLIC void nk_sparse_dot_u16bf16_sve2(                  //
    nk_u16_t const *a, nk_u16_t const *b,                   //
    nk_bf16_t const *a_weights, nk_bf16_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,                 //
    nk_f32_t *product) {

    // Mirrors `nk_sparse_dot_u32f32_sve2`, widening the 16-bit indices on load. `svmatch_u16`
    // only reports THAT a lane matched, never which, so pairing weights by lane index is wrong;
    // `svhistcnt_u32_z` is whole-vector and yields both overlap masks, and compacting each
    // weight vector by its own mask aligns the k-th survivors because both inputs are sorted.
    nk_size_t const register_size = svcntw();
    nk_size_t a_idx = 0, b_idx = 0;
    svbool_t const predicate_all_b32x = svptrue_b32();
    svfloat32_t product_f32x = svdup_f32(0.f);

    while (a_idx < a_length && b_idx < b_length) {
        svbool_t a_progress_b32x = svwhilelt_b32_u64(a_idx, a_length);
        svbool_t b_progress_b32x = svwhilelt_b32_u64(b_idx, b_length);
        svuint32_t a_u32x = svld1uh_u32(a_progress_b32x, a + a_idx);
        svuint32_t b_u32x = svld1uh_u32(b_progress_b32x, b + b_idx);

        // Avoid the intersection entirely when the slices cannot overlap.
        nk_u32_t a_min;
        nk_u32_t a_max = svlastb(a_progress_b32x, a_u32x);
        nk_u32_t b_min = svlasta(svpfalse_b(), b_u32x);
        nk_u32_t b_max = svlastb(b_progress_b32x, b_u32x);
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress_b32x = svwhilelt_b32_u64(a_idx, a_length);
            a_u32x = svld1uh_u32(a_progress_b32x, a + a_idx);
            a_max = svlastb(a_progress_b32x, a_u32x);
        }
        a_min = svlasta(svpfalse_b(), a_u32x);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress_b32x = svwhilelt_b32_u64(b_idx, b_length);
            b_u32x = svld1uh_u32(b_progress_b32x, b + b_idx);
            b_max = svlastb(b_progress_b32x, b_u32x);
        }
        b_min = svlasta(svpfalse_b(), b_u32x);

        svbool_t a_mask_b32x = svcmple_n_u32(a_progress_b32x, a_u32x, b_max);
        svbool_t b_mask_b32x = svcmple_n_u32(b_progress_b32x, b_u32x, a_max);
        nk_u64_t a_step = svcntp_b32(a_progress_b32x, a_mask_b32x);
        nk_u64_t b_step = svcntp_b32(b_progress_b32x, b_mask_b32x);

        // `svhistcnt` ignores the predicate on its second operand; replace the zero tail with a real member.
        a_u32x = svsel_u32(a_progress_b32x, a_u32x, svdup_n_u32(a_max));
        b_u32x = svsel_u32(b_progress_b32x, b_u32x, svdup_n_u32(b_max));

        svuint32_t a_rev_u32x = svrev_u32(a_u32x);
        svuint32_t b_rev_u32x = svrev_u32(b_u32x);
        svuint32_t hist_low_u32x = svhistcnt_u32_z(a_progress_b32x, a_u32x, b_u32x);
        svuint32_t hist_high_u32x = svrev_u32(svhistcnt_u32_z(predicate_all_b32x, a_rev_u32x, b_rev_u32x));
        svuint32_t hist_u32x = svorr_u32_x(a_progress_b32x, hist_low_u32x, hist_high_u32x);
        svbool_t a_overlap_b32x = svcmpne_n_u32(a_progress_b32x, hist_u32x, 0);

        if (!svptest_any(a_progress_b32x, a_overlap_b32x)) {
            a_idx += a_step;
            b_idx += b_step;
            continue;
        }

        svuint32_t b_hist_low_u32x = svhistcnt_u32_z(b_progress_b32x, b_u32x, a_u32x);
        svuint32_t b_hist_high_u32x = svrev_u32(svhistcnt_u32_z(predicate_all_b32x, b_rev_u32x, a_rev_u32x));
        svuint32_t b_hist_u32x = svorr_u32_x(b_progress_b32x, b_hist_low_u32x, b_hist_high_u32x);
        svbool_t b_overlap_b32x = svcmpne_n_u32(b_progress_b32x, b_hist_u32x, 0);

        // `bf16` is the top half of `f32`, so a widening load plus a 16-bit shift is exact.
        svfloat32_t a_w_f32x = svreinterpret_f32_u32(
            svlsl_n_u32_x(a_progress_b32x, svld1uh_u32(a_progress_b32x, (nk_u16_t const *)a_weights + a_idx), 16));
        svfloat32_t b_w_f32x = svreinterpret_f32_u32(
            svlsl_n_u32_x(b_progress_b32x, svld1uh_u32(b_progress_b32x, (nk_u16_t const *)b_weights + b_idx), 16));

        // Both inputs are sorted, so compaction preserves order and the k-th survivors pair up.
        svfloat32_t a_matched_f32x = svcompact_f32(a_overlap_b32x, a_w_f32x);
        svfloat32_t b_matched_f32x = svcompact_f32(b_overlap_b32x, b_w_f32x);
        product_f32x = svmla_f32_m(predicate_all_b32x, product_f32x, a_matched_f32x, b_matched_f32x);

        a_idx += a_step;
        b_idx += b_step;
    }
    *product = nk_svaddv_f32_(svptrue_b32(), product_f32x);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVE2

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ARM64_
#endif // NK_SPARSE_SVE2_H
