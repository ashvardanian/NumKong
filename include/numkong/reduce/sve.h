/**
 *  @brief SVE horizontal reduction helpers with MSan unpoisoning.
 *  @file include/numkong/reduce/sve.h
 *  @author Ash Vardanian
 *  @date April 12, 2026
 *
 *  LLVM's MSan does not instrument ARM SVE intrinsics — `svaddv` moves data
 *  from vector to scalar registers via architecture-specific paths invisible
 *  to the compiler, causing false-positive uninitialized-value reports.
 *  These macros wrap the reduction and unpoison the scalar result.
 *
 *  The `svaddv` intrinsic stays inside a macro so it expands in the caller's
 *  target context — SVE and SME streaming translation units carry incompatible
 *  target attributes. The unpoisoning runs on the already-reduced scalar, so it
 *  lives in a target-agnostic `NK_HELPER_INLINE` helper called from the macro.
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_SVE_H
#define NK_REDUCE_SVE_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVE || NK_TARGET_SVE2 || NK_TARGET_SME

#include "numkong/types.h"

NK_HELPER_INLINE nk_f64_t nk_unpoison_f64_(nk_f64_t v) NK_STREAMING_COMPATIBLE_ {
    nk_unpoison_(&v, sizeof(v));
    return v;
}
NK_HELPER_INLINE nk_f32_t nk_unpoison_f32_(nk_f32_t v) NK_STREAMING_COMPATIBLE_ {
    nk_unpoison_(&v, sizeof(v));
    return v;
}
NK_HELPER_INLINE nk_u64_t nk_unpoison_u64_(nk_u64_t v) NK_STREAMING_COMPATIBLE_ {
    nk_unpoison_(&v, sizeof(v));
    return v;
}
NK_HELPER_INLINE nk_i64_t nk_unpoison_i64_(nk_i64_t v) NK_STREAMING_COMPATIBLE_ {
    nk_unpoison_(&v, sizeof(v));
    return v;
}

#define nk_svaddv_f64_(predicate, vector) nk_unpoison_f64_(svaddv_f64((predicate), (vector)))
#define nk_svaddv_f32_(predicate, vector) nk_unpoison_f32_(svaddv_f32((predicate), (vector)))
#define nk_svaddv_u32_(predicate, vector) nk_unpoison_u64_(svaddv_u32((predicate), (vector)))
#define nk_svaddv_s32_(predicate, vector) nk_unpoison_i64_(svaddv_s32((predicate), (vector)))
#define nk_svaddv_u8_(predicate, vector)  nk_unpoison_u64_(svaddv_u8((predicate), (vector)))

#endif // NK_TARGET_SVE || NK_TARGET_SVE2 || NK_TARGET_SME
#endif // NK_TARGET_ARM64_
#endif // NK_REDUCE_SVE_H
