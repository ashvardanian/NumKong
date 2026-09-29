/**
 *  @file include/numkong/reduce/sve.h
 *  @author Ash Vardanian
 *  @date April 12, 2026
 *  @brief SVE horizontal reduction helpers with MSan unpoisoning.
 *
 *  LLVM's MSan does not instrument ARM SVE intrinsics — @c svaddv moves data from vector to scalar
 *  registers via architecture-specific paths invisible to the compiler, causing false-positive
 *  uninitialized-value reports. These macros wrap the reduction and unpoison the scalar result.
 *
 *  The @c svaddv intrinsic stays inside a macro so it expands in the caller's target context — SVE
 *  and SME streaming translation units carry incompatible target attributes. The unpoisoning runs
 *  on the already-reduced scalar, so it lives in a target-agnostic @c NUMKONG_INLINE helper called
 *  from the macro.
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NUMKONG_REDUCE_SVE_H
#define NUMKONG_REDUCE_SVE_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_SVE_

#include "numkong/types.h"

NUMKONG_INLINE nk_f64_t nk_unpoison_f64_(nk_f64_t v) NUMKONG_STREAMABLE_ {
    nk_unpoison_(&v, sizeof(v));
    return v;
}
NUMKONG_INLINE nk_f32_t nk_unpoison_f32_(nk_f32_t v) NUMKONG_STREAMABLE_ {
    nk_unpoison_(&v, sizeof(v));
    return v;
}
NUMKONG_INLINE nk_u64_t nk_unpoison_u64_(nk_u64_t v) NUMKONG_STREAMABLE_ {
    nk_unpoison_(&v, sizeof(v));
    return v;
}
NUMKONG_INLINE nk_i64_t nk_unpoison_i64_(nk_i64_t v) NUMKONG_STREAMABLE_ {
    nk_unpoison_(&v, sizeof(v));
    return v;
}

#define nk_svaddv_f64_(predicate, vector) nk_unpoison_f64_(svaddv_f64((predicate), (vector)))
#define nk_svaddv_f32_(predicate, vector) nk_unpoison_f32_(svaddv_f32((predicate), (vector)))
#define nk_svaddv_u32_(predicate, vector) nk_unpoison_u64_(svaddv_u32((predicate), (vector)))
#define nk_svaddv_s32_(predicate, vector) nk_unpoison_i64_(svaddv_s32((predicate), (vector)))
#define nk_svaddv_u8_(predicate, vector)  nk_unpoison_u64_(svaddv_u8((predicate), (vector)))

#endif // NUMKONG_ARCH_ARM64_SVE_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_REDUCE_SVE_H
