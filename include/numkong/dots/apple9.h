/**
 *  @file include/numkong/dots/apple9.h
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched Dot Products for Apple GPUs of Metal family 9, launched from C.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/simt.h
 *  @sa include/numkong/dots/apple9.metal, the kernels this embeds
 *
 *  The @c metal pack and launch contract over @c simdgroup_matrix products, the fastest multiply M3
 *  and M4 expose to Metal. Every signature matches @c metal's, and the packed B is @c metal's byte
 *  for byte. The kernels travel as their own `.metal` source, embedded after `simt.metal`, whose
 *  formats they widen, and compile on the queue at first use under Metal 3.1, which adds @c bfloat,
 *  so M3 and M4 run them from macOS 14.
 *
 *  Only floats are here: their products are exact in the @c float accumulators, while integer sums
 *  past 2²⁴ would not be, and the @c metal capability already accumulates those exactly.
 */
#ifndef NUMKONG_DOTS_APPLE9_H
#define NUMKONG_DOTS_APPLE9_H

#if NUMKONG_TARGET_APPLE9
#include "numkong/dots/simt.h" // `nk_cross_launch_msl_`, `nk_define_device_cross_msl_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per queue. */
static char const nk_dots_apple9_source_[] = {
#embed "simt.metal" suffix(, )
#embed "apple9.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#pragma region Launchers

/** Threads one threadgroup runs, four simdgroups, as in the kernels. */
enum { nk_cross_threads_apple9_k = 128 };

/** @ref nk_cross_launch_msl_ over the @c apple9 kernels, which read B in aligned 16-byte chunks. */
NUMKONG_INLINE nk_status_t nk_cross_launch_apple9_(char const *kernel, void const *a, void const *b,
                                                   nk_size_t b_extra_offset, void *c, nk_size_t result_bytes,
                                                   nk_size_t row_start, nk_size_t row_end, nk_size_t column_count,
                                                   nk_size_t depth, nk_size_t a_stride, nk_size_t b_stride,
                                                   nk_size_t c_stride, nk_u32_t upper_triangle, void *stream) {
    if ((((nk_size_t)b + b_extra_offset) | b_stride) & 15) return nk_misaligned_k;
    return nk_cross_launch_msl_(nk_dots_apple9_source_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cross_threads_apple9_k, kernel,
                                a, b, b_extra_offset, c, result_bytes, row_start, row_end, column_count, depth,
                                a_stride, b_stride, c_stride, upper_triangle, stream);
}

#pragma endregion Launchers

nk_define_device_cross_msl_(f16, apple9, f16, f32, f32, 16, 1)
nk_define_device_cross_msl_(bf16, apple9, bf16, f32, f32, 16, 1)
nk_define_device_cross_msl_(e4m3, apple9, e4m3, f32, f32, 16, 1)
nk_define_device_cross_msl_(e5m2, apple9, e5m2, f32, f32, 16, 1)
nk_define_device_cross_msl_(e3m2, apple9, e3m2, f32, f32, 16, 1)
nk_define_device_cross_msl_(e2m3, apple9, e2m3, f32, f32, 16, 1)
nk_define_device_cross_msl_(e2m1, apple9, e2m1x2, f32, f32, 32, 2)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_APPLE9
#endif // NUMKONG_DOTS_APPLE9_H
