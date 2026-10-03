/**
 *  @file include/numkong/dots/apple10.h
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Batched Dot Products for Apple GPUs of Metal family 10, launched from C.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/apple10.metal, the kernels this embeds
 *  @sa include/numkong/dots/ampere.cuh, the CUDA sibling
 *
 *  Every signature matches @c metal's, whose pack, shape reader and launch core these reuse with
 *  wider padding. The kernels travel as their own `.metal` source, embedded after `metal.metal`,
 *  whose formats they widen, and compile on the device at first use under Metal 4.0, the first
 *  language version with tensor operations.
 */
#ifndef NUMKONG_DOTS_APPLE10_H
#define NUMKONG_DOTS_APPLE10_H

#if NUMKONG_TARGET_APPLE10
#include "numkong/dots/metal.h" // `nk_cross_encode_metal_`, `nk_define_cross_metal_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per device. */
static char const nk_dots_source_apple10_[] = {
#embed "metal.metal" suffix(, )
#embed "apple10.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#pragma region Launchers

/** Threads one threadgroup runs: the four simdgroups each kernel's @c matmul2d spans. */
enum { nk_cross_threads_apple10_k = 4 * 32 };

/** @ref nk_cross_encode_metal_ over the @c apple10 kernels. */
NUMKONG_INLINE nk_status_t nk_cross_launch_apple10_(char const *kernel, void const *a, void const *b,
                                                    nk_size_t b_extra_offset, void *c, nk_size_t result_bytes,
                                                    nk_size_t row_start, nk_size_t row_end, nk_size_t column_count,
                                                    nk_size_t depth, nk_size_t a_stride, nk_size_t b_stride,
                                                    nk_size_t c_stride, nk_u32_t upper_triangle, void *stream) {
    return nk_cross_encode_metal_(nk_dots_source_apple10_, NUMKONG_METAL_LANGUAGE_4_0_, nk_cross_threads_apple10_k,
                                  kernel, a, b, b_extra_offset, c, result_bytes, row_start, row_end, column_count,
                                  depth, a_stride, b_stride, c_stride, upper_triangle, stream);
}

#pragma endregion Launchers

nk_define_cross_metal_(i8, apple10, i8, i32, u32, 64, 1)
nk_define_cross_metal_(u8, apple10, u8, u32, u32, 64, 1)
nk_define_cross_metal_(f16, apple10, f16, f32, f32, 32, 1)
nk_define_cross_metal_(bf16, apple10, bf16, f32, f32, 32, 1)
nk_define_cross_metal_(e4m3, apple10, e4m3, f32, f32, 64, 1)
nk_define_cross_metal_(e5m2, apple10, e5m2, f32, f32, 64, 1)
nk_define_cross_metal_(e3m2, apple10, e3m2, f32, f32, 64, 1)
nk_define_cross_metal_(e2m3, apple10, e2m3, f32, f32, 64, 1)
nk_define_cross_metal_(e2m1, apple10, e2m1x2, f32, f32, 128, 2)

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_TARGET_APPLE10
#endif // NUMKONG_DOTS_APPLE10_H
