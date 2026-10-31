/**
 *  @file include/numkong/shapes/apple10.h
 *  @author Ash Vardanian
 *  @date October 24, 2026
 *  @brief Chamfer, Hausdorff and superposed RMSD of F16 clouds for Apple GPUs of Metal family 10,
 *      launched from C.
 *
 *  @sa include/numkong/shapes.h
 *  @sa include/numkong/shapes/apple10.metal, the kernels this embeds
 *  @sa include/numkong/shapes/blackwell.cuh, the CUDA sibling
 *
 *  Every signature matches @c metal's, whose launcher and memory contract these reuse. The kernels
 *  travel as their own `.metal` source, embedded after `metal.metal`, whose pack readers and SVD
 *  they call, and compile on the device at first use under Metal 4.0, the first language version
 *  to offer the tensor operations they multiply with.
 */
#ifndef NUMKONG_SHAPES_APPLE10_H
#define NUMKONG_SHAPES_APPLE10_H

#if NUMKONG_ARCH_METAL_
#if NUMKONG_TARGET_APPLE10
#include "numkong/shapes/metal.h" // `nk_shapes_launch_metal_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per device. */
static char const nk_shapes_source_apple10_[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/shapes/metal.metal" suffix(, )
#embed "numkong/shapes/apple10.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** Frames per side of an RMSD tile, as the kernels stage them. */
enum { nk_rmsds_frames_apple10_k = 21 };

NUMKONG_API nk_status_t nk_chamfers_packed_f16_apple10(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f32_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_stream_t stream) {
    return nk_shapes_launch_metal_(nk_shapes_source_apple10_, NUMKONG_METAL_LANGUAGE_4_0_,
                                   "nk_chamfers_packed_f16_apple10_kernel_", rows * columns, NUMKONG_SIZE_MAX, a_packed,
                                   rows, b_packed, columns, results, results_stride, rotations, stream);
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_apple10(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f32_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_stream_t stream) {
    return nk_shapes_launch_metal_(nk_shapes_source_apple10_, NUMKONG_METAL_LANGUAGE_4_0_,
                                   "nk_hausdorffs_packed_f16_apple10_kernel_", rows * columns, NUMKONG_SIZE_MAX,
                                   a_packed, rows, b_packed, columns, results, results_stride, rotations, stream);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_apple10(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                    nk_size_t results_stride, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    nk_size_t const tiles = nk_size_divide_round_up_(rows, nk_rmsds_frames_apple10_k) *
                            nk_size_divide_round_up_(columns, nk_rmsds_frames_apple10_k);
    return nk_shapes_launch_metal_(nk_shapes_source_apple10_, NUMKONG_METAL_LANGUAGE_4_0_,
                                   "nk_rmsds_packed_f16_apple10_kernel_", tiles,
                                   ((nk_shapes_packed_header_f32_t const *)a_packed[0])->points, a_packed, rows,
                                   b_packed, columns, results, results_stride, NUMKONG_NULL, stream);
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_TARGET_APPLE10
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_SHAPES_APPLE10_H
