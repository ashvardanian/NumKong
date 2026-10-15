/**
 *  @file include/numkong/shapes/blackwell.cuh
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Many-to-many shape distances for the NVIDIA compute capability 10.x family.
 *
 *  @sa include/numkong/shapes.h
 *  @sa include/numkong/dots/blackwell.cuh
 *
 *  Superposed RMSD multiplies frames as one product, C = A × Bᵀ, over their points. A holds the x,
 *  y and z rows of 32 row frames as lanes 0-31, 32-63 and 64-95, and B those of 32 column frames as
 *  columns 0-31, 32-63 and 64-95, so a frame's three axes land in one lane of warps 0, 1 and 2, and
 *  each pair's 3 × 3 cross-covariance in one 3 × 3 block of C. `tcgen05.mma.kind::f16` sums it in
 *  F32 tensor memory over 64-point slabs, which every thread copies from the packs into the
 *  128-byte swizzle, two slabs in flight. The three warps drain it through shared memory, and all
 *  four solve Theobald's quaternion polynomial in F32 per pair, folding into Ga + Gb − 2λ as the
 *  CPU capabilities do.
 *
 *  Chamfer and Hausdorff fold the norms into the depth, so the products are squared distances, [a,
 *  ‖a‖²_high, ‖a‖²_low, 1, 1] · [−2b, 1, 1, ‖b‖²_high, ‖b‖²_low] over 7 of 16 depth values, and the
 *  8 axis flips into the columns as 8 flipped copies of B: one product compares 128 outer points
 *  with 32 inner points under every flip. The expansion only searches: per flip, each thread keeps
 *  its outer point's least expanded value, the group of 8 inner points holding it and the least of
 *  all other groups, then recomputes that group in the difference form of the CPU capabilities, and
 *  rescans every inner point when another group comes within the expansion's rounding. The reverse
 *  direction runs as its own pass with the clouds swapped, as ‖S × b − a‖ = ‖b − S × a‖, and both
 *  clouds scale by one power of two that keeps the norms within F16.
 */
#ifndef NUMKONG_SHAPES_BLACKWELL_CUH
#define NUMKONG_SHAPES_BLACKWELL_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_CUDA_BLACKWELL_

#include "numkong/dots/blackwell.cuh"
#include "numkong/shapes/serial.h" // `nk_shapes_packed_header_f32_t`, `nk_shapes_alignment_k_`
#include "numkong/shapes/simt.cuh" // `nk_svd3x3_f32_simt_`, `nk_rotation_from_svd_f32_simt_`

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_BLACKWELL
#pragma region Configuration

enum {
    nk_shapes_threads_blackwell_k = 128,
    // RMSD tiles pair 32 row frames with 32 column frames, whose 3 axes fill 96 rows and columns.
    nk_rmsds_frames_blackwell_k = 32,
    nk_rmsds_operand_rows_blackwell_k = 96,
    nk_rmsds_slab_points_blackwell_k = 64,
    nk_rmsds_a_bytes_blackwell_k = 128 * 128,
    nk_rmsds_b_bytes_blackwell_k = nk_rmsds_operand_rows_blackwell_k * 128,
    // Each slab stages 16-byte chunks of both operands' 96 rows, 12 per thread.
    nk_rmsds_staged_chunks_blackwell_k = 2 * nk_rmsds_operand_rows_blackwell_k * 8,
    nk_rmsds_thread_chunks_blackwell_k = nk_rmsds_staged_chunks_blackwell_k / 128,
    nk_rmsds_accumulator_column_blackwell_k = 0,
    nk_rmsds_tensor_columns_blackwell_k = 128,
    // Chamfer and Hausdorff products: 128 outer points by 8 flips of 32 inner points, 32-byte rows.
    nk_shapes_outer_points_blackwell_k = 128,
    nk_shapes_inner_points_blackwell_k = 32,
    nk_shapes_flips_blackwell_k = 8,
    nk_shapes_group_points_blackwell_k = 8,
    nk_shapes_row_bytes_blackwell_k = 32,
    nk_shapes_accumulator_column_blackwell_k = 0,
    nk_shapes_tensor_columns_blackwell_k = 256,
    // The 32-byte swizzle's code in shared-memory descriptors.
    nk_smem_swizzle_32_blackwell_k = 6,
};

/** RMSD shared memory: two slabs of both operands, each 1024-byte aligned for the swizzle, the
 *  drained cross-covariances, padded against bank conflicts, and each tile's packs and norms. */
typedef struct {
    unsigned char a_operands[2][nk_rmsds_a_bytes_blackwell_k];
    unsigned char b_operands[2][nk_rmsds_b_bytes_blackwell_k];
    nk_f32_t cross_covariances[9][nk_rmsds_frames_blackwell_k][nk_rmsds_frames_blackwell_k + 1];
    nk_f32_t chunk_norms[nk_rmsds_staged_chunks_blackwell_k];
    nk_f32_t row_norms[nk_rmsds_frames_blackwell_k];
    nk_f32_t column_norms[nk_rmsds_frames_blackwell_k];
    unsigned char const *row_packs[nk_rmsds_frames_blackwell_k];
    unsigned char const *column_packs[nk_rmsds_frames_blackwell_k];
    nk_u64_t multiplied[2];
    nk_u32_t tensor_memory;
} nk_rmsds_shared_blackwell_t;

/** One RMSD launch: the packs, the output, and the tiles every block walks. */
typedef struct {
    void const *const *a_packed;
    void const *const *b_packed;
    nk_f32_t *results;
    nk_size_t rows, columns, results_stride;
    nk_size_t column_tiles, tiles;
} nk_rmsds_schedule_blackwell_t;

/** Chamfer and Hausdorff shared memory: one outer and one inner operand in the 32-byte swizzle, and
 *  each warp's per-flip sums and maxima. */
typedef struct {
    unsigned char outer_operand[nk_shapes_outer_points_blackwell_k * nk_shapes_row_bytes_blackwell_k];
    unsigned char inner_operand[nk_shapes_flips_blackwell_k * nk_shapes_inner_points_blackwell_k *
                                nk_shapes_row_bytes_blackwell_k];
    nk_f32_t warp_sums[4][nk_shapes_flips_blackwell_k];
    nk_f32_t warp_maxima[4][nk_shapes_flips_blackwell_k];
    nk_u64_t multiplied;
    nk_u32_t tensor_memory;
} nk_shapes_shared_blackwell_t;

/** One Chamfer or Hausdorff launch: the packs, the outputs, and the pairs every block walks. */
typedef struct {
    void const *const *a_packed;
    void const *const *b_packed;
    nk_f32_t *results;
    nk_f32_t *rotations;
    nk_size_t rows, columns, results_stride;
} nk_shapes_schedule_blackwell_t;

/** The per-flip sums and maxima of nearest distances, from one cloud's points to the other's. */
typedef struct {
    nk_f32_t sums[nk_shapes_flips_blackwell_k];
    nk_f32_t maxima[nk_shapes_flips_blackwell_k];
} nk_shapes_direction_blackwell_t;

#pragma endregion Configuration

#pragma region Helpers

NUMKONG_DEVICE nk_f32_t nk_f16_bits_to_f32_blackwell_(nk_u16_t bits) {
    nk_f32_t value;
    asm("cvt.f32.f16 %0, %1;\n" : "=f"(value) : "h"(bits));
    return value;
}

/** Σ‖aᵢ − R × bᵢ‖² at the best rotation, folded into Ga + Gb − 2λ and clamped at zero, as
 *  @c nk_rmsds_folded_f32_serial_ computes it from the row-major M = Σ aᵢ × bᵢᵀ. */
NUMKONG_DEVICE nk_f32_t nk_rmsds_folded_f32_blackwell_(nk_f32_t const *cross_covariance, nk_f32_t norms_squared) {
    nk_f32_t const upper = norms_squared / 2;
    if (!(upper > 0)) return 0;
    nk_f32_t const inverse = 1 / upper;
    nk_f32_t const xx = cross_covariance[0] * inverse, xy = cross_covariance[1] * inverse;
    nk_f32_t const xz = cross_covariance[2] * inverse, yx = cross_covariance[3] * inverse;
    nk_f32_t const yy = cross_covariance[4] * inverse, yz = cross_covariance[5] * inverse;
    nk_f32_t const zx = cross_covariance[6] * inverse, zy = cross_covariance[7] * inverse;
    nk_f32_t const zz = cross_covariance[8] * inverse;
    nk_f32_t const xx2 = xx * xx, yy2 = yy * yy, zz2 = zz * zz, xy2 = xy * xy, yz2 = yz * yz;
    nk_f32_t const xz2 = xz * xz, yx2 = yx * yx, zy2 = zy * zy, zx2 = zx * zx;
    nk_f32_t const xz_plus_zx = xz + zx, yz_plus_zy = yz + zy, xy_plus_yx = xy + yx;
    nk_f32_t const yz_minus_zy = yz - zy, xz_minus_zx = xz - zx, xy_minus_yx = xy - yx;
    nk_f32_t const xx_plus_yy = xx + yy, xx_minus_yy = xx - yy;
    nk_f32_t const off_diagonal = xy2 + xz2 - yx2 - zx2;
    nk_f32_t const quadratic = -2 * (xx2 + yy2 + zz2 + xy2 + yx2 + xz2 + zx2 + yz2 + zy2);
    nk_f32_t const linear = 8 *
                            (xx * yz * zy + yy * zx * xz + zz * xy * yx - xx * yy * zz - yz * zx * xy - zy * yx * xz);
    nk_f32_t const twice_minor = 2 * (yz * zy - yy * zz);
    nk_f32_t const squares = yy2 + zz2 - xx2 + yz2 + zy2;
    nk_f32_t const constant = off_diagonal * off_diagonal + (squares + twice_minor) * (squares - twice_minor) +
                              (-xz_plus_zx * yz_minus_zy + xy_minus_yx * (xx_minus_yy - zz)) *
                                  (-xz_minus_zx * yz_plus_zy + xy_minus_yx * (xx_minus_yy + zz)) +
                              (-xz_plus_zx * yz_plus_zy - xy_plus_yx * (xx_plus_yy - zz)) *
                                  (-xz_minus_zx * yz_minus_zy - xy_plus_yx * (xx_plus_yy + zz)) +
                              (xy_plus_yx * yz_plus_zy + xz_plus_zx * (xx_minus_yy + zz)) *
                                  (-xy_minus_yx * yz_minus_zy + xz_plus_zx * (xx_plus_yy + zz)) +
                              (xy_plus_yx * yz_minus_zy + xz_minus_zx * (xx_minus_yy - zz)) *
                                  (-xy_minus_yx * yz_plus_zy + xz_minus_zx * (xx_plus_yy - zz));
    nk_f32_t const k00 = xx + yy + zz, k01 = yz_minus_zy, k02 = -xz_minus_zx, k03 = xy_minus_yx;
    nk_f32_t const k11 = xx - yy - zz, k12 = xy_plus_yx, k13 = xz_plus_zx;
    nk_f32_t const k22 = -xx + yy - zz, k23 = yz_plus_zy, k33 = -xx - yy + zz;
    // λ never falls below the largest diagonal entry of K, which stops a rounded step running away
    nk_f32_t const floor_01 = k00 > k11 ? k00 : k11, floor_23 = k22 > k33 ? k22 : k33;
    nk_f32_t const floor = floor_01 > floor_23 ? floor_01 : floor_23;
    // ‖M‖ sets the scale of K's eigenvalues, so the adjugate's rounding noise is that cubed
    nk_f32_t const frobenius = __fsqrt_rn(-quadratic / 2),
                   noise = 64 * 0x1p-24f * 8 * frobenius * frobenius * frobenius;
    nk_f32_t eigenvalue = 1, slope = 0;
    for (int iteration = 0; iteration != 50; ++iteration) {
        nk_f32_t const squared = eigenvalue * eigenvalue;
        nk_f32_t const cubic = (squared + quadratic) * eigenvalue;
        nk_f32_t const shifted = cubic + linear;
        nk_f32_t const derivative = 2 * squared * eigenvalue + cubic + shifted;
        // A repeated largest root leaves no slope, and the iterate already sits on it
        if (!(derivative > 0)) break;
        slope = derivative;
        nk_f32_t const step = (shifted * eigenvalue + constant) / derivative;
        eigenvalue -= step;
        if (eigenvalue < floor) {
            eigenvalue = floor;
            break;
        }
        if (!(nk_f32_abs_(step) > 1e-7f)) break;
    }
    // Newton lands within ε of a root as steep as an isotropic or planar frame's, so it stays
    int refined = slope > 4 * frobenius * frobenius * frobenius;
    for (int refinement = refined ? 2 : 0; refinement != 2; ++refinement) {
        nk_f32_t const a00 = k00 - eigenvalue, a11 = k11 - eigenvalue, a22 = k22 - eigenvalue;
        nk_f32_t const a33 = k33 - eigenvalue;
        nk_f32_t const s0 = a00 * a11 - k01 * k01, s1 = a00 * k12 - k01 * k02, s2 = a00 * k13 - k01 * k03;
        nk_f32_t const s3 = k01 * k12 - a11 * k02, s4 = k01 * k13 - a11 * k03, s5 = k02 * k13 - k12 * k03;
        nk_f32_t const c0 = k02 * k13 - k03 * k12, c1 = k02 * k23 - k03 * a22, c2 = k02 * a33 - k03 * k23;
        nk_f32_t const c3 = k12 * k23 - k13 * a22, c4 = k12 * a33 - k13 * k23, c5 = a22 * a33 - k23 * k23;
        nk_f32_t const adjugate[4][4] = {{a11 * c5 - k12 * c4 + k13 * c3, -k01 * c5 + k02 * c4 - k03 * c3,
                                          k13 * s5 - k23 * s4 + a33 * s3, -k12 * s5 + a22 * s4 - k23 * s3},
                                         {-k01 * c5 + k12 * c2 - k13 * c1, a00 * c5 - k02 * c2 + k03 * c1,
                                          -k03 * s5 + k23 * s2 - a33 * s1, k02 * s5 - a22 * s2 + k23 * s1},
                                         {k01 * c4 - a11 * c2 + k13 * c0, -a00 * c4 + k01 * c2 - k03 * c0,
                                          k03 * s4 - k13 * s2 + a33 * s0, -k02 * s4 + k12 * s2 - k23 * s0},
                                         {-k01 * c3 + a11 * c1 - k12 * c0, a00 * c3 - k01 * c1 + k02 * c0,
                                          -k03 * s3 + k13 * s1 - k23 * s0, k02 * s3 - k12 * s1 + a22 * s0}};
        // The eigenvalue nearest λ dominates the adjugate, with either sign once λ lands below it
        int column = 0;
        for (int candidate = 1; candidate != 4; ++candidate)
            if (nk_f32_abs_(adjugate[candidate][candidate]) > nk_f32_abs_(adjugate[column][column])) column = candidate;
        if (!(nk_f32_abs_(adjugate[column][column]) > noise)) break;
        nk_f32_t const *vector = adjugate[column];
        nk_f32_t const image[4] = {k00 * vector[0] + k01 * vector[1] + k02 * vector[2] + k03 * vector[3],
                                   k01 * vector[0] + k11 * vector[1] + k12 * vector[2] + k13 * vector[3],
                                   k02 * vector[0] + k12 * vector[1] + k22 * vector[2] + k23 * vector[3],
                                   k03 * vector[0] + k13 * vector[1] + k23 * vector[2] + k33 * vector[3]};
        nk_f32_t const length = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2] +
                                vector[3] * vector[3];
        nk_f32_t const quotient =
            (vector[0] * image[0] + vector[1] * image[1] + vector[2] * image[2] + vector[3] * image[3]) / length;
        // λ lies between the diagonal bound and the iterate, so a quotient outside is noise
        if (!(quotient >= floor && quotient <= eigenvalue)) break;
        eigenvalue = quotient, refined = 1;
    }
    nk_f32_t residual = 2 * upper * (1 - eigenvalue);
    // Exactly linear frames repeat the root, so they fold over the Kabsch SVD rotation instead
    if (!refined) {
        nk_f32_t const transposed[9] = {cross_covariance[0], cross_covariance[3], cross_covariance[6],
                                        cross_covariance[1], cross_covariance[4], cross_covariance[7],
                                        cross_covariance[2], cross_covariance[5], cross_covariance[8]};
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9], rotation[9], trace = 0;
        nk_svd3x3_f32_simt_(transposed, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f32_simt_(svd_left, svd_right, rotation);
        if (nk_det3x3_f32_simt_(rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f32_simt_(svd_left, svd_right, rotation);
        }
        for (int cell = 0; cell != 9; ++cell) trace += rotation[cell] * cross_covariance[cell];
        residual = norms_squared - 2 * trace;
    }
    return residual > 0 ? residual : 0;
}

/** The x array of a shape pack of @c f16 points, with the elements from each axis to the next. */
NUMKONG_DEVICE nk_f16_t const *nk_shapes_packed_axes_f16_blackwell_(void const *packed, nk_size_t *axis_stride) {
    nk_size_t const points = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f32_t),
                                                                 nk_shapes_alignment_k_);
    *axis_stride = nk_size_round_up_to_multiple_(points * sizeof(nk_f16_t), nk_shapes_alignment_k_) / sizeof(nk_f16_t);
    return (nk_f16_t const *)((char const *)packed + header_bytes);
}

/** The row-major rotation of the unit quaternion (x, y, z, w), as the serial packs store frames. */
NUMKONG_DEVICE void nk_quaternion_to_mat3x3_f32_blackwell_(nk_f32_t const *quaternion, nk_f32_t *matrix) {
    nk_f32_t const w = quaternion[3], x = quaternion[0], y = quaternion[1], z = quaternion[2];
    matrix[0] = 1 - 2 * (y * y + z * z), matrix[1] = 2 * (x * y - w * z), matrix[2] = 2 * (x * z + w * y);
    matrix[3] = 2 * (x * y + w * z), matrix[4] = 1 - 2 * (x * x + z * z), matrix[5] = 2 * (y * z - w * x);
    matrix[6] = 2 * (x * z - w * y), matrix[7] = 2 * (y * z + w * x), matrix[8] = 1 - 2 * (x * x + y * y);
}

#pragma endregion Helpers

#pragma region RMSD

/** Zeroes the unused A lanes, sets up the barriers and allocates the accumulator. */
NUMKONG_DEVICE nk_u32_t nk_rmsds_initialize_blackwell_(nk_rmsds_shared_blackwell_t *shared) {
    uint4 const zeros = {0, 0, 0, 0};
    for (unsigned index = threadIdx.x; index < 2 * 32 * 8; index += nk_shapes_threads_blackwell_k)
        *(uint4 *)(shared->a_operands[index >> 8] + nk_rmsds_operand_rows_blackwell_k * 128 +
                   (index & 255) * 16) = zeros;
    if (threadIdx.x == 0) {
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&shared->multiplied[0]), 1);
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&shared->multiplied[1]), 1);
        nk_mbarrier_init_fence_blackwell_();
    }
    if (threadIdx.x < 32)
        nk_tmem_alloc_blackwell_(nk_shared_address_ampere_(&shared->tensor_memory),
                                 nk_rmsds_tensor_columns_blackwell_k);
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    nk_tmem_fence_after_blackwell_();
    return shared->tensor_memory;
}

/** Reads the packs of one tile's 32 row and 32 column frames, null past either edge. A frame of
 *  another size than the first row frame is never read, and its NaN norm marks its pairs, as a
 *  launch cannot refuse it. */
NUMKONG_DEVICE void nk_rmsds_gather_f16_blackwell_(nk_rmsds_schedule_blackwell_t const *schedule,
                                                   nk_rmsds_shared_blackwell_t *shared, nk_size_t row_begin,
                                                   nk_size_t column_begin, nk_size_t points) {
    if (threadIdx.x < 2 * nk_rmsds_frames_blackwell_k) {
        int const rows_side = threadIdx.x < nk_rmsds_frames_blackwell_k;
        unsigned const frame = threadIdx.x & 31;
        nk_size_t const index = (rows_side ? row_begin : column_begin) + frame;
        int const inside = index < (rows_side ? schedule->rows : schedule->columns);
        unsigned char const *pack = inside ? (unsigned char const *)(rows_side ? schedule->a_packed
                                                                               : schedule->b_packed)[index]
                                           : (unsigned char const *)NUMKONG_NULL;
        int const matching = inside && ((nk_shapes_packed_header_f32_t const *)pack)->points == points;
        nk_f32_t const mark = inside && !matching ? __uint_as_float(0x7FC00000u) : 0;
        if (rows_side) shared->row_packs[frame] = matching ? pack : NUMKONG_NULL, shared->row_norms[frame] = mark;
        else shared->column_packs[frame] = matching ? pack : NUMKONG_NULL, shared->column_norms[frame] = mark;
    }
    __syncthreads();
}

/** Copies one 64-point slab of both operands into @p buffer: the axes of each frame as rows of 128
 *  bytes, each 16-byte chunk at its place in the 128-byte swizzle, zeros past the points. Each
 *  thread adds the squares of its chunks to @p partials, as it stages the same rows every slab. */
NUMKONG_DEVICE void nk_rmsds_stage_f16_blackwell_(nk_rmsds_shared_blackwell_t *shared, unsigned buffer, nk_size_t slab,
                                                  nk_size_t points, nk_size_t axis_stride, nk_f32_t *partials) {
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f32_t),
                                                                 nk_shapes_alignment_k_);
    unsigned const chunks_per_side = nk_rmsds_operand_rows_blackwell_k * 8;
#pragma unroll
    for (unsigned slot = 0; slot != nk_rmsds_thread_chunks_blackwell_k; ++slot) {
        unsigned const index = threadIdx.x + slot * nk_shapes_threads_blackwell_k;
        unsigned const side = index >= chunks_per_side, within = index - side * chunks_per_side;
        unsigned const row = within >> 3, chunk = within & 7, axis = row >> 5, frame = row & 31;
        unsigned char const *pack = side ? shared->column_packs[frame] : shared->row_packs[frame];
        nk_size_t const point = slab * nk_rmsds_slab_points_blackwell_k + chunk * 8;
        uint4 value = {0, 0, 0, 0};
        if (pack && point < points)
            value = *(uint4 const *)(pack + header_bytes + (axis * axis_stride + point) * sizeof(nk_f16_t));
        unsigned char *operand = side ? shared->b_operands[buffer] : shared->a_operands[buffer];
        *(uint4 *)(operand + row * 128 + ((chunk ^ (row & 7)) << 4)) = value;
        nk_u32_t const pairs[4] = {value.x, value.y, value.z, value.w};
        for (unsigned pair = 0; pair != 4; ++pair) {
            nk_f32_t const low = nk_f16_bits_to_f32_blackwell_((nk_u16_t)(pairs[pair] & 0xFFFFu));
            nk_f32_t const high = nk_f16_bits_to_f32_blackwell_((nk_u16_t)(pairs[pair] >> 16));
            partials[slot] += low * low + high * high;
        }
    }
}

/** Accumulates one tile's cross-covariances over every slab, staging each while the tensor cores
 *  multiply the one before, and waits for the last product. @p slabs_issued counts the slabs this
 *  block has multiplied, whose parity picks each barrier's phase. */
NUMKONG_DEVICE void nk_rmsds_multiply_f16_blackwell_(nk_rmsds_shared_blackwell_t *shared, nk_u32_t tensor_memory,
                                                     nk_size_t points, nk_size_t axis_stride, nk_size_t *slabs_issued) {
    nk_size_t const slabs = nk_size_divide_round_up_(points, nk_rmsds_slab_points_blackwell_k);
    nk_u32_t const instruction = nk_mma_instruction_blackwell_(0, 0, 128, nk_rmsds_operand_rows_blackwell_k,
                                                               nk_major_k_k, nk_major_k_k);
    nk_f32_t partials[nk_rmsds_thread_chunks_blackwell_k];
    for (unsigned slot = 0; slot != nk_rmsds_thread_chunks_blackwell_k; ++slot) partials[slot] = 0;
    for (nk_size_t slab = 0; slab != slabs; ++slab) {
        nk_size_t const issued = *slabs_issued + slab;
        unsigned const buffer = (unsigned)(issued & 1);
        nk_u32_t const barrier = nk_shared_address_ampere_(&shared->multiplied[buffer]);
        if (issued >= 2) nk_mbarrier_wait_blackwell_(barrier, (nk_u32_t)((issued - 2) >> 1) & 1);
        nk_rmsds_stage_f16_blackwell_(shared, buffer, slab, points, axis_stride, partials);
        nk_fence_async_shared_blackwell_();
        __syncthreads();
        if (threadIdx.x == 0) {
            nk_tmem_fence_after_blackwell_();
            nk_u64_t const a = nk_smem_descriptor_blackwell_(nk_shared_address_ampere_(shared->a_operands[buffer]),
                                                             nk_smem_swizzle_128_blackwell_k, 16, 1024);
            nk_u64_t const b = nk_smem_descriptor_blackwell_(nk_shared_address_ampere_(shared->b_operands[buffer]),
                                                             nk_smem_swizzle_128_blackwell_k, 16, 1024);
            for (unsigned step = 0; step != 4; ++step)
                nk_mma_f16_blackwell_(tensor_memory + nk_rmsds_accumulator_column_blackwell_k, a + 2 * step,
                                      b + 2 * step, instruction, slab != 0 || step != 0);
            nk_mma_commit_blackwell_(barrier);
        }
    }
    if (slabs) {
        nk_size_t const last = *slabs_issued + slabs - 1;
        nk_mbarrier_wait_blackwell_(nk_shared_address_ampere_(&shared->multiplied[last & 1]),
                                    (nk_u32_t)(last >> 1) & 1);
        *slabs_issued += slabs;
    }
    nk_tmem_fence_after_blackwell_();
    // Every frame's squared norm sums its 3 rows of 8 chunk columns in one order
    for (unsigned slot = 0; slot != nk_rmsds_thread_chunks_blackwell_k; ++slot)
        shared->chunk_norms[threadIdx.x + slot * nk_shapes_threads_blackwell_k] = partials[slot];
    __syncthreads();
    if (threadIdx.x < 2 * nk_rmsds_frames_blackwell_k) {
        unsigned const side = threadIdx.x >> 5, frame = threadIdx.x & 31;
        nk_f32_t norm = side ? shared->column_norms[frame] : shared->row_norms[frame];
        for (unsigned axis = 0; axis != 3; ++axis)
            for (unsigned chunk = 0; chunk != 8; ++chunk)
                norm +=
                    shared->chunk_norms[side * nk_rmsds_operand_rows_blackwell_k * 8 + (axis * 32 + frame) * 8 + chunk];
        if (side) shared->column_norms[frame] = norm;
        else shared->row_norms[frame] = norm;
    }
}

/** Warps 0, 1 and 2 read the x, y and z rows of C and file each 3 × 3 block by pair. */
NUMKONG_DEVICE void nk_rmsds_drain_blackwell_(nk_rmsds_shared_blackwell_t *shared, nk_u32_t tensor_memory) {
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp < 3) {
        nk_u32_t values[32];
        for (unsigned column_axis = 0; column_axis != 3; ++column_axis) {
            nk_tmem_load_x32_blackwell_(
                nk_tmem_offset_blackwell_(tensor_memory, warp * 32,
                                          nk_rmsds_accumulator_column_blackwell_k + column_axis * 32),
                values);
            for (unsigned column = 0; column != 32; ++column)
                shared->cross_covariances[warp * 3 + column_axis][lane][column] = __uint_as_float(values[column]);
        }
    }
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
}

/** Every thread solves 8 pairs, consecutive lanes on consecutive columns, and stores their RMSD. */
NUMKONG_DEVICE void nk_rmsds_superpose_blackwell_(nk_rmsds_schedule_blackwell_t const *schedule,
                                                  nk_rmsds_shared_blackwell_t *shared, nk_size_t row_begin,
                                                  nk_size_t column_begin, nk_size_t points) {
    for (unsigned pair = threadIdx.x; pair < nk_rmsds_frames_blackwell_k * nk_rmsds_frames_blackwell_k;
         pair += nk_shapes_threads_blackwell_k) {
        unsigned const row = pair >> 5, column = pair & 31;
        if (row_begin + row >= schedule->rows || column_begin + column >= schedule->columns) continue;
        nk_f32_t cross_covariance[9];
        for (unsigned cell = 0; cell != 9; ++cell)
            cross_covariance[cell] = shared->cross_covariances[cell][row][column];
        nk_f32_t const norms = shared->row_norms[row] + shared->column_norms[column];
        nk_f32_t const residual = nk_rmsds_folded_f32_blackwell_(cross_covariance, norms);
        nk_f32_t *results_row = (nk_f32_t *)((char *)schedule->results + (row_begin + row) * schedule->results_stride);
        results_row[column_begin + column] = norms != norms ? norms
                                             : points       ? __fsqrt_rn(residual / (nk_f32_t)points)
                                                            : 0.0f;
    }
    __syncthreads();
}

static __global__ void __launch_bounds__(nk_shapes_threads_blackwell_k)
    nk_rmsds_packed_f16_blackwell_kernel_(nk_rmsds_schedule_blackwell_t const schedule) {
    extern __shared__ unsigned char nk_rmsds_dynamic_shared_blackwell_[];
    nk_rmsds_shared_blackwell_t *shared = (nk_rmsds_shared_blackwell_t *)nk_shared_aligned_ampere_(
        nk_rmsds_dynamic_shared_blackwell_, 1024);
    nk_u32_t const tensor_memory = nk_rmsds_initialize_blackwell_(shared);
    nk_size_t const points = ((nk_shapes_packed_header_f32_t const *)schedule.a_packed[0])->points;
    nk_size_t const axis_stride = nk_size_round_up_to_multiple_(points * sizeof(nk_f16_t), nk_shapes_alignment_k_) /
                                  sizeof(nk_f16_t);
    nk_size_t slabs_issued = 0;
    for (nk_size_t tile = blockIdx.x; tile < schedule.tiles; tile += gridDim.x) {
        nk_size_t const row_begin = tile / schedule.column_tiles * nk_rmsds_frames_blackwell_k;
        nk_size_t const column_begin = tile % schedule.column_tiles * nk_rmsds_frames_blackwell_k;
        nk_rmsds_gather_f16_blackwell_(&schedule, shared, row_begin, column_begin, points);
        nk_rmsds_multiply_f16_blackwell_(shared, tensor_memory, points, axis_stride, &slabs_issued);
        nk_rmsds_drain_blackwell_(shared, tensor_memory);
        nk_rmsds_superpose_blackwell_(&schedule, shared, row_begin, column_begin, points);
    }
    if (threadIdx.x < 32) nk_tmem_dealloc_blackwell_(tensor_memory, nk_rmsds_tensor_columns_blackwell_k);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_blackwell(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                      nk_size_t results_stride, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f32_t) - 1)) return nk_misaligned_k;
    nk_rmsds_schedule_blackwell_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rows = rows, schedule.columns = columns, schedule.results_stride = results_stride;
    schedule.column_tiles = nk_size_divide_round_up_(columns, nk_rmsds_frames_blackwell_k);
    schedule.tiles = nk_size_divide_round_up_(rows, nk_rmsds_frames_blackwell_k) * schedule.column_tiles;
    nk_size_t const shared_bytes = sizeof(nk_rmsds_shared_blackwell_t) + 1024;
    return nk_launch_resident_cuda_((void const *)nk_rmsds_packed_f16_blackwell_kernel_, nk_shapes_threads_blackwell_k,
                                    shared_bytes, shared_bytes, schedule.tiles, &schedule, stream);
}

#pragma endregion RMSD

#pragma region Chamfer and Hausdorff

/** Zeroes both operands, sets up the barrier and allocates the accumulator's tensor memory. */
NUMKONG_DEVICE nk_u32_t nk_shapes_initialize_blackwell_(nk_shapes_shared_blackwell_t *shared) {
    uint4 const zeros = {0, 0, 0, 0};
    for (unsigned index = threadIdx.x; index < sizeof(shared->outer_operand) / 16;
         index += nk_shapes_threads_blackwell_k)
        *(uint4 *)(shared->outer_operand + index * 16) = zeros;
    for (unsigned index = threadIdx.x; index < sizeof(shared->inner_operand) / 16;
         index += nk_shapes_threads_blackwell_k)
        *(uint4 *)(shared->inner_operand + index * 16) = zeros;
    if (threadIdx.x == 0) {
        nk_mbarrier_init_blackwell_(nk_shared_address_ampere_(&shared->multiplied), 1);
        nk_mbarrier_init_fence_blackwell_();
    }
    if (threadIdx.x < 32)
        nk_tmem_alloc_blackwell_(nk_shared_address_ampere_(&shared->tensor_memory),
                                 nk_shapes_tensor_columns_blackwell_k);
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    nk_tmem_fence_after_blackwell_();
    return shared->tensor_memory;
}

/** The power of two both clouds of a pair divide by, so that every squared norm stays below 2¹⁴
 *  and every doubled coordinate below 2⁸ in F16: 2^−k for k = max(0, ⌊log₂ r⌋ − 6). */
NUMKONG_DEVICE nk_f32_t nk_shapes_scale_blackwell_(nk_f32_t largest_radius) {
    int const exponent = (int)((__float_as_uint(largest_radius) >> 23) & 0xFF) - 127 - 6;
    return exponent > 0 ? __uint_as_float((nk_u32_t)(127 - exponent) << 23) : 1.0f;
}

/** Reads one packed point scaled by @p scale and rounded into F16, with its squared norm summed in
 *  F32 from the rounded coordinates, or zeros past the cloud. */
NUMKONG_DEVICE void nk_shapes_point_f16_blackwell_(nk_f16_t const *axes, nk_size_t axis_stride, nk_size_t point,
                                                   nk_size_t count, nk_f32_t scale, nk_u32_t *coordinates,
                                                   nk_f32_t *norm) {
    nk_u16_t const *axes_bits = (nk_u16_t const *)axes;
    *norm = 0;
    for (unsigned axis = 0; axis != 3; ++axis) {
        nk_u16_t const bits = point < count ? axes_bits[axis * axis_stride + point] : 0;
        coordinates[axis] = nk_f32_to_f16_bits_ampere_(nk_f16_bits_to_f32_blackwell_(bits) * scale);
        nk_f32_t const rounded = nk_f16_bits_to_f32_blackwell_((nk_u16_t)coordinates[axis]);
        *norm += rounded * rounded;
    }
}

/** Writes the first 16-byte chunk of a 32-byte operand row in the 32-byte swizzle; the second
 *  chunk stays zero from initialization. */
NUMKONG_DEVICE void nk_shapes_store_row_blackwell_(unsigned char *operand, unsigned row, uint4 chunk) {
    *(uint4 *)(operand + row * nk_shapes_row_bytes_blackwell_k + (((row >> 2) & 1) << 4)) = chunk;
}

/** Stages 128 outer points as [x, y, z, ‖x‖²_high, ‖x‖²_low, 1, 1, 0], returning this thread's
 *  ‖x‖². */
NUMKONG_DEVICE nk_f32_t nk_shapes_stage_outer_f16_blackwell_(nk_shapes_shared_blackwell_t *shared, nk_f16_t const *axes,
                                                             nk_size_t axis_stride, nk_size_t first, nk_size_t count,
                                                             nk_f32_t scale) {
    nk_u32_t coordinates[3];
    nk_f32_t norm;
    nk_shapes_point_f16_blackwell_(axes, axis_stride, first + threadIdx.x, count, scale, coordinates, &norm);
    nk_u32_t const high = nk_f32_to_f16_bits_ampere_(norm);
    nk_u32_t const low = nk_f32_to_f16_bits_ampere_(norm - nk_f16_bits_to_f32_blackwell_((nk_u16_t)high));
    nk_u32_t const one = 0x3C00u;
    uint4 chunk;
    chunk.x = coordinates[0] | (coordinates[1] << 16), chunk.y = coordinates[2] | (high << 16);
    chunk.z = low | (one << 16), chunk.w = one;
    nk_shapes_store_row_blackwell_(shared->outer_operand, threadIdx.x, chunk);
    return norm;
}

/** Stages 32 inner points under all 8 flips, flip s at rows 32s to 32s + 31, as
 *  [−2 × S × y, 1, 1, ‖y‖²_high, ‖y‖²_low, 0]. */
NUMKONG_DEVICE void nk_shapes_stage_inner_f16_blackwell_(nk_shapes_shared_blackwell_t *shared, nk_f16_t const *axes,
                                                         nk_size_t axis_stride, nk_size_t first, nk_size_t count,
                                                         nk_f32_t scale) {
    unsigned const point = threadIdx.x & 31, flip_group = threadIdx.x >> 5;
    nk_u32_t coordinates[3];
    nk_f32_t norm;
    nk_shapes_point_f16_blackwell_(axes, axis_stride, first + point, count, scale, coordinates, &norm);
    nk_u32_t const high = nk_f32_to_f16_bits_ampere_(norm);
    nk_u32_t const low = nk_f32_to_f16_bits_ampere_(norm - nk_f16_bits_to_f32_blackwell_((nk_u16_t)high));
    nk_u32_t const one = 0x3C00u;
    nk_u32_t doubled[3];
    // Doubling adds one to the exponent of every nonzero half, exactly, and the sign bit flips it
    for (unsigned axis = 0; axis != 3; ++axis)
        doubled[axis] = coordinates[axis] & 0x7FFFu ? (coordinates[axis] + 0x0400u) ^ 0x8000u : 0;
    for (unsigned flip = flip_group; flip < nk_shapes_flips_blackwell_k; flip += 4) {
        nk_u32_t flipped[3];
        for (unsigned axis = 0; axis != 3; ++axis) flipped[axis] = doubled[axis] ^ ((flip >> axis) & 1 ? 0x8000u : 0);
        uint4 chunk;
        chunk.x = flipped[0] | (flipped[1] << 16), chunk.y = flipped[2] | (one << 16);
        chunk.z = one | (high << 16), chunk.w = low;
        nk_shapes_store_row_blackwell_(shared->inner_operand, flip * nk_shapes_inner_points_blackwell_k + point, chunk);
    }
}

/** One product of the staged operands into tensor memory, waited for by every thread. */
NUMKONG_DEVICE void nk_shapes_multiply_f16_blackwell_(nk_shapes_shared_blackwell_t *shared, nk_u32_t tensor_memory,
                                                      nk_u32_t *phase) {
    nk_u32_t const barrier = nk_shared_address_ampere_(&shared->multiplied);
    nk_fence_async_shared_blackwell_();
    __syncthreads();
    if (threadIdx.x == 0) {
        nk_tmem_fence_after_blackwell_();
        nk_u64_t const a = nk_smem_descriptor_blackwell_(nk_shared_address_ampere_(shared->outer_operand),
                                                         nk_smem_swizzle_32_blackwell_k, 16, 256);
        nk_u64_t const b = nk_smem_descriptor_blackwell_(nk_shared_address_ampere_(shared->inner_operand),
                                                         nk_smem_swizzle_32_blackwell_k, 16, 256);
        nk_mma_f16_blackwell_(
            tensor_memory + nk_shapes_accumulator_column_blackwell_k, a, b,
            nk_mma_instruction_blackwell_(0, 0, nk_shapes_outer_points_blackwell_k,
                                          nk_shapes_tensor_columns_blackwell_k, nk_major_k_k, nk_major_k_k),
            0);
        nk_mma_commit_blackwell_(barrier);
    }
    nk_mbarrier_wait_blackwell_(barrier, *phase);
    *phase ^= 1;
    nk_tmem_fence_after_blackwell_();
}

/** ‖a − S × b‖² between @p outer_point and packed inner point @p index under @p flip, unfused in
 *  the serial order, so it matches the CPU capabilities bit for bit. */
NUMKONG_DEVICE nk_f32_t nk_shapes_squared_distance_f16_blackwell_(nk_f32_t const *outer_point,
                                                                  nk_f16_t const *inner_axes, nk_size_t inner_stride,
                                                                  nk_size_t index, unsigned flip) {
    nk_u16_t const *inner_bits = (nk_u16_t const *)inner_axes;
    nk_f32_t squares[3];
    for (unsigned axis = 0; axis != 3; ++axis) {
        nk_f32_t const inner = nk_f16_bits_to_f32_blackwell_(inner_bits[axis * inner_stride + index]);
        nk_f32_t const difference = (flip >> axis) & 1 ? outer_point[axis] + inner : outer_point[axis] - inner;
        squares[axis] = __fmul_rn(difference, difference);
    }
    return __fadd_rn(__fadd_rn(squares[0], squares[1]), squares[2]);
}

/** Lowers each flip's least expanded value of this thread's outer point over groups of the first @p
 *  valid inner points of the product, keeping the group that holds it and the least value of every
 *  other group, then lets the next product overwrite it. Non-negative F32 bits order like their
 *  values and negative ones clamp to zero, so a single integer minimum takes each point. */
NUMKONG_DEVICE void nk_shapes_drain_nearest_f16_blackwell_(nk_u32_t tensor_memory, nk_u32_t first_group, unsigned valid,
                                                           nk_u32_t *least, nk_u32_t *groups, nk_u32_t *others) {
    unsigned const warp = threadIdx.x >> 5;
    nk_i32_t values[32];
#pragma unroll
    for (unsigned flip = 0; flip != nk_shapes_flips_blackwell_k; ++flip) {
        nk_tmem_load_x32_blackwell_(nk_tmem_offset_blackwell_(tensor_memory, warp * 32,
                                                              nk_shapes_accumulator_column_blackwell_k +
                                                                  flip * nk_shapes_inner_points_blackwell_k),
                                    (nk_u32_t *)values);
        if (valid != 32)
#pragma unroll
            for (unsigned point = 0; point != 32; ++point) values[point] = point < valid ? values[point] : 0x7FFFFFFF;
#pragma unroll
        for (unsigned group = 0; group != 32 / nk_shapes_group_points_blackwell_k; ++group) {
            nk_i32_t group_least = 0x7FFFFFFF;
#pragma unroll
            for (unsigned point = 0; point != nk_shapes_group_points_blackwell_k; ++point) {
                nk_i32_t const value = values[group * nk_shapes_group_points_blackwell_k + point];
                group_least = value < group_least ? value : group_least;
            }
            nk_u32_t const clamped = group_least > 0 ? (nk_u32_t)group_least : 0u;
            int const below = clamped < least[flip];
            nk_u32_t const displaced = below ? least[flip] : clamped;
            others[flip] = displaced < others[flip] ? displaced : others[flip];
            groups[flip] = below ? first_group + group : groups[flip];
            least[flip] = below ? clamped : least[flip];
        }
    }
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
}

/** Sums and maxima over every outer point of its distance to the nearest inner point, per flip.
 *  Expanded values lie within a tolerance of the exact squares, so the nearest point sits in the
 *  group of the least one unless another group's least falls within twice that: then the thread
 *  rescans every inner point. Distances are recomputed from the packed coordinates in the
 *  difference form of the CPU capabilities. */
NUMKONG_DEVICE void nk_shapes_direction_f16_blackwell_(nk_shapes_shared_blackwell_t *shared, nk_u32_t tensor_memory,
                                                       nk_u32_t *phase, void const *outer_packed,
                                                       void const *inner_packed, nk_f32_t scale,
                                                       nk_shapes_direction_blackwell_t *direction) {
    nk_size_t const outer_count = ((nk_shapes_packed_header_f32_t const *)outer_packed)->points;
    nk_size_t const inner_count = ((nk_shapes_packed_header_f32_t const *)inner_packed)->points;
    nk_size_t outer_stride, inner_stride;
    nk_f16_t const *outer_axes = nk_shapes_packed_axes_f16_blackwell_(outer_packed, &outer_stride);
    nk_f16_t const *inner_axes = nk_shapes_packed_axes_f16_blackwell_(inner_packed, &inner_stride);
    nk_f32_t const inner_reach = ((nk_shapes_packed_header_f32_t const *)inner_packed)->max_radius * scale;
    nk_f32_t sums[nk_shapes_flips_blackwell_k], maxima[nk_shapes_flips_blackwell_k];
    for (unsigned flip = 0; flip != nk_shapes_flips_blackwell_k; ++flip) sums[flip] = 0, maxima[flip] = 0;
    for (nk_size_t outer = 0; outer < outer_count; outer += nk_shapes_outer_points_blackwell_k) {
        nk_f32_t const norm = nk_shapes_stage_outer_f16_blackwell_(shared, outer_axes, outer_stride, outer, outer_count,
                                                                   scale);
        // Rounding of the 7-term expansion and the split norms, with room for the tensor cores
        nk_f32_t const tolerance = 0x1p-19f * (norm + inner_reach * inner_reach);
        nk_u32_t least[nk_shapes_flips_blackwell_k], groups[nk_shapes_flips_blackwell_k],
            others[nk_shapes_flips_blackwell_k];
#pragma unroll
        for (unsigned flip = 0; flip != nk_shapes_flips_blackwell_k; ++flip)
            least[flip] = ~0u, groups[flip] = 0, others[flip] = ~0u;
        for (nk_size_t inner = 0; inner < inner_count; inner += nk_shapes_inner_points_blackwell_k) {
            nk_shapes_stage_inner_f16_blackwell_(shared, inner_axes, inner_stride, inner, inner_count, scale);
            nk_shapes_multiply_f16_blackwell_(shared, tensor_memory, phase);
            nk_size_t const remaining = inner_count - inner;
            nk_shapes_drain_nearest_f16_blackwell_(tensor_memory,
                                                   (nk_u32_t)(inner / nk_shapes_group_points_blackwell_k),
                                                   remaining < 32 ? (unsigned)remaining : 32u, least, groups, others);
        }
        nk_size_t const point = outer + threadIdx.x;
        if (point >= outer_count) continue;
        nk_u16_t const *outer_bits = (nk_u16_t const *)outer_axes;
        nk_f32_t outer_point[3];
        for (unsigned axis = 0; axis != 3; ++axis)
            outer_point[axis] = nk_f16_bits_to_f32_blackwell_(outer_bits[axis * outer_stride + point]);
#pragma unroll
        for (unsigned flip = 0; flip != nk_shapes_flips_blackwell_k; ++flip) {
            nk_f32_t exact = NUMKONG_F32_INF;
            nk_f32_t const threshold = __fadd_ru(__uint_as_float(least[flip]), 2 * tolerance);
            if (others[flip] <= __float_as_uint(threshold))
                for (nk_size_t index = 0; index != inner_count; ++index) {
                    nk_f32_t const candidate = nk_shapes_squared_distance_f16_blackwell_(outer_point, inner_axes,
                                                                                         inner_stride, index, flip);
                    exact = candidate < exact ? candidate : exact;
                }
            else
#pragma unroll
                for (unsigned member = 0; member != nk_shapes_group_points_blackwell_k; ++member) {
                    nk_size_t const index = (nk_size_t)groups[flip] * nk_shapes_group_points_blackwell_k + member;
                    nk_f32_t const candidate = index < inner_count
                                                   ? nk_shapes_squared_distance_f16_blackwell_(
                                                         outer_point, inner_axes, inner_stride, index, flip)
                                                   : NUMKONG_F32_INF;
                    exact = candidate < exact ? candidate : exact;
                }
            nk_f32_t const distance = __fsqrt_rn(exact);
            sums[flip] += distance, maxima[flip] = distance > maxima[flip] ? distance : maxima[flip];
        }
    }
    unsigned const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (unsigned flip = 0; flip != nk_shapes_flips_blackwell_k; ++flip) {
        for (unsigned offset = 16; offset; offset >>= 1) {
            nk_f32_t const other_maximum = __shfl_xor_sync(0xFFFFFFFFu, maxima[flip], offset);
            sums[flip] += __shfl_xor_sync(0xFFFFFFFFu, sums[flip], offset);
            maxima[flip] = other_maximum > maxima[flip] ? other_maximum : maxima[flip];
        }
        if (lane == 0) shared->warp_sums[warp][flip] = sums[flip], shared->warp_maxima[warp][flip] = maxima[flip];
    }
    __syncthreads();
    for (unsigned flip = 0; flip != nk_shapes_flips_blackwell_k; ++flip) {
        direction->sums[flip] = shared->warp_sums[0][flip] + shared->warp_sums[1][flip] + shared->warp_sums[2][flip] +
                                shared->warp_sums[3][flip];
        nk_f32_t maximum = shared->warp_maxima[0][flip];
        for (unsigned other = 1; other != 4; ++other)
            maximum = shared->warp_maxima[other][flip] > maximum ? shared->warp_maxima[other][flip] : maximum;
        direction->maxima[flip] = maximum;
    }
    __syncthreads();
}

/** Writes the best flip's distance and, when asked, its rotation 𝑄ₐᵀ × S × 𝑄ᵦ, or the identity for
 *  an empty cloud. */
NUMKONG_DEVICE void nk_shapes_write_best_flip_blackwell_(nk_shapes_schedule_blackwell_t const *schedule, nk_size_t row,
                                                         nk_size_t column, nk_f32_t const *flip_distances, int clouds) {
    unsigned best_flip = 0;
    for (unsigned flip = 1; flip != nk_shapes_flips_blackwell_k; ++flip)
        if (flip_distances[flip] < flip_distances[best_flip]) best_flip = flip;
    ((nk_f32_t *)((char *)schedule->results + row * schedule->results_stride))[column] = flip_distances[best_flip];
    if (!schedule->rotations) return;
    nk_f32_t *rotation = schedule->rotations + (row * schedule->columns + column) * 9;
    nk_f32_t row_axes[9], column_axes[9];
    nk_quaternion_to_mat3x3_f32_blackwell_(((nk_shapes_packed_header_f32_t const *)schedule->a_packed[row])->frame,
                                           row_axes);
    nk_quaternion_to_mat3x3_f32_blackwell_(((nk_shapes_packed_header_f32_t const *)schedule->b_packed[column])->frame,
                                           column_axes);
    for (unsigned output_row = 0; output_row != 3; ++output_row)
        for (unsigned output_column = 0; output_column != 3; ++output_column) {
            nk_f32_t sum = 0;
            for (unsigned axis = 0; axis != 3; ++axis)
                sum += row_axes[axis * 3 + output_row] * ((best_flip >> axis) & 1 ? -1.0f : 1.0f) *
                       column_axes[axis * 3 + output_column];
            rotation[output_row * 3 + output_column] = clouds ? sum : (output_row == output_column ? 1.0f : 0.0f);
        }
}

/** Both directions of one pair, or the empty-cloud convention, into per-flip Chamfer and Hausdorff
 *  distances; every thread returns them. Returns whether both clouds hold points. */
NUMKONG_DEVICE int nk_shapes_flips_f16_blackwell_(nk_shapes_shared_blackwell_t *shared, nk_u32_t tensor_memory,
                                                  nk_u32_t *phase, void const *row_packed, void const *column_packed,
                                                  nk_f32_t *chamfers, nk_f32_t *hausdorffs) {
    nk_shapes_packed_header_f32_t const *row_header = (nk_shapes_packed_header_f32_t const *)row_packed;
    nk_shapes_packed_header_f32_t const *column_header = (nk_shapes_packed_header_f32_t const *)column_packed;
    nk_size_t const rows = row_header->points, columns = column_header->points;
    if (!rows || !columns) {
        nk_f32_t const distance = rows == columns ? 0 : NUMKONG_F32_INF;
        for (unsigned flip = 0; flip != nk_shapes_flips_blackwell_k; ++flip)
            chamfers[flip] = distance, hausdorffs[flip] = distance;
        return 0;
    }
    nk_f32_t const scale = nk_shapes_scale_blackwell_(
        row_header->max_radius > column_header->max_radius ? row_header->max_radius : column_header->max_radius);
    nk_shapes_direction_blackwell_t forward, backward;
    nk_shapes_direction_f16_blackwell_(shared, tensor_memory, phase, row_packed, column_packed, scale, &forward);
    nk_shapes_direction_f16_blackwell_(shared, tensor_memory, phase, column_packed, row_packed, scale, &backward);
    for (unsigned flip = 0; flip != nk_shapes_flips_blackwell_k; ++flip) {
        chamfers[flip] = (forward.sums[flip] / (nk_f32_t)rows + backward.sums[flip] / (nk_f32_t)columns) / 2;
        hausdorffs[flip] = forward.maxima[flip] > backward.maxima[flip] ? forward.maxima[flip] : backward.maxima[flip];
    }
    return 1;
}

static __global__ void __launch_bounds__(nk_shapes_threads_blackwell_k)
    nk_chamfers_packed_f16_blackwell_kernel_(nk_shapes_schedule_blackwell_t const schedule) {
    extern __shared__ unsigned char nk_shapes_dynamic_shared_blackwell_[];
    nk_shapes_shared_blackwell_t *shared = (nk_shapes_shared_blackwell_t *)nk_shared_aligned_ampere_(
        nk_shapes_dynamic_shared_blackwell_, 1024);
    nk_u32_t const tensor_memory = nk_shapes_initialize_blackwell_(shared);
    nk_u32_t phase = 0;
    nk_f32_t chamfers[nk_shapes_flips_blackwell_k], hausdorffs[nk_shapes_flips_blackwell_k];
    for (nk_size_t pair = blockIdx.x; pair < schedule.rows * schedule.columns; pair += gridDim.x) {
        nk_size_t const row = pair / schedule.columns, column = pair % schedule.columns;
        int const clouds = nk_shapes_flips_f16_blackwell_(shared, tensor_memory, &phase, schedule.a_packed[row],
                                                          schedule.b_packed[column], chamfers, hausdorffs);
        if (threadIdx.x == 0) nk_shapes_write_best_flip_blackwell_(&schedule, row, column, chamfers, clouds);
    }
    __syncthreads();
    if (threadIdx.x < 32) nk_tmem_dealloc_blackwell_(tensor_memory, nk_shapes_tensor_columns_blackwell_k);
}

static __global__ void __launch_bounds__(nk_shapes_threads_blackwell_k)
    nk_hausdorffs_packed_f16_blackwell_kernel_(nk_shapes_schedule_blackwell_t const schedule) {
    extern __shared__ unsigned char nk_shapes_dynamic_shared_blackwell_[];
    nk_shapes_shared_blackwell_t *shared = (nk_shapes_shared_blackwell_t *)nk_shared_aligned_ampere_(
        nk_shapes_dynamic_shared_blackwell_, 1024);
    nk_u32_t const tensor_memory = nk_shapes_initialize_blackwell_(shared);
    nk_u32_t phase = 0;
    nk_f32_t chamfers[nk_shapes_flips_blackwell_k], hausdorffs[nk_shapes_flips_blackwell_k];
    for (nk_size_t pair = blockIdx.x; pair < schedule.rows * schedule.columns; pair += gridDim.x) {
        nk_size_t const row = pair / schedule.columns, column = pair % schedule.columns;
        int const clouds = nk_shapes_flips_f16_blackwell_(shared, tensor_memory, &phase, schedule.a_packed[row],
                                                          schedule.b_packed[column], chamfers, hausdorffs);
        if (threadIdx.x == 0) nk_shapes_write_best_flip_blackwell_(&schedule, row, column, hausdorffs, clouds);
    }
    __syncthreads();
    if (threadIdx.x < 32) nk_tmem_dealloc_blackwell_(tensor_memory, nk_shapes_tensor_columns_blackwell_k);
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_blackwell(void const *const *a_packed, nk_size_t rows,
                                                         void const *const *b_packed, nk_size_t columns,
                                                         nk_f32_t *results, nk_size_t results_stride,
                                                         nk_f32_t *rotations, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride | (nk_size_t)rotations) & (sizeof(nk_f32_t) - 1)) return nk_misaligned_k;
    nk_shapes_schedule_blackwell_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    nk_size_t const shared_bytes = sizeof(nk_shapes_shared_blackwell_t) + 1024;
    return nk_launch_resident_cuda_((void const *)nk_chamfers_packed_f16_blackwell_kernel_,
                                    nk_shapes_threads_blackwell_k, shared_bytes, shared_bytes, rows * columns,
                                    &schedule, stream);
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_blackwell(void const *const *a_packed, nk_size_t rows,
                                                           void const *const *b_packed, nk_size_t columns,
                                                           nk_f32_t *results, nk_size_t results_stride,
                                                           nk_f32_t *rotations, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride | (nk_size_t)rotations) & (sizeof(nk_f32_t) - 1)) return nk_misaligned_k;
    nk_shapes_schedule_blackwell_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    nk_size_t const shared_bytes = sizeof(nk_shapes_shared_blackwell_t) + 1024;
    return nk_launch_resident_cuda_((void const *)nk_hausdorffs_packed_f16_blackwell_kernel_,
                                    nk_shapes_threads_blackwell_k, shared_bytes, shared_bytes, rows * columns,
                                    &schedule, stream);
}

#pragma endregion Chamfer and Hausdorff
#endif // NUMKONG_TARGET_BLACKWELL

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_BLACKWELL_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_SHAPES_BLACKWELL_CUH
