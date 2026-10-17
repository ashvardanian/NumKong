/**
 *  @file include/numkong/shapes/simt.cuh
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Exact optimal transport between 3D point clouds on the SIMT cores of CUDA and ROCm.
 *
 *  @sa include/numkong/shapes.h
 *  @sa include/numkong/shapes/serial.h
 *
 *  The serial network simplex solves the transport between uniform weights as integer masses, each
 *  of n row points supplying m units and each of m column points taking n. Here one block, a team,
 *  solves the same linear program by successive shortest paths with potentials: Dijkstra over the
 *  reduced costs closes one column per step, every thread scanning its columns a team apart, and a
 *  closed column without demand passes its distance on to the rows shipping into it, as those arcs
 *  are tight. Each path ships the least of the source's supply, the sink's demand and the flows it
 *  runs back along. Arc costs come from the coordinates on the fly, the potentials stay in F64 for
 *  both input types like the serial duals, and the positive arcs live in one list per column.
 *
 *  The first solve of a flip starts from column and row reductions, shipping along every arc they
 *  leave tight. Each later round walks potentials out of the last plan that leave it tight under
 *  the new rotation, and returns only the rows a cheaper column now undercuts.
 *
 *  Equal clouds make assignments, where those paths stay short, but unequal ones make staircases
 *  of small flows, where each path ships a few units over dozens of steps. Those run the serial
 *  network simplex instead, from its northwest corner and warm across rounds, with every block of
 *  arcs priced across the team and each pivot on one thread. Team reductions run through shared
 *  memory, and the 3 × 3 SVD repeats the serial McAdams iterations operation by operation.
 */
#ifndef NUMKONG_SHAPES_SIMT_CUH
#define NUMKONG_SHAPES_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/shapes/serial.h" // `nk_shapes_packed_header_f64_t`, `nk_shapes_alignment_k_`
#include "numkong/cast/simt.cuh"   // `nk_f16_to_f32_simt_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Serial Arithmetic

/** Reciprocal square root as @c nk_rsqrt_f64_serial_ seeds and refines it. */
NUMKONG_DEVICE nk_f64_t nk_rsqrt_f64_simt_(nk_f64_t number) {
    nk_u64_t bits = (nk_u64_t)__double_as_longlong(number);
    nk_u64_t const result_mask = (nk_u64_t)0 - (bits != 0x7FF0000000000000ULL);
    int const subnormal = bits - 1 < 0x000FFFFFFFFFFFFFULL;
    number = (number == NUMKONG_F64_INF ? 1.0 : number) * (subnormal ? 0x1p54 : 1.0);
    bits = 0x5FE6EB50C7B537A9ULL - ((nk_u64_t)__double_as_longlong(number) >> 1);
    nk_f64_t y = __longlong_as_double((long long)bits), half_number_y_squared;
    half_number_y_squared = 0.5 * number * y * y;
    y = y * (1.5 - half_number_y_squared);
    half_number_y_squared = 0.5 * number * y * y;
    y = y * (1.5 - half_number_y_squared);
    half_number_y_squared = 0.5 * number * y * y;
    y = y * (1.5 - half_number_y_squared);
    half_number_y_squared = 0.5 * number * y * y;
    y = y * (1.5 - half_number_y_squared);
    bits = (nk_u64_t)__double_as_longlong(subnormal ? 0x1p27 : 1.0) & result_mask;
    return y * __longlong_as_double((long long)bits);
}

NUMKONG_DEVICE nk_f64_t nk_sqrt_f64_simt_(nk_f64_t number) {
    nk_f64_t const positive = number > 0 ? number : 0;
    nk_f64_t const finite = number > 0 && number < NUMKONG_F64_INF ? number : 1.0;
    return positive * nk_rsqrt_f64_simt_(finite);
}

/** Reciprocal square root as @c nk_rsqrt_f32_serial_ seeds and refines it. */
NUMKONG_DEVICE nk_f32_t nk_rsqrt_f32_simt_(nk_f32_t number) {
    nk_u32_t bits = (nk_u32_t)__float_as_uint(number);
    nk_u32_t const result_mask = (nk_u32_t)0 - (bits != 0x7F800000u);
    int const subnormal = bits - 1 < 0x007FFFFFu;
    number = (number == NUMKONG_F32_INF ? 1.0f : number) * (subnormal ? 0x1p24f : 1.0f);
    bits = 0x5F375A86u - (__float_as_uint(number) >> 1);
    nk_f32_t y = __uint_as_float(bits), half_number_y_squared;
    half_number_y_squared = 0.5f * number * y * y;
    y = y * (1.5f - half_number_y_squared);
    half_number_y_squared = 0.5f * number * y * y;
    y = y * (1.5f - half_number_y_squared);
    half_number_y_squared = 0.5f * number * y * y;
    y = y * (1.5f - half_number_y_squared);
    bits = __float_as_uint(subnormal ? 0x1p12f : 1.0f) & result_mask;
    return y * __uint_as_float(bits);
}

NUMKONG_DEVICE nk_f32_t nk_sqrt_f32_simt_(nk_f32_t number) {
    nk_f32_t const positive = number > 0 ? number : 0;
    nk_f32_t const finite = number > 0 && number < NUMKONG_F32_INF ? number : 1.0f;
    return positive * nk_rsqrt_f32_simt_(finite);
}

/** Adds @p a × @p b into the Dot2 pair, as @c nk_dot2_f64_serial_ does. */
NUMKONG_DEVICE void nk_dot2_f64_simt_(nk_f64_t *sum, nk_f64_t *compensation, nk_f64_t a, nk_f64_t b) {
    nk_f64_t const product = __dmul_rn(a, b);
    nk_f64_t const product_error = __fma_rn(a, b, -product);
    nk_f64_t const running_sum = *sum + product;
    nk_f64_t const recovered_addend = running_sum - *sum;
    nk_f64_t const sum_error = (*sum - (running_sum - recovered_addend)) + (product - recovered_addend);
    *sum = running_sum;
    *compensation += sum_error + product_error;
}

NUMKONG_DEVICE nk_f64_t nk_sum_three_products_f64_simt_(nk_f64_t left_0, nk_f64_t right_0, nk_f64_t left_1,
                                                        nk_f64_t right_1, nk_f64_t left_2, nk_f64_t right_2) {
    nk_f64_t sum = 0, compensation = 0;
    nk_dot2_f64_simt_(&sum, &compensation, left_0, right_0);
    nk_dot2_f64_simt_(&sum, &compensation, left_1, right_1);
    nk_dot2_f64_simt_(&sum, &compensation, left_2, right_2);
    return sum + compensation;
}

NUMKONG_DEVICE nk_f32_t nk_sum_three_products_f32_simt_(nk_f32_t left_0, nk_f32_t right_0, nk_f32_t left_1,
                                                        nk_f32_t right_1, nk_f32_t left_2, nk_f32_t right_2) {
    return left_0 * right_0 + left_1 * right_1 + left_2 * right_2;
}

NUMKONG_DEVICE nk_f64_t nk_sum_two_products_f64_simt_(nk_f64_t left_0, nk_f64_t right_0, nk_f64_t left_1,
                                                      nk_f64_t right_1) {
    return __dmul_rn(left_0, right_0) + __dmul_rn(left_1, right_1);
}

NUMKONG_DEVICE nk_f32_t nk_sum_two_products_f32_simt_(nk_f32_t left_0, nk_f32_t right_0, nk_f32_t left_1,
                                                      nk_f32_t right_1) {
    return __fmul_rn(left_0, right_0) + __fmul_rn(left_1, right_1);
}

#pragma endregion Serial Arithmetic

#pragma region SVD F64

NUMKONG_DEVICE void nk_approximate_givens_quaternion_f64_simt_(nk_f64_t a11, nk_f64_t a12, nk_f64_t a22,
                                                               nk_f64_t *cos_half, nk_f64_t *sin_half) {
    *cos_half = 2 * (a11 - a22), *sin_half = a12;
    int const use_givens = 5.828427124746190 * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);
    nk_f64_t const w = nk_rsqrt_f64_simt_(nk_sum_two_products_f64_simt_(*cos_half, *cos_half, *sin_half, *sin_half));
    *cos_half = use_givens ? w * (*cos_half) : 0.9238795325112867;
    *sin_half = use_givens ? w * (*sin_half) : 0.3826834323650898;
}

NUMKONG_DEVICE void nk_jacobi_conjugation_f64_simt_(int index_x, int index_y, int index_z, nk_f64_t *s11, nk_f64_t *s21,
                                                    nk_f64_t *s22, nk_f64_t *s31, nk_f64_t *s32, nk_f64_t *s33,
                                                    nk_f64_t *quaternion) {
    nk_f64_t cos_half, sin_half;
    nk_approximate_givens_quaternion_f64_simt_(*s11, *s21, *s22, &cos_half, &sin_half);
    nk_f64_t const scale = nk_sum_two_products_f64_simt_(cos_half, cos_half, sin_half, sin_half);
    nk_f64_t const cos_theta = nk_sum_two_products_f64_simt_(cos_half, cos_half, -sin_half, sin_half) / scale;
    nk_f64_t const sin_theta = (2 * sin_half * cos_half) / scale;
    nk_f64_t s11_old = *s11, s21_old = *s21, s22_old = *s22;
    nk_f64_t s31_old = *s31, s32_old = *s32, s33_old = *s33;
    nk_f64_t const left_11 = nk_sum_two_products_f64_simt_(cos_theta, s11_old, sin_theta, s21_old);
    nk_f64_t const left_12 = nk_sum_two_products_f64_simt_(cos_theta, s21_old, sin_theta, s22_old);
    nk_f64_t const left_21 = nk_sum_two_products_f64_simt_(-sin_theta, s11_old, cos_theta, s21_old);
    nk_f64_t const left_22 = nk_sum_two_products_f64_simt_(-sin_theta, s21_old, cos_theta, s22_old);
    *s11 = nk_sum_two_products_f64_simt_(cos_theta, left_11, sin_theta, left_12);
    *s21 = nk_sum_two_products_f64_simt_(cos_theta, left_21, sin_theta, left_22);
    *s22 = nk_sum_two_products_f64_simt_(-sin_theta, left_21, cos_theta, left_22);
    *s31 = nk_sum_two_products_f64_simt_(cos_theta, s31_old, sin_theta, s32_old);
    *s32 = nk_sum_two_products_f64_simt_(-sin_theta, s31_old, cos_theta, s32_old);
    *s33 = s33_old;
    nk_f64_t quaternion_sines[3];
    quaternion_sines[0] = quaternion[0] * sin_half;
    quaternion_sines[1] = quaternion[1] * sin_half;
    quaternion_sines[2] = quaternion[2] * sin_half;
    sin_half *= quaternion[3];
    quaternion[0] *= cos_half, quaternion[1] *= cos_half;
    quaternion[2] *= cos_half, quaternion[3] *= cos_half;
    quaternion[index_z] += sin_half, quaternion[3] -= quaternion_sines[index_z];
    quaternion[index_x] += quaternion_sines[index_y], quaternion[index_y] -= quaternion_sines[index_x];
    s11_old = *s22, s21_old = *s32, s22_old = *s33;
    s31_old = *s21, s32_old = *s31, s33_old = *s11;
    *s11 = s11_old, *s21 = s21_old, *s22 = s22_old;
    *s31 = s31_old, *s32 = s32_old, *s33 = s33_old;
}

NUMKONG_DEVICE void nk_quaternion_to_mat3x3_f64_simt_(nk_f64_t const *quaternion, nk_f64_t *matrix) {
    nk_f64_t const w = quaternion[3], x = quaternion[0], y = quaternion[1], z = quaternion[2];
    nk_f64_t const xx = x * x, yy = y * y, zz = z * z, xz = x * z, xy = x * y, yz = y * z;
    nk_f64_t const wx = w * x, wy = w * y, wz = w * z;
    matrix[0] = 1 - 2 * (yy + zz), matrix[1] = 2 * (xy - wz), matrix[2] = 2 * (xz + wy);
    matrix[3] = 2 * (xy + wz), matrix[4] = 1 - 2 * (xx + zz), matrix[5] = 2 * (yz - wx);
    matrix[6] = 2 * (xz - wy), matrix[7] = 2 * (yz + wx), matrix[8] = 1 - 2 * (xx + yy);
}

NUMKONG_DEVICE void nk_jacobi_eigenanalysis_f64_simt_(nk_f64_t *s11, nk_f64_t *s21, nk_f64_t *s22, nk_f64_t *s31,
                                                      nk_f64_t *s32, nk_f64_t *s33, nk_f64_t *quaternion) {
    quaternion[0] = 0, quaternion[1] = 0, quaternion[2] = 0, quaternion[3] = 1;
    for (int iteration = 0; iteration < 16; iteration++) {
        nk_jacobi_conjugation_f64_simt_(0, 1, 2, s11, s21, s22, s31, s32, s33, quaternion);
        nk_jacobi_conjugation_f64_simt_(1, 2, 0, s11, s21, s22, s31, s32, s33, quaternion);
        nk_jacobi_conjugation_f64_simt_(2, 0, 1, s11, s21, s22, s31, s32, s33, quaternion);
    }
    nk_f64_t const norm = nk_rsqrt_f64_simt_(
        nk_sum_two_products_f64_simt_(quaternion[0], quaternion[0], quaternion[1], quaternion[1]) +
        nk_sum_two_products_f64_simt_(quaternion[2], quaternion[2], quaternion[3], quaternion[3]));
    quaternion[0] *= norm, quaternion[1] *= norm, quaternion[2] *= norm, quaternion[3] *= norm;
}

NUMKONG_DEVICE void nk_qr_givens_quaternion_f64_simt_(nk_f64_t a1, nk_f64_t a2, nk_f64_t *cos_half,
                                                      nk_f64_t *sin_half) {
    nk_f64_t const epsilon = 1e-12;
    nk_f64_t const squares = a1 * a1 + a2 * a2;
    nk_f64_t rho = squares * nk_rsqrt_f64_simt_(squares);
    rho = squares > epsilon ? rho : 0;
    *sin_half = rho > epsilon ? a2 : 0;
    nk_f64_t const absolute_a1 = a1 < 0 ? -a1 : a1;
    *cos_half = absolute_a1 + (rho > epsilon ? rho : epsilon);
    if (a1 < 0) {
        nk_f64_t const swapped = *sin_half;
        *sin_half = *cos_half, *cos_half = swapped;
    }
    nk_f64_t const w = nk_rsqrt_f64_simt_((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));
    *cos_half *= w, *sin_half *= w;
}

NUMKONG_DEVICE void nk_negating_swap_f64_simt_(int condition, nk_f64_t *x, nk_f64_t *y) {
    nk_f64_t const negated = -*x;
    *x = condition ? *y : *x;
    *y = condition ? negated : *y;
}

NUMKONG_DEVICE void nk_sort_singular_values_f64_simt_(nk_f64_t *b, nk_f64_t *v) {
    nk_f64_t norm_0 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];
    nk_f64_t norm_1 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];
    nk_f64_t norm_2 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];
    int swap = norm_0 < norm_1;
    for (int row = 0; row != 9; row += 3)
        nk_negating_swap_f64_simt_(swap, &b[row], &b[row + 1]), nk_negating_swap_f64_simt_(swap, &v[row], &v[row + 1]);
    if (swap) {
        nk_f64_t const kept = norm_0;
        norm_0 = norm_1, norm_1 = kept;
    }
    swap = norm_0 < norm_2;
    for (int row = 0; row != 9; row += 3)
        nk_negating_swap_f64_simt_(swap, &b[row], &b[row + 2]), nk_negating_swap_f64_simt_(swap, &v[row], &v[row + 2]);
    if (swap) {
        nk_f64_t const kept = norm_0;
        norm_0 = norm_2, norm_2 = kept;
    }
    swap = norm_1 < norm_2;
    for (int row = 0; row != 9; row += 3)
        nk_negating_swap_f64_simt_(swap, &b[row + 1], &b[row + 2]),
            nk_negating_swap_f64_simt_(swap, &v[row + 1], &v[row + 2]);
}

NUMKONG_DEVICE void nk_qr_orthogonal_factor_f64_simt_(nk_f64_t const *input, nk_f64_t *q) {
    nk_f64_t cos_half_1, sin_half_1, cos_half_2, sin_half_2, cos_half_3, sin_half_3, cos_theta, sin_theta;
    nk_qr_givens_quaternion_f64_simt_(input[0], input[3], &cos_half_1, &sin_half_1);
    cos_theta = 1 - 2 * sin_half_1 * sin_half_1;
    sin_theta = 2 * cos_half_1 * sin_half_1;
    nk_f64_t const rotated_1 = cos_theta * input[1] + sin_theta * input[4];
    nk_f64_t const rotated_4 = -sin_theta * input[1] + cos_theta * input[4];
    nk_f64_t const rotated_7 = input[7];
    nk_qr_givens_quaternion_f64_simt_(input[0] * cos_theta + sin_theta * input[3], input[6], &cos_half_2, &sin_half_2);
    cos_theta = 1 - 2 * sin_half_2 * sin_half_2;
    sin_theta = 2 * cos_half_2 * sin_half_2;
    nk_f64_t const remaining_7 = -sin_theta * rotated_1 + cos_theta * rotated_7;
    nk_qr_givens_quaternion_f64_simt_(rotated_4, remaining_7, &cos_half_3, &sin_half_3);
    nk_f64_t const sin_half_1_squared = sin_half_1 * sin_half_1;
    nk_f64_t const sin_half_2_squared = sin_half_2 * sin_half_2;
    nk_f64_t const sin_half_3_squared = sin_half_3 * sin_half_3;
    q[0] = (-1 + 2 * sin_half_1_squared) * (-1 + 2 * sin_half_2_squared);
    q[1] = 4 * cos_half_2 * cos_half_3 * (-1 + 2 * sin_half_1_squared) * sin_half_2 * sin_half_3 +
           2 * cos_half_1 * sin_half_1 * (-1 + 2 * sin_half_3_squared);
    q[2] = 4 * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -
           2 * cos_half_2 * (-1 + 2 * sin_half_1_squared) * sin_half_2 * (-1 + 2 * sin_half_3_squared);
    q[3] = 2 * cos_half_1 * sin_half_1 * (1 - 2 * sin_half_2_squared);
    q[4] = -8 * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +
           (-1 + 2 * sin_half_1_squared) * (-1 + 2 * sin_half_3_squared);
    q[5] = -2 * cos_half_3 * sin_half_3 + 4 * sin_half_1 *
                                              (cos_half_3 * sin_half_1 * sin_half_3 +
                                               cos_half_1 * cos_half_2 * sin_half_2 * (-1 + 2 * sin_half_3_squared));
    q[6] = 2 * cos_half_2 * sin_half_2;
    q[7] = 2 * cos_half_3 * (1 - 2 * sin_half_2_squared) * sin_half_3;
    q[8] = (-1 + 2 * sin_half_2_squared) * (-1 + 2 * sin_half_3_squared);
}

/** The McAdams SVD of @c nk_svd3x3_f64_serial_ : left factor, singular values on the diagonal
 *  positions, and right factor of the row-major @p a . */
NUMKONG_DEVICE void nk_svd3x3_f64_simt_(nk_f64_t const *a, nk_f64_t *svd_left, nk_f64_t *svd_diagonal,
                                        nk_f64_t *svd_right) {
    nk_f64_t ata[9], quaternion[4], product[9];
    for (int row = 0; row != 3; ++row)
        for (int column = row; column != 3; ++column)
            ata[row * 3 + column] = nk_sum_three_products_f64_simt_(a[row], a[column], a[3 + row], a[3 + column],
                                                                    a[6 + row], a[6 + column]);
    ata[3] = ata[1], ata[6] = ata[2], ata[7] = ata[5];
    nk_jacobi_eigenanalysis_f64_simt_(&ata[0], &ata[1], &ata[4], &ata[2], &ata[5], &ata[8], quaternion);
    nk_quaternion_to_mat3x3_f64_simt_(quaternion, svd_right);
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            product[row * 3 + column] = nk_sum_three_products_f64_simt_(a[row * 3], svd_right[column], a[row * 3 + 1],
                                                                        svd_right[3 + column], a[row * 3 + 2],
                                                                        svd_right[6 + column]);
    nk_sort_singular_values_f64_simt_(product, svd_right);
    nk_f64_t const squared_0 = nk_sum_three_products_f64_simt_(product[0], product[0], product[3], product[3],
                                                               product[6], product[6]);
    nk_f64_t const squared_1 = nk_sum_three_products_f64_simt_(product[1], product[1], product[4], product[4],
                                                               product[7], product[7]);
    nk_f64_t const squared_2 = nk_sum_three_products_f64_simt_(product[2], product[2], product[5], product[5],
                                                               product[8], product[8]);
    nk_qr_orthogonal_factor_f64_simt_(product, svd_left);
    svd_diagonal[0] = nk_sqrt_f64_simt_(squared_0);
    svd_diagonal[4] = nk_sqrt_f64_simt_(squared_1);
    svd_diagonal[8] = nk_sqrt_f64_simt_(squared_2);
}

NUMKONG_DEVICE nk_f64_t nk_det3x3_f64_simt_(nk_f64_t const *m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/** V × Uᵀ from the SVD factors, as @c nk_rotation_from_svd_f64_serial_ sums it. */
NUMKONG_DEVICE void nk_rotation_from_svd_f64_simt_(nk_f64_t const *svd_left, nk_f64_t const *svd_right,
                                                   nk_f64_t *rotation) {
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            rotation[row * 3 + column] = nk_sum_three_products_f64_simt_(
                svd_right[row * 3], svd_left[column * 3], svd_right[row * 3 + 1], svd_left[column * 3 + 1],
                svd_right[row * 3 + 2], svd_left[column * 3 + 2]);
}

#pragma endregion SVD F64

#pragma region SVD F32

NUMKONG_DEVICE void nk_approximate_givens_quaternion_f32_simt_(nk_f32_t a11, nk_f32_t a12, nk_f32_t a22,
                                                               nk_f32_t *cos_half, nk_f32_t *sin_half) {
    *cos_half = 2 * (a11 - a22), *sin_half = a12;
    int const use_givens = 5.828427124f * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);
    nk_f32_t const w = nk_rsqrt_f32_simt_(nk_sum_two_products_f32_simt_(*cos_half, *cos_half, *sin_half, *sin_half));
    *cos_half = use_givens ? w * (*cos_half) : 0.923879532f;
    *sin_half = use_givens ? w * (*sin_half) : 0.3826834323f;
}

NUMKONG_DEVICE void nk_jacobi_conjugation_f32_simt_(int index_x, int index_y, int index_z, nk_f32_t *s11, nk_f32_t *s21,
                                                    nk_f32_t *s22, nk_f32_t *s31, nk_f32_t *s32, nk_f32_t *s33,
                                                    nk_f32_t *quaternion) {
    nk_f32_t cos_half, sin_half;
    nk_approximate_givens_quaternion_f32_simt_(*s11, *s21, *s22, &cos_half, &sin_half);
    nk_f32_t const scale = nk_sum_two_products_f32_simt_(cos_half, cos_half, sin_half, sin_half);
    nk_f32_t const cos_theta = nk_sum_two_products_f32_simt_(cos_half, cos_half, -sin_half, sin_half) / scale;
    nk_f32_t const sin_theta = (2 * sin_half * cos_half) / scale;
    nk_f32_t s11_old = *s11, s21_old = *s21, s22_old = *s22;
    nk_f32_t s31_old = *s31, s32_old = *s32, s33_old = *s33;
    nk_f32_t const left_11 = nk_sum_two_products_f32_simt_(cos_theta, s11_old, sin_theta, s21_old);
    nk_f32_t const left_12 = nk_sum_two_products_f32_simt_(cos_theta, s21_old, sin_theta, s22_old);
    nk_f32_t const left_21 = nk_sum_two_products_f32_simt_(-sin_theta, s11_old, cos_theta, s21_old);
    nk_f32_t const left_22 = nk_sum_two_products_f32_simt_(-sin_theta, s21_old, cos_theta, s22_old);
    *s11 = nk_sum_two_products_f32_simt_(cos_theta, left_11, sin_theta, left_12);
    *s21 = nk_sum_two_products_f32_simt_(cos_theta, left_21, sin_theta, left_22);
    *s22 = nk_sum_two_products_f32_simt_(-sin_theta, left_21, cos_theta, left_22);
    *s31 = nk_sum_two_products_f32_simt_(cos_theta, s31_old, sin_theta, s32_old);
    *s32 = nk_sum_two_products_f32_simt_(-sin_theta, s31_old, cos_theta, s32_old);
    *s33 = s33_old;
    nk_f32_t quaternion_sines[3];
    quaternion_sines[0] = quaternion[0] * sin_half;
    quaternion_sines[1] = quaternion[1] * sin_half;
    quaternion_sines[2] = quaternion[2] * sin_half;
    sin_half *= quaternion[3];
    quaternion[0] *= cos_half, quaternion[1] *= cos_half;
    quaternion[2] *= cos_half, quaternion[3] *= cos_half;
    quaternion[index_z] += sin_half, quaternion[3] -= quaternion_sines[index_z];
    quaternion[index_x] += quaternion_sines[index_y], quaternion[index_y] -= quaternion_sines[index_x];
    s11_old = *s22, s21_old = *s32, s22_old = *s33;
    s31_old = *s21, s32_old = *s31, s33_old = *s11;
    *s11 = s11_old, *s21 = s21_old, *s22 = s22_old;
    *s31 = s31_old, *s32 = s32_old, *s33 = s33_old;
}

NUMKONG_DEVICE void nk_quaternion_to_mat3x3_f32_simt_(nk_f32_t const *quaternion, nk_f32_t *matrix) {
    nk_f32_t const w = quaternion[3], x = quaternion[0], y = quaternion[1], z = quaternion[2];
    nk_f32_t const xx = x * x, yy = y * y, zz = z * z, xz = x * z, xy = x * y, yz = y * z;
    nk_f32_t const wx = w * x, wy = w * y, wz = w * z;
    matrix[0] = 1 - 2 * (yy + zz), matrix[1] = 2 * (xy - wz), matrix[2] = 2 * (xz + wy);
    matrix[3] = 2 * (xy + wz), matrix[4] = 1 - 2 * (xx + zz), matrix[5] = 2 * (yz - wx);
    matrix[6] = 2 * (xz - wy), matrix[7] = 2 * (yz + wx), matrix[8] = 1 - 2 * (xx + yy);
}

NUMKONG_DEVICE void nk_jacobi_eigenanalysis_f32_simt_(nk_f32_t *s11, nk_f32_t *s21, nk_f32_t *s22, nk_f32_t *s31,
                                                      nk_f32_t *s32, nk_f32_t *s33, nk_f32_t *quaternion) {
    quaternion[0] = 0, quaternion[1] = 0, quaternion[2] = 0, quaternion[3] = 1;
    for (int iteration = 0; iteration < 16; iteration++) {
        nk_jacobi_conjugation_f32_simt_(0, 1, 2, s11, s21, s22, s31, s32, s33, quaternion);
        nk_jacobi_conjugation_f32_simt_(1, 2, 0, s11, s21, s22, s31, s32, s33, quaternion);
        nk_jacobi_conjugation_f32_simt_(2, 0, 1, s11, s21, s22, s31, s32, s33, quaternion);
    }
    nk_f32_t const norm = nk_rsqrt_f32_simt_(
        nk_sum_two_products_f32_simt_(quaternion[0], quaternion[0], quaternion[1], quaternion[1]) +
        nk_sum_two_products_f32_simt_(quaternion[2], quaternion[2], quaternion[3], quaternion[3]));
    quaternion[0] *= norm, quaternion[1] *= norm, quaternion[2] *= norm, quaternion[3] *= norm;
}

NUMKONG_DEVICE void nk_qr_givens_quaternion_f32_simt_(nk_f32_t a1, nk_f32_t a2, nk_f32_t *cos_half,
                                                      nk_f32_t *sin_half) {
    nk_f32_t const epsilon = 1e-6f;
    nk_f32_t const squares = a1 * a1 + a2 * a2;
    nk_f32_t rho = squares * nk_rsqrt_f32_simt_(squares);
    rho = squares > epsilon ? rho : 0;
    *sin_half = rho > epsilon ? a2 : 0;
    nk_f32_t const absolute_a1 = a1 < 0 ? -a1 : a1;
    *cos_half = absolute_a1 + (rho > epsilon ? rho : epsilon);
    if (a1 < 0) {
        nk_f32_t const swapped = *sin_half;
        *sin_half = *cos_half, *cos_half = swapped;
    }
    nk_f32_t const w = nk_rsqrt_f32_simt_((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));
    *cos_half *= w, *sin_half *= w;
}

NUMKONG_DEVICE void nk_negating_swap_f32_simt_(int condition, nk_f32_t *x, nk_f32_t *y) {
    nk_f32_t const negated = -*x;
    *x = condition ? *y : *x;
    *y = condition ? negated : *y;
}

NUMKONG_DEVICE void nk_sort_singular_values_f32_simt_(nk_f32_t *b, nk_f32_t *v) {
    nk_f32_t norm_0 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];
    nk_f32_t norm_1 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];
    nk_f32_t norm_2 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];
    int swap = norm_0 < norm_1;
    for (int row = 0; row != 9; row += 3)
        nk_negating_swap_f32_simt_(swap, &b[row], &b[row + 1]), nk_negating_swap_f32_simt_(swap, &v[row], &v[row + 1]);
    if (swap) {
        nk_f32_t const kept = norm_0;
        norm_0 = norm_1, norm_1 = kept;
    }
    swap = norm_0 < norm_2;
    for (int row = 0; row != 9; row += 3)
        nk_negating_swap_f32_simt_(swap, &b[row], &b[row + 2]), nk_negating_swap_f32_simt_(swap, &v[row], &v[row + 2]);
    if (swap) {
        nk_f32_t const kept = norm_0;
        norm_0 = norm_2, norm_2 = kept;
    }
    swap = norm_1 < norm_2;
    for (int row = 0; row != 9; row += 3)
        nk_negating_swap_f32_simt_(swap, &b[row + 1], &b[row + 2]),
            nk_negating_swap_f32_simt_(swap, &v[row + 1], &v[row + 2]);
}

NUMKONG_DEVICE void nk_qr_orthogonal_factor_f32_simt_(nk_f32_t const *input, nk_f32_t *q) {
    nk_f32_t cos_half_1, sin_half_1, cos_half_2, sin_half_2, cos_half_3, sin_half_3, cos_theta, sin_theta;
    nk_qr_givens_quaternion_f32_simt_(input[0], input[3], &cos_half_1, &sin_half_1);
    cos_theta = 1 - 2 * sin_half_1 * sin_half_1;
    sin_theta = 2 * cos_half_1 * sin_half_1;
    nk_f32_t const rotated_1 = cos_theta * input[1] + sin_theta * input[4];
    nk_f32_t const rotated_4 = -sin_theta * input[1] + cos_theta * input[4];
    nk_f32_t const rotated_7 = input[7];
    nk_qr_givens_quaternion_f32_simt_(input[0] * cos_theta + sin_theta * input[3], input[6], &cos_half_2, &sin_half_2);
    cos_theta = 1 - 2 * sin_half_2 * sin_half_2;
    sin_theta = 2 * cos_half_2 * sin_half_2;
    nk_f32_t const remaining_7 = -sin_theta * rotated_1 + cos_theta * rotated_7;
    nk_qr_givens_quaternion_f32_simt_(rotated_4, remaining_7, &cos_half_3, &sin_half_3);
    nk_f32_t const sin_half_1_squared = sin_half_1 * sin_half_1;
    nk_f32_t const sin_half_2_squared = sin_half_2 * sin_half_2;
    nk_f32_t const sin_half_3_squared = sin_half_3 * sin_half_3;
    q[0] = (-1 + 2 * sin_half_1_squared) * (-1 + 2 * sin_half_2_squared);
    q[1] = 4 * cos_half_2 * cos_half_3 * (-1 + 2 * sin_half_1_squared) * sin_half_2 * sin_half_3 +
           2 * cos_half_1 * sin_half_1 * (-1 + 2 * sin_half_3_squared);
    q[2] = 4 * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -
           2 * cos_half_2 * (-1 + 2 * sin_half_1_squared) * sin_half_2 * (-1 + 2 * sin_half_3_squared);
    q[3] = 2 * cos_half_1 * sin_half_1 * (1 - 2 * sin_half_2_squared);
    q[4] = -8 * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +
           (-1 + 2 * sin_half_1_squared) * (-1 + 2 * sin_half_3_squared);
    q[5] = -2 * cos_half_3 * sin_half_3 + 4 * sin_half_1 *
                                              (cos_half_3 * sin_half_1 * sin_half_3 +
                                               cos_half_1 * cos_half_2 * sin_half_2 * (-1 + 2 * sin_half_3_squared));
    q[6] = 2 * cos_half_2 * sin_half_2;
    q[7] = 2 * cos_half_3 * (1 - 2 * sin_half_2_squared) * sin_half_3;
    q[8] = (-1 + 2 * sin_half_2_squared) * (-1 + 2 * sin_half_3_squared);
}

/** The McAdams SVD of @c nk_svd3x3_f32_serial_ : left factor, singular values on the diagonal
 *  positions, and right factor of the row-major @p a . */
NUMKONG_DEVICE void nk_svd3x3_f32_simt_(nk_f32_t const *a, nk_f32_t *svd_left, nk_f32_t *svd_diagonal,
                                        nk_f32_t *svd_right) {
    nk_f32_t ata[9], quaternion[4], product[9];
    for (int row = 0; row != 3; ++row)
        for (int column = row; column != 3; ++column)
            ata[row * 3 + column] = nk_sum_three_products_f32_simt_(a[row], a[column], a[3 + row], a[3 + column],
                                                                    a[6 + row], a[6 + column]);
    ata[3] = ata[1], ata[6] = ata[2], ata[7] = ata[5];
    nk_jacobi_eigenanalysis_f32_simt_(&ata[0], &ata[1], &ata[4], &ata[2], &ata[5], &ata[8], quaternion);
    nk_quaternion_to_mat3x3_f32_simt_(quaternion, svd_right);
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            product[row * 3 + column] = nk_sum_three_products_f32_simt_(a[row * 3], svd_right[column], a[row * 3 + 1],
                                                                        svd_right[3 + column], a[row * 3 + 2],
                                                                        svd_right[6 + column]);
    nk_sort_singular_values_f32_simt_(product, svd_right);
    nk_f32_t const squared_0 = product[0] * product[0] + product[3] * product[3] + product[6] * product[6];
    nk_f32_t const squared_1 = product[1] * product[1] + product[4] * product[4] + product[7] * product[7];
    nk_f32_t const squared_2 = product[2] * product[2] + product[5] * product[5] + product[8] * product[8];
    nk_qr_orthogonal_factor_f32_simt_(product, svd_left);
    svd_diagonal[0] = nk_sqrt_f32_simt_(squared_0);
    svd_diagonal[4] = nk_sqrt_f32_simt_(squared_1);
    svd_diagonal[8] = nk_sqrt_f32_simt_(squared_2);
}

NUMKONG_DEVICE nk_f32_t nk_det3x3_f32_simt_(nk_f32_t const *m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

NUMKONG_DEVICE void nk_rotation_from_svd_f32_simt_(nk_f32_t const *svd_left, nk_f32_t const *svd_right,
                                                   nk_f32_t *rotation) {
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            rotation[row * 3 + column] = nk_sum_three_products_f32_simt_(
                svd_right[row * 3], svd_left[column * 3], svd_right[row * 3 + 1], svd_left[column * 3 + 1],
                svd_right[row * 3 + 2], svd_left[column * 3 + 2]);
}

#pragma endregion SVD F32

#pragma region Transport Network

/** One team's transport between up to @c rows row points and @c columns column points: residual
 *  supplies and demands as integer masses, F64 potentials, the Dijkstra state, and the positive
 *  arcs, each in the list of its column. A plan never holds more than n + m arcs: the simplex keeps
 *  a spanning tree, and shortest paths run only between equal clouds, where every path ships a
 *  whole row and each row and column keeps one arc. The state lives in the block's shared memory
 *  when it fits, or in global memory for a team of one or more blocks. */
typedef struct {
    nk_f64_t *row_potentials;
    nk_f64_t *column_potentials;
    nk_f64_t *distances;
    nk_f64_t *team_values;
    nk_u32_t *row_supplies;
    nk_u32_t *row_reached_columns;
    nk_u32_t *row_reached_arcs;
    nk_u32_t *column_demands;
    nk_u32_t *column_predecessors;
    nk_u32_t *column_arcs;
    nk_u32_t *column_closed;
    nk_u32_t *arc_rows;
    nk_u32_t *arc_flows;
    nk_u32_t *arc_next;

    /** The plan as rows' lists, rebuilt by each warm start: every arc's column, and its next arc of
     *  the same row, from each row's first, and the order the start visits rows and columns in. */
    nk_u32_t *arc_columns;
    nk_u32_t *arc_row_next;
    nk_u32_t *row_arcs;
    nk_u32_t *visits;
    nk_u32_t *team_indices;

    /** The head of the free arcs. */
    nk_u32_t *free_arcs;

    /** The network simplex's spanning tree over row then column nodes, kept as the serial solver
     *  keeps it: each node's parent, the flow on the arc to it, its depth, its first child and its
     *  siblings, and each node's potential. */
    nk_u32_t *tree_parents, *tree_flows, *tree_depths, *tree_first_children, *tree_next_siblings,
        *tree_previous_siblings;
    nk_f64_t *tree_potentials;

    /** Where a team of several blocks meets in global memory: their arrival count and generation,
     *  then each block's argmins and sums for the others to fold. Null for a team of one block. */
    nk_u32_t *team_meeting;

    /** The serial tolerance of the pair's scale: no reduced cost above its negation enters. */
    nk_f64_t tolerance;
    nk_u32_t rows, columns, arcs;

    /** This thread's index in the team and the team's threads, a whole number of blocks. */
    unsigned thread, team;
} nk_shapes_network_simt_t;

enum {

    /** Doubles each thread keeps in the team scratch: an argmin's two halves, or a plan's cost and
     *  cross-covariance. */
    nk_shapes_team_values_simt_k = 10,

    /** The most blocks a team spans. */
    nk_shapes_team_blocks_simt_k = 16,

    /** Bytes where a team of several blocks meets: the barrier, then 2 argmins and 10 sums of
     *  doubles per block, then the argmins' indices. */
    nk_shapes_meeting_bytes_simt_k = 64 + (2 + nk_shapes_team_values_simt_k) * nk_shapes_team_blocks_simt_k * 8 +
                                     2 * nk_shapes_team_blocks_simt_k * 4,
};

/** Bytes of a block's team scratch in shared memory for @p threads threads. */
NUMKONG_CONSTEXPR nk_size_t nk_shapes_scratch_bytes_simt_(nk_size_t threads) {
    return nk_shapes_team_values_simt_k * threads * sizeof(nk_f64_t) + 2 * threads * sizeof(nk_u32_t);
}

/** Bytes of one team's network and its two point arrays of @p point_bytes per coordinate, for
 *  clouds of up to @p rows and @p columns points. The shortest-path arrays and the simplex tree
 *  share one region, as a pair runs only one of the two solvers. */
NUMKONG_INLINE nk_size_t nk_shapes_state_bytes_simt_(nk_size_t rows, nk_size_t columns, nk_size_t point_bytes) {
    nk_size_t const arcs = rows + columns;
    nk_size_t const common_bytes = nk_size_divide_round_up_(3 * (rows + columns) * point_bytes, sizeof(nk_f64_t)) *
                                       sizeof(nk_f64_t) +
                                   (columns + 3 * arcs + 1) * sizeof(nk_u32_t);
    nk_size_t const paths_bytes = (rows + 2 * columns) * sizeof(nk_f64_t) +
                                  (5 * rows + 4 * columns + 2 * arcs) * sizeof(nk_u32_t);
    nk_size_t const tree_bytes = (rows + columns) * (sizeof(nk_f64_t) + 6 * sizeof(nk_u32_t));
    return nk_size_round_up_to_multiple_(common_bytes, sizeof(nk_f64_t)) +
           (paths_bytes > tree_bytes ? paths_bytes : tree_bytes);
}

/** Bytes of shared memory for a block of @p threads holding the whole state of its clouds. */
NUMKONG_INLINE nk_size_t nk_shapes_team_bytes_simt_(nk_size_t rows, nk_size_t columns, nk_size_t threads,
                                                    nk_size_t point_bytes) {
    return nk_shapes_scratch_bytes_simt_(threads) + nk_shapes_state_bytes_simt_(rows, columns, point_bytes);
}

/** Carves the block's team scratch from @p scratch and the network of @p rows and @p columns
 *  points from @p state , after the two point arrays of @p point_bytes per coordinate, sized by
 *  the launch's largest clouds, for a team of this block alone. */
NUMKONG_DEVICE void nk_shapes_network_carve_simt_(void *scratch, void *state, nk_u32_t largest_rows,
                                                  nk_u32_t largest_columns, nk_u32_t rows, nk_u32_t columns,
                                                  nk_size_t point_bytes, nk_shapes_network_simt_t *network) {
    nk_u32_t const arcs = largest_rows + largest_columns, nodes = largest_rows + largest_columns;
    network->team_values = (nk_f64_t *)scratch;
    network->team_indices = (nk_u32_t *)(network->team_values + nk_shapes_team_values_simt_k * blockDim.x);
    nk_f64_t *doubles = (nk_f64_t *)state +
                        nk_size_divide_round_up_(3 * (largest_rows + largest_columns) * point_bytes, sizeof(nk_f64_t));
    nk_u32_t *words = (nk_u32_t *)doubles;
    network->column_arcs = words, words += largest_columns;
    network->arc_rows = words, words += arcs;
    network->arc_flows = words, words += arcs;
    network->arc_next = words, words += arcs;
    network->free_arcs = words, words += 1;
    char *shared_region = (char *)state +
                          nk_size_round_up_to_multiple_((nk_size_t)((char *)words - (char *)state), sizeof(nk_f64_t));
    doubles = (nk_f64_t *)shared_region;
    network->row_potentials = doubles, doubles += largest_rows;
    network->column_potentials = doubles, doubles += largest_columns;
    network->distances = doubles, doubles += largest_columns;
    words = (nk_u32_t *)doubles;
    network->row_supplies = words, words += largest_rows;
    network->row_reached_columns = words, words += largest_rows;
    network->row_reached_arcs = words, words += largest_rows;
    network->column_demands = words, words += largest_columns;
    network->column_predecessors = words, words += largest_columns;
    network->column_closed = words, words += largest_columns;
    network->arc_columns = words, words += arcs;
    network->arc_row_next = words, words += arcs;
    network->row_arcs = words, words += largest_rows;
    network->visits = words;
    network->tree_potentials = (nk_f64_t *)shared_region;
    nk_u32_t *tree_words = (nk_u32_t *)(network->tree_potentials + nodes);
    network->tree_parents = tree_words, network->tree_flows = tree_words + nodes;
    network->tree_depths = tree_words + 2 * nodes, network->tree_first_children = tree_words + 3 * nodes;
    network->tree_next_siblings = tree_words + 4 * nodes, network->tree_previous_siblings = tree_words + 5 * nodes;
    network->tolerance = 0;
    network->rows = rows, network->columns = columns, network->arcs = arcs;
    network->team_meeting = NUMKONG_NULL, network->thread = threadIdx.x, network->team = blockDim.x;
}

/** Widens the team to @p blocks blocks meeting at @p meeting in global memory, this one of
 *  @p rank , each scan splitting its elements over all of their threads. */
NUMKONG_DEVICE void nk_shapes_network_join_simt_(nk_shapes_network_simt_t *network, void *meeting, unsigned blocks,
                                                 unsigned rank) {
    network->team_meeting = (nk_u32_t *)meeting;
    network->thread = rank * blockDim.x + threadIdx.x, network->team = blocks * blockDim.x;
}

/** A team's merge slots past its barrier: 2 argmins, then 10 sums, of a value per block. */
NUMKONG_DEVICE nk_f64_t volatile *nk_shapes_team_merges_simt_(nk_shapes_network_simt_t const *network) {
    return (nk_f64_t volatile *)((char *)network->team_meeting + 64);
}

/** Waits for every thread of the team: the block's barrier, then across blocks an arrival count
 *  in global memory, whose last arrival opens the next generation. */
NUMKONG_DEVICE void nk_shapes_team_sync_simt_(nk_shapes_network_simt_t const *network) {
    __syncthreads();
    if (!network->team_meeting) return;
    if (threadIdx.x == 0) {
        nk_u32_t volatile *generation = network->team_meeting + 1;
        nk_u32_t const seen = *generation;
        __threadfence();
        if (atomicAdd(network->team_meeting, 1u) == network->team / blockDim.x - 1) {
            atomicExch(network->team_meeting, 0u);
            __threadfence();
            atomicAdd(network->team_meeting + 1, 1u);
        }
        else
            while (*generation == seen) {}
        __threadfence();
    }
    __syncthreads();
}

/** Empties the plan: every row supplies @c columns units and every column takes @c rows . */
NUMKONG_DEVICE void nk_shapes_network_reset_simt_(nk_shapes_network_simt_t *network) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    unsigned const thread = network->thread, team = network->team;
    for (nk_u32_t row = thread; row < network->rows; row += team) network->row_supplies[row] = network->columns;
    for (nk_u32_t column = thread; column < network->columns; column += team)
        network->column_demands[column] = network->rows, network->column_arcs[column] = no_node;
    for (nk_u32_t arc = thread; arc < network->arcs; arc += team)
        network->arc_next[arc] = arc + 1 < network->arcs ? arc + 1 : no_node;
    if (thread == 0) network->free_arcs[0] = 0;
    nk_shapes_team_sync_simt_(network);
}

/** The least @p value over the team, the lower @p index breaking ties, left in every thread. */
NUMKONG_DEVICE void nk_shapes_team_argmin_simt_(nk_shapes_network_simt_t *network, nk_f64_t *value, nk_u32_t *index,
                                                unsigned *parity) {
    unsigned const thread = threadIdx.x, threads = blockDim.x;
    nk_f64_t *values = network->team_values + *parity * threads;
    nk_u32_t *indices = network->team_indices + *parity * threads;
    values[thread] = *value, indices[thread] = *index;
    __syncthreads();
    // Two barriers per step: every eighth thread folds its 8 entries in place, then all fold those
    if (thread < threads / 8) {
        nk_f64_t best_value = values[thread * 8];
        nk_u32_t best_index = indices[thread * 8];
        for (unsigned other = thread * 8 + 1; other != thread * 8 + 8; ++other)
            if (values[other] < best_value || (values[other] == best_value && indices[other] < best_index))
                best_value = values[other], best_index = indices[other];
        values[thread * 8] = best_value, indices[thread * 8] = best_index;
    }
    __syncthreads();
    nk_f64_t best_value = values[0];
    nk_u32_t best_index = indices[0];
    for (unsigned group = 8; group < threads; group += 8)
        if (values[group] < best_value || (values[group] == best_value && indices[group] < best_index))
            best_value = values[group], best_index = indices[group];
    if (network->team_meeting) {
        // The two parities keep a late block's reads apart from the next argmin's writes
        unsigned const blocks = network->team / threads, block = network->thread / threads;
        nk_f64_t volatile *merges = nk_shapes_team_merges_simt_(network);
        nk_u32_t volatile *merge_indices = (nk_u32_t volatile *)(merges + (2 + nk_shapes_team_values_simt_k) *
                                                                              nk_shapes_team_blocks_simt_k);
        merges += *parity * nk_shapes_team_blocks_simt_k, merge_indices += *parity * nk_shapes_team_blocks_simt_k;
        if (thread == 0) {
            merges[block] = best_value;
            merge_indices[block] = best_index;
        }
        nk_shapes_team_sync_simt_(network);
        best_value = merges[0], best_index = merge_indices[0];
        for (unsigned rank = 1; rank != blocks; ++rank) {
            nk_f64_t const other_value = merges[rank];
            nk_u32_t const other_index = merge_indices[rank];
            if (other_value < best_value || (other_value == best_value && other_index < best_index))
                best_value = other_value, best_index = other_index;
        }
    }
    *value = best_value, *index = best_index;
    *parity ^= 1u;
}

/** Sums each of the @p count values every thread holds over the team, in one fixed order. */
NUMKONG_DEVICE void nk_shapes_team_sum_f64_simt_(nk_shapes_network_simt_t *network, nk_f64_t *values, unsigned count) {
    unsigned const thread = threadIdx.x, threads = blockDim.x;
    __syncthreads();
    for (unsigned entry = 0; entry != count; ++entry) network->team_values[entry * threads + thread] = values[entry];
    __syncthreads();
    if (thread < count) {
        nk_f64_t total = 0;
        for (unsigned other = 0; other != threads; ++other) total += network->team_values[thread * threads + other];
        network->team_values[thread * threads] = total;
    }
    __syncthreads();
    for (unsigned entry = 0; entry != count; ++entry) values[entry] = network->team_values[entry * threads];
    __syncthreads();
    if (!network->team_meeting) return;
    unsigned const blocks = network->team / threads, block = network->thread / threads;
    nk_f64_t volatile *merges = nk_shapes_team_merges_simt_(network) + 2 * nk_shapes_team_blocks_simt_k;
    if (thread < count) merges[thread * nk_shapes_team_blocks_simt_k + block] = values[thread];
    nk_shapes_team_sync_simt_(network);
    for (unsigned entry = 0; entry != count; ++entry) {
        nk_f64_t total = 0;
        for (unsigned rank = 0; rank != blocks; ++rank) total += merges[entry * nk_shapes_team_blocks_simt_k + rank];
        values[entry] = total;
    }
    nk_shapes_team_sync_simt_(network);
}

NUMKONG_DEVICE void nk_shapes_team_sum_f32_simt_(nk_shapes_network_simt_t *network, nk_f32_t *values, unsigned count) {
    unsigned const thread = threadIdx.x, threads = blockDim.x;
    nk_f32_t *scratch = (nk_f32_t *)network->team_values;
    __syncthreads();
    for (unsigned entry = 0; entry != count; ++entry) scratch[entry * threads + thread] = values[entry];
    __syncthreads();
    if (thread < count) {
        nk_f32_t total = 0;
        for (unsigned other = 0; other != threads; ++other) total += scratch[thread * threads + other];
        scratch[thread * threads] = total;
    }
    __syncthreads();
    for (unsigned entry = 0; entry != count; ++entry) values[entry] = scratch[entry * threads];
    __syncthreads();
    if (!network->team_meeting) return;
    unsigned const blocks = network->team / threads, block = network->thread / threads;
    nk_f32_t volatile *merges = (nk_f32_t volatile *)(nk_shapes_team_merges_simt_(network) +
                                                      2 * nk_shapes_team_blocks_simt_k);
    if (thread < count) merges[thread * nk_shapes_team_blocks_simt_k + block] = values[thread];
    nk_shapes_team_sync_simt_(network);
    for (unsigned entry = 0; entry != count; ++entry) {
        nk_f32_t total = 0;
        for (unsigned rank = 0; rank != blocks; ++rank) total += merges[entry * nk_shapes_team_blocks_simt_k + rank];
        values[entry] = total;
    }
    nk_shapes_team_sync_simt_(network);
}

/** Ships @p amount more from @p row to @p column , opening the arc if the plan had none. */
NUMKONG_DEVICE void nk_shapes_network_ship_simt_(nk_shapes_network_simt_t *network, nk_u32_t row, nk_u32_t column,
                                                 nk_u32_t amount) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc])
        if (network->arc_rows[arc] == row) {
            network->arc_flows[arc] += amount;
            return;
        }
    nk_u32_t const arc = network->free_arcs[0];
    network->free_arcs[0] = network->arc_next[arc];
    network->arc_rows[arc] = row, network->arc_flows[arc] = amount;
    network->arc_next[arc] = network->column_arcs[column], network->column_arcs[column] = arc;
}

/** Ships @p amount less along @p arc of @p column , closing it once empty. */
NUMKONG_DEVICE void nk_shapes_network_return_simt_(nk_shapes_network_simt_t *network, nk_u32_t arc, nk_u32_t column,
                                                   nk_u32_t amount) {
    network->arc_flows[arc] -= amount;
    if (network->arc_flows[arc]) return;
    if (network->column_arcs[column] == arc) network->column_arcs[column] = network->arc_next[arc];
    else {
        nk_u32_t previous = network->column_arcs[column];
        while (network->arc_next[previous] != arc) previous = network->arc_next[previous];
        network->arc_next[previous] = network->arc_next[arc];
    }
    network->arc_next[arc] = network->free_arcs[0], network->free_arcs[0] = arc;
}

/** Lists the plan's arcs by row as well as by column, and clears the visit marks of a warm start.
 *  Runs on one thread. */
NUMKONG_DEVICE void nk_shapes_network_index_rows_simt_(nk_shapes_network_simt_t *network) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    for (nk_u32_t row = 0; row != network->rows; ++row)
        network->row_arcs[row] = no_node, network->row_reached_columns[row] = 0;
    for (nk_u32_t column = 0; column != network->columns; ++column) {
        network->column_closed[column] = 0;
        for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc]) {
            nk_u32_t const row = network->arc_rows[arc];
            network->arc_columns[arc] = column;
            network->arc_row_next[arc] = network->row_arcs[row], network->row_arcs[row] = arc;
        }
    }
}

/** Returns every arc of the rows a warm start marked with 2 to the supplies and demands. Runs on
 *  one thread. */
NUMKONG_DEVICE void nk_shapes_network_drop_rows_simt_(nk_shapes_network_simt_t *network) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    for (nk_u32_t row = 0; row != network->rows; ++row) {
        if (network->row_reached_columns[row] != 2) continue;
        for (nk_u32_t arc = network->row_arcs[row]; arc != no_node;) {
            nk_u32_t const next = network->arc_row_next[arc], column = network->arc_columns[arc];
            nk_u32_t const flow = network->arc_flows[arc];
            network->row_supplies[row] += flow, network->column_demands[column] += flow;
            nk_shapes_network_return_simt_(network, arc, column, flow);
            arc = next;
        }
        network->row_arcs[row] = no_node;
    }
}

/** Copies the plan's arcs to @p saved as rows, columns and flows, @c arcs apart, empty slots
 *  without flow, for Gromov-Wasserstein to start from. */
NUMKONG_DEVICE void nk_shapes_network_save_simt_(nk_shapes_network_simt_t const *network, nk_u32_t *saved) {
    nk_u32_t const no_node = 0xFFFFFFFFu, arcs = network->arcs;
    unsigned const thread = network->thread, team = network->team;
    for (nk_u32_t arc = thread; arc < arcs; arc += team) saved[2 * arcs + arc] = 0;
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t column = thread; column < network->columns; column += team)
        for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc])
            saved[arc] = network->arc_rows[arc], saved[arcs + arc] = column,
            saved[2 * arcs + arc] = network->arc_flows[arc];
    nk_shapes_team_sync_simt_(network);
}

/** Shifts the potentials by the distances of a Dijkstra that reached a sink at @p nearest , which
 *  leaves every arc of the tree tight and none negative. */
NUMKONG_DEVICE void nk_shapes_network_lift_simt_(nk_shapes_network_simt_t *network, nk_f64_t nearest) {
    nk_u32_t const no_node = 0xFFFFFFFFu, source_marker = 0xFFFFFFFEu;
    unsigned const thread = network->thread, team = network->team;
    for (nk_u32_t column = thread; column < network->columns; column += team)
        if (network->column_closed[column]) network->column_potentials[column] -= nearest - network->distances[column];
    for (nk_u32_t row = thread; row < network->rows; row += team) {
        nk_u32_t const reached = network->row_reached_columns[row];
        if (reached == no_node) continue;
        network->row_potentials[row] += nearest - (reached == source_marker ? 0 : network->distances[reached]);
    }
    nk_shapes_team_sync_simt_(network);
}

/** Ships the most the path from @p source to @p sink carries: the least of the source's supply,
 *  the sink's demand, and the flows it runs back along. Runs on one thread. */
NUMKONG_DEVICE void nk_shapes_network_augment_simt_(nk_shapes_network_simt_t *network, nk_u32_t source, nk_u32_t sink) {
    nk_u32_t amount = network->row_supplies[source] < network->column_demands[sink] ? network->row_supplies[source]
                                                                                    : network->column_demands[sink];
    for (nk_u32_t column = sink;;) {
        nk_u32_t const row = network->column_predecessors[column];
        if (row == source) break;
        nk_u32_t const flow = network->arc_flows[network->row_reached_arcs[row]];
        amount = flow < amount ? flow : amount;
        column = network->row_reached_columns[row];
    }
    for (nk_u32_t column = sink;;) {
        nk_u32_t const row = network->column_predecessors[column];
        nk_shapes_network_ship_simt_(network, row, column, amount);
        if (row == source) break;
        nk_u32_t const previous_column = network->row_reached_columns[row];
        nk_shapes_network_return_simt_(network, network->row_reached_arcs[row], previous_column, amount);
        column = previous_column;
    }
    network->row_supplies[source] -= amount, network->column_demands[sink] -= amount;
}

/** Appends @p child to the children of @p parent in the simplex tree. */
NUMKONG_DEVICE void nk_shapes_tree_attach_simt_(nk_shapes_network_simt_t *network, nk_u32_t parent, nk_u32_t child) {
    nk_u32_t const no_node = 0xFFFFFFFFu, after = network->tree_first_children[parent];
    network->tree_next_siblings[child] = after, network->tree_previous_siblings[child] = no_node;
    if (after != no_node) network->tree_previous_siblings[after] = child;
    network->tree_first_children[parent] = child;
}

/** Removes @p child from the children of @p parent in the simplex tree. */
NUMKONG_DEVICE void nk_shapes_tree_detach_simt_(nk_shapes_network_simt_t *network, nk_u32_t parent, nk_u32_t child) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    nk_u32_t const before = network->tree_previous_siblings[child], after = network->tree_next_siblings[child];
    if (before != no_node) network->tree_next_siblings[before] = after;
    else network->tree_first_children[parent] = after;
    if (after != no_node) network->tree_previous_siblings[after] = before;
}

/** Seeds the simplex tree with the northwest-corner plan rooted at the first row point, as
 *  @c nk_shapes_northwest_corner_serial_ does. Runs on one thread. */
NUMKONG_DEVICE void nk_shapes_tree_northwest_simt_(nk_shapes_network_simt_t *network) {
    nk_u32_t const no_node = 0xFFFFFFFFu, row_nodes = network->rows, column_nodes = network->columns;
    for (nk_u32_t node = 0; node != row_nodes + column_nodes; ++node) network->tree_first_children[node] = no_node;
    network->tree_parents[0] = no_node, network->tree_flows[0] = 0, network->tree_depths[0] = 0;
    nk_u32_t row = 0, column = 0, supply = column_nodes, demand = row_nodes, newest = row_nodes;
    for (;;) {
        nk_u32_t const shipped = supply < demand ? supply : demand;
        nk_u32_t const above = newest < row_nodes ? row_nodes + column : row;
        network->tree_parents[newest] = above, network->tree_flows[newest] = shipped;
        network->tree_depths[newest] = network->tree_depths[above] + 1;
        nk_shapes_tree_attach_simt_(network, above, newest);
        supply -= shipped, demand -= shipped;
        if (row + 1 == row_nodes && column + 1 == column_nodes) return;
        if (!supply && row + 1 != row_nodes) ++row, supply = column_nodes, newest = row;
        else ++column, demand = row_nodes, newest = row_nodes + column;
    }
}

/** One pivot of the serial network simplex: the entering arc closes a cycle through the join of its
 *  ends, Cunningham's rule picks the last blocking arc around it, the flow shifts, and the subtree
 *  cut off hangs from the entering arc with updated depths and potentials. Runs on one thread. */
NUMKONG_DEVICE void nk_shapes_tree_pivot_simt_(nk_shapes_network_simt_t *network, nk_u32_t entering_row,
                                               nk_u32_t entering_column, nk_f64_t entering_cost) {
    nk_u32_t const no_node = 0xFFFFFFFFu, row_nodes = network->rows;
    nk_u32_t *parent = network->tree_parents, *flow = network->tree_flows, *depth = network->tree_depths;
    nk_u32_t const row_node = entering_row, column_node = row_nodes + entering_column;
    nk_u32_t row_side = row_node, column_side = column_node;
    while (row_side != column_side) {
        if (depth[row_side] > depth[column_side]) row_side = parent[row_side];
        else if (depth[column_side] > depth[row_side]) column_side = parent[column_side];
        else row_side = parent[row_side], column_side = parent[column_side];
    }
    nk_u32_t const join = row_side;
    nk_u32_t row_side_flow = no_node, row_side_leaving = no_node;
    nk_u32_t column_side_flow = no_node, column_side_leaving = no_node;
    for (nk_u32_t node = row_node; node != join; node = parent[node])
        if (node < row_nodes && flow[node] < row_side_flow) row_side_flow = flow[node], row_side_leaving = node;
    for (nk_u32_t node = column_node; node != join; node = parent[node])
        if (node >= row_nodes && flow[node] <= column_side_flow)
            column_side_flow = flow[node], column_side_leaving = node;
    int const leaves_column_side = column_side_flow <= row_side_flow;
    nk_u32_t const delta = leaves_column_side ? column_side_flow : row_side_flow;
    nk_u32_t const leaving = leaves_column_side ? column_side_leaving : row_side_leaving;
    for (nk_u32_t node = row_node; node != join; node = parent[node])
        flow[node] = node < row_nodes ? flow[node] - delta : flow[node] + delta;
    for (nk_u32_t node = column_node; node != join; node = parent[node])
        flow[node] = node < row_nodes ? flow[node] + delta : flow[node] - delta;
    nk_u32_t const moved_root = leaves_column_side ? column_node : row_node;
    nk_u32_t const anchor = leaves_column_side ? row_node : column_node;
    nk_f64_t const shift = leaves_column_side ? -entering_cost : entering_cost;
    nk_u32_t node = moved_root, new_parent = anchor, carried_flow = delta;
    for (;;) {
        nk_u32_t const old_parent = parent[node], old_flow = flow[node];
        nk_shapes_tree_detach_simt_(network, old_parent, node);
        parent[node] = new_parent, flow[node] = carried_flow;
        nk_shapes_tree_attach_simt_(network, new_parent, node);
        if (node == leaving) break;
        new_parent = node, carried_flow = old_flow, node = old_parent;
    }
    depth[moved_root] = depth[anchor] + 1, network->tree_potentials[moved_root] += shift;
    node = moved_root;
    for (;;) {
        if (network->tree_first_children[node] != no_node) node = network->tree_first_children[node];
        else {
            while (node != moved_root && network->tree_next_siblings[node] == no_node) node = parent[node];
            if (node == moved_root) break;
            node = network->tree_next_siblings[node];
        }
        depth[node] = depth[parent[node]] + 1, network->tree_potentials[node] += shift;
    }
}

/** Rebuilds the plan's column lists from the simplex tree's arcs with flow. Runs on one thread. */
NUMKONG_DEVICE void nk_shapes_tree_plan_simt_(nk_shapes_network_simt_t *network) {
    nk_u32_t const no_node = 0xFFFFFFFFu, rows = network->rows, nodes = network->rows + network->columns;
    for (nk_u32_t column = 0; column != network->columns; ++column) network->column_arcs[column] = no_node;
    nk_u32_t arc = 0;
    for (nk_u32_t node = 1; node != nodes; ++node) {
        nk_u32_t const flow = network->tree_flows[node];
        if (!flow) continue;
        nk_u32_t const above = network->tree_parents[node];
        nk_u32_t const row = node < rows ? node : above, column = (node < rows ? above : node) - rows;
        network->arc_rows[arc] = row, network->arc_flows[arc] = flow;
        network->arc_next[arc] = network->column_arcs[column], network->column_arcs[column] = arc;
        ++arc;
    }
    for (nk_u32_t free_arc = arc; free_arc < network->arcs; ++free_arc)
        network->arc_next[free_arc] = free_arc + 1 < network->arcs ? free_arc + 1 : no_node;
    network->free_arcs[0] = arc < network->arcs ? arc : no_node;
}

#pragma endregion Transport Network

#pragma region Transport F32

/** The f64 coordinates of an f32 pair: the row points as packed, the column points rotated. */
typedef struct {
    nk_f64_t *row_x, *row_y, *row_z;
    nk_f64_t *column_x, *column_y, *column_z;
} nk_shapes_points_f32_simt_t;

/** Lays an f32 pair's point arrays at the start of @p shared , sized by the largest clouds. */
NUMKONG_DEVICE void nk_shapes_points_carve_f32_simt_(void *shared, nk_u32_t largest_rows, nk_u32_t largest_columns,
                                                     nk_shapes_points_f32_simt_t *points) {
    nk_f64_t *doubles = (nk_f64_t *)shared;
    points->row_x = doubles, points->row_y = doubles + largest_rows, points->row_z = doubles + 2 * largest_rows;
    doubles += 3 * largest_rows;
    points->column_x = doubles, points->column_y = doubles + largest_columns;
    points->column_z = doubles + 2 * largest_columns;
}

NUMKONG_DEVICE nk_f64_t nk_shapes_point_cost_f32_simt_(nk_shapes_points_f32_simt_t const *points, nk_u32_t row,
                                                       nk_u32_t column) {
    nk_f64_t const delta_x = points->row_x[row] - points->column_x[column];
    nk_f64_t const delta_y = points->row_y[row] - points->column_y[column];
    nk_f64_t const delta_z = points->row_z[row] - points->column_z[column];
    // Unfused like the serial cost, as near-ties between plans follow its last bit
    return __dadd_rn(__dadd_rn(__dmul_rn(delta_x, delta_x), __dmul_rn(delta_y, delta_y)), __dmul_rn(delta_z, delta_z));
}

/** Widens the row pack's points into the team's state. */
NUMKONG_DEVICE void nk_shapes_load_rows_f32_simt_(nk_shapes_network_simt_t const *network,
                                                  nk_shapes_points_f32_simt_t *points, nk_f32_t const *axes,
                                                  nk_size_t axis_stride, nk_u32_t rows) {
    for (nk_u32_t row = network->thread; row < rows; row += network->team)
        points->row_x[row] = axes[row], points->row_y[row] = axes[axis_stride + row],
        points->row_z[row] = axes[2 * axis_stride + row];
}

/** Rotates the column pack's points by @p rotation into the team's state like the serial solver. */
NUMKONG_DEVICE void nk_shapes_rotate_columns_f32_simt_(nk_shapes_network_simt_t const *network,
                                                       nk_shapes_points_f32_simt_t *points, nk_f32_t const *axes,
                                                       nk_size_t axis_stride, nk_u32_t columns,
                                                       nk_f64_t const *rotation) {
    for (nk_u32_t column = network->thread; column < columns; column += network->team) {
        nk_f64_t const x = axes[column], y = axes[axis_stride + column], z = axes[2 * axis_stride + column];
        points->column_x[column] = __dadd_rn(__dadd_rn(__dmul_rn(rotation[0], x), __dmul_rn(rotation[1], y)),
                                             __dmul_rn(rotation[2], z));
        points->column_y[column] = __dadd_rn(__dadd_rn(__dmul_rn(rotation[3], x), __dmul_rn(rotation[4], y)),
                                             __dmul_rn(rotation[5], z));
        points->column_z[column] = __dadd_rn(__dadd_rn(__dmul_rn(rotation[6], x), __dmul_rn(rotation[7], y)),
                                             __dmul_rn(rotation[8], z));
    }
    nk_shapes_team_sync_simt_(network);
}

/** The first solve's start: column then row reductions, shipping along each arc left tight. */
NUMKONG_DEVICE void nk_shapes_seed_f32_simt_(nk_shapes_network_simt_t *network,
                                             nk_shapes_points_f32_simt_t const *points) {
    unsigned const thread = network->thread, team = network->team;
    for (nk_u32_t column = thread; column < network->columns; column += team) {
        nk_f64_t best = NUMKONG_F64_INF;
        nk_u32_t best_row = 0;
        for (nk_u32_t row = 0; row != network->rows; ++row) {
            nk_f64_t const cost = nk_shapes_point_cost_f32_simt_(points, row, column);
            if (cost < best) best = cost, best_row = row;
        }
        network->column_potentials[column] = best, network->column_predecessors[column] = best_row;
    }
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t row = thread; row < network->rows; row += team) {
        nk_f64_t best = NUMKONG_F64_INF;
        for (nk_u32_t column = 0; column != network->columns; ++column) {
            nk_f64_t const reduced = nk_shapes_point_cost_f32_simt_(points, row, column) -
                                     network->column_potentials[column];
            best = reduced < best ? reduced : best;
        }
        network->row_potentials[row] = best;
    }
    nk_shapes_team_sync_simt_(network);
    if (thread == 0)
        for (nk_u32_t column = 0; column != network->columns; ++column) {
            nk_u32_t const row = network->column_predecessors[column];
            nk_f64_t const reduced = nk_shapes_point_cost_f32_simt_(points, row, column) -
                                     network->column_potentials[column] - network->row_potentials[row];
            nk_u32_t const supply = network->row_supplies[row], demand = network->column_demands[column];
            nk_u32_t const amount = supply < demand ? supply : demand;
            if (!(reduced <= 0) || !amount) continue;
            nk_shapes_network_ship_simt_(network, row, column, amount);
            network->row_supplies[row] -= amount, network->column_demands[column] -= amount;
        }
    nk_shapes_team_sync_simt_(network);
}

/** A later round's start from the last plan: potentials that leave every arc of the plan tight
 *  under the new rotation, walked out from each tree's first row, which keeps its potential. Rows
 *  that a cheaper column now undercuts return their flows and drop to their least reduced cost. */
NUMKONG_DEVICE void nk_shapes_reseed_f32_simt_(nk_shapes_network_simt_t *network,
                                               nk_shapes_points_f32_simt_t const *points) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    unsigned const thread = network->thread, team = network->team;
    nk_u32_t const rows = network->rows;
    if (thread == 0) {
        nk_shapes_network_index_rows_simt_(network);
        nk_u32_t head = 0, tail = 0;
        for (nk_u32_t root = 0; root != rows; ++root) {
            if (network->row_reached_columns[root] || network->row_arcs[root] == no_node) continue;
            network->row_reached_columns[root] = 1, network->visits[tail++] = root;
            while (head != tail) {
                nk_u32_t const node = network->visits[head++];
                if (node < rows) {
                    for (nk_u32_t arc = network->row_arcs[node]; arc != no_node; arc = network->arc_row_next[arc]) {
                        nk_u32_t const column = network->arc_columns[arc];
                        if (network->column_closed[column]) continue;
                        network->column_closed[column] = 1, network->visits[tail++] = rows + column;
                        network->column_potentials[column] = nk_shapes_point_cost_f32_simt_(points, node, column) -
                                                             network->row_potentials[node];
                    }
                }
                else {
                    nk_u32_t const column = node - rows;
                    for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc]) {
                        nk_u32_t const row = network->arc_rows[arc];
                        if (network->row_reached_columns[row]) continue;
                        network->row_reached_columns[row] = 1, network->visits[tail++] = row;
                        network->row_potentials[row] = nk_shapes_point_cost_f32_simt_(points, row, column) -
                                                       network->column_potentials[column];
                    }
                }
            }
        }
    }
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t row = thread; row < rows; row += team) {
        nk_f64_t best = NUMKONG_F64_INF;
        for (nk_u32_t column = 0; column != network->columns; ++column) {
            nk_f64_t const reduced = nk_shapes_point_cost_f32_simt_(points, row, column) -
                                     network->column_potentials[column];
            best = reduced < best ? reduced : best;
        }
        nk_f64_t const potential = network->row_potentials[row];
        nk_f64_t const scale = (best < 0 ? -best : best) + (potential < 0 ? -potential : potential);
        int const undercut = best < potential - 1e-12 * scale;
        network->row_reached_columns[row] = undercut ? 2 : 0;
        network->row_potentials[row] = best < potential ? best : potential;
    }
    nk_shapes_team_sync_simt_(network);
    if (thread == 0) nk_shapes_network_drop_rows_simt_(network);
    nk_shapes_team_sync_simt_(network);
}

/** Dijkstra from @p source over the reduced costs until it closes a column with demand, which it
 *  returns with its distance in @p nearest . Closing a column without demand reaches every row
 *  shipping into it at the same distance, as their arcs are tight. */
NUMKONG_DEVICE nk_u32_t nk_shapes_shortest_path_f32_simt_(nk_shapes_network_simt_t *network,
                                                          nk_shapes_points_f32_simt_t const *points, nk_u32_t source,
                                                          nk_f64_t *nearest, unsigned *parity) {
    nk_u32_t const no_node = 0xFFFFFFFFu, source_marker = 0xFFFFFFFEu;
    unsigned const thread = network->thread, team = network->team;
    nk_u32_t const columns = network->columns;
    nk_f64_t const source_potential = network->row_potentials[source];
    for (nk_u32_t column = thread; column < columns; column += team) {
        network->distances[column] = nk_shapes_point_cost_f32_simt_(points, source, column) -
                                     network->column_potentials[column] - source_potential;
        network->column_predecessors[column] = source, network->column_closed[column] = 0;
    }
    for (nk_u32_t row = thread; row < network->rows; row += team)
        network->row_reached_columns[row] = row == source ? source_marker : no_node;
    for (;;) {
        nk_f64_t value = NUMKONG_F64_INF;
        nk_u32_t index = no_node;
        for (nk_u32_t column = thread; column < columns; column += team)
            if (!network->column_closed[column] && network->distances[column] < value)
                value = network->distances[column], index = column;
        nk_shapes_team_argmin_simt_(network, &value, &index, parity);
        if (index % team == thread) network->column_closed[index] = 1;
        if (network->column_demands[index]) {
            *nearest = value;
            nk_shapes_team_sync_simt_(network);
            return index;
        }
        for (nk_u32_t arc = network->column_arcs[index]; arc != no_node; arc = network->arc_next[arc]) {
            nk_u32_t const row = network->arc_rows[arc];
            if (thread == 0 && network->row_reached_columns[row] == no_node)
                network->row_reached_columns[row] = index, network->row_reached_arcs[row] = arc;
            nk_f64_t const row_potential = network->row_potentials[row];
            for (nk_u32_t column = thread; column < columns; column += team) {
                if (network->column_closed[column]) continue;
                nk_f64_t const candidate = value + (nk_shapes_point_cost_f32_simt_(points, row, column) -
                                                    network->column_potentials[column] - row_potential);
                if (candidate < network->distances[column])
                    network->distances[column] = candidate, network->column_predecessors[column] = row;
            }
        }
    }
}

/** The optimal plan for the current rotation, from the reductions on the first round or from the
 *  last plan after. */
NUMKONG_DEVICE void nk_shapes_transport_f32_simt_(nk_shapes_network_simt_t *network,
                                                  nk_shapes_points_f32_simt_t const *points, int first,
                                                  unsigned *parity) {
    if (first) nk_shapes_seed_f32_simt_(network, points);
    else nk_shapes_reseed_f32_simt_(network, points);
    for (nk_u32_t source = 0; source != network->rows; ++source)
        while (network->row_supplies[source]) {
            nk_f64_t nearest;
            nk_u32_t const sink = nk_shapes_shortest_path_f32_simt_(network, points, source, &nearest, parity);
            nk_shapes_network_lift_simt_(network, nearest);
            if (network->thread == 0) nk_shapes_network_augment_simt_(network, source, sink);
            nk_shapes_team_sync_simt_(network);
        }
}

/** The plan's mean cost, Σ flow × cost / (n × m), and its cross-covariance, Σ flow / (n × m) × bⱼ ×
 *  aᵢᵀ over the unrotated column points at @p column_axes , summed in F64 like the serial ones. */
NUMKONG_DEVICE nk_f64_t nk_shapes_plan_moments_f32_simt_(nk_shapes_network_simt_t *network,
                                                         nk_shapes_points_f32_simt_t const *points,
                                                         nk_f32_t const *column_axes, nk_size_t axis_stride,
                                                         nk_f64_t *cross_covariance) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    nk_f64_t const mass = (nk_f64_t)((nk_size_t)network->rows * network->columns);
    nk_f64_t sums[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (nk_u32_t column = network->thread; column < network->columns; column += network->team) {
        nk_f64_t const column_point[3] = {column_axes[column], column_axes[axis_stride + column],
                                          column_axes[2 * axis_stride + column]};
        for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc]) {
            nk_u32_t const row = network->arc_rows[arc];
            nk_f64_t const flow = (nk_f64_t)network->arc_flows[arc];
            nk_f64_t const row_point[3] = {points->row_x[row], points->row_y[row], points->row_z[row]};
            sums[9] = __dadd_rn(sums[9], __dmul_rn(flow, nk_shapes_point_cost_f32_simt_(points, row, column)));
            nk_f64_t const weight = flow / mass;
            for (int column_axis = 0; column_axis != 3; ++column_axis) {
                nk_f64_t const weighted = weight * column_point[column_axis];
                for (int row_axis = 0; row_axis != 3; ++row_axis)
                    sums[column_axis * 3 + row_axis] = __dadd_rn(sums[column_axis * 3 + row_axis],
                                                                 __dmul_rn(weighted, row_point[row_axis]));
            }
        }
    }
    nk_shapes_team_sum_f64_simt_(network, sums, 10);
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = sums[cell];
    return sums[9] / mass;
}

/** The simplex tree's potentials walked from its root, as @c nk_shapes_tree_potentials_f64_serial_
 *  sets them. Runs on one thread. */
NUMKONG_DEVICE void nk_shapes_tree_potentials_f32_simt_(nk_shapes_network_simt_t *network,
                                                        nk_shapes_points_f32_simt_t const *points) {
    nk_u32_t const no_node = 0xFFFFFFFFu, row_nodes = network->rows;
    nk_u32_t node = 0;
    network->tree_potentials[0] = 0;
    for (;;) {
        if (network->tree_first_children[node] != no_node) node = network->tree_first_children[node];
        else {
            while (node && network->tree_next_siblings[node] == no_node) node = network->tree_parents[node];
            if (!node) return;
            node = network->tree_next_siblings[node];
        }
        nk_u32_t const above = network->tree_parents[node];
        if (node < row_nodes)
            network->tree_potentials[node] = network->tree_potentials[above] +
                                             nk_shapes_point_cost_f32_simt_(points, node, above - row_nodes);
        else
            network->tree_potentials[node] = network->tree_potentials[above] -
                                             nk_shapes_point_cost_f32_simt_(points, above, node - row_nodes);
    }
}

/** The exact plan by the serial network simplex from the tree in @p network : the team prices each
 *  block of √(n × m) arcs from the cursor at once, taking its first most negative reduced cost, and
 *  one thread pivots. Leaves the plan's arcs in the column lists. */
NUMKONG_DEVICE void nk_shapes_simplex_f32_simt_(nk_shapes_network_simt_t *network,
                                                nk_shapes_points_f32_simt_t const *points, unsigned *parity) {
    nk_u32_t const no_node = 0xFFFFFFFFu, rows = network->rows, columns = network->columns;
    unsigned const thread = network->thread, team = network->team;
    nk_size_t const arcs = (nk_size_t)rows * columns;
    nk_f64_t const tolerance = network->tolerance;
    nk_size_t block = 10, cursor = 0;
    while (block * block < arcs) ++block;
    if (thread == 0) nk_shapes_tree_potentials_f32_simt_(network, points);
    nk_shapes_team_sync_simt_(network);
    for (;;) {
        nk_f64_t entering_cost = -tolerance;
        nk_size_t entering = arcs;
        for (nk_size_t scanned = 0; scanned < arcs;) {
            nk_size_t const count = arcs - scanned < block ? arcs - scanned : block;
            nk_f64_t value = -tolerance;
            nk_u32_t index = no_node;
            for (nk_size_t offset = thread; offset < count; offset += team) {
                nk_size_t position = cursor + offset;
                position = position >= arcs ? position - arcs : position;
                nk_u32_t const row = (nk_u32_t)(position / columns), column = (nk_u32_t)(position % columns);
                nk_f64_t const reduced = nk_shapes_point_cost_f32_simt_(points, row, column) -
                                         network->tree_potentials[row] + network->tree_potentials[rows + column];
                if (reduced < value) value = reduced, index = (nk_u32_t)offset;
            }
            nk_shapes_team_argmin_simt_(network, &value, &index, parity);
            nk_size_t const block_start = cursor;
            cursor += count, cursor = cursor >= arcs ? cursor - arcs : cursor;
            scanned += count;
            if (value < -tolerance) {
                entering_cost = value, entering = block_start + index;
                entering = entering >= arcs ? entering - arcs : entering;
                break;
            }
        }
        if (entering == arcs) break;
        if (thread == 0)
            nk_shapes_tree_pivot_simt_(network, (nk_u32_t)(entering / columns), (nk_u32_t)(entering % columns),
                                       entering_cost);
        nk_shapes_team_sync_simt_(network);
    }
    if (thread == 0) nk_shapes_tree_plan_simt_(network);
    nk_shapes_team_sync_simt_(network);
}

/** One flip of Wasserstein-Procrustes: exact transport alternating with the Kabsch fit of its plan
 *  while the mean cost falls, from the axis signs of @p flip . Leaves the least cost and its
 *  rotation of the column points in @p best and @p best_rotation , and its plan in @p saved_plan
 *  unless null. @p shared_rotation holds the 9 entries each block's first thread fits for it, all
 *  alike, as every block sums the same moments in the same order. */
NUMKONG_DEVICE void nk_wassersteinprocrustes_flip_f32_simt_(nk_shapes_network_simt_t *network,
                                                            nk_shapes_points_f32_simt_t *points,
                                                            nk_f32_t const *column_axes, nk_size_t axis_stride,
                                                            int flip, nk_f64_t *shared_rotation, nk_f64_t *best,
                                                            nk_f64_t *best_rotation, nk_u32_t *saved_plan,
                                                            unsigned *parity) {
    nk_f64_t current[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    current[0] = (flip & 1) ? -1 : 1, current[4] = (flip & 2) ? -1 : 1, current[8] = (flip & 4) ? -1 : 1;
    nk_f64_t const flip_determinant = current[0] * current[4] * current[8];
    *best = NUMKONG_F64_INF;
    for (int cell = 0; cell != 9; ++cell) best_rotation[cell] = current[cell];
    // Unequal clouds make staircases of small flows, which one pivot ships but paths take steps for
    int const simplex = network->rows != network->columns;
    nk_shapes_network_reset_simt_(network);
    if (simplex && network->thread == 0) nk_shapes_tree_northwest_simt_(network);
    for (int round = 0;; ++round) {
        nk_shapes_rotate_columns_f32_simt_(network, points, column_axes, axis_stride, network->columns, current);
        if (simplex) nk_shapes_simplex_f32_simt_(network, points, parity);
        else nk_shapes_transport_f32_simt_(network, points, round == 0, parity);
        nk_f64_t cross_covariance[9];
        nk_f64_t const cost = nk_shapes_plan_moments_f32_simt_(network, points, column_axes, axis_stride,
                                                               cross_covariance);
        if (!(cost < *best)) return;
        *best = cost;
        for (int cell = 0; cell != 9; ++cell) best_rotation[cell] = current[cell];
        if (saved_plan) nk_shapes_network_save_simt_(network, saved_plan);
        if (threadIdx.x == 0) {
            nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9], fitted[9];
            nk_svd3x3_f64_simt_(cross_covariance, svd_left, svd_diagonal, svd_right);
            nk_rotation_from_svd_f64_simt_(svd_left, svd_right, fitted);
            // Each start stays among the rotations, or among the reflections, it began with
            if (nk_det3x3_f64_simt_(fitted) * flip_determinant < 0) {
                svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
                nk_rotation_from_svd_f64_simt_(svd_left, svd_right, fitted);
            }
            for (int cell = 0; cell != 9; ++cell) shared_rotation[cell] = fitted[cell];
        }
        __syncthreads();
        for (int cell = 0; cell != 9; ++cell) current[cell] = shared_rotation[cell];
        __syncthreads();
    }
}

#pragma endregion Transport F32

#pragma region Transport F16

/** The f32 coordinates of an f16 pair: the row points as packed, the column points rotated. */
typedef struct {
    nk_f32_t *row_x, *row_y, *row_z;
    nk_f32_t *column_x, *column_y, *column_z;
} nk_shapes_points_f16_simt_t;

NUMKONG_DEVICE void nk_shapes_points_carve_f16_simt_(void *shared, nk_u32_t largest_rows, nk_u32_t largest_columns,
                                                     nk_shapes_points_f16_simt_t *points) {
    nk_f32_t *floats = (nk_f32_t *)shared;
    points->row_x = floats, points->row_y = floats + largest_rows, points->row_z = floats + 2 * largest_rows;
    floats += 3 * largest_rows;
    points->column_x = floats, points->column_y = floats + largest_columns;
    points->column_z = floats + 2 * largest_columns;
}

NUMKONG_DEVICE nk_f32_t nk_shapes_point_cost_f16_simt_(nk_shapes_points_f16_simt_t const *points, nk_u32_t row,
                                                       nk_u32_t column) {
    nk_f32_t const delta_x = points->row_x[row] - points->column_x[column];
    nk_f32_t const delta_y = points->row_y[row] - points->column_y[column];
    nk_f32_t const delta_z = points->row_z[row] - points->column_z[column];
    // Unfused like the serial cost, as near-ties between plans follow its last bit
    return __fadd_rn(__fadd_rn(__fmul_rn(delta_x, delta_x), __fmul_rn(delta_y, delta_y)), __fmul_rn(delta_z, delta_z));
}

NUMKONG_DEVICE void nk_shapes_load_rows_f16_simt_(nk_shapes_network_simt_t const *network,
                                                  nk_shapes_points_f16_simt_t *points, nk_f16_t const *axes,
                                                  nk_size_t axis_stride, nk_u32_t rows) {
    for (nk_u32_t row = network->thread; row < rows; row += network->team) {
        nk_f16_to_f32_simt_(axes + row, points->row_x + row);
        nk_f16_to_f32_simt_(axes + axis_stride + row, points->row_y + row);
        nk_f16_to_f32_simt_(axes + 2 * axis_stride + row, points->row_z + row);
    }
}

NUMKONG_DEVICE void nk_shapes_rotate_columns_f16_simt_(nk_shapes_network_simt_t const *network,
                                                       nk_shapes_points_f16_simt_t *points, nk_f16_t const *axes,
                                                       nk_size_t axis_stride, nk_u32_t columns,
                                                       nk_f32_t const *rotation) {
    for (nk_u32_t column = network->thread; column < columns; column += network->team) {
        nk_f32_t x, y, z;
        nk_f16_to_f32_simt_(axes + column, &x);
        nk_f16_to_f32_simt_(axes + axis_stride + column, &y);
        nk_f16_to_f32_simt_(axes + 2 * axis_stride + column, &z);
        points->column_x[column] = __fadd_rn(__fadd_rn(__fmul_rn(rotation[0], x), __fmul_rn(rotation[1], y)),
                                             __fmul_rn(rotation[2], z));
        points->column_y[column] = __fadd_rn(__fadd_rn(__fmul_rn(rotation[3], x), __fmul_rn(rotation[4], y)),
                                             __fmul_rn(rotation[5], z));
        points->column_z[column] = __fadd_rn(__fadd_rn(__fmul_rn(rotation[6], x), __fmul_rn(rotation[7], y)),
                                             __fmul_rn(rotation[8], z));
    }
    nk_shapes_team_sync_simt_(network);
}

NUMKONG_DEVICE void nk_shapes_seed_f16_simt_(nk_shapes_network_simt_t *network,
                                             nk_shapes_points_f16_simt_t const *points) {
    unsigned const thread = network->thread, team = network->team;
    for (nk_u32_t column = thread; column < network->columns; column += team) {
        nk_f64_t best = NUMKONG_F64_INF;
        nk_u32_t best_row = 0;
        for (nk_u32_t row = 0; row != network->rows; ++row) {
            nk_f64_t const cost = (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, row, column);
            if (cost < best) best = cost, best_row = row;
        }
        network->column_potentials[column] = best, network->column_predecessors[column] = best_row;
    }
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t row = thread; row < network->rows; row += team) {
        nk_f64_t best = NUMKONG_F64_INF;
        for (nk_u32_t column = 0; column != network->columns; ++column) {
            nk_f64_t const reduced = (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, row, column) -
                                     network->column_potentials[column];
            best = reduced < best ? reduced : best;
        }
        network->row_potentials[row] = best;
    }
    nk_shapes_team_sync_simt_(network);
    if (thread == 0)
        for (nk_u32_t column = 0; column != network->columns; ++column) {
            nk_u32_t const row = network->column_predecessors[column];
            nk_f64_t const reduced = (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, row, column) -
                                     network->column_potentials[column] - network->row_potentials[row];
            nk_u32_t const supply = network->row_supplies[row], demand = network->column_demands[column];
            nk_u32_t const amount = supply < demand ? supply : demand;
            if (!(reduced <= 0) || !amount) continue;
            nk_shapes_network_ship_simt_(network, row, column, amount);
            network->row_supplies[row] -= amount, network->column_demands[column] -= amount;
        }
    nk_shapes_team_sync_simt_(network);
}

NUMKONG_DEVICE void nk_shapes_reseed_f16_simt_(nk_shapes_network_simt_t *network,
                                               nk_shapes_points_f16_simt_t const *points) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    unsigned const thread = network->thread, team = network->team;
    nk_u32_t const rows = network->rows;
    if (thread == 0) {
        nk_shapes_network_index_rows_simt_(network);
        nk_u32_t head = 0, tail = 0;
        for (nk_u32_t root = 0; root != rows; ++root) {
            if (network->row_reached_columns[root] || network->row_arcs[root] == no_node) continue;
            network->row_reached_columns[root] = 1, network->visits[tail++] = root;
            while (head != tail) {
                nk_u32_t const node = network->visits[head++];
                if (node < rows) {
                    for (nk_u32_t arc = network->row_arcs[node]; arc != no_node; arc = network->arc_row_next[arc]) {
                        nk_u32_t const column = network->arc_columns[arc];
                        if (network->column_closed[column]) continue;
                        network->column_closed[column] = 1, network->visits[tail++] = rows + column;
                        network->column_potentials[column] = (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, node,
                                                                                                      column) -
                                                             network->row_potentials[node];
                    }
                }
                else {
                    nk_u32_t const column = node - rows;
                    for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc]) {
                        nk_u32_t const row = network->arc_rows[arc];
                        if (network->row_reached_columns[row]) continue;
                        network->row_reached_columns[row] = 1, network->visits[tail++] = row;
                        network->row_potentials[row] = (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, row, column) -
                                                       network->column_potentials[column];
                    }
                }
            }
        }
    }
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t row = thread; row < rows; row += team) {
        nk_f64_t best = NUMKONG_F64_INF;
        for (nk_u32_t column = 0; column != network->columns; ++column) {
            nk_f64_t const reduced = (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, row, column) -
                                     network->column_potentials[column];
            best = reduced < best ? reduced : best;
        }
        nk_f64_t const potential = network->row_potentials[row];
        nk_f64_t const scale = (best < 0 ? -best : best) + (potential < 0 ? -potential : potential);
        int const undercut = best < potential - 1e-12 * scale;
        network->row_reached_columns[row] = undercut ? 2 : 0;
        network->row_potentials[row] = best < potential ? best : potential;
    }
    nk_shapes_team_sync_simt_(network);
    if (thread == 0) nk_shapes_network_drop_rows_simt_(network);
    nk_shapes_team_sync_simt_(network);
}

NUMKONG_DEVICE nk_u32_t nk_shapes_shortest_path_f16_simt_(nk_shapes_network_simt_t *network,
                                                          nk_shapes_points_f16_simt_t const *points, nk_u32_t source,
                                                          nk_f64_t *nearest, unsigned *parity) {
    nk_u32_t const no_node = 0xFFFFFFFFu, source_marker = 0xFFFFFFFEu;
    unsigned const thread = network->thread, team = network->team;
    nk_u32_t const columns = network->columns;
    nk_f64_t const source_potential = network->row_potentials[source];
    for (nk_u32_t column = thread; column < columns; column += team) {
        network->distances[column] = (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, source, column) -
                                     network->column_potentials[column] - source_potential;
        network->column_predecessors[column] = source, network->column_closed[column] = 0;
    }
    for (nk_u32_t row = thread; row < network->rows; row += team)
        network->row_reached_columns[row] = row == source ? source_marker : no_node;
    for (;;) {
        nk_f64_t value = NUMKONG_F64_INF;
        nk_u32_t index = no_node;
        for (nk_u32_t column = thread; column < columns; column += team)
            if (!network->column_closed[column] && network->distances[column] < value)
                value = network->distances[column], index = column;
        nk_shapes_team_argmin_simt_(network, &value, &index, parity);
        if (index % team == thread) network->column_closed[index] = 1;
        if (network->column_demands[index]) {
            *nearest = value;
            nk_shapes_team_sync_simt_(network);
            return index;
        }
        for (nk_u32_t arc = network->column_arcs[index]; arc != no_node; arc = network->arc_next[arc]) {
            nk_u32_t const row = network->arc_rows[arc];
            if (thread == 0 && network->row_reached_columns[row] == no_node)
                network->row_reached_columns[row] = index, network->row_reached_arcs[row] = arc;
            nk_f64_t const row_potential = network->row_potentials[row];
            for (nk_u32_t column = thread; column < columns; column += team) {
                if (network->column_closed[column]) continue;
                nk_f64_t const candidate = value + ((nk_f64_t)nk_shapes_point_cost_f16_simt_(points, row, column) -
                                                    network->column_potentials[column] - row_potential);
                if (candidate < network->distances[column])
                    network->distances[column] = candidate, network->column_predecessors[column] = row;
            }
        }
    }
}

NUMKONG_DEVICE void nk_shapes_transport_f16_simt_(nk_shapes_network_simt_t *network,
                                                  nk_shapes_points_f16_simt_t const *points, int first,
                                                  unsigned *parity) {
    if (first) nk_shapes_seed_f16_simt_(network, points);
    else nk_shapes_reseed_f16_simt_(network, points);
    for (nk_u32_t source = 0; source != network->rows; ++source)
        while (network->row_supplies[source]) {
            nk_f64_t nearest;
            nk_u32_t const sink = nk_shapes_shortest_path_f16_simt_(network, points, source, &nearest, parity);
            nk_shapes_network_lift_simt_(network, nearest);
            if (network->thread == 0) nk_shapes_network_augment_simt_(network, source, sink);
            nk_shapes_team_sync_simt_(network);
        }
}

/** The plan's mean cost and cross-covariance over the unrotated column points, summed in F32 like
 *  the serial ones of f16 inputs. */
NUMKONG_DEVICE nk_f32_t nk_shapes_plan_moments_f16_simt_(nk_shapes_network_simt_t *network,
                                                         nk_shapes_points_f16_simt_t const *points,
                                                         nk_f16_t const *column_axes, nk_size_t axis_stride,
                                                         nk_f32_t *cross_covariance) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    nk_f32_t const mass = (nk_f32_t)((nk_size_t)network->rows * network->columns);
    nk_f32_t sums[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (nk_u32_t column = network->thread; column < network->columns; column += network->team) {
        nk_f32_t column_point[3];
        nk_f16_to_f32_simt_(column_axes + column, &column_point[0]);
        nk_f16_to_f32_simt_(column_axes + axis_stride + column, &column_point[1]);
        nk_f16_to_f32_simt_(column_axes + 2 * axis_stride + column, &column_point[2]);
        for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc]) {
            nk_u32_t const row = network->arc_rows[arc];
            nk_f32_t const flow = (nk_f32_t)network->arc_flows[arc];
            nk_f32_t const row_point[3] = {points->row_x[row], points->row_y[row], points->row_z[row]};
            sums[9] = __fadd_rn(sums[9], __fmul_rn(flow, nk_shapes_point_cost_f16_simt_(points, row, column)));
            nk_f32_t const weight = flow / mass;
            for (int column_axis = 0; column_axis != 3; ++column_axis) {
                nk_f32_t const weighted = weight * column_point[column_axis];
                for (int row_axis = 0; row_axis != 3; ++row_axis)
                    sums[column_axis * 3 + row_axis] = __fadd_rn(sums[column_axis * 3 + row_axis],
                                                                 __fmul_rn(weighted, row_point[row_axis]));
            }
        }
    }
    nk_shapes_team_sum_f32_simt_(network, sums, 10);
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = sums[cell];
    return sums[9] / mass;
}

NUMKONG_DEVICE void nk_shapes_tree_potentials_f16_simt_(nk_shapes_network_simt_t *network,
                                                        nk_shapes_points_f16_simt_t const *points) {
    nk_u32_t const no_node = 0xFFFFFFFFu, row_nodes = network->rows;
    nk_u32_t node = 0;
    network->tree_potentials[0] = 0;
    for (;;) {
        if (network->tree_first_children[node] != no_node) node = network->tree_first_children[node];
        else {
            while (node && network->tree_next_siblings[node] == no_node) node = network->tree_parents[node];
            if (!node) return;
            node = network->tree_next_siblings[node];
        }
        nk_u32_t const above = network->tree_parents[node];
        if (node < row_nodes)
            network->tree_potentials[node] = network->tree_potentials[above] +
                                             (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, node, above - row_nodes);
        else
            network->tree_potentials[node] = network->tree_potentials[above] -
                                             (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, above, node - row_nodes);
    }
}

NUMKONG_DEVICE void nk_shapes_simplex_f16_simt_(nk_shapes_network_simt_t *network,
                                                nk_shapes_points_f16_simt_t const *points, unsigned *parity) {
    nk_u32_t const no_node = 0xFFFFFFFFu, rows = network->rows, columns = network->columns;
    unsigned const thread = network->thread, team = network->team;
    nk_size_t const arcs = (nk_size_t)rows * columns;
    nk_f64_t const tolerance = network->tolerance;
    nk_size_t block = 10, cursor = 0;
    while (block * block < arcs) ++block;
    if (thread == 0) nk_shapes_tree_potentials_f16_simt_(network, points);
    nk_shapes_team_sync_simt_(network);
    for (;;) {
        nk_f64_t entering_cost = -tolerance;
        nk_size_t entering = arcs;
        for (nk_size_t scanned = 0; scanned < arcs;) {
            nk_size_t const count = arcs - scanned < block ? arcs - scanned : block;
            nk_f64_t value = -tolerance;
            nk_u32_t index = no_node;
            for (nk_size_t offset = thread; offset < count; offset += team) {
                nk_size_t position = cursor + offset;
                position = position >= arcs ? position - arcs : position;
                nk_u32_t const row = (nk_u32_t)(position / columns), column = (nk_u32_t)(position % columns);
                nk_f64_t const reduced = (nk_f64_t)nk_shapes_point_cost_f16_simt_(points, row, column) -
                                         network->tree_potentials[row] + network->tree_potentials[rows + column];
                if (reduced < value) value = reduced, index = (nk_u32_t)offset;
            }
            nk_shapes_team_argmin_simt_(network, &value, &index, parity);
            nk_size_t const block_start = cursor;
            cursor += count, cursor = cursor >= arcs ? cursor - arcs : cursor;
            scanned += count;
            if (value < -tolerance) {
                entering_cost = value, entering = block_start + index;
                entering = entering >= arcs ? entering - arcs : entering;
                break;
            }
        }
        if (entering == arcs) break;
        if (thread == 0)
            nk_shapes_tree_pivot_simt_(network, (nk_u32_t)(entering / columns), (nk_u32_t)(entering % columns),
                                       entering_cost);
        nk_shapes_team_sync_simt_(network);
    }
    if (thread == 0) nk_shapes_tree_plan_simt_(network);
    nk_shapes_team_sync_simt_(network);
}

NUMKONG_DEVICE void nk_wassersteinprocrustes_flip_f16_simt_(nk_shapes_network_simt_t *network,
                                                            nk_shapes_points_f16_simt_t *points,
                                                            nk_f16_t const *column_axes, nk_size_t axis_stride,
                                                            int flip, nk_f32_t *shared_rotation, nk_f32_t *best,
                                                            nk_f32_t *best_rotation, nk_u32_t *saved_plan,
                                                            unsigned *parity) {
    nk_f32_t current[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    current[0] = (flip & 1) ? -1 : 1, current[4] = (flip & 2) ? -1 : 1, current[8] = (flip & 4) ? -1 : 1;
    nk_f32_t const flip_determinant = current[0] * current[4] * current[8];
    *best = NUMKONG_F32_INF;
    for (int cell = 0; cell != 9; ++cell) best_rotation[cell] = current[cell];
    int const simplex = network->rows != network->columns;
    nk_shapes_network_reset_simt_(network);
    if (simplex && network->thread == 0) nk_shapes_tree_northwest_simt_(network);
    for (int round = 0;; ++round) {
        nk_shapes_rotate_columns_f16_simt_(network, points, column_axes, axis_stride, network->columns, current);
        if (simplex) nk_shapes_simplex_f16_simt_(network, points, parity);
        else nk_shapes_transport_f16_simt_(network, points, round == 0, parity);
        nk_f32_t cross_covariance[9];
        nk_f32_t const cost = nk_shapes_plan_moments_f16_simt_(network, points, column_axes, axis_stride,
                                                               cross_covariance);
        if (!(cost < *best)) return;
        *best = cost;
        for (int cell = 0; cell != 9; ++cell) best_rotation[cell] = current[cell];
        if (saved_plan) nk_shapes_network_save_simt_(network, saved_plan);
        if (threadIdx.x == 0) {
            nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9], fitted[9];
            nk_svd3x3_f32_simt_(cross_covariance, svd_left, svd_diagonal, svd_right);
            nk_rotation_from_svd_f32_simt_(svd_left, svd_right, fitted);
            // Each start stays among the rotations, or among the reflections, it began with
            if (nk_det3x3_f32_simt_(fitted) * flip_determinant < 0) {
                svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
                nk_rotation_from_svd_f32_simt_(svd_left, svd_right, fitted);
            }
            for (int cell = 0; cell != 9; ++cell) shared_rotation[cell] = fitted[cell];
        }
        __syncthreads();
        for (int cell = 0; cell != 9; ++cell) current[cell] = shared_rotation[cell];
        __syncthreads();
    }
}

#pragma endregion Transport F16

#pragma region Gromov Wasserstein

/** One Gromov-Wasserstein start's dense F64 state in global memory: the plan, its gradient, a
 *  vertex's gradient and the vertex times the column distances, rows × columns each, then the
 *  distances within each cloud. */
typedef struct {
    nk_f64_t *plan;
    nk_f64_t *gradient;
    nk_f64_t *vertex_gradient;
    nk_f64_t *products;
    nk_f64_t *row_distances;
    nk_f64_t *column_distances;
} nk_shapes_gromov_simt_t;

/** Bytes of one start's dense state for clouds of up to @p largest points. */
NUMKONG_INLINE nk_size_t nk_shapes_gromov_bytes_simt_(nk_size_t largest) {
    return 6 * largest * largest * sizeof(nk_f64_t);
}

NUMKONG_DEVICE void nk_shapes_gromov_carve_simt_(void *memory, nk_u32_t largest, nk_shapes_gromov_simt_t *gromov) {
    nk_f64_t *doubles = (nk_f64_t *)memory;
    nk_size_t const cells = (nk_size_t)largest * largest;
    gromov->plan = doubles, gromov->gradient = doubles + cells, gromov->vertex_gradient = doubles + 2 * cells;
    gromov->products = doubles + 3 * cells, gromov->row_distances = doubles + 4 * cells;
    gromov->column_distances = doubles + 5 * cells;
}

/** An arc's cost in the Frank-Wolfe transport: the negated gradient. */
NUMKONG_DEVICE nk_f64_t nk_shapes_gradient_cost_simt_(nk_shapes_gromov_simt_t const *gromov, nk_u32_t columns,
                                                      nk_u32_t row, nk_u32_t column) {
    return -gromov->gradient[(nk_size_t)row * columns + column];
}

NUMKONG_DEVICE void nk_shapes_seed_gradient_simt_(nk_shapes_network_simt_t *network,
                                                  nk_shapes_gromov_simt_t const *gromov) {
    unsigned const thread = network->thread, team = network->team;
    nk_u32_t const columns = network->columns;
    for (nk_u32_t column = thread; column < columns; column += team) {
        nk_f64_t best = NUMKONG_F64_INF;
        nk_u32_t best_row = 0;
        for (nk_u32_t row = 0; row != network->rows; ++row) {
            nk_f64_t const cost = nk_shapes_gradient_cost_simt_(gromov, columns, row, column);
            if (cost < best) best = cost, best_row = row;
        }
        network->column_potentials[column] = best, network->column_predecessors[column] = best_row;
    }
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t row = thread; row < network->rows; row += team) {
        nk_f64_t best = NUMKONG_F64_INF;
        for (nk_u32_t column = 0; column != columns; ++column) {
            nk_f64_t const reduced = nk_shapes_gradient_cost_simt_(gromov, columns, row, column) -
                                     network->column_potentials[column];
            best = reduced < best ? reduced : best;
        }
        network->row_potentials[row] = best;
    }
    nk_shapes_team_sync_simt_(network);
    if (thread == 0)
        for (nk_u32_t column = 0; column != columns; ++column) {
            nk_u32_t const row = network->column_predecessors[column];
            nk_f64_t const reduced = nk_shapes_gradient_cost_simt_(gromov, columns, row, column) -
                                     network->column_potentials[column] - network->row_potentials[row];
            nk_u32_t const supply = network->row_supplies[row], demand = network->column_demands[column];
            nk_u32_t const amount = supply < demand ? supply : demand;
            if (!(reduced <= 0) || !amount) continue;
            nk_shapes_network_ship_simt_(network, row, column, amount);
            network->row_supplies[row] -= amount, network->column_demands[column] -= amount;
        }
    nk_shapes_team_sync_simt_(network);
}

NUMKONG_DEVICE void nk_shapes_reseed_gradient_simt_(nk_shapes_network_simt_t *network,
                                                    nk_shapes_gromov_simt_t const *gromov) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    unsigned const thread = network->thread, team = network->team;
    nk_u32_t const rows = network->rows, columns = network->columns;
    if (thread == 0) {
        nk_shapes_network_index_rows_simt_(network);
        nk_u32_t head = 0, tail = 0;
        for (nk_u32_t root = 0; root != rows; ++root) {
            if (network->row_reached_columns[root] || network->row_arcs[root] == no_node) continue;
            network->row_reached_columns[root] = 1, network->visits[tail++] = root;
            while (head != tail) {
                nk_u32_t const node = network->visits[head++];
                if (node < rows) {
                    for (nk_u32_t arc = network->row_arcs[node]; arc != no_node; arc = network->arc_row_next[arc]) {
                        nk_u32_t const column = network->arc_columns[arc];
                        if (network->column_closed[column]) continue;
                        network->column_closed[column] = 1, network->visits[tail++] = rows + column;
                        network->column_potentials[column] = nk_shapes_gradient_cost_simt_(gromov, columns, node,
                                                                                           column) -
                                                             network->row_potentials[node];
                    }
                }
                else {
                    nk_u32_t const column = node - rows;
                    for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc]) {
                        nk_u32_t const row = network->arc_rows[arc];
                        if (network->row_reached_columns[row]) continue;
                        network->row_reached_columns[row] = 1, network->visits[tail++] = row;
                        network->row_potentials[row] = nk_shapes_gradient_cost_simt_(gromov, columns, row, column) -
                                                       network->column_potentials[column];
                    }
                }
            }
        }
    }
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t row = thread; row < rows; row += team) {
        nk_f64_t best = NUMKONG_F64_INF;
        for (nk_u32_t column = 0; column != columns; ++column) {
            nk_f64_t const reduced = nk_shapes_gradient_cost_simt_(gromov, columns, row, column) -
                                     network->column_potentials[column];
            best = reduced < best ? reduced : best;
        }
        nk_f64_t const potential = network->row_potentials[row];
        nk_f64_t const scale = (best < 0 ? -best : best) + (potential < 0 ? -potential : potential);
        int const undercut = best < potential - 1e-12 * scale;
        network->row_reached_columns[row] = undercut ? 2 : 0;
        network->row_potentials[row] = best < potential ? best : potential;
    }
    nk_shapes_team_sync_simt_(network);
    if (thread == 0) nk_shapes_network_drop_rows_simt_(network);
    nk_shapes_team_sync_simt_(network);
}

NUMKONG_DEVICE nk_u32_t nk_shapes_shortest_path_gradient_simt_(nk_shapes_network_simt_t *network,
                                                               nk_shapes_gromov_simt_t const *gromov, nk_u32_t source,
                                                               nk_f64_t *nearest, unsigned *parity) {
    nk_u32_t const no_node = 0xFFFFFFFFu, source_marker = 0xFFFFFFFEu;
    unsigned const thread = network->thread, team = network->team;
    nk_u32_t const columns = network->columns;
    nk_f64_t const source_potential = network->row_potentials[source];
    for (nk_u32_t column = thread; column < columns; column += team) {
        network->distances[column] = nk_shapes_gradient_cost_simt_(gromov, columns, source, column) -
                                     network->column_potentials[column] - source_potential;
        network->column_predecessors[column] = source, network->column_closed[column] = 0;
    }
    for (nk_u32_t row = thread; row < network->rows; row += team)
        network->row_reached_columns[row] = row == source ? source_marker : no_node;
    for (;;) {
        nk_f64_t value = NUMKONG_F64_INF;
        nk_u32_t index = no_node;
        for (nk_u32_t column = thread; column < columns; column += team)
            if (!network->column_closed[column] && network->distances[column] < value)
                value = network->distances[column], index = column;
        nk_shapes_team_argmin_simt_(network, &value, &index, parity);
        if (index % team == thread) network->column_closed[index] = 1;
        if (network->column_demands[index]) {
            *nearest = value;
            nk_shapes_team_sync_simt_(network);
            return index;
        }
        for (nk_u32_t arc = network->column_arcs[index]; arc != no_node; arc = network->arc_next[arc]) {
            nk_u32_t const row = network->arc_rows[arc];
            if (thread == 0 && network->row_reached_columns[row] == no_node)
                network->row_reached_columns[row] = index, network->row_reached_arcs[row] = arc;
            nk_f64_t const row_potential = network->row_potentials[row];
            for (nk_u32_t column = thread; column < columns; column += team) {
                if (network->column_closed[column]) continue;
                nk_f64_t const candidate = value + (nk_shapes_gradient_cost_simt_(gromov, columns, row, column) -
                                                    network->column_potentials[column] - row_potential);
                if (candidate < network->distances[column])
                    network->distances[column] = candidate, network->column_predecessors[column] = row;
            }
        }
    }
}

/** The transport plan of least cost −G, the vertex of a Frank-Wolfe step. */
NUMKONG_DEVICE void nk_shapes_transport_gradient_simt_(nk_shapes_network_simt_t *network,
                                                       nk_shapes_gromov_simt_t const *gromov, int first,
                                                       unsigned *parity) {
    if (first) nk_shapes_seed_gradient_simt_(network, gromov);
    else nk_shapes_reseed_gradient_simt_(network, gromov);
    for (nk_u32_t source = 0; source != network->rows; ++source)
        while (network->row_supplies[source]) {
            nk_f64_t nearest;
            nk_u32_t const sink = nk_shapes_shortest_path_gradient_simt_(network, gromov, source, &nearest, parity);
            nk_shapes_network_lift_simt_(network, nearest);
            if (network->thread == 0) nk_shapes_network_augment_simt_(network, source, sink);
            nk_shapes_team_sync_simt_(network);
        }
}

NUMKONG_DEVICE void nk_shapes_tree_potentials_gradient_simt_(nk_shapes_network_simt_t *network,
                                                             nk_shapes_gromov_simt_t const *gromov) {
    nk_u32_t const no_node = 0xFFFFFFFFu, row_nodes = network->rows, columns = network->columns;
    nk_u32_t node = 0;
    network->tree_potentials[0] = 0;
    for (;;) {
        if (network->tree_first_children[node] != no_node) node = network->tree_first_children[node];
        else {
            while (node && network->tree_next_siblings[node] == no_node) node = network->tree_parents[node];
            if (!node) return;
            node = network->tree_next_siblings[node];
        }
        nk_u32_t const above = network->tree_parents[node];
        if (node < row_nodes)
            network->tree_potentials[node] = network->tree_potentials[above] +
                                             nk_shapes_gradient_cost_simt_(gromov, columns, node, above - row_nodes);
        else
            network->tree_potentials[node] = network->tree_potentials[above] -
                                             nk_shapes_gradient_cost_simt_(gromov, columns, above, node - row_nodes);
    }
}

/** The Frank-Wolfe vertex by the serial network simplex over the costs −G, team-priced. */
NUMKONG_DEVICE void nk_shapes_simplex_gradient_simt_(nk_shapes_network_simt_t *network,
                                                     nk_shapes_gromov_simt_t const *gromov, unsigned *parity) {
    nk_u32_t const no_node = 0xFFFFFFFFu, rows = network->rows, columns = network->columns;
    unsigned const thread = network->thread, team = network->team;
    nk_size_t const arcs = (nk_size_t)rows * columns;
    nk_f64_t const tolerance = network->tolerance;
    nk_size_t block = 10, cursor = 0;
    while (block * block < arcs) ++block;
    if (thread == 0) nk_shapes_tree_potentials_gradient_simt_(network, gromov);
    nk_shapes_team_sync_simt_(network);
    for (;;) {
        nk_f64_t entering_cost = -tolerance;
        nk_size_t entering = arcs;
        for (nk_size_t scanned = 0; scanned < arcs;) {
            nk_size_t const count = arcs - scanned < block ? arcs - scanned : block;
            nk_f64_t value = -tolerance;
            nk_u32_t index = no_node;
            for (nk_size_t offset = thread; offset < count; offset += team) {
                nk_size_t position = cursor + offset;
                position = position >= arcs ? position - arcs : position;
                nk_u32_t const row = (nk_u32_t)(position / columns), column = (nk_u32_t)(position % columns);
                nk_f64_t const reduced = nk_shapes_gradient_cost_simt_(gromov, columns, row, column) -
                                         network->tree_potentials[row] + network->tree_potentials[rows + column];
                if (reduced < value) value = reduced, index = (nk_u32_t)offset;
            }
            nk_shapes_team_argmin_simt_(network, &value, &index, parity);
            nk_size_t const block_start = cursor;
            cursor += count, cursor = cursor >= arcs ? cursor - arcs : cursor;
            scanned += count;
            if (value < -tolerance) {
                entering_cost = value, entering = block_start + index;
                entering = entering >= arcs ? entering - arcs : entering;
                break;
            }
        }
        if (entering == arcs) break;
        if (thread == 0)
            nk_shapes_tree_pivot_simt_(network, (nk_u32_t)(entering / columns), (nk_u32_t)(entering % columns),
                                       entering_cost);
        nk_shapes_team_sync_simt_(network);
    }
    if (thread == 0) nk_shapes_tree_plan_simt_(network);
    nk_shapes_team_sync_simt_(network);
}

/** Dx × V × Dy into @c vertex_gradient for the plan V in @p network : first V × Dy into
 *  @c products from the arcs, then the product by the symmetric Dx, each thread a 4 × 4 tile. */
NUMKONG_DEVICE void nk_shapes_vertex_gradient_simt_(nk_shapes_network_simt_t const *network,
                                                    nk_shapes_gromov_simt_t *gromov) {
    nk_u32_t const no_node = 0xFFFFFFFFu, rows = network->rows, columns = network->columns;
    unsigned const thread = network->thread, team = network->team;
    nk_f64_t const mass = (nk_f64_t)((nk_size_t)rows * columns);
    for (nk_size_t cell = thread; cell < (nk_size_t)rows * columns; cell += team) gromov->products[cell] = 0;
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t column = thread; column < columns; column += team)
        for (nk_u32_t arc_column = 0; arc_column != columns; ++arc_column)
            for (nk_u32_t arc = network->column_arcs[arc_column]; arc != no_node; arc = network->arc_next[arc]) {
                nk_f64_t const weight = (nk_f64_t)network->arc_flows[arc] / mass;
                gromov->products[(nk_size_t)network->arc_rows[arc] * columns + column] +=
                    weight * gromov->column_distances[(nk_size_t)arc_column * columns + column];
            }
    nk_shapes_team_sync_simt_(network);
    nk_u32_t const tile_rows = (nk_u32_t)nk_size_divide_round_up_(rows, 4),
                   tile_columns = (nk_u32_t)nk_size_divide_round_up_(columns, 4);
    for (nk_u32_t tile = thread; tile < tile_rows * tile_columns; tile += team) {
        nk_u32_t const row_begin = tile / tile_columns * 4, column_begin = tile % tile_columns * 4;
        nk_f64_t sums[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
        for (nk_u32_t inner = 0; inner != rows; ++inner) {
            nk_f64_t left[4], right[4];
            for (int lane = 0; lane != 4; ++lane) {
                left[lane] = row_begin + lane < rows ? gromov->row_distances[(nk_size_t)inner * rows + row_begin + lane]
                                                     : 0;
                right[lane] = column_begin + lane < columns
                                  ? gromov->products[(nk_size_t)inner * columns + column_begin + lane]
                                  : 0;
            }
            for (int row = 0; row != 4; ++row)
                for (int column = 0; column != 4; ++column) sums[row][column] += left[row] * right[column];
        }
        for (int row = 0; row != 4; ++row)
            for (int column = 0; column != 4; ++column)
                if (row_begin + row < rows && column_begin + column < columns)
                    gromov->vertex_gradient[(nk_size_t)(row_begin + row) * columns + column_begin + column] =
                        sums[row][column];
    }
    nk_shapes_team_sync_simt_(network);
}

/** ⟨G, P⟩ over the dense plan and gradient, the same in every thread. */
NUMKONG_DEVICE nk_f64_t nk_shapes_plan_score_simt_(nk_shapes_network_simt_t *network,
                                                   nk_shapes_gromov_simt_t const *gromov, unsigned *parity) {
    nk_f64_t score[1] = {0}, largest = 0;
    for (nk_size_t cell = network->thread; cell < (nk_size_t)network->rows * network->columns; cell += network->team) {
        nk_f64_t const gradient = gromov->gradient[cell];
        score[0] += gradient * gromov->plan[cell];
        nk_f64_t const magnitude = gradient < 0 ? -gradient : gradient;
        largest = magnitude > largest ? magnitude : largest;
    }
    nk_shapes_team_sum_f64_simt_(network, score, 1);
    // The largest magnitude, the least negated one, scales the next simplex's tolerance
    nk_f64_t negated = -largest;
    nk_u32_t ignored = 0;
    nk_shapes_team_argmin_simt_(network, &negated, &ignored, parity);
    network->tolerance = -negated * 1e-12;
    return score[0];
}

/** Σ G × w and Σ G_vertex × w over the vertex's arcs, the line search's score and curvature. */
NUMKONG_DEVICE void nk_shapes_vertex_scores_simt_(nk_shapes_network_simt_t *network,
                                                  nk_shapes_gromov_simt_t const *gromov, nk_f64_t *scores) {
    nk_u32_t const no_node = 0xFFFFFFFFu, columns = network->columns;
    nk_f64_t const mass = (nk_f64_t)((nk_size_t)network->rows * columns);
    scores[0] = 0, scores[1] = 0;
    for (nk_u32_t column = network->thread; column < columns; column += network->team)
        for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc]) {
            nk_size_t const cell = (nk_size_t)network->arc_rows[arc] * columns + column;
            nk_f64_t const weight = (nk_f64_t)network->arc_flows[arc] / mass;
            scores[0] += gromov->gradient[cell] * weight, scores[1] += gromov->vertex_gradient[cell] * weight;
        }
    nk_shapes_team_sum_f64_simt_(network, scores, 2);
}

/** Moves the plan and its gradient a Frank-Wolfe @p step toward the vertex in @p network . */
NUMKONG_DEVICE void nk_shapes_plan_step_simt_(nk_shapes_network_simt_t const *network, nk_shapes_gromov_simt_t *gromov,
                                              nk_f64_t step) {
    nk_u32_t const no_node = 0xFFFFFFFFu, columns = network->columns;
    nk_f64_t const keep = 1 - step, mass = (nk_f64_t)((nk_size_t)network->rows * columns);
    for (nk_size_t cell = network->thread; cell < (nk_size_t)network->rows * columns; cell += network->team)
        gromov->plan[cell] *= keep,
            gromov->gradient[cell] = keep * gromov->gradient[cell] + step * gromov->vertex_gradient[cell];
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t column = network->thread; column < columns; column += network->team)
        for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc])
            gromov->plan[(nk_size_t)network->arc_rows[arc] * columns + column] +=
                step * ((nk_f64_t)network->arc_flows[arc] / mass);
    nk_shapes_team_sync_simt_(network);
}

/** The product plan and its gradient, the outer product of the two matrices' row sums. */
NUMKONG_DEVICE void nk_shapes_product_start_simt_(nk_shapes_network_simt_t const *network,
                                                  nk_shapes_gromov_simt_t *gromov) {
    nk_u32_t const rows = network->rows, columns = network->columns;
    nk_f64_t const mass = (nk_f64_t)((nk_size_t)rows * columns);
    // The vertex gradient holds the column sums until the steps need it
    for (nk_u32_t column = network->thread; column < columns; column += network->team) {
        nk_f64_t column_sum = 0;
        for (nk_u32_t other = 0; other != columns; ++other)
            column_sum += gromov->column_distances[(nk_size_t)other * columns + column];
        gromov->vertex_gradient[column] = column_sum;
    }
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t row = network->thread; row < rows; row += network->team) {
        nk_f64_t row_sum = 0;
        for (nk_u32_t other = 0; other != rows; ++other)
            row_sum += gromov->row_distances[(nk_size_t)row * rows + other];
        for (nk_u32_t column = 0; column != columns; ++column) {
            gromov->plan[(nk_size_t)row * columns + column] = 1 / mass;
            gromov->gradient[(nk_size_t)row * columns + column] = row_sum * gromov->vertex_gradient[column] / mass;
        }
    }
    nk_shapes_team_sync_simt_(network);
}

/** Starts from the plan in @p network : its dense copy and its gradient. */
NUMKONG_DEVICE void nk_shapes_plan_start_simt_(nk_shapes_network_simt_t const *network,
                                               nk_shapes_gromov_simt_t *gromov) {
    nk_u32_t const no_node = 0xFFFFFFFFu, columns = network->columns;
    nk_f64_t const mass = (nk_f64_t)((nk_size_t)network->rows * columns);
    for (nk_size_t cell = network->thread; cell < (nk_size_t)network->rows * columns; cell += network->team)
        gromov->plan[cell] = 0;
    nk_shapes_team_sync_simt_(network);
    for (nk_u32_t column = network->thread; column < columns; column += network->team)
        for (nk_u32_t arc = network->column_arcs[column]; arc != no_node; arc = network->arc_next[arc])
            gromov->plan[(nk_size_t)network->arc_rows[arc] * columns + column] = (nk_f64_t)network->arc_flows[arc] /
                                                                                 mass;
    nk_shapes_vertex_gradient_simt_(network, gromov);
    for (nk_size_t cell = network->thread; cell < (nk_size_t)network->rows * columns; cell += network->team)
        gromov->gradient[cell] = gromov->vertex_gradient[cell];
    nk_shapes_team_sync_simt_(network);
}

/** Loads a plan saved by @c nk_shapes_network_save_simt_ into the emptied @p network . */
NUMKONG_DEVICE void nk_shapes_network_load_simt_(nk_shapes_network_simt_t *network, nk_u32_t const *saved,
                                                 nk_u32_t saved_arcs) {
    nk_shapes_network_reset_simt_(network);
    if (network->thread == 0)
        for (nk_u32_t arc = 0; arc != saved_arcs; ++arc) {
            nk_u32_t const flow = saved[2 * saved_arcs + arc];
            if (!flow) continue;
            nk_u32_t const row = saved[arc], column = saved[saved_arcs + arc];
            nk_shapes_network_ship_simt_(network, row, column, flow);
            network->row_supplies[row] -= flow, network->column_demands[column] -= flow;
        }
    nk_shapes_team_sync_simt_(network);
}

/** Frank-Wolfe from the start already in @p gromov : up to 32 steps toward exact transport
 *  vertices with closed-form line searches, as the serial solver takes them, returning ⟨G, P⟩. */
NUMKONG_DEVICE nk_f64_t nk_shapes_frank_wolfe_simt_(nk_shapes_network_simt_t *network, nk_shapes_gromov_simt_t *gromov,
                                                    unsigned *parity) {
    nk_f64_t plan_score = nk_shapes_plan_score_simt_(network, gromov, parity);
    int const simplex = network->rows != network->columns;
    nk_shapes_network_reset_simt_(network);
    if (simplex && network->thread == 0) nk_shapes_tree_northwest_simt_(network);
    for (int step_index = 0; step_index != 32; ++step_index) {
        if (simplex) nk_shapes_simplex_gradient_simt_(network, gromov, parity);
        else nk_shapes_transport_gradient_simt_(network, gromov, step_index == 0, parity);
        nk_shapes_vertex_gradient_simt_(network, gromov);
        nk_f64_t scores[2];
        nk_shapes_vertex_scores_simt_(network, gromov, scores);
        nk_f64_t const slope = scores[0] - plan_score;
        nk_f64_t const curvature = scores[1] - 2 * scores[0] + plan_score;
        nk_f64_t const quadratic = -2 * curvature, linear = -4 * slope;
        nk_f64_t step = quadratic + linear < 0 ? 1 : 0;
        if (quadratic > 0) {
            step = -linear / (2 * quadratic);
            step = step < 0 ? 0 : step > 1 ? 1 : step;
        }
        if (step == 0) break;
        nk_shapes_plan_step_simt_(network, gromov, step);
        plan_score = nk_shapes_plan_score_simt_(network, gromov, parity);
    }
    return plan_score;
}

/** Σ Dᵢⱼ² / count² of one cloud's distances, the same in every thread. */
NUMKONG_DEVICE nk_f64_t nk_shapes_distance_spread_simt_(nk_shapes_network_simt_t *network, nk_f64_t const *distances,
                                                        nk_u32_t count) {
    nk_f64_t spread[1] = {0};
    for (nk_size_t cell = network->thread; cell < (nk_size_t)count * count; cell += network->team)
        spread[0] += distances[cell] * distances[cell];
    nk_shapes_team_sum_f64_simt_(network, spread, 1);
    return spread[0] / (nk_f64_t)((nk_size_t)count * count);
}

/** The distances within a cloud of f32 points, as @c nk_shapes_distances_f64_serial_ sums them. */
NUMKONG_DEVICE void nk_shapes_distances_f32_simt_(nk_shapes_network_simt_t const *network, nk_f32_t const *axes,
                                                  nk_size_t axis_stride, nk_u32_t count, nk_f64_t *distances) {
    for (nk_size_t cell = network->thread; cell < (nk_size_t)count * count; cell += network->team) {
        nk_size_t const first = cell / count, second = cell % count;
        nk_f64_t const delta_x = (nk_f64_t)axes[first] - (nk_f64_t)axes[second];
        nk_f64_t const delta_y = (nk_f64_t)axes[axis_stride + first] - (nk_f64_t)axes[axis_stride + second];
        nk_f64_t const delta_z = (nk_f64_t)axes[2 * axis_stride + first] - (nk_f64_t)axes[2 * axis_stride + second];
        distances[cell] = nk_sqrt_f64_simt_(__dadd_rn(
            __dadd_rn(__dmul_rn(delta_x, delta_x), __dmul_rn(delta_y, delta_y)), __dmul_rn(delta_z, delta_z)));
    }
    nk_shapes_team_sync_simt_(network);
}

NUMKONG_DEVICE void nk_shapes_distances_f16_simt_(nk_shapes_network_simt_t const *network, nk_f16_t const *axes,
                                                  nk_size_t axis_stride, nk_u32_t count, nk_f64_t *distances) {
    for (nk_size_t cell = network->thread; cell < (nk_size_t)count * count; cell += network->team) {
        nk_size_t const first = cell / count, second = cell % count;
        nk_f32_t first_x, first_y, first_z, second_x, second_y, second_z;
        nk_f16_to_f32_simt_(axes + first, &first_x);
        nk_f16_to_f32_simt_(axes + axis_stride + first, &first_y);
        nk_f16_to_f32_simt_(axes + 2 * axis_stride + first, &first_z);
        nk_f16_to_f32_simt_(axes + second, &second_x);
        nk_f16_to_f32_simt_(axes + axis_stride + second, &second_y);
        nk_f16_to_f32_simt_(axes + 2 * axis_stride + second, &second_z);
        nk_f64_t const delta_x = (nk_f64_t)first_x - (nk_f64_t)second_x;
        nk_f64_t const delta_y = (nk_f64_t)first_y - (nk_f64_t)second_y;
        nk_f64_t const delta_z = (nk_f64_t)first_z - (nk_f64_t)second_z;
        distances[cell] = nk_sqrt_f64_simt_(__dadd_rn(
            __dadd_rn(__dmul_rn(delta_x, delta_x), __dmul_rn(delta_y, delta_y)), __dmul_rn(delta_z, delta_z)));
    }
    nk_shapes_team_sync_simt_(network);
}

/** The orthogonal Procrustes fit of the dense plan over f32 points, in F64 with the determinant's
 *  sign moved into V, as the serial fit takes it. Left in @p fit in every thread. */
NUMKONG_DEVICE void nk_shapes_plan_fit_f32_simt_(nk_shapes_network_simt_t *network,
                                                 nk_shapes_gromov_simt_t const *gromov, nk_f32_t const *row_axes,
                                                 nk_size_t row_stride, nk_f32_t const *column_axes,
                                                 nk_size_t column_stride, nk_f64_t *shared_fit, nk_f64_t *fit) {
    nk_u32_t const rows = network->rows, columns = network->columns;
    nk_f64_t sums[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (nk_u32_t column = network->thread; column < columns; column += network->team) {
        nk_f64_t const column_point[3] = {column_axes[column], column_axes[column_stride + column],
                                          column_axes[2 * column_stride + column]};
        for (nk_u32_t row = 0; row != rows; ++row) {
            nk_f64_t const weight = gromov->plan[(nk_size_t)row * columns + column];
            nk_f64_t const row_point[3] = {row_axes[row], row_axes[row_stride + row], row_axes[2 * row_stride + row]};
            for (int column_axis = 0; column_axis != 3; ++column_axis) {
                nk_f64_t const weighted = weight * column_point[column_axis];
                for (int row_axis = 0; row_axis != 3; ++row_axis)
                    sums[column_axis * 3 + row_axis] += weighted * row_point[row_axis];
            }
        }
    }
    nk_shapes_team_sum_f64_simt_(network, sums, 9);
    if (threadIdx.x == 0) {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_simt_(sums, svd_left, svd_diagonal, svd_right);
        // The McAdams factors are proper, so det H's sign moves into V to fit over O(3)
        if (nk_det3x3_f64_simt_(sums) < 0)
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f64_simt_(svd_left, svd_right, shared_fit);
    }
    __syncthreads();
    for (int cell = 0; cell != 9; ++cell) fit[cell] = shared_fit[cell];
    __syncthreads();
}

NUMKONG_DEVICE void nk_shapes_plan_fit_f16_simt_(nk_shapes_network_simt_t *network,
                                                 nk_shapes_gromov_simt_t const *gromov, nk_f16_t const *row_axes,
                                                 nk_size_t row_stride, nk_f16_t const *column_axes,
                                                 nk_size_t column_stride, nk_f64_t *shared_fit, nk_f64_t *fit) {
    nk_u32_t const rows = network->rows, columns = network->columns;
    nk_f64_t sums[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (nk_u32_t column = network->thread; column < columns; column += network->team) {
        nk_f32_t column_point[3];
        nk_f16_to_f32_simt_(column_axes + column, &column_point[0]);
        nk_f16_to_f32_simt_(column_axes + column_stride + column, &column_point[1]);
        nk_f16_to_f32_simt_(column_axes + 2 * column_stride + column, &column_point[2]);
        for (nk_u32_t row = 0; row != rows; ++row) {
            nk_f64_t const weight = gromov->plan[(nk_size_t)row * columns + column];
            nk_f32_t row_point[3];
            nk_f16_to_f32_simt_(row_axes + row, &row_point[0]);
            nk_f16_to_f32_simt_(row_axes + row_stride + row, &row_point[1]);
            nk_f16_to_f32_simt_(row_axes + 2 * row_stride + row, &row_point[2]);
            for (int column_axis = 0; column_axis != 3; ++column_axis) {
                nk_f64_t const weighted = weight * (nk_f64_t)column_point[column_axis];
                for (int row_axis = 0; row_axis != 3; ++row_axis)
                    sums[column_axis * 3 + row_axis] += weighted * (nk_f64_t)row_point[row_axis];
            }
        }
    }
    nk_shapes_team_sum_f64_simt_(network, sums, 9);
    if (threadIdx.x == 0) {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_simt_(sums, svd_left, svd_diagonal, svd_right);
        // The McAdams factors are proper, so det H's sign moves into V to fit over O(3)
        if (nk_det3x3_f64_simt_(sums) < 0)
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f64_simt_(svd_left, svd_right, shared_fit);
    }
    __syncthreads();
    for (int cell = 0; cell != 9; ++cell) fit[cell] = shared_fit[cell];
    __syncthreads();
}

#pragma endregion Gromov Wasserstein

#pragma region Chamfer and Hausdorff

/** The x array of a pack of F32 points, with the elements from each axis to the next. */
NUMKONG_DEVICE nk_f32_t const *nk_shapes_packed_axes_f32_simt_(void const *packed, nk_size_t *axis_stride) {
    nk_size_t const points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)packed)->points;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f64_t),
                                                                 nk_shapes_alignment_k_);
    *axis_stride = nk_size_round_up_to_multiple_(points * sizeof(nk_f32_t), nk_shapes_alignment_k_) / sizeof(nk_f32_t);
    return (nk_f32_t const *)((char const *)packed + header_bytes);
}

/** The x array of a pack of F16 points, with the elements from each axis to the next. */
NUMKONG_DEVICE nk_f16_t const *nk_shapes_packed_axes_f16_simt_(void const *packed, nk_size_t *axis_stride) {
    nk_size_t const points = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f32_t),
                                                                 nk_shapes_alignment_k_);
    *axis_stride = nk_size_round_up_to_multiple_(points * sizeof(nk_f16_t), nk_shapes_alignment_k_) / sizeof(nk_f16_t);
    return (nk_f16_t const *)((char const *)packed + header_bytes);
}

/** One packed F16 coordinate in F32, exactly. Packs hold finite coordinates, so it skips the
 *  compare and select per value that @c nk_f16_to_f32_simt_ spends keeping NaN payloads. */
NUMKONG_DEVICE nk_f32_t nk_shapes_coordinate_f16_simt_(nk_f16_t const *coordinate) {
    return __half2float(__ushort_as_half(*(unsigned short const *)coordinate));
}

/** The least squared distance from the outer point to the inner points of F32 under each of the 8
 *  axis flips, widened to F64 and rounded in the serial order, so each equals the serial one. */
NUMKONG_DEVICE void nk_shapes_nearest_squares_f32_simt_(nk_f64_t outer_x, nk_f64_t outer_y, nk_f64_t outer_z,
                                                        nk_f32_t const *inner_axes, nk_size_t inner_stride,
                                                        nk_size_t inner_count, nk_f64_t *nearest) {
    for (int flip = 0; flip != 8; ++flip) nearest[flip] = NUMKONG_F64_INF;
    for (nk_size_t index = 0; index != inner_count; ++index) {
        nk_f64_t const inner_x = inner_axes[index], inner_y = inner_axes[inner_stride + index],
                       inner_z = inner_axes[2 * inner_stride + index];
        // A flip negates an inner axis, turning its difference into a sum
        nk_f64_t const minus_x = outer_x - inner_x, plus_x = outer_x + inner_x;
        nk_f64_t const minus_y = outer_y - inner_y, plus_y = outer_y + inner_y;
        nk_f64_t const minus_z = outer_z - inner_z, plus_z = outer_z + inner_z;
        nk_f64_t const minus_xx = __dmul_rn(minus_x, minus_x), plus_xx = __dmul_rn(plus_x, plus_x);
        nk_f64_t const minus_yy = __dmul_rn(minus_y, minus_y), plus_yy = __dmul_rn(plus_y, plus_y);
        nk_f64_t const minus_zz = __dmul_rn(minus_z, minus_z), plus_zz = __dmul_rn(plus_z, plus_z);
        nk_f64_t const planar[4] = {minus_xx + minus_yy, plus_xx + minus_yy, minus_xx + plus_yy, plus_xx + plus_yy};
        for (int flip = 0; flip != 8; ++flip) {
            nk_f64_t const distance_squared = planar[flip & 3] + ((flip & 4) ? plus_zz : minus_zz);
            nearest[flip] = distance_squared < nearest[flip] ? distance_squared : nearest[flip];
        }
    }
}

/** The least squared distance from the outer point to the inner points of F16 under each of the 8
 *  axis flips, widened to F32 and rounded in the serial order, so each equals the serial one. The
 *  minimum is @c fminf , one instruction where a select takes two, as no square is NaN or −0. */
NUMKONG_DEVICE void nk_shapes_nearest_squares_f16_simt_(nk_f32_t outer_x, nk_f32_t outer_y, nk_f32_t outer_z,
                                                        nk_f16_t const *inner_axes, nk_size_t inner_stride,
                                                        nk_size_t inner_count, nk_f32_t *nearest) {
    for (int flip = 0; flip != 8; ++flip) nearest[flip] = NUMKONG_F32_INF;
    for (nk_size_t index = 0; index != inner_count; ++index) {
        nk_f32_t const inner_x = nk_shapes_coordinate_f16_simt_(inner_axes + index);
        nk_f32_t const inner_y = nk_shapes_coordinate_f16_simt_(inner_axes + inner_stride + index);
        nk_f32_t const inner_z = nk_shapes_coordinate_f16_simt_(inner_axes + 2 * inner_stride + index);
        // A flip negates an inner axis, turning its difference into a sum
        nk_f32_t const minus_x = outer_x - inner_x, plus_x = outer_x + inner_x;
        nk_f32_t const minus_y = outer_y - inner_y, plus_y = outer_y + inner_y;
        nk_f32_t const minus_z = outer_z - inner_z, plus_z = outer_z + inner_z;
        nk_f32_t const minus_xx = __fmul_rn(minus_x, minus_x), plus_xx = __fmul_rn(plus_x, plus_x);
        nk_f32_t const minus_yy = __fmul_rn(minus_y, minus_y), plus_yy = __fmul_rn(plus_y, plus_y);
        nk_f32_t const minus_zz = __fmul_rn(minus_z, minus_z), plus_zz = __fmul_rn(plus_z, plus_z);
        nk_f32_t const planar[4] = {minus_xx + minus_yy, plus_xx + minus_yy, minus_xx + plus_yy, plus_xx + plus_yy};
        for (int flip = 0; flip != 8; ++flip)
            nearest[flip] = fminf(planar[flip & 3] + ((flip & 4) ? plus_zz : minus_zz), nearest[flip]);
    }
}

/** A lower bound of the Hausdorff distance between two packs of F32 points from their headers
 *  alone: the gap between their largest radii. */
NUMKONG_DEVICE nk_f64_t nk_hausdorff_bound_packed_f32_simt_(void const *a_packed, void const *b_packed) {
    nk_f64_t const gap = ((nk_shapes_packed_header_f64_t const *)a_packed)->max_radius -
                         ((nk_shapes_packed_header_f64_t const *)b_packed)->max_radius;
    return gap < 0 ? -gap : gap;
}

/** A lower bound of the Hausdorff distance between two packs of F16 points from their headers
 *  alone: the gap between their largest radii. */
NUMKONG_DEVICE nk_f32_t nk_hausdorff_bound_packed_f16_simt_(void const *a_packed, void const *b_packed) {
    nk_f32_t const gap = ((nk_shapes_packed_header_f32_t const *)a_packed)->max_radius -
                         ((nk_shapes_packed_header_f32_t const *)b_packed)->max_radius;
    return gap < 0 ? -gap : gap;
}

/** The Chamfer distance between two packs of F32 points on one thread, for clouds too small to
 *  split across a warp. Distances up to @p limit are exact; past it, the thread may stop early and
 *  return a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f64_t nk_chamfer_packed_f32_simt_(void const *a_packed, void const *b_packed, nk_f64_t limit) {
    nk_size_t const a_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed)->points;
    nk_size_t const b_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : NUMKONG_F64_INF;
    nk_size_t a_stride, b_stride;
    nk_f32_t const *a_axes = nk_shapes_packed_axes_f32_simt_(a_packed, &a_stride);
    nk_f32_t const *b_axes = nk_shapes_packed_axes_f32_simt_(b_packed, &b_stride);
    // Per flip, the finished directions' mean distances: F/n after one, F/n + B/m after both
    nk_f64_t means[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int direction = 0; direction != 2; ++direction) {
        nk_f32_t const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        nk_size_t const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        nk_size_t const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        nk_f64_t const count = (nk_f64_t)outer_count;
        nk_f64_t sums[8] = {0, 0, 0, 0, 0, 0, 0, 0}, nearest[8];
        for (nk_size_t point = 0; point != outer_count; ++point) {
            nk_shapes_nearest_squares_f32_simt_(outer_axes[point], outer_axes[outer_stride + point],
                                                outer_axes[2 * outer_stride + point], inner_axes, inner_stride,
                                                inner_count, nearest);
            int exceeded = 1;
            for (int flip = 0; flip != 8; ++flip) {
                sums[flip] += __dsqrt_rn(nearest[flip]);
                exceeded &= sums[flip] > (2 * limit - means[flip]) * count;
            }
            if (!exceeded) continue;
            nk_f64_t partial = NUMKONG_F64_INF;
            for (int flip = 0; flip != 8; ++flip) {
                nk_f64_t const chamfer = (means[flip] + sums[flip] / count) / 2;
                partial = chamfer < partial ? chamfer : partial;
            }
            if (partial > limit) return partial;
        }
        for (int flip = 0; flip != 8; ++flip) means[flip] += sums[flip] / count;
    }
    nk_f64_t best = means[0] / 2;
    for (int flip = 1; flip != 8; ++flip) best = means[flip] / 2 < best ? means[flip] / 2 : best;
    return best;
}

/** The Chamfer distance between two packs of F16 points on one thread, for clouds too small to
 *  split across a warp. Distances up to @p limit are exact; past it, the thread may stop early and
 *  return a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f32_t nk_chamfer_packed_f16_simt_(void const *a_packed, void const *b_packed, nk_f32_t limit) {
    nk_size_t const a_count = ((nk_shapes_packed_header_f32_t const *)a_packed)->points;
    nk_size_t const b_count = ((nk_shapes_packed_header_f32_t const *)b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : NUMKONG_F32_INF;
    nk_size_t a_stride, b_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_simt_(a_packed, &a_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_simt_(b_packed, &b_stride);
    // Per flip, the finished directions' mean distances: F/n after one, F/n + B/m after both
    nk_f32_t means[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int direction = 0; direction != 2; ++direction) {
        nk_f16_t const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        nk_size_t const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        nk_size_t const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        nk_f32_t const count = (nk_f32_t)outer_count;
        nk_f32_t sums[8] = {0, 0, 0, 0, 0, 0, 0, 0}, nearest[8];
        for (nk_size_t point = 0; point != outer_count; ++point) {
            nk_shapes_nearest_squares_f16_simt_(nk_shapes_coordinate_f16_simt_(outer_axes + point),
                                                nk_shapes_coordinate_f16_simt_(outer_axes + outer_stride + point),
                                                nk_shapes_coordinate_f16_simt_(outer_axes + 2 * outer_stride + point),
                                                inner_axes, inner_stride, inner_count, nearest);
            int exceeded = 1;
            for (int flip = 0; flip != 8; ++flip) {
                sums[flip] += __fsqrt_rn(nearest[flip]);
                exceeded &= sums[flip] > (2 * limit - means[flip]) * count;
            }
            if (!exceeded) continue;
            nk_f32_t partial = NUMKONG_F32_INF;
            for (int flip = 0; flip != 8; ++flip) {
                nk_f32_t const chamfer = (means[flip] + sums[flip] / count) / 2;
                partial = chamfer < partial ? chamfer : partial;
            }
            if (partial > limit) return partial;
        }
        for (int flip = 0; flip != 8; ++flip) means[flip] += sums[flip] / count;
    }
    nk_f32_t best = means[0] / 2;
    for (int flip = 1; flip != 8; ++flip) best = means[flip] / 2 < best ? means[flip] / 2 : best;
    return best;
}

/** The Hausdorff distance between two packs of F32 points on one thread, for clouds too small to
 *  split across a warp. Distances up to @p limit are exact; past it, the thread may return early,
 *  even before reading a point, with a lower bound that also lies past @p limit . A square root
 *  keeps the order of its inputs, so the farthest nearest point is found among squares. */
NUMKONG_DEVICE nk_f64_t nk_hausdorff_packed_f32_simt_(void const *a_packed, void const *b_packed, nk_f64_t limit) {
    nk_size_t const a_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed)->points;
    nk_size_t const b_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : NUMKONG_F64_INF;
    nk_f64_t const bound = nk_hausdorff_bound_packed_f32_simt_(a_packed, b_packed);
    if (bound > limit) return bound;
    nk_size_t a_stride, b_stride;
    nk_f32_t const *a_axes = nk_shapes_packed_axes_f32_simt_(a_packed, &a_stride);
    nk_f32_t const *b_axes = nk_shapes_packed_axes_f32_simt_(b_packed, &b_stride);
    nk_f64_t const limit_squared = limit * limit;
    nk_f64_t farthest[8] = {0, 0, 0, 0, 0, 0, 0, 0}, nearest[8];
    for (int direction = 0; direction != 2; ++direction) {
        nk_f32_t const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        nk_size_t const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        nk_size_t const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        for (nk_size_t point = 0; point != outer_count; ++point) {
            nk_shapes_nearest_squares_f32_simt_(outer_axes[point], outer_axes[outer_stride + point],
                                                outer_axes[2 * outer_stride + point], inner_axes, inner_stride,
                                                inner_count, nearest);
            nk_f64_t least = NUMKONG_F64_INF;
            for (int flip = 0; flip != 8; ++flip) {
                farthest[flip] = nearest[flip] > farthest[flip] ? nearest[flip] : farthest[flip];
                least = farthest[flip] < least ? farthest[flip] : least;
            }
            if (least > limit_squared && __dsqrt_rn(least) > limit) return __dsqrt_rn(least);
        }
    }
    nk_f64_t least = farthest[0];
    for (int flip = 1; flip != 8; ++flip) least = farthest[flip] < least ? farthest[flip] : least;
    return __dsqrt_rn(least);
}

/** The Hausdorff distance between two packs of F16 points on one thread, for clouds too small to
 *  split across a warp. Distances up to @p limit are exact; past it, the thread may return early,
 *  even before reading a point, with a lower bound that also lies past @p limit . A square root
 *  keeps the order of its inputs, so the farthest nearest point is found among squares. */
NUMKONG_DEVICE nk_f32_t nk_hausdorff_packed_f16_simt_(void const *a_packed, void const *b_packed, nk_f32_t limit) {
    nk_size_t const a_count = ((nk_shapes_packed_header_f32_t const *)a_packed)->points;
    nk_size_t const b_count = ((nk_shapes_packed_header_f32_t const *)b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : NUMKONG_F32_INF;
    nk_f32_t const bound = nk_hausdorff_bound_packed_f16_simt_(a_packed, b_packed);
    if (bound > limit) return bound;
    nk_size_t a_stride, b_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_simt_(a_packed, &a_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_simt_(b_packed, &b_stride);
    nk_f32_t const limit_squared = limit * limit;
    nk_f32_t farthest[8] = {0, 0, 0, 0, 0, 0, 0, 0}, nearest[8];
    for (int direction = 0; direction != 2; ++direction) {
        nk_f16_t const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        nk_size_t const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        nk_size_t const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        for (nk_size_t point = 0; point != outer_count; ++point) {
            nk_shapes_nearest_squares_f16_simt_(nk_shapes_coordinate_f16_simt_(outer_axes + point),
                                                nk_shapes_coordinate_f16_simt_(outer_axes + outer_stride + point),
                                                nk_shapes_coordinate_f16_simt_(outer_axes + 2 * outer_stride + point),
                                                inner_axes, inner_stride, inner_count, nearest);
            nk_f32_t least = NUMKONG_F32_INF;
            for (int flip = 0; flip != 8; ++flip) {
                farthest[flip] = nearest[flip] > farthest[flip] ? nearest[flip] : farthest[flip];
                least = farthest[flip] < least ? farthest[flip] : least;
            }
            if (least > limit_squared && __fsqrt_rn(least) > limit) return __fsqrt_rn(least);
        }
    }
    nk_f32_t least = farthest[0];
    for (int flip = 1; flip != 8; ++flip) least = farthest[flip] < least ? farthest[flip] : least;
    return __fsqrt_rn(least);
}

#pragma endregion Chamfer and Hausdorff

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_SHAPES_SIMT_CUH
