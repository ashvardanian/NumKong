/**
 *  @file include/numkong/meshes/serial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief Many-to-many shape distances between 3D point clouds for SIMD-free CPUs.
 *
 *  @sa include/numkong/meshes.h
 *
 *  Every capability packs through the serial routine here and solves exact transport through its
 *  own network simplex: a northwest-corner tree rooted at the first row point, block pricing, and
 *  the leaving arc by Cunningham's rule, which keeps the tree strongly feasible so that degenerate
 *  pivots never cycle.
 */
#ifndef NUMKONG_MESHES_SERIAL_H
#define NUMKONG_MESHES_SERIAL_H

#include "numkong/types.h"
#include "numkong/meshes.h"        // `nk_shapes_header_f32_t`, `nk_shapes_axes_f16`
#include "numkong/cast/serial.h"   // `nk_f16_to_f32_serial_`, `nk_f32_to_f16_serial_`
#include "numkong/scalar/serial.h" // `nk_sqrt_f32_serial_`, `nk_rsqrt_f64_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

/** Constants for the McAdams 3×3 SVD algorithm: γ = (√8 + 3)² / 4 = 5.828427124, with cstar =
 *  cos(π/8) and sstar = sin(π/8). */
#define NUMKONG_F32_SVD_GAMMA_   5.828427124f
#define NUMKONG_F32_SVD_CSTAR_   0.923879532f
#define NUMKONG_F32_SVD_SSTAR_   0.3826834323f
#define NUMKONG_F32_SVD_EPSILON_ 1e-6f

#define NUMKONG_F64_SVD_GAMMA_   5.828427124746190
#define NUMKONG_F64_SVD_CSTAR_   0.9238795325112867
#define NUMKONG_F64_SVD_SSTAR_   0.3826834323650898
#define NUMKONG_F64_SVD_EPSILON_ 1e-12

/** Type-generic SVD helper macros, starting with this conditional swap, generate f32 and f64
 *  versions of the 3×3 helper functions used by packing and the transport kinds. */
#define nk_define_cond_swap_(type)                                                                 \
    NUMKONG_INLINE void nk_cond_swap_##type##_serial_(int c, nk_##type##_t *x, nk_##type##_t *y) { \
        nk_##type##_t temp = *x;                                                                   \
        *x = c ? *y : *x;                                                                          \
        *y = c ? temp : *y;                                                                        \
    }

#define nk_define_conditional_negating_swap_(type)                                                                 \
    NUMKONG_INLINE void nk_conditional_negating_swap_##type##_serial_(int c, nk_##type##_t *x, nk_##type##_t *y) { \
        nk_##type##_t neg_x = -*x;                                                                                 \
        *x = c ? *y : *x;                                                                                          \
        *y = c ? neg_x : *y;                                                                                       \
    }

/** a × b + c × d with each product in its own statement, which no compiler contracts into an
 *  FMA under this project's flags, so packing rounds alike with and without FMA units. */
#define nk_define_sum_two_products_(type)                                                                            \
    NUMKONG_INLINE nk_##type##_t nk_sum_two_products_##type##_serial_(nk_##type##_t left_0, nk_##type##_t right_0,   \
                                                                      nk_##type##_t left_1, nk_##type##_t right_1) { \
        nk_##type##_t const product_0 = left_0 * right_0;                                                            \
        nk_##type##_t const product_1 = left_1 * right_1;                                                            \
        return product_0 + product_1;                                                                                \
    }

#define nk_define_approximate_givens_quaternion_(type, gamma, cstar, sstar, compute_rsqrt)                           \
    NUMKONG_INLINE void nk_approximate_givens_quaternion_##type##_serial_(                                           \
        nk_##type##_t a11, nk_##type##_t a12, nk_##type##_t a22, nk_##type##_t *cos_half, nk_##type##_t *sin_half) { \
        *cos_half = 2 * (a11 - a22), *sin_half = a12;                                                                \
        int use_givens = gamma * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);                              \
        nk_##type##_t w = compute_rsqrt(                                                                             \
            nk_sum_two_products_##type##_serial_(*cos_half, *cos_half, *sin_half, *sin_half));                       \
        *cos_half = use_givens ? w * (*cos_half) : cstar;                                                            \
        *sin_half = use_givens ? w * (*sin_half) : sstar;                                                            \
    }

#define nk_define_jacobi_conjugation_(type)                                                                       \
    NUMKONG_INLINE void nk_jacobi_conjugation_##type##_serial_(                                                   \
        int idx_x, int idx_y, int idx_z, nk_##type##_t *s11, nk_##type##_t *s21, nk_##type##_t *s22,              \
        nk_##type##_t *s31, nk_##type##_t *s32, nk_##type##_t *s33, nk_##type##_t *quaternion) {                  \
        nk_##type##_t cos_half, sin_half;                                                                         \
        nk_approximate_givens_quaternion_##type##_serial_(*s11, *s21, *s22, &cos_half, &sin_half);                \
        nk_##type##_t scale = nk_sum_two_products_##type##_serial_(cos_half, cos_half, sin_half, sin_half);       \
        nk_##type##_t cos_theta = nk_sum_two_products_##type##_serial_(cos_half, cos_half, -sin_half, sin_half) / \
                                  scale;                                                                          \
        nk_##type##_t sin_theta = (2 * sin_half * cos_half) / scale;                                              \
        nk_##type##_t s11_old = *s11, s21_old = *s21, s22_old = *s22;                                             \
        nk_##type##_t s31_old = *s31, s32_old = *s32, s33_old = *s33;                                             \
        /* The rotation applied from the left, before the one from the right */                                   \
        nk_##type##_t left_11 = nk_sum_two_products_##type##_serial_(cos_theta, s11_old, sin_theta, s21_old);     \
        nk_##type##_t left_12 = nk_sum_two_products_##type##_serial_(cos_theta, s21_old, sin_theta, s22_old);     \
        nk_##type##_t left_21 = nk_sum_two_products_##type##_serial_(-sin_theta, s11_old, cos_theta, s21_old);    \
        nk_##type##_t left_22 = nk_sum_two_products_##type##_serial_(-sin_theta, s21_old, cos_theta, s22_old);    \
        *s11 = nk_sum_two_products_##type##_serial_(cos_theta, left_11, sin_theta, left_12);                      \
        *s21 = nk_sum_two_products_##type##_serial_(cos_theta, left_21, sin_theta, left_22);                      \
        *s22 = nk_sum_two_products_##type##_serial_(-sin_theta, left_21, cos_theta, left_22);                     \
        *s31 = nk_sum_two_products_##type##_serial_(cos_theta, s31_old, sin_theta, s32_old);                      \
        *s32 = nk_sum_two_products_##type##_serial_(-sin_theta, s31_old, cos_theta, s32_old);                     \
        *s33 = s33_old;                                                                                           \
        /* Update quaternion accumulator */                                                                       \
        nk_##type##_t quat_temp[3];                                                                               \
        quat_temp[0] = quaternion[0] * sin_half;                                                                  \
        quat_temp[1] = quaternion[1] * sin_half;                                                                  \
        quat_temp[2] = quaternion[2] * sin_half;                                                                  \
        sin_half *= quaternion[3];                                                                                \
        quaternion[0] *= cos_half, quaternion[1] *= cos_half;                                                     \
        quaternion[2] *= cos_half, quaternion[3] *= cos_half;                                                     \
        quaternion[idx_z] += sin_half, quaternion[3] -= quat_temp[idx_z];                                         \
        quaternion[idx_x] += quat_temp[idx_y], quaternion[idx_y] -= quat_temp[idx_x];                             \
        /* Cyclic permutation of matrix elements */                                                               \
        s11_old = *s22, s21_old = *s32, s22_old = *s33;                                                           \
        s31_old = *s21, s32_old = *s31, s33_old = *s11;                                                           \
        *s11 = s11_old, *s21 = s21_old, *s22 = s22_old;                                                           \
        *s31 = s31_old, *s32 = s32_old, *s33 = s33_old;                                                           \
    }

#define nk_define_quaternion_to_mat3x3_(type)                                                                        \
    NUMKONG_INLINE void nk_quaternion_to_mat3x3_##type##_serial_(nk_##type##_t const *quat, nk_##type##_t *matrix) { \
        nk_##type##_t w = quat[3], x = quat[0], y = quat[1], z = quat[2];                                            \
        nk_##type##_t q_xx = x * x, q_yy = y * y, q_zz = z * z;                                                      \
        nk_##type##_t q_xz = x * z, q_xy = x * y, q_yz = y * z;                                                      \
        nk_##type##_t q_wx = w * x, q_wy = w * y, q_wz = w * z;                                                      \
        matrix[0] = 1 - 2 * (q_yy + q_zz), matrix[1] = 2 * (q_xy - q_wz);                                            \
        matrix[2] = 2 * (q_xz + q_wy);                                                                               \
        matrix[3] = 2 * (q_xy + q_wz), matrix[4] = 1 - 2 * (q_xx + q_zz);                                            \
        matrix[5] = 2 * (q_yz - q_wx);                                                                               \
        matrix[6] = 2 * (q_xz - q_wy), matrix[7] = 2 * (q_yz + q_wx);                                                \
        matrix[8] = 1 - 2 * (q_xx + q_yy);                                                                           \
    }

#define nk_define_jacobi_eigenanalysis_(type, compute_rsqrt)                                                   \
    NUMKONG_INLINE void nk_jacobi_eigenanalysis_##type##_serial_(                                              \
        nk_##type##_t *s11, nk_##type##_t *s21, nk_##type##_t *s22, nk_##type##_t *s31, nk_##type##_t *s32,    \
        nk_##type##_t *s33, nk_##type##_t *quaternion) {                                                       \
        quaternion[0] = 0, quaternion[1] = 0, quaternion[2] = 0, quaternion[3] = 1;                            \
        /* 16 iterations converge better with repeated eigenvalues and identity-like matrices */               \
        for (int iter = 0; iter < 16; iter++) {                                                                \
            nk_jacobi_conjugation_##type##_serial_(0, 1, 2, s11, s21, s22, s31, s32, s33, quaternion);         \
            nk_jacobi_conjugation_##type##_serial_(1, 2, 0, s11, s21, s22, s31, s32, s33, quaternion);         \
            nk_jacobi_conjugation_##type##_serial_(2, 0, 1, s11, s21, s22, s31, s32, s33, quaternion);         \
        }                                                                                                      \
        nk_##type##_t norm = compute_rsqrt(                                                                    \
            nk_sum_two_products_##type##_serial_(quaternion[0], quaternion[0], quaternion[1], quaternion[1]) + \
            nk_sum_two_products_##type##_serial_(quaternion[2], quaternion[2], quaternion[3], quaternion[3])); \
        quaternion[0] *= norm, quaternion[1] *= norm;                                                          \
        quaternion[2] *= norm, quaternion[3] *= norm;                                                          \
    }

#define nk_define_qr_givens_quaternion_(type, epsilon, compute_rsqrt)                                                \
    NUMKONG_INLINE void nk_qr_givens_quaternion_##type##_serial_(nk_##type##_t a1, nk_##type##_t a2,                 \
                                                                 nk_##type##_t *cos_half, nk_##type##_t *sin_half) { \
        nk_##type##_t a1_sq_plus_a2_sq = a1 * a1 + a2 * a2;                                                          \
        nk_##type##_t rho = a1_sq_plus_a2_sq * compute_rsqrt(a1_sq_plus_a2_sq);                                      \
        rho = a1_sq_plus_a2_sq > epsilon ? rho : 0;                                                                  \
        *sin_half = rho > epsilon ? a2 : 0;                                                                          \
        nk_##type##_t abs_a1 = a1 < 0 ? -a1 : a1;                                                                    \
        nk_##type##_t max_rho = rho > epsilon ? rho : epsilon;                                                       \
        *cos_half = abs_a1 + max_rho;                                                                                \
        int should_swap = a1 < 0;                                                                                    \
        nk_cond_swap_##type##_serial_(should_swap, sin_half, cos_half);                                              \
        nk_##type##_t w = compute_rsqrt((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));                      \
        *cos_half *= w, *sin_half *= w;                                                                              \
    }

#define nk_define_sort_singular_values_(type)                                                          \
    NUMKONG_INLINE void nk_sort_singular_values_##type##_serial_(nk_##type##_t *b, nk_##type##_t *v) { \
        nk_##type##_t column_norm_squared_0 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];                 \
        nk_##type##_t column_norm_squared_1 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];                 \
        nk_##type##_t column_norm_squared_2 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];                 \
        int should_swap;                                                                               \
        /* Sort columns by descending singular value magnitude */                                      \
        should_swap = column_norm_squared_0 < column_norm_squared_1;                                   \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &b[0], &b[1]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &v[0], &v[1]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &b[3], &b[4]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &v[3], &v[4]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &b[6], &b[7]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &v[6], &v[7]);                      \
        nk_cond_swap_##type##_serial_(should_swap, &column_norm_squared_0, &column_norm_squared_1);    \
        should_swap = column_norm_squared_0 < column_norm_squared_2;                                   \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &b[0], &b[2]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &v[0], &v[2]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &b[3], &b[5]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &v[3], &v[5]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &b[6], &b[8]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &v[6], &v[8]);                      \
        nk_cond_swap_##type##_serial_(should_swap, &column_norm_squared_0, &column_norm_squared_2);    \
        should_swap = column_norm_squared_1 < column_norm_squared_2;                                   \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &b[1], &b[2]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &v[1], &v[2]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &b[4], &b[5]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &v[4], &v[5]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &b[7], &b[8]);                      \
        nk_conditional_negating_swap_##type##_serial_(should_swap, &v[7], &v[8]);                      \
    }

/** Q-only QR via three Givens rotations. The R upper-triangular factor is unused by the SVD caller,
 *  so we skip computing it (saves the third cos_theta/sin_theta and 9 R stores per call). */
#define nk_define_qr_orthogonal_factor_(type)                                                                        \
    NUMKONG_INLINE void nk_qr_orthogonal_factor_##type##_serial_(nk_##type##_t const *input, nk_##type##_t *q) {     \
        nk_##type##_t cos_half_1, sin_half_1;                                                                        \
        nk_##type##_t cos_half_2, sin_half_2;                                                                        \
        nk_##type##_t cos_half_3, sin_half_3;                                                                        \
        nk_##type##_t cos_theta, sin_theta;                                                                          \
        nk_##type##_t rotation_temp[9];                                                                              \
        /* First Givens rotation (zero input[3]) */                                                                  \
        nk_qr_givens_quaternion_##type##_serial_(input[0], input[3], &cos_half_1, &sin_half_1);                      \
        cos_theta = 1 - 2 * sin_half_1 * sin_half_1;                                                                 \
        sin_theta = 2 * cos_half_1 * sin_half_1;                                                                     \
        rotation_temp[1] = cos_theta * input[1] + sin_theta * input[4];                                              \
        rotation_temp[4] = -sin_theta * input[1] + cos_theta * input[4];                                             \
        rotation_temp[7] = input[7];                                                                                 \
        /* Second Givens rotation (only the [4]/[7] cells flow into the third Givens) */                             \
        nk_qr_givens_quaternion_##type##_serial_(input[0] * cos_theta + sin_theta * input[3], input[6], &cos_half_2, \
                                                 &sin_half_2);                                                       \
        cos_theta = 1 - 2 * sin_half_2 * sin_half_2;                                                                 \
        sin_theta = 2 * cos_half_2 * sin_half_2;                                                                     \
        nk_##type##_t matrix_temp_4 = rotation_temp[4];                                                              \
        nk_##type##_t matrix_temp_7 = -sin_theta * rotation_temp[1] + cos_theta * rotation_temp[7];                  \
        /* Third Givens rotation zeroes matrix_temp_7; Q needs only sin_half_3 and cos_half_3 */                     \
        nk_qr_givens_quaternion_##type##_serial_(matrix_temp_4, matrix_temp_7, &cos_half_3, &sin_half_3);            \
        /* Construct Q = Q1 * Q2 * Q3 (closed-form expressions) */                                                   \
        nk_##type##_t sin_half_1_sq = sin_half_1 * sin_half_1;                                                       \
        nk_##type##_t sin_half_2_sq = sin_half_2 * sin_half_2;                                                       \
        nk_##type##_t sin_half_3_sq = sin_half_3 * sin_half_3;                                                       \
        q[0] = (-1 + 2 * sin_half_1_sq) * (-1 + 2 * sin_half_2_sq);                                                  \
        q[1] = 4 * cos_half_2 * cos_half_3 * (-1 + 2 * sin_half_1_sq) * sin_half_2 * sin_half_3 +                    \
               2 * cos_half_1 * sin_half_1 * (-1 + 2 * sin_half_3_sq);                                               \
        q[2] = 4 * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -                                               \
               2 * cos_half_2 * (-1 + 2 * sin_half_1_sq) * sin_half_2 * (-1 + 2 * sin_half_3_sq);                    \
        q[3] = 2 * cos_half_1 * sin_half_1 * (1 - 2 * sin_half_2_sq);                                                \
        q[4] = -8 * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +                    \
               (-1 + 2 * sin_half_1_sq) * (-1 + 2 * sin_half_3_sq);                                                  \
        q[5] = -2 * cos_half_3 * sin_half_3 + 4 * sin_half_1 *                                                       \
                                                  (cos_half_3 * sin_half_1 * sin_half_3 +                            \
                                                   cos_half_1 * cos_half_2 * sin_half_2 * (-1 + 2 * sin_half_3_sq)); \
        q[6] = 2 * cos_half_2 * sin_half_2;                                                                          \
        q[7] = 2 * cos_half_3 * (1 - 2 * sin_half_2_sq) * sin_half_3;                                                \
        q[8] = (-1 + 2 * sin_half_2_sq) * (-1 + 2 * sin_half_3_sq);                                                  \
    }

#define nk_define_svd3x3_(type, compute_sqrt)                                                                       \
    NUMKONG_INLINE void nk_svd3x3_##type##_serial_(nk_##type##_t const *a, nk_##type##_t *svd_left,                 \
                                                   nk_##type##_t *svd_diagonal, nk_##type##_t *svd_right) {         \
        /* Compute Aᵀ * A (symmetric) */                                                                            \
        nk_##type##_t ata[9];                                                                                       \
        ata[0] = nk_sum_three_squares_##type##_serial_(a[0], a[3], a[6]);                                           \
        ata[1] = nk_sum_three_products_##type##_serial_(a[0], a[1], a[3], a[4], a[6], a[7]);                        \
        ata[2] = nk_sum_three_products_##type##_serial_(a[0], a[2], a[3], a[5], a[6], a[8]);                        \
        ata[3] = ata[1];                                                                                            \
        ata[4] = nk_sum_three_squares_##type##_serial_(a[1], a[4], a[7]);                                           \
        ata[5] = nk_sum_three_products_##type##_serial_(a[1], a[2], a[4], a[5], a[7], a[8]);                        \
        ata[6] = ata[2];                                                                                            \
        ata[7] = ata[5];                                                                                            \
        ata[8] = nk_sum_three_squares_##type##_serial_(a[2], a[5], a[8]);                                           \
        /* Jacobi eigenanalysis of Aᵀ * A */                                                                        \
        nk_##type##_t quaternion[4];                                                                                \
        nk_jacobi_eigenanalysis_##type##_serial_(&ata[0], &ata[1], &ata[4], &ata[2], &ata[5], &ata[8], quaternion); \
        nk_quaternion_to_mat3x3_##type##_serial_(quaternion, svd_right);                                            \
        /* B = A * V */                                                                                             \
        nk_##type##_t product[9];                                                                                   \
        product[0] = nk_sum_three_products_##type##_serial_(a[0], svd_right[0], a[1], svd_right[3], a[2],           \
                                                            svd_right[6]);                                          \
        product[1] = nk_sum_three_products_##type##_serial_(a[0], svd_right[1], a[1], svd_right[4], a[2],           \
                                                            svd_right[7]);                                          \
        product[2] = nk_sum_three_products_##type##_serial_(a[0], svd_right[2], a[1], svd_right[5], a[2],           \
                                                            svd_right[8]);                                          \
        product[3] = nk_sum_three_products_##type##_serial_(a[3], svd_right[0], a[4], svd_right[3], a[5],           \
                                                            svd_right[6]);                                          \
        product[4] = nk_sum_three_products_##type##_serial_(a[3], svd_right[1], a[4], svd_right[4], a[5],           \
                                                            svd_right[7]);                                          \
        product[5] = nk_sum_three_products_##type##_serial_(a[3], svd_right[2], a[4], svd_right[5], a[5],           \
                                                            svd_right[8]);                                          \
        product[6] = nk_sum_three_products_##type##_serial_(a[6], svd_right[0], a[7], svd_right[3], a[8],           \
                                                            svd_right[6]);                                          \
        product[7] = nk_sum_three_products_##type##_serial_(a[6], svd_right[1], a[7], svd_right[4], a[8],           \
                                                            svd_right[7]);                                          \
        product[8] = nk_sum_three_products_##type##_serial_(a[6], svd_right[2], a[7], svd_right[5], a[8],           \
                                                            svd_right[8]);                                          \
        /* Sort singular values and update V */                                                                     \
        nk_sort_singular_values_##type##_serial_(product, svd_right);                                               \
        /* Compute singular values from column norms of sorted B (before QR orthogonalizes them) */                 \
        /* These are the true singular values: √(‖colᵢ‖²) */                                                        \
        nk_##type##_t singular_value_squared_0 = nk_sum_three_squares_##type##_serial_(product[0], product[3],      \
                                                                                       product[6]);                 \
        nk_##type##_t singular_value_squared_1 = nk_sum_three_squares_##type##_serial_(product[1], product[4],      \
                                                                                       product[7]);                 \
        nk_##type##_t singular_value_squared_2 = nk_sum_three_squares_##type##_serial_(product[2], product[5],      \
                                                                                       product[8]);                 \
        /* Q-only QR: extract U (the orthogonal factor); R is unused by SVD */                                      \
        nk_qr_orthogonal_factor_##type##_serial_(product, svd_left);                                                \
        /* Singular values go on diagonal positions [0], [4], [8]; callers never read the rest */                   \
        svd_diagonal[0] = compute_sqrt(singular_value_squared_0);                                                   \
        svd_diagonal[4] = compute_sqrt(singular_value_squared_1);                                                   \
        svd_diagonal[8] = compute_sqrt(singular_value_squared_2);                                                   \
    }

#define nk_define_det3x3_(type)                                                          \
    NUMKONG_INLINE nk_##type##_t nk_det3x3_##type##_serial_(nk_##type##_t const *m) {    \
        return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + \
               m[2] * (m[3] * m[7] - m[4] * m[6]);                                       \
    }

/*  GCC inlines a helper only into callers whose targets include its own, so serial code builds at
 *  the Armv8-A floor. */
#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

/* Size bias for release. Gated on NDEBUG so Debug builds keep -O0 for stepping. */
#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("s", on)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("Os")
#endif
#endif

NUMKONG_INLINE nk_f32_t nk_sum_three_products_f32_serial_(nk_f32_t left_0, nk_f32_t right_0, nk_f32_t left_1,
                                                          nk_f32_t right_1, nk_f32_t left_2, nk_f32_t right_2) {
    return left_0 * right_0 + left_1 * right_1 + left_2 * right_2;
}
NUMKONG_INLINE nk_f64_t nk_sum_three_products_f64_serial_(nk_f64_t left_0, nk_f64_t right_0, nk_f64_t left_1,
                                                          nk_f64_t right_1, nk_f64_t left_2, nk_f64_t right_2) {
    nk_f64_t sum = 0.0, compensation = 0.0;
    nk_dot2_f64_serial_(&sum, &compensation, left_0, right_0);
    nk_dot2_f64_serial_(&sum, &compensation, left_1, right_1);
    nk_dot2_f64_serial_(&sum, &compensation, left_2, right_2);
    return sum + compensation;
}

NUMKONG_INLINE nk_f32_t nk_sum_three_squares_f32_serial_(nk_f32_t value_0, nk_f32_t value_1, nk_f32_t value_2) {
    return value_0 * value_0 + value_1 * value_1 + value_2 * value_2;
}
NUMKONG_INLINE nk_f64_t nk_sum_three_squares_f64_serial_(nk_f64_t value_0, nk_f64_t value_1, nk_f64_t value_2) {
    nk_f64_t sum = 0.0, compensation = 0.0;
    nk_dot2_f64_serial_(&sum, &compensation, value_0, value_0);
    nk_dot2_f64_serial_(&sum, &compensation, value_1, value_1);
    nk_dot2_f64_serial_(&sum, &compensation, value_2, value_2);
    return sum + compensation;
}

NUMKONG_INLINE void nk_rotation_from_svd_f32_serial_(nk_f32_t const *svd_left, nk_f32_t const *svd_right,
                                                     nk_f32_t *rotation) {
    rotation[0] = nk_sum_three_products_f32_serial_( //
        svd_right[0], svd_left[0], svd_right[1], svd_left[1], svd_right[2], svd_left[2]);
    rotation[1] = nk_sum_three_products_f32_serial_( //
        svd_right[0], svd_left[3], svd_right[1], svd_left[4], svd_right[2], svd_left[5]);
    rotation[2] = nk_sum_three_products_f32_serial_( //
        svd_right[0], svd_left[6], svd_right[1], svd_left[7], svd_right[2], svd_left[8]);
    rotation[3] = nk_sum_three_products_f32_serial_( //
        svd_right[3], svd_left[0], svd_right[4], svd_left[1], svd_right[5], svd_left[2]);
    rotation[4] = nk_sum_three_products_f32_serial_( //
        svd_right[3], svd_left[3], svd_right[4], svd_left[4], svd_right[5], svd_left[5]);
    rotation[5] = nk_sum_three_products_f32_serial_( //
        svd_right[3], svd_left[6], svd_right[4], svd_left[7], svd_right[5], svd_left[8]);
    rotation[6] = nk_sum_three_products_f32_serial_( //
        svd_right[6], svd_left[0], svd_right[7], svd_left[1], svd_right[8], svd_left[2]);
    rotation[7] = nk_sum_three_products_f32_serial_( //
        svd_right[6], svd_left[3], svd_right[7], svd_left[4], svd_right[8], svd_left[5]);
    rotation[8] = nk_sum_three_products_f32_serial_( //
        svd_right[6], svd_left[6], svd_right[7], svd_left[7], svd_right[8], svd_left[8]);
}
NUMKONG_INLINE void nk_rotation_from_svd_f64_serial_(nk_f64_t const *svd_left, nk_f64_t const *svd_right,
                                                     nk_f64_t *rotation) {
    rotation[0] = nk_sum_three_products_f64_serial_( //
        svd_right[0], svd_left[0], svd_right[1], svd_left[1], svd_right[2], svd_left[2]);
    rotation[1] = nk_sum_three_products_f64_serial_( //
        svd_right[0], svd_left[3], svd_right[1], svd_left[4], svd_right[2], svd_left[5]);
    rotation[2] = nk_sum_three_products_f64_serial_( //
        svd_right[0], svd_left[6], svd_right[1], svd_left[7], svd_right[2], svd_left[8]);
    rotation[3] = nk_sum_three_products_f64_serial_( //
        svd_right[3], svd_left[0], svd_right[4], svd_left[1], svd_right[5], svd_left[2]);
    rotation[4] = nk_sum_three_products_f64_serial_( //
        svd_right[3], svd_left[3], svd_right[4], svd_left[4], svd_right[5], svd_left[5]);
    rotation[5] = nk_sum_three_products_f64_serial_( //
        svd_right[3], svd_left[6], svd_right[4], svd_left[7], svd_right[5], svd_left[8]);
    rotation[6] = nk_sum_three_products_f64_serial_( //
        svd_right[6], svd_left[0], svd_right[7], svd_left[1], svd_right[8], svd_left[2]);
    rotation[7] = nk_sum_three_products_f64_serial_( //
        svd_right[6], svd_left[3], svd_right[7], svd_left[4], svd_right[8], svd_left[5]);
    rotation[8] = nk_sum_three_products_f64_serial_( //
        svd_right[6], svd_left[6], svd_right[7], svd_left[7], svd_right[8], svd_left[8]);
}

nk_define_cond_swap_(f32)
nk_define_conditional_negating_swap_(f32)
nk_define_sum_two_products_(f32)
nk_define_approximate_givens_quaternion_(f32, NUMKONG_F32_SVD_GAMMA_, NUMKONG_F32_SVD_CSTAR_, NUMKONG_F32_SVD_SSTAR_,
                                         nk_rsqrt_f32_serial_)
nk_define_jacobi_conjugation_(f32)
nk_define_quaternion_to_mat3x3_(f32)
nk_define_jacobi_eigenanalysis_(f32, nk_rsqrt_f32_serial_)
nk_define_qr_givens_quaternion_(f32, NUMKONG_F32_SVD_EPSILON_, nk_rsqrt_f32_serial_)
nk_define_sort_singular_values_(f32)
nk_define_qr_orthogonal_factor_(f32)
nk_define_svd3x3_(f32, nk_sqrt_f32_serial_)
nk_define_det3x3_(f32)

nk_define_cond_swap_(f64)
nk_define_conditional_negating_swap_(f64)
nk_define_sum_two_products_(f64)
nk_define_approximate_givens_quaternion_(f64, NUMKONG_F64_SVD_GAMMA_, NUMKONG_F64_SVD_CSTAR_, NUMKONG_F64_SVD_SSTAR_,
                                         nk_rsqrt_f64_serial_)
nk_define_jacobi_conjugation_(f64)
nk_define_quaternion_to_mat3x3_(f64)
nk_define_jacobi_eigenanalysis_(f64, nk_rsqrt_f64_serial_)
nk_define_qr_givens_quaternion_(f64, NUMKONG_F64_SVD_EPSILON_, nk_rsqrt_f64_serial_)
nk_define_sort_singular_values_(f64)
nk_define_qr_orthogonal_factor_(f64)
nk_define_svd3x3_(f64, nk_sqrt_f64_serial_)
nk_define_det3x3_(f64)

/** A cloud's centroid and the 6 unique entries of its covariance in the order Jacobi takes them,
 *  xx, yx, yy, zx, zy and zz, returning the RMS distance from the centroid, or 0 when every point
 *  coincides, as the rounded centroid of coincident points still leaves a spread. Then the pack of
 *  one cloud from them: the header, then the x, y and z arrays of its points centered, rotated onto
 *  the principal axes and divided by @p scale. Every product sits in a statement of its own, as
 *  Clang contracts only within one and GCC's ISO modes not at all, so the arithmetic rounds alike
 *  with and without FMA units. */
#define nk_define_shapes_pack_(input_type, math_type, points_type, load_and_convert, convert_and_store, compute_sqrt, \
                               compute_rsqrt)                                                                         \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_moments_##input_type##_serial_(                                       \
        nk_##input_type##_t const *points, nk_size_t count, nk_##math_type##_t *centroid,                             \
        nk_##math_type##_t *covariance) {                                                                             \
        nk_##math_type##_t x, y, z, product, sum_x = 0, sum_y = 0, sum_z = 0, first[3];                               \
        centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;                                                            \
        for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;                                               \
        if (!count) return 0;                                                                                         \
        load_and_convert(points + 0, &first[0]), load_and_convert(points + 1, &first[1]);                             \
        load_and_convert(points + 2, &first[2]);                                                                      \
        int coincident = 1;                                                                                           \
        for (nk_size_t index = 0; index != count; ++index) {                                                          \
            load_and_convert(points + index * 3 + 0, &x), load_and_convert(points + index * 3 + 1, &y);               \
            load_and_convert(points + index * 3 + 2, &z);                                                             \
            sum_x += x, sum_y += y, sum_z += z;                                                                       \
            coincident &= x == first[0] && y == first[1] && z == first[2];                                            \
        }                                                                                                             \
        nk_##math_type##_t const points_count = (nk_##math_type##_t)count;                                            \
        centroid[0] = sum_x / points_count, centroid[1] = sum_y / points_count, centroid[2] = sum_z / points_count;   \
        for (nk_size_t index = 0; index != count; ++index) {                                                          \
            load_and_convert(points + index * 3 + 0, &x), load_and_convert(points + index * 3 + 1, &y);               \
            load_and_convert(points + index * 3 + 2, &z);                                                             \
            x -= centroid[0], y -= centroid[1], z -= centroid[2];                                                     \
            product = x * x, covariance[0] += product, product = y * x, covariance[1] += product;                     \
            product = y * y, covariance[2] += product, product = z * x, covariance[3] += product;                     \
            product = z * y, covariance[4] += product, product = z * z, covariance[5] += product;                     \
        }                                                                                                             \
        for (int entry = 0; entry != 6; ++entry) covariance[entry] /= points_count;                                   \
        return coincident ? 0 : compute_sqrt(covariance[0] + covariance[2] + covariance[5]);                          \
    }                                                                                                                 \
    NUMKONG_INLINE void nk_shapes_pack_##input_type##_serial_(                                                        \
        nk_##input_type##_t const *points, nk_size_t count, nk_##math_type##_t const *centroid,                       \
        nk_##math_type##_t const *covariance, nk_##math_type##_t scale, void *packed) {                               \
        nk_shapes_header_##math_type##_t *header = (nk_shapes_header_##math_type##_t *)packed;                        \
        nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_header_##math_type##_t),        \
                                                                     nk_shapes_alignment_k);                          \
        nk_size_t const axis_bytes = nk_size_round_up_to_multiple_(count * sizeof(nk_##input_type##_t),               \
                                                                   nk_shapes_alignment_k);                            \
        nk_u8_t *packed_bytes = (nk_u8_t *)packed;                                                                    \
        for (nk_size_t byte_index = 0; byte_index != header_bytes + 3 * axis_bytes; ++byte_index)                     \
            packed_bytes[byte_index] = 0;                                                                             \
        header->frame[3] = 1, header->scale = scale;                                                                  \
        if (!count) return;                                                                                           \
        nk_size_t const axis_stride = axis_bytes / sizeof(nk_##input_type##_t);                                       \
        nk_##input_type##_t *packed_x = (nk_##input_type##_t *)(packed_bytes + header_bytes);                         \
        nk_##input_type##_t *packed_y = packed_x + axis_stride, *packed_z = packed_y + axis_stride;                   \
        nk_##math_type##_t const points_count = (nk_##math_type##_t)count;                                            \
        nk_##math_type##_t x, y, z, product;                                                                          \
        nk_##math_type##_t covariance_xx = covariance[0], covariance_yx = covariance[1];                              \
        nk_##math_type##_t covariance_yy = covariance[2], covariance_zx = covariance[3];                              \
        nk_##math_type##_t covariance_zy = covariance[4], covariance_zz = covariance[5];                              \
        nk_##math_type##_t quaternion[4], axes[9];                                                                    \
        nk_jacobi_eigenanalysis_##math_type##_serial_(&covariance_xx, &covariance_yx, &covariance_yy, &covariance_zx, \
                                                      &covariance_zy, &covariance_zz, quaternion);                    \
        nk_quaternion_to_mat3x3_##math_type##_serial_(quaternion, axes);                                              \
        nk_##math_type##_t eigenvalue_0 = covariance_xx, eigenvalue_1 = covariance_yy, eigenvalue_2 = covariance_zz;  \
        /* Each swap moves one eigenvector and negates the other, so the axes stay a rotation */                      \
        int should_swap = eigenvalue_0 < eigenvalue_1;                                                                \
        for (int row = 0; row != 3; ++row)                                                                            \
            nk_conditional_negating_swap_##math_type##_serial_(should_swap, &axes[row * 3 + 0], &axes[row * 3 + 1]);  \
        nk_cond_swap_##math_type##_serial_(should_swap, &eigenvalue_0, &eigenvalue_1);                                \
        should_swap = eigenvalue_0 < eigenvalue_2;                                                                    \
        for (int row = 0; row != 3; ++row)                                                                            \
            nk_conditional_negating_swap_##math_type##_serial_(should_swap, &axes[row * 3 + 0], &axes[row * 3 + 2]);  \
        nk_cond_swap_##math_type##_serial_(should_swap, &eigenvalue_0, &eigenvalue_2);                                \
        should_swap = eigenvalue_1 < eigenvalue_2;                                                                    \
        for (int row = 0; row != 3; ++row)                                                                            \
            nk_conditional_negating_swap_##math_type##_serial_(should_swap, &axes[row * 3 + 1], &axes[row * 3 + 2]);  \
        /* Shepperd's method divides by the largest of 4w², 4x², 4y² and 4z² */                                       \
        nk_##math_type##_t const trace = axes[0] + axes[4] + axes[8];                                                 \
        nk_##math_type##_t divisor;                                                                                   \
        if (trace > 0) {                                                                                              \
            divisor = 2 * compute_sqrt(trace + 1);                                                                    \
            quaternion[0] = (axes[7] - axes[5]) / divisor, quaternion[1] = (axes[2] - axes[6]) / divisor;             \
            quaternion[2] = (axes[3] - axes[1]) / divisor, quaternion[3] = divisor / 4;                               \
        }                                                                                                             \
        else if (axes[0] > axes[4] && axes[0] > axes[8]) {                                                            \
            divisor = 2 * compute_sqrt(1 + axes[0] - axes[4] - axes[8]);                                              \
            quaternion[0] = divisor / 4, quaternion[1] = (axes[1] + axes[3]) / divisor;                               \
            quaternion[2] = (axes[2] + axes[6]) / divisor, quaternion[3] = (axes[7] - axes[5]) / divisor;             \
        }                                                                                                             \
        else if (axes[4] > axes[8]) {                                                                                 \
            divisor = 2 * compute_sqrt(1 + axes[4] - axes[0] - axes[8]);                                              \
            quaternion[0] = (axes[1] + axes[3]) / divisor, quaternion[1] = divisor / 4;                               \
            quaternion[2] = (axes[5] + axes[7]) / divisor, quaternion[3] = (axes[2] - axes[6]) / divisor;             \
        }                                                                                                             \
        else {                                                                                                        \
            divisor = 2 * compute_sqrt(1 + axes[8] - axes[0] - axes[4]);                                              \
            quaternion[0] = (axes[2] + axes[6]) / divisor, quaternion[1] = (axes[5] + axes[7]) / divisor;             \
            quaternion[2] = divisor / 4, quaternion[3] = (axes[3] - axes[1]) / divisor;                               \
        }                                                                                                             \
        nk_##math_type##_t const inverse_norm = compute_rsqrt(                                                        \
            nk_sum_two_products_##math_type##_serial_(quaternion[0], quaternion[0], quaternion[1], quaternion[1]) +   \
            nk_sum_two_products_##math_type##_serial_(quaternion[2], quaternion[2], quaternion[3], quaternion[3]));   \
        /* The header holds the frame 𝑄 = Vᵀ, whose quaternion is the conjugate of V's */                             \
        header->frame[0] = -quaternion[0] * inverse_norm, header->frame[1] = -quaternion[1] * inverse_norm;           \
        header->frame[2] = -quaternion[2] * inverse_norm, header->frame[3] = quaternion[3] * inverse_norm;            \
        nk_quaternion_to_mat3x3_##math_type##_serial_(header->frame, axes);                                           \
        /* Dividing 𝑄 by the scale divides every rotated point, and dividing by 1 changes no bit */                   \
        for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;                                                    \
        nk_##math_type##_t moment_xx = 0, moment_yx = 0, moment_yy = 0, moment_zx = 0, moment_zy = 0, moment_zz = 0;  \
        nk_##math_type##_t max_norm_squared = 0, rotated[3];                                                          \
        for (nk_size_t index = 0; index != count; ++index) {                                                          \
            load_and_convert(points + index * 3 + 0, &x), load_and_convert(points + index * 3 + 1, &y);               \
            load_and_convert(points + index * 3 + 2, &z);                                                             \
            x -= centroid[0], y -= centroid[1], z -= centroid[2];                                                     \
            for (int axis = 0; axis != 3; ++axis) {                                                                   \
                rotated[axis] = axes[axis * 3 + 0] * x;                                                               \
                product = axes[axis * 3 + 1] * y, rotated[axis] += product;                                           \
                product = axes[axis * 3 + 2] * z, rotated[axis] += product;                                           \
            }                                                                                                         \
            convert_and_store(&rotated[0], packed_x + index), convert_and_store(&rotated[1], packed_y + index);       \
            convert_and_store(&rotated[2], packed_z + index);                                                         \
            load_and_convert(packed_x + index, &x), load_and_convert(packed_y + index, &y);                           \
            load_and_convert(packed_z + index, &z);                                                                   \
            nk_##math_type##_t const xx = x * x, yy = y * y, zz = z * z, norm_squared = xx + yy + zz;                 \
            moment_xx += xx, moment_yy += yy, moment_zz += zz;                                                        \
            product = y * x, moment_yx += product, product = z * x, moment_zx += product;                             \
            product = z * y, moment_zy += product;                                                                    \
            max_norm_squared = norm_squared > max_norm_squared ? norm_squared : max_norm_squared;                     \
        }                                                                                                             \
        moment_xx /= points_count, moment_yx /= points_count, moment_yy /= points_count;                              \
        moment_zx /= points_count, moment_zy /= points_count, moment_zz /= points_count;                              \
        nk_jacobi_eigenanalysis_##math_type##_serial_(&moment_xx, &moment_yx, &moment_yy, &moment_zx, &moment_zy,     \
                                                      &moment_zz, quaternion);                                        \
        nk_cond_swap_##math_type##_serial_(moment_xx < moment_yy, &moment_xx, &moment_yy);                            \
        nk_cond_swap_##math_type##_serial_(moment_xx < moment_zz, &moment_xx, &moment_zz);                            \
        nk_cond_swap_##math_type##_serial_(moment_yy < moment_zz, &moment_yy, &moment_zz);                            \
        header->points = (nk_##points_type##_t)count;                                                                 \
        header->centroid[0] = centroid[0], header->centroid[1] = centroid[1], header->centroid[2] = centroid[2];      \
        header->spreads[0] = moment_xx, header->spreads[1] = moment_yy, header->spreads[2] = moment_zz;               \
        header->max_radius = compute_sqrt(max_norm_squared);                                                          \
    }

nk_define_shapes_pack_(f16, f32, u32, nk_f16_to_f32_serial_, nk_f32_to_f16_serial_, nk_sqrt_f32_serial_,
                       nk_rsqrt_f32_serial_)
nk_define_shapes_pack_(f32, f64, u64, nk_assign_from_to_, nk_assign_from_to_, nk_sqrt_f64_serial_, nk_rsqrt_f64_serial_)

/** What every kind reads from the pack headers: the rotation 𝑄ₐᵀ × inner × 𝑄ᵦ taking the column
 *  cloud onto the row cloud, the best of 8 per-flip distances with its rotation, and the largest
 *  cloud of a batch. */
#define nk_define_shapes_headers_(math_type)                                                                          \
    NUMKONG_INLINE void nk_shapes_rotation_##math_type##_serial_(                                                     \
        nk_##math_type##_t const *row_frame, nk_##math_type##_t const *inner, nk_##math_type##_t const *column_frame, \
        nk_##math_type##_t *rotation) {                                                                               \
        nk_##math_type##_t row_axes[9], column_axes[9], inner_column[9];                                              \
        nk_quaternion_to_mat3x3_##math_type##_serial_(row_frame, row_axes);                                           \
        nk_quaternion_to_mat3x3_##math_type##_serial_(column_frame, column_axes);                                     \
        for (int row = 0; row != 3; ++row)                                                                            \
            for (int column = 0; column != 3; ++column)                                                               \
                inner_column[row * 3 + column] = inner[row * 3] * column_axes[column] +                               \
                                                 inner[row * 3 + 1] * column_axes[3 + column] +                       \
                                                 inner[row * 3 + 2] * column_axes[6 + column];                        \
        for (int row = 0; row != 3; ++row)                                                                            \
            for (int column = 0; column != 3; ++column)                                                               \
                rotation[row * 3 + column] = row_axes[row] * inner_column[column] +                                   \
                                             row_axes[3 + row] * inner_column[3 + column] +                           \
                                             row_axes[6 + row] * inner_column[6 + column];                            \
    }                                                                                                                 \
    NUMKONG_INLINE void nk_shapes_best_flip_##math_type##_serial_(nk_##math_type##_t const *flip_distances,           \
                                                                  void const *row_packed, void const *column_packed,  \
                                                                  nk_##math_type##_t *result, nk_f32_t *rotation) {   \
        int best_flip = 0;                                                                                            \
        for (int flip = 1; flip != 8; ++flip)                                                                         \
            if (flip_distances[flip] < flip_distances[best_flip]) best_flip = flip;                                   \
        *result = flip_distances[best_flip];                                                                          \
        if (!rotation) return;                                                                                        \
        nk_shapes_header_##math_type##_t const *row_header = (nk_shapes_header_##math_type##_t const *)row_packed;    \
        nk_shapes_header_##math_type##_t const *column_header =                                                       \
            (nk_shapes_header_##math_type##_t const *)column_packed;                                                  \
        nk_##math_type##_t flip_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};                                            \
        nk_##math_type##_t pair_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};                                            \
        if (row_header->points && column_header->points) {                                                            \
            flip_rotation[0] = (best_flip & 1) ? -1 : 1, flip_rotation[4] = (best_flip & 2) ? -1 : 1;                 \
            flip_rotation[8] = (best_flip & 4) ? -1 : 1;                                                              \
            nk_shapes_rotation_##math_type##_serial_(row_header->frame, flip_rotation, column_header->frame,          \
                                                     pair_rotation);                                                  \
        }                                                                                                             \
        for (int cell = 0; cell != 9; ++cell) rotation[cell] = (nk_f32_t)pair_rotation[cell];                         \
    }                                                                                                                 \
    NUMKONG_INLINE nk_size_t nk_shapes_largest_cloud_##math_type##_serial_(void const *const *packs,                  \
                                                                           nk_size_t count) {                         \
        nk_size_t largest = 0;                                                                                        \
        for (nk_size_t index = 0; index != count; ++index) {                                                          \
            nk_shapes_header_##math_type##_t const *header = (nk_shapes_header_##math_type##_t const *)packs[index];  \
            largest = header->points > largest ? (nk_size_t)header->points : largest;                                 \
        }                                                                                                             \
        return largest;                                                                                               \
    }

nk_define_shapes_headers_(f32)
nk_define_shapes_headers_(f64)

/** One directed pass of Chamfer and Hausdorff under the 8 axis flips: adds each outer point's
 *  distance to its nearest inner point onto @p sums and raises @p maxima to it. The squared
 *  differences and sums read the same from either point, so swapped clouds swap the passes. */
#define nk_define_shapes_nearest_(input_type, math_type, load_and_convert, compute_sqrt, infinity)                    \
    NUMKONG_INLINE void nk_shapes_nearest_##input_type##_serial_(                                                     \
        nk_##input_type##_t const *outer_axes, nk_size_t outer_stride, nk_size_t outer_count,                         \
        nk_##input_type##_t const *inner_axes, nk_size_t inner_stride, nk_size_t inner_count,                         \
        nk_##math_type##_t *sums, nk_##math_type##_t *maxima) {                                                       \
        nk_##math_type##_t outer_x, outer_y, outer_z, inner_x, inner_y, inner_z, nearest[8];                          \
        for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {                                  \
            load_and_convert(outer_axes + outer_index, &outer_x);                                                     \
            load_and_convert(outer_axes + outer_stride + outer_index, &outer_y);                                      \
            load_and_convert(outer_axes + 2 * outer_stride + outer_index, &outer_z);                                  \
            for (int flip = 0; flip != 8; ++flip) nearest[flip] = infinity;                                           \
            for (nk_size_t inner_index = 0; inner_index != inner_count; ++inner_index) {                              \
                load_and_convert(inner_axes + inner_index, &inner_x);                                                 \
                load_and_convert(inner_axes + inner_stride + inner_index, &inner_y);                                  \
                load_and_convert(inner_axes + 2 * inner_stride + inner_index, &inner_z);                              \
                /* A flip negates an inner axis, turning its difference into a sum */                                 \
                nk_##math_type##_t const minus_x = outer_x - inner_x, plus_x = outer_x + inner_x;                     \
                nk_##math_type##_t const minus_y = outer_y - inner_y, plus_y = outer_y + inner_y;                     \
                nk_##math_type##_t const minus_z = outer_z - inner_z, plus_z = outer_z + inner_z;                     \
                nk_##math_type##_t const minus_xx = minus_x * minus_x, plus_xx = plus_x * plus_x;                     \
                nk_##math_type##_t const minus_yy = minus_y * minus_y, plus_yy = plus_y * plus_y;                     \
                nk_##math_type##_t const minus_zz = minus_z * minus_z, plus_zz = plus_z * plus_z;                     \
                nk_##math_type##_t const planar[4] = {minus_xx + minus_yy, plus_xx + minus_yy, minus_xx + plus_yy,    \
                                                      plus_xx + plus_yy};                                             \
                for (int flip = 0; flip != 8; ++flip) {                                                               \
                    nk_##math_type##_t const distance_squared = planar[flip & 3] + ((flip & 4) ? plus_zz : minus_zz); \
                    nearest[flip] = distance_squared < nearest[flip] ? distance_squared : nearest[flip];              \
                }                                                                                                     \
            }                                                                                                         \
            for (int flip = 0; flip != 8; ++flip) {                                                                   \
                nk_##math_type##_t const distance = compute_sqrt(nearest[flip]);                                      \
                sums[flip] += distance;                                                                               \
                maxima[flip] = distance > maxima[flip] ? distance : maxima[flip];                                     \
            }                                                                                                         \
        }                                                                                                             \
    }

nk_define_shapes_nearest_(f16, f32, nk_f16_to_f32_serial_, nk_sqrt_f32_serial_, NUMKONG_F32_INF)
nk_define_shapes_nearest_(f32, f64, nk_assign_from_to_, nk_sqrt_f64_serial_, NUMKONG_F64_INF)

/** Appends @p child to the children of @p parent_node in a transport tree. */
NUMKONG_INLINE void nk_shapes_attach_child_serial_(nk_u32_t *tree, nk_size_t tree_stride, nk_u32_t parent_node,
                                                   nk_u32_t child) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    nk_u32_t *first_child = tree + 3 * tree_stride, *next_sibling = tree + 4 * tree_stride;
    nk_u32_t *previous_sibling = tree + 5 * tree_stride;
    nk_u32_t const after = first_child[parent_node];
    next_sibling[child] = after, previous_sibling[child] = no_node;
    if (after != no_node) previous_sibling[after] = child;
    first_child[parent_node] = child;
}

/** Removes @p child from the children of @p parent_node in a transport tree. */
NUMKONG_INLINE void nk_shapes_detach_child_serial_(nk_u32_t *tree, nk_size_t tree_stride, nk_u32_t parent_node,
                                                   nk_u32_t child) {
    nk_u32_t const no_node = 0xFFFFFFFFu;
    nk_u32_t *first_child = tree + 3 * tree_stride, *next_sibling = tree + 4 * tree_stride;
    nk_u32_t *previous_sibling = tree + 5 * tree_stride;
    nk_u32_t const before = previous_sibling[child], after = next_sibling[child];
    if (before != no_node) next_sibling[before] = after;
    else first_child[parent_node] = after;
    if (after != no_node) previous_sibling[after] = before;
}

/** Seeds a transport tree with the northwest-corner plan, rooted at the first row point. Row
 *  nodes come first, and each node keeps the flow on the arc to its parent. */
NUMKONG_INLINE void nk_shapes_northwest_corner_serial_(nk_size_t rows, nk_size_t columns, nk_u32_t *tree,
                                                       nk_size_t tree_stride) {
    nk_u32_t const no_node = 0xFFFFFFFFu, row_nodes = (nk_u32_t)rows, column_nodes = (nk_u32_t)columns;
    nk_u32_t *parent = tree, *flow = tree + tree_stride, *depth = tree + 2 * tree_stride;
    nk_u32_t *first_child = tree + 3 * tree_stride;
    for (nk_u32_t node = 0; node != row_nodes + column_nodes; ++node) first_child[node] = no_node;
    parent[0] = no_node, flow[0] = 0, depth[0] = 0;
    nk_u32_t row = 0, column = 0, supply = column_nodes, demand = row_nodes, newest = row_nodes;
    for (;;) {
        nk_u32_t const shipped = supply < demand ? supply : demand;
        nk_u32_t const above = newest < row_nodes ? row_nodes + column : row;
        parent[newest] = above, flow[newest] = shipped, depth[newest] = depth[above] + 1;
        nk_shapes_attach_child_serial_(tree, tree_stride, above, newest);
        supply -= shipped, demand -= shipped;
        if (row + 1 == row_nodes && column + 1 == column_nodes) return;
        if (!supply && row + 1 != row_nodes) ++row, supply = column_nodes, newest = row;
        else ++column, demand = row_nodes, newest = row_nodes + column;
    }
}

/** Copies the n + m − 1 arcs of a transport tree into the row, column and flow arrays of a plan. */
NUMKONG_INLINE void nk_shapes_tree_arcs_serial_(nk_size_t rows, nk_size_t columns, nk_u32_t const *tree,
                                                nk_size_t tree_stride, nk_u32_t *plan, nk_size_t plan_stride) {
    nk_u32_t const *parent = tree, *flow = tree + tree_stride;
    nk_u32_t const row_nodes = (nk_u32_t)rows, nodes = (nk_u32_t)(rows + columns);
    for (nk_u32_t node = 1; node != nodes; ++node) {
        plan[node - 1] = node < row_nodes ? node : parent[node];
        plan[plan_stride + node - 1] = (node < row_nodes ? parent[node] : node) - row_nodes;
        plan[2 * plan_stride + node - 1] = flow[node];
    }
}

/** The transport workspace: the widened row and column points and the rotated column points, each
 *  as x, y and z arrays @c points_stride apart; the tree's parent, flow, depth, first child, next
 *  and previous sibling arrays, @c tree_stride apart, with their f64 potentials; the best proper
 *  and improper plans as row, column and flow arrays, @c plan_stride apart; and the f64
 *  Gromov-Wasserstein matrices, @c matrix_stride apart. Its layout function carves one when given
 *  memory, leaving null views without it, and returns the bytes up to the matrices either way. */
#define nk_define_shapes_workspace_(math_type)                                                                       \
    typedef struct nk_shapes_workspace_##math_type##_serial_t {                                                      \
        nk_##math_type##_t *row_points, *column_points, *rotated_points;                                             \
        nk_f64_t *matrices, *potential;                                                                              \
        nk_u32_t *tree, *proper_plan, *improper_plan;                                                                \
        nk_size_t points_stride, tree_stride, plan_stride, matrix_stride;                                            \
    } nk_shapes_workspace_##math_type##_serial_t;                                                                    \
    NUMKONG_INLINE nk_size_t nk_shapes_workspace_##math_type##_serial_(                                              \
        nk_size_t points, nk_u8_t *workspace, nk_shapes_workspace_##math_type##_serial_t *views) {                   \
        nk_size_t const points_bytes = nk_size_round_up_to_multiple_(3 * points * sizeof(nk_##math_type##_t),        \
                                                                     nk_shapes_alignment_k);                         \
        nk_size_t const nodes_bytes = nk_size_round_up_to_multiple_(2 * points * sizeof(nk_u32_t),                   \
                                                                    nk_shapes_alignment_k);                          \
        nk_size_t const potential_bytes = nk_size_round_up_to_multiple_(2 * points * sizeof(nk_f64_t),               \
                                                                        nk_shapes_alignment_k);                      \
        nk_size_t const plan_bytes = nk_size_round_up_to_multiple_((points ? 2 * points - 1 : 0) * sizeof(nk_u32_t), \
                                                                   nk_shapes_alignment_k);                           \
        nk_size_t const matrix_bytes = nk_size_round_up_to_multiple_(points * points * sizeof(nk_f64_t),             \
                                                                     nk_shapes_alignment_k);                         \
        nk_size_t const transport_bytes = 3 * points_bytes + 6 * nodes_bytes + potential_bytes + 6 * plan_bytes;     \
        views->points_stride = points, views->tree_stride = nodes_bytes / sizeof(nk_u32_t);                          \
        views->plan_stride = plan_bytes / sizeof(nk_u32_t), views->matrix_stride = matrix_bytes / sizeof(nk_f64_t);  \
        views->row_points = views->column_points = views->rotated_points = NUMKONG_NULL;                             \
        views->matrices = views->potential = NUMKONG_NULL;                                                           \
        views->tree = views->proper_plan = views->improper_plan = NUMKONG_NULL;                                      \
        if (!workspace) return transport_bytes;                                                                      \
        views->row_points = (nk_##math_type##_t *)workspace;                                                         \
        views->column_points = (nk_##math_type##_t *)(workspace + points_bytes);                                     \
        views->rotated_points = (nk_##math_type##_t *)(workspace + 2 * points_bytes);                                \
        views->tree = (nk_u32_t *)(workspace + 3 * points_bytes);                                                    \
        views->potential = (nk_f64_t *)(workspace + 3 * points_bytes + 6 * nodes_bytes);                             \
        views->proper_plan = (nk_u32_t *)(workspace + 3 * points_bytes + 6 * nodes_bytes + potential_bytes);         \
        views->improper_plan = views->proper_plan + 3 * views->plan_stride;                                          \
        views->matrices = (nk_f64_t *)(workspace + transport_bytes);                                                 \
        return transport_bytes;                                                                                      \
    }

nk_define_shapes_workspace_(f32)
nk_define_shapes_workspace_(f64)

/** The serial costs of the transport simplex: an arc's squared distance between a row point and a
 *  rotated column point in the math type, and a block of @p count reduced costs scanned arc by arc
 *  from the cursor in row-major order, keeping the first most negative one below
 *  @p entering_cost. */
#define nk_define_shapes_pricing_(math_type)                                                                          \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_point_cost_##math_type##_serial_(                                     \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t row, nk_size_t column) {                   \
        nk_size_t const stride = views->points_stride;                                                                \
        nk_##math_type##_t const delta_x = views->row_points[row] - views->rotated_points[column];                    \
        nk_##math_type##_t const delta_y = views->row_points[stride + row] - views->rotated_points[stride + column];  \
        nk_##math_type##_t const delta_z = views->row_points[2 * stride + row] -                                      \
                                           views->rotated_points[2 * stride + column];                                \
        return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;                                             \
    }                                                                                                                 \
    NUMKONG_INLINE void nk_shapes_price_block_##math_type##_serial_(                                                  \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,            \
        nk_size_t columns, nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column, nk_f64_t *entering_cost, \
        nk_size_t *entering_row, nk_size_t *entering_column) {                                                        \
        nk_f64_t const *potential = views->potential;                                                                 \
        nk_size_t row = *cursor_row, column = *cursor_column;                                                         \
        for (nk_size_t scanned = 0; scanned != count; ++scanned) {                                                    \
            nk_f64_t const arc_cost = gradient                                                                        \
                                          ? -gradient[row * columns + column]                                         \
                                          : (nk_f64_t)nk_shapes_point_cost_##math_type##_serial_(views, row, column); \
            nk_f64_t const reduced_cost = arc_cost - potential[row] + potential[rows + column];                       \
            if (reduced_cost < *entering_cost)                                                                        \
                *entering_cost = reduced_cost, *entering_row = row, *entering_column = column;                        \
            if (++column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;                                 \
        }                                                                                                             \
        *cursor_row = row, *cursor_column = column;                                                                   \
    }

nk_define_shapes_pricing_(f32)
nk_define_shapes_pricing_(f64)

/** The exact transport plan between uniform weights as integer masses, each of n row points
 *  supplying m units and each of m column points taking n. An arc costs ‖aᵢ − R × bⱼ‖² between the
 *  row points and the rotated column points, or −Gᵢⱼ when a Gromov-Wasserstein @p gradient is
 *  given. A solve starts from the tree in the workspace, a northwest corner or the last solve's
 *  plan, and recomputes its potentials. Costs round in the math type, but potentials and reduced
 *  costs stay in f64 for both types, so the duals solve the LP of those rounded costs down to the
 *  f64 tolerance, where an f32 one would stop short of near-ties. Each capability passes its @p
 *  point_cost and its @p price_block scan, which must round alike, as the potentials come from one
 *  and the reduced costs from the other. */
#define nk_define_shapes_network_simplex_(math_type, suffix, point_cost, price_block)                           \
    NUMKONG_INLINE nk_f64_t nk_shapes_arc_cost_##math_type##_##suffix##_(                                       \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t columns,   \
        nk_size_t row, nk_size_t column) {                                                                      \
        if (gradient) return -gradient[row * columns + column];                                                 \
        return (nk_f64_t)point_cost(views, row, column);                                                        \
    }                                                                                                           \
    NUMKONG_INLINE void nk_shapes_tree_potentials_##math_type##_##suffix##_(                                    \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,      \
        nk_size_t columns) {                                                                                    \
        nk_u32_t const no_node = 0xFFFFFFFFu, row_nodes = (nk_u32_t)rows;                                       \
        nk_u32_t const *parent = views->tree, *first_child = views->tree + 3 * views->tree_stride;              \
        nk_u32_t const *next_sibling = views->tree + 4 * views->tree_stride;                                    \
        nk_f64_t *potential = views->potential;                                                                 \
        nk_u32_t node = 0;                                                                                      \
        potential[0] = 0;                                                                                       \
        for (;;) {                                                                                              \
            if (first_child[node] != no_node) node = first_child[node];                                         \
            else {                                                                                              \
                while (node && next_sibling[node] == no_node) node = parent[node];                              \
                if (!node) return;                                                                              \
                node = next_sibling[node];                                                                      \
            }                                                                                                   \
            nk_u32_t const above = parent[node];                                                                \
            if (node < row_nodes)                                                                               \
                potential[node] = potential[above] + nk_shapes_arc_cost_##math_type##_##suffix##_(              \
                                                         views, gradient, columns, node, above - row_nodes);    \
            else                                                                                                \
                potential[node] = potential[above] - nk_shapes_arc_cost_##math_type##_##suffix##_(              \
                                                         views, gradient, columns, above, node - row_nodes);    \
        }                                                                                                       \
    }                                                                                                           \
    NUMKONG_INLINE void nk_shapes_network_simplex_##math_type##_##suffix##_(                                    \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,      \
        nk_size_t columns, nk_f64_t cost_scale) {                                                               \
        nk_u32_t const no_node = 0xFFFFFFFFu, row_nodes = (nk_u32_t)rows;                                       \
        nk_u32_t *tree = views->tree;                                                                           \
        nk_size_t const tree_stride = views->tree_stride;                                                       \
        nk_u32_t *parent = tree, *flow = tree + tree_stride, *depth = tree + 2 * tree_stride;                   \
        nk_u32_t const *first_child = tree + 3 * tree_stride, *next_sibling = tree + 4 * tree_stride;           \
        nk_f64_t *potential = views->potential;                                                                 \
        nk_f64_t const tolerance = cost_scale * 1e-12;                                                          \
        nk_size_t const arcs = rows * columns;                                                                  \
        nk_size_t block = 10, cursor_row = 0, cursor_column = 0;                                                \
        while (block * block < arcs) ++block;                                                                   \
        nk_shapes_tree_potentials_##math_type##_##suffix##_(views, gradient, rows, columns);                    \
        for (;;) {                                                                                              \
            nk_f64_t entering_cost = -tolerance;                                                                \
            nk_size_t entering_row = 0, entering_column = 0, scanned = 0;                                       \
            while (scanned != arcs && !(entering_cost < -tolerance)) {                                          \
                nk_size_t const count = arcs - scanned > block ? block : arcs - scanned;                        \
                price_block(views, gradient, rows, columns, count, &cursor_row, &cursor_column, &entering_cost, \
                            &entering_row, &entering_column);                                                   \
                scanned += count;                                                                               \
            }                                                                                                   \
            if (!(entering_cost < -tolerance)) return;                                                          \
            nk_u32_t const row_node = (nk_u32_t)entering_row, column_node = (nk_u32_t)(rows + entering_column); \
            nk_u32_t row_side = row_node, column_side = column_node;                                            \
            while (row_side != column_side) {                                                                   \
                if (depth[row_side] > depth[column_side]) row_side = parent[row_side];                          \
                else if (depth[column_side] > depth[row_side]) column_side = parent[column_side];               \
                else row_side = parent[row_side], column_side = parent[column_side];                            \
            }                                                                                                   \
            nk_u32_t const join = row_side;                                                                     \
            /* Cunningham's rule: the last blocking arc around join → row → column → join */                    \
            nk_u32_t row_side_flow = no_node, row_side_leaving = no_node;                                       \
            nk_u32_t column_side_flow = no_node, column_side_leaving = no_node;                                 \
            for (nk_u32_t node = row_node; node != join; node = parent[node])                                   \
                if (node < row_nodes && flow[node] < row_side_flow)                                             \
                    row_side_flow = flow[node], row_side_leaving = node;                                        \
            for (nk_u32_t node = column_node; node != join; node = parent[node])                                \
                if (node >= row_nodes && flow[node] <= column_side_flow)                                        \
                    column_side_flow = flow[node], column_side_leaving = node;                                  \
            int const leaves_column_side = column_side_flow <= row_side_flow;                                   \
            nk_u32_t const delta = leaves_column_side ? column_side_flow : row_side_flow;                       \
            nk_u32_t const leaving = leaves_column_side ? column_side_leaving : row_side_leaving;               \
            for (nk_u32_t node = row_node; node != join; node = parent[node])                                   \
                flow[node] = node < row_nodes ? flow[node] - delta : flow[node] + delta;                        \
            for (nk_u32_t node = column_node; node != join; node = parent[node])                                \
                flow[node] = node < row_nodes ? flow[node] + delta : flow[node] - delta;                        \
            /* The subtree cut off by the leaving arc hangs from the entering one, path reversed */             \
            nk_u32_t const moved_root = leaves_column_side ? column_node : row_node;                            \
            nk_u32_t const anchor = leaves_column_side ? row_node : column_node;                                \
            nk_f64_t const shift = leaves_column_side ? -entering_cost : entering_cost;                         \
            nk_u32_t node = moved_root, new_parent = anchor, carried_flow = delta;                              \
            for (;;) {                                                                                          \
                nk_u32_t const old_parent = parent[node], old_flow = flow[node];                                \
                nk_shapes_detach_child_serial_(tree, tree_stride, old_parent, node);                            \
                parent[node] = new_parent, flow[node] = carried_flow;                                           \
                nk_shapes_attach_child_serial_(tree, tree_stride, new_parent, node);                            \
                if (node == leaving) break;                                                                     \
                new_parent = node, carried_flow = old_flow, node = old_parent;                                  \
            }                                                                                                   \
            depth[moved_root] = depth[anchor] + 1, potential[moved_root] += shift;                              \
            node = moved_root;                                                                                  \
            for (;;) {                                                                                          \
                if (first_child[node] != no_node) node = first_child[node];                                     \
                else {                                                                                          \
                    while (node != moved_root && next_sibling[node] == no_node) node = parent[node];            \
                    if (node == moved_root) break;                                                              \
                    node = next_sibling[node];                                                                  \
                }                                                                                               \
                depth[node] = depth[parent[node]] + 1, potential[node] += shift;                                \
            }                                                                                                   \
        }                                                                                                       \
    }                                                                                                           \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_tree_cost_##math_type##_##suffix##_(                            \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t rows, nk_size_t columns) {           \
        nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;                         \
        nk_##math_type##_t cost = 0;                                                                            \
        for (nk_size_t node = 1; node != rows + columns; ++node) {                                              \
            nk_size_t const row = node < rows ? node : parent[node];                                            \
            nk_size_t const column = (node < rows ? parent[node] : node) - rows;                                \
            cost += (nk_##math_type##_t)flow[node] * point_cost(views, row, column);                            \
        }                                                                                                       \
        return cost / (nk_##math_type##_t)(rows * columns);                                                     \
    }

nk_define_shapes_network_simplex_(f32, serial, nk_shapes_point_cost_f32_serial_, nk_shapes_price_block_f32_serial_)
nk_define_shapes_network_simplex_(f64, serial, nk_shapes_point_cost_f64_serial_, nk_shapes_price_block_f64_serial_)

/** Wasserstein-Procrustes of one canonically ordered pair, widened into the workspace: from each
 *  of the 8 flips, exact transport alternates with the Kabsch fit of its plan while the cost
 *  falls. Returns the least mean squared cost and its rotation of the column points, and keeps
 *  the best proper and improper plans for Gromov-Wasserstein. */
#define nk_define_shapes_wasserstein_procrustes_(math_type, suffix, infinity)                                        \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_wasserstein_procrustes_##math_type##_##suffix##_(                    \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t rows, nk_size_t columns,                  \
        nk_##math_type##_t cost_scale, nk_##math_type##_t *rotation) {                                               \
        nk_size_t const stride = views->points_stride;                                                               \
        nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;                              \
        nk_##math_type##_t const mass = (nk_##math_type##_t)(rows * columns);                                        \
        nk_##math_type##_t best = infinity, best_proper = infinity, best_improper = infinity;                        \
        for (int flip = 0; flip != 8; ++flip) {                                                                      \
            nk_##math_type##_t current[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};                                             \
            nk_##math_type##_t start_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};                                      \
            current[0] = (flip & 1) ? -1 : 1, current[4] = (flip & 2) ? -1 : 1, current[8] = (flip & 4) ? -1 : 1;    \
            nk_##math_type##_t const flip_determinant = current[0] * current[4] * current[8];                        \
            nk_u32_t *parity_plan = flip_determinant > 0 ? views->proper_plan : views->improper_plan;                \
            nk_##math_type##_t *parity_best = flip_determinant > 0 ? &best_proper : &best_improper;                  \
            nk_##math_type##_t start_best = infinity;                                                                \
            nk_shapes_northwest_corner_serial_(rows, columns, views->tree, views->tree_stride);                      \
            for (;;) {                                                                                               \
                for (nk_size_t column = 0; column != columns; ++column) {                                            \
                    nk_##math_type##_t const x = views->column_points[column];                                       \
                    nk_##math_type##_t const y = views->column_points[stride + column];                              \
                    nk_##math_type##_t const z = views->column_points[2 * stride + column];                          \
                    for (int axis = 0; axis != 3; ++axis)                                                            \
                        views->rotated_points[axis * stride + column] = current[axis * 3] * x +                      \
                                                                        current[axis * 3 + 1] * y +                  \
                                                                        current[axis * 3 + 2] * z;                   \
                }                                                                                                    \
                nk_shapes_network_simplex_##math_type##_##suffix##_(views, NUMKONG_NULL, rows, columns, cost_scale); \
                nk_##math_type##_t const cost = nk_shapes_tree_cost_##math_type##_##suffix##_(views, rows, columns); \
                if (!(cost < start_best)) break;                                                                     \
                start_best = cost;                                                                                   \
                for (int cell = 0; cell != 9; ++cell) start_rotation[cell] = current[cell];                          \
                if (cost < *parity_best) {                                                                           \
                    *parity_best = cost;                                                                             \
                    nk_shapes_tree_arcs_serial_(rows, columns, views->tree, views->tree_stride, parity_plan,         \
                                                views->plan_stride);                                                 \
                }                                                                                                    \
                nk_##math_type##_t cross_covariance[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};                                \
                for (nk_size_t node = 1; node != rows + columns; ++node) {                                           \
                    if (!flow[node]) continue;                                                                       \
                    nk_size_t const row = node < rows ? node : parent[node];                                         \
                    nk_size_t const column = (node < rows ? parent[node] : node) - rows;                             \
                    nk_##math_type##_t const weight = (nk_##math_type##_t)flow[node] / mass;                         \
                    for (int column_axis = 0; column_axis != 3; ++column_axis) {                                     \
                        nk_##math_type##_t const weighted = weight *                                                 \
                                                            views->column_points[column_axis * stride + column];     \
                        for (int row_axis = 0; row_axis != 3; ++row_axis)                                            \
                            cross_covariance[column_axis * 3 + row_axis] +=                                          \
                                weighted * views->row_points[row_axis * stride + row];                               \
                    }                                                                                                \
                }                                                                                                    \
                nk_##math_type##_t svd_left[9], svd_diagonal[9], svd_right[9];                                       \
                nk_svd3x3_##math_type##_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);                \
                nk_rotation_from_svd_##math_type##_serial_(svd_left, svd_right, current);                            \
                /* Each start stays among the rotations, or among the reflections, it began with */                  \
                if (nk_det3x3_##math_type##_serial_(current) * flip_determinant < 0) {                               \
                    svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];        \
                    nk_rotation_from_svd_##math_type##_serial_(svd_left, svd_right, current);                        \
                }                                                                                                    \
            }                                                                                                        \
            if (start_best < best) {                                                                                 \
                best = start_best;                                                                                   \
                for (int cell = 0; cell != 9; ++cell) rotation[cell] = start_rotation[cell];                         \
            }                                                                                                        \
        }                                                                                                            \
        return best;                                                                                                 \
    }

nk_define_shapes_wasserstein_procrustes_(f32, serial, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f64, serial, NUMKONG_F64_INF)

/** Dx × V × Dy into @p gradient for a plan V given by its arcs, each adding its mass times a row of
 *  Dx and a row of Dy. */
NUMKONG_INLINE void nk_shapes_distance_gradient_f64_serial_(nk_u32_t const *plan, nk_size_t plan_stride, nk_size_t arcs,
                                                            nk_f64_t mass, nk_f64_t const *row_distances,
                                                            nk_size_t rows, nk_f64_t const *column_distances,
                                                            nk_size_t columns, nk_f64_t *gradient) {
    for (nk_size_t cell = 0; cell != rows * columns; ++cell) gradient[cell] = 0;
    for (nk_size_t arc = 0; arc != arcs; ++arc) {
        if (!plan[2 * plan_stride + arc]) continue;
        nk_f64_t const weight = (nk_f64_t)plan[2 * plan_stride + arc] / mass;
        nk_f64_t const *row_distances_from = row_distances + plan[arc] * rows;
        nk_f64_t const *column_distances_from = column_distances + plan[plan_stride + arc] * columns;
        for (nk_size_t row = 0; row != rows; ++row) {
            nk_f64_t const weighted = weight * row_distances_from[row];
            nk_f64_t *gradient_row = gradient + row * columns;
            for (nk_size_t column = 0; column != columns; ++column)
                gradient_row[column] += weighted * column_distances_from[column];
        }
    }
}

/** Moves the dense plan and its gradient a Frank-Wolfe @p step toward a vertex: the plan scales by
 *  @p keep, and the gradient becomes G × @p keep + @p vertex_gradient × @p step. The caller then
 *  adds the vertex's own arcs to the plan. */
NUMKONG_INLINE void nk_shapes_plan_step_f64_serial_(nk_f64_t *plan, nk_f64_t *gradient, nk_f64_t const *vertex_gradient,
                                                    nk_size_t cells, nk_f64_t keep, nk_f64_t step) {
    for (nk_size_t cell = 0; cell != cells; ++cell)
        plan[cell] *= keep, gradient[cell] = keep * gradient[cell] + step * vertex_gradient[cell];
}

/** Returns ⟨G, P⟩ and sets @p gradient_scale to the largest |G|, the tolerance scale of the next
 *  transport step. */
NUMKONG_INLINE nk_f64_t nk_shapes_plan_score_f64_serial_(nk_f64_t const *gradient, nk_f64_t const *plan,
                                                         nk_size_t cells, nk_f64_t *gradient_scale) {
    nk_f64_t score = 0, scale = 0;
    for (nk_size_t cell = 0; cell != cells; ++cell) score += gradient[cell] * plan[cell];
    for (nk_size_t cell = 0; cell != cells; ++cell) {
        nk_f64_t const magnitude = gradient[cell] < 0 ? -gradient[cell] : gradient[cell];
        scale = magnitude > scale ? magnitude : scale;
    }
    *gradient_scale = scale;
    return score;
}

/** Gromov-Wasserstein of one canonically ordered pair over plain distances within each cloud:
 *  Frank-Wolfe with exact transport steps from the product plan and the best proper and improper
 *  Wasserstein-Procrustes plans. Returns the least square loss and the orthogonal Procrustes fit of
 *  its plan. Only the coordinates come in the math type: the distances, plans, gradients, line
 *  search, loss and fit all stay in f64, as a non-convex search follows f32 rounding near plan ties
 *  into other optima. Each capability passes its f64 @p distance_gradient, @p step_plan and
 *  @p score_plan passes over the dense matrices. */
#define nk_define_shapes_gromov_wasserstein_(math_type, suffix, distance_gradient, step_plan, score_plan)              \
    NUMKONG_INLINE void nk_shapes_distances_##math_type##_##suffix##_(                                                 \
        nk_##math_type##_t const *points, nk_size_t count, nk_size_t stride, nk_f64_t *distances) {                    \
        for (nk_size_t first = 0; first != count; ++first)                                                             \
            for (nk_size_t second = 0; second != count; ++second) {                                                    \
                nk_f64_t const delta_x = (nk_f64_t)points[first] - (nk_f64_t)points[second];                           \
                nk_f64_t const delta_y = (nk_f64_t)points[stride + first] - (nk_f64_t)points[stride + second];         \
                nk_f64_t const delta_z = (nk_f64_t)points[2 * stride + first] - (nk_f64_t)points[2 * stride + second]; \
                distances[first * count + second] = nk_sqrt_f64_serial_(delta_x * delta_x + delta_y * delta_y +        \
                                                                        delta_z * delta_z);                            \
            }                                                                                                          \
    }                                                                                                                  \
    NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_##math_type##_##suffix##_(                                    \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t rows, nk_size_t columns,                    \
        nk_##math_type##_t cost_scale, nk_f64_t const *row_distances, nk_f64_t const *column_distances,                \
        nk_##math_type##_t *rotation) {                                                                                \
        nk_size_t const cells = rows * columns, arcs = rows + columns - 1, stride = views->points_stride;              \
        nk_size_t const plan_stride = views->plan_stride;                                                              \
        nk_f64_t *plan = views->matrices + 2 * views->matrix_stride;                                                   \
        nk_f64_t *gradient = views->matrices + 3 * views->matrix_stride;                                               \
        nk_f64_t *vertex_gradient = views->matrices + 4 * views->matrix_stride;                                        \
        nk_f64_t const mass = (nk_f64_t)cells;                                                                         \
        nk_shapes_wasserstein_procrustes_##math_type##_##suffix##_(views, rows, columns, cost_scale, rotation);        \
        nk_f64_t row_spread = 0, column_spread = 0;                                                                    \
        for (nk_size_t cell = 0; cell != rows * rows; ++cell) row_spread += row_distances[cell] * row_distances[cell]; \
        for (nk_size_t cell = 0; cell != columns * columns; ++cell)                                                    \
            column_spread += column_distances[cell] * column_distances[cell];                                          \
        row_spread /= (nk_f64_t)(rows * rows), column_spread /= (nk_f64_t)(columns * columns);                         \
        /* Once its start spreads into the dense plan, the proper plan holds each step's vertex */                     \
        nk_u32_t *vertex = views->proper_plan;                                                                         \
        nk_f64_t best = NUMKONG_F64_INF;                                                                               \
        for (int start = 0; start != 3; ++start) {                                                                     \
            if (start == 2) {                                                                                          \
                /* The product plan's gradient is the outer product of the two matrices' row sums */                   \
                for (nk_size_t column = 0; column != columns; ++column) {                                              \
                    nk_f64_t column_sum = 0;                                                                           \
                    for (nk_size_t other = 0; other != columns; ++other)                                               \
                        column_sum += column_distances[other * columns + column];                                      \
                    vertex_gradient[column] = column_sum;                                                              \
                }                                                                                                      \
                for (nk_size_t row = 0; row != rows; ++row) {                                                          \
                    nk_f64_t row_sum = 0;                                                                              \
                    for (nk_size_t other = 0; other != rows; ++other) row_sum += row_distances[row * rows + other];    \
                    for (nk_size_t column = 0; column != columns; ++column) {                                          \
                        plan[row * columns + column] = 1 / mass;                                                       \
                        gradient[row * columns + column] = row_sum * vertex_gradient[column] / mass;                   \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            else {                                                                                                     \
                nk_u32_t const *start_plan = start ? views->improper_plan : views->proper_plan;                        \
                for (nk_size_t cell = 0; cell != cells; ++cell) plan[cell] = 0;                                        \
                for (nk_size_t arc = 0; arc != arcs; ++arc)                                                            \
                    plan[start_plan[arc] * columns + start_plan[plan_stride + arc]] =                                  \
                        (nk_f64_t)start_plan[2 * plan_stride + arc] / mass;                                            \
                distance_gradient(start_plan, plan_stride, arcs, mass, row_distances, rows, column_distances, columns, \
                                  gradient);                                                                           \
            }                                                                                                          \
            nk_f64_t gradient_scale;                                                                                   \
            nk_f64_t plan_score = score_plan(gradient, plan, cells, &gradient_scale);                                  \
            for (int step_index = 0; step_index != 32; ++step_index) {                                                 \
                if (!step_index) nk_shapes_northwest_corner_serial_(rows, columns, views->tree, views->tree_stride);   \
                nk_shapes_network_simplex_##math_type##_##suffix##_(views, gradient, rows, columns, gradient_scale);   \
                nk_shapes_tree_arcs_serial_(rows, columns, views->tree, views->tree_stride, vertex, plan_stride);      \
                distance_gradient(vertex, plan_stride, arcs, mass, row_distances, rows, column_distances, columns,     \
                                  vertex_gradient);                                                                    \
                nk_f64_t vertex_score = 0, vertex_curvature = 0;                                                       \
                for (nk_size_t arc = 0; arc != arcs; ++arc) {                                                          \
                    nk_size_t const cell = vertex[arc] * columns + vertex[plan_stride + arc];                          \
                    nk_f64_t const weight = (nk_f64_t)vertex[2 * plan_stride + arc] / mass;                            \
                    vertex_score += gradient[cell] * weight;                                                           \
                    vertex_curvature += vertex_gradient[cell] * weight;                                                \
                }                                                                                                      \
                nk_f64_t const slope = vertex_score - plan_score;                                                      \
                nk_f64_t const curvature = vertex_curvature - 2 * vertex_score + plan_score;                           \
                nk_f64_t const quadratic = -2 * curvature, linear = -4 * slope;                                        \
                nk_f64_t step = quadratic + linear < 0 ? 1 : 0;                                                        \
                if (quadratic > 0) {                                                                                   \
                    step = -linear / (2 * quadratic);                                                                  \
                    step = step < 0 ? 0 : step > 1 ? 1 : step;                                                         \
                }                                                                                                      \
                if (step == 0) break;                                                                                  \
                step_plan(plan, gradient, vertex_gradient, cells, 1 - step, step);                                     \
                for (nk_size_t arc = 0; arc != arcs; ++arc)                                                            \
                    plan[vertex[arc] * columns + vertex[plan_stride + arc]] +=                                         \
                        step * ((nk_f64_t)vertex[2 * plan_stride + arc] / mass);                                       \
                plan_score = score_plan(gradient, plan, cells, &gradient_scale);                                       \
            }                                                                                                          \
            nk_f64_t const loss = row_spread + column_spread - 2 * plan_score;                                         \
            if (!(loss < best)) continue;                                                                              \
            best = loss;                                                                                               \
            nk_f64_t cross_covariance[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};                                                \
            for (nk_size_t row = 0; row != rows; ++row)                                                                \
                for (nk_size_t column = 0; column != columns; ++column)                                                \
                    for (int column_axis = 0; column_axis != 3; ++column_axis) {                                       \
                        nk_f64_t const weighted = plan[row * columns + column] *                                       \
                                                  (nk_f64_t)views->column_points[column_axis * stride + column];       \
                        for (int row_axis = 0; row_axis != 3; ++row_axis)                                              \
                            cross_covariance[column_axis * 3 + row_axis] +=                                            \
                                weighted * (nk_f64_t)views->row_points[row_axis * stride + row];                       \
                    }                                                                                                  \
            nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9], fit[9];                                               \
            nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);                                \
            /* The McAdams factors are proper, so det H's sign moves into V to fit over O(3) */                        \
            if (nk_det3x3_f64_serial_(cross_covariance) < 0)                                                           \
                svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];              \
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, fit);                                                \
            for (int cell = 0; cell != 9; ++cell) rotation[cell] = (nk_##math_type##_t)fit[cell];                      \
        }                                                                                                              \
        return best;                                                                                                   \
    }

nk_define_shapes_gromov_wasserstein_(f32, serial, nk_shapes_distance_gradient_f64_serial_,
                                     nk_shapes_plan_step_f64_serial_, nk_shapes_plan_score_f64_serial_)
nk_define_shapes_gromov_wasserstein_(f64, serial, nk_shapes_distance_gradient_f64_serial_,
                                     nk_shapes_plan_step_f64_serial_, nk_shapes_plan_score_f64_serial_)

/** The canonical order of a transport pair, by points, spreads, radius and then packed bytes, so
 *  swapping the sides solves the same problem; and the widening of a pack into workspace points. */
#define nk_define_shapes_transport_(input_type, math_type, load_and_convert)                                       \
    NUMKONG_INLINE int nk_shapes_column_first_##input_type##_serial_(void const *row_packed,                       \
                                                                     void const *column_packed) {                  \
        nk_shapes_header_##math_type##_t const *row_header = (nk_shapes_header_##math_type##_t const *)row_packed; \
        nk_shapes_header_##math_type##_t const *column_header =                                                    \
            (nk_shapes_header_##math_type##_t const *)column_packed;                                               \
        if (row_header->points != column_header->points) return column_header->points < row_header->points;        \
        for (int axis = 0; axis != 3; ++axis)                                                                      \
            if (row_header->spreads[axis] != column_header->spreads[axis])                                         \
                return column_header->spreads[axis] < row_header->spreads[axis];                                   \
        if (row_header->max_radius != column_header->max_radius)                                                   \
            return column_header->max_radius < row_header->max_radius;                                             \
        nk_size_t axis_stride;                                                                                     \
        nk_u8_t const *row_bytes = (nk_u8_t const *)nk_shapes_axes_##input_type(row_packed, &axis_stride);         \
        nk_u8_t const *column_bytes = (nk_u8_t const *)nk_shapes_axes_##input_type(column_packed, &axis_stride);   \
        nk_size_t const axis_bytes = axis_stride * sizeof(nk_##input_type##_t);                                    \
        nk_size_t const points_bytes = (nk_size_t)row_header->points * sizeof(nk_##input_type##_t);                \
        for (nk_size_t axis_offset = 0; axis_offset != 3 * axis_bytes; axis_offset += axis_bytes)                  \
            for (nk_size_t byte_index = axis_offset; byte_index != axis_offset + points_bytes; ++byte_index)       \
                if (row_bytes[byte_index] != column_bytes[byte_index])                                             \
                    return column_bytes[byte_index] < row_bytes[byte_index];                                       \
        return 0;                                                                                                  \
    }                                                                                                              \
    NUMKONG_INLINE void nk_shapes_widen_##input_type##_serial_(void const *packed, nk_##math_type##_t *points,     \
                                                               nk_size_t points_stride) {                          \
        nk_size_t axis_stride;                                                                                     \
        nk_##input_type##_t const *axes = nk_shapes_axes_##input_type(packed, &axis_stride);                       \
        nk_size_t const count = (nk_size_t)((nk_shapes_header_##math_type##_t const *)packed)->points;             \
        for (int axis = 0; axis != 3; ++axis)                                                                      \
            for (nk_size_t index = 0; index != count; ++index)                                                     \
                load_and_convert(axes + axis * axis_stride + index, points + axis * points_stride + index);        \
    }

nk_define_shapes_transport_(f16, f32, nk_f16_to_f32_serial_)
nk_define_shapes_transport_(f32, f64, nk_assign_from_to_)

/** The @c _serial pack and workspace-size kernels of one input type, which every capability
 *  shares. */
#define nk_define_shapes_(input_type, math_type)                                                                     \
    NUMKONG_API nk_status_t nk_shapes_pack_size_##input_type##_serial(nk_size_t points, nk_size_t *bytes) {          \
        *bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_header_##math_type##_t), nk_shapes_alignment_k) +    \
                 3 * nk_size_round_up_to_multiple_(points * sizeof(nk_##input_type##_t), nk_shapes_alignment_k);     \
        return nk_success_k;                                                                                         \
    }                                                                                                                \
    NUMKONG_API nk_status_t nk_shapes_pack_##input_type##_serial(nk_##input_type##_t const *points, nk_size_t count, \
                                                                 void *packed, nk_stream_t stream) {                 \
        nk_assert_(stream == NUMKONG_NULL);                                                                          \
        nk_##math_type##_t centroid[3], covariance[6];                                                               \
        nk_shapes_moments_##input_type##_serial_(points, count, centroid, covariance);                               \
        nk_shapes_pack_##input_type##_serial_(points, count, centroid, covariance, 1, packed);                       \
        return nk_success_k;                                                                                         \
    }                                                                                                                \
    NUMKONG_API nk_status_t nk_shapes_pack_normalized_##input_type##_serial(                                         \
        nk_##input_type##_t const *points, nk_size_t count, void *packed, nk_stream_t stream) {                      \
        nk_assert_(stream == NUMKONG_NULL);                                                                          \
        nk_##math_type##_t centroid[3], covariance[6];                                                               \
        nk_##math_type##_t const radius = nk_shapes_moments_##input_type##_serial_(points, count, centroid,          \
                                                                                   covariance);                      \
        nk_shapes_pack_##input_type##_serial_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed); \
        return nk_success_k;                                                                                         \
    }                                                                                                                \
    NUMKONG_API nk_status_t nk_shapes_wassersteinprocrustes_workspace_size_##input_type##_serial(                    \
        nk_size_t row_points, nk_size_t column_points, nk_size_t *bytes) {                                           \
        nk_shapes_workspace_##math_type##_serial_t views;                                                            \
        *bytes = nk_shapes_workspace_##math_type##_serial_(row_points > column_points ? row_points : column_points,  \
                                                           NUMKONG_NULL, &views);                                    \
        return nk_success_k;                                                                                         \
    }                                                                                                                \
    NUMKONG_API nk_status_t nk_shapes_gromovwassersteins_workspace_size_##input_type##_serial(                       \
        nk_size_t row_points, nk_size_t column_points, nk_size_t *bytes) {                                           \
        nk_shapes_workspace_##math_type##_serial_t views;                                                            \
        *bytes = nk_shapes_workspace_##math_type##_serial_(row_points > column_points ? row_points : column_points,  \
                                                           NUMKONG_NULL, &views) +                                   \
                 5 * views.matrix_stride * sizeof(nk_f64_t);                                                         \
        return nk_success_k;                                                                                         \
    }

/** The Chamfer and Hausdorff kernels of one input type for the capability @p suffix, whose
 *  @p nearest pass runs once each way per pair. */
#define nk_define_shapes_flip_kernels_(input_type, math_type, suffix, nearest, infinity)                               \
    NUMKONG_INLINE void nk_shapes_flips_##input_type##_##suffix##_(void const *row_packed, void const *column_packed,  \
                                                                   nk_##math_type##_t *chamfers,                       \
                                                                   nk_##math_type##_t *hausdorffs) {                   \
        nk_size_t const rows = (nk_size_t)((nk_shapes_header_##math_type##_t const *)row_packed)->points;              \
        nk_size_t const columns = (nk_size_t)((nk_shapes_header_##math_type##_t const *)column_packed)->points;        \
        if (!rows || !columns) {                                                                                       \
            nk_##math_type##_t const distance = rows == columns ? 0 : infinity;                                        \
            for (int flip = 0; flip != 8; ++flip) chamfers[flip] = distance, hausdorffs[flip] = distance;              \
            return;                                                                                                    \
        }                                                                                                              \
        nk_size_t row_stride, column_stride;                                                                           \
        nk_##input_type##_t const *row_axes = nk_shapes_axes_##input_type(row_packed, &row_stride);                    \
        nk_##input_type##_t const *column_axes = nk_shapes_axes_##input_type(column_packed, &column_stride);           \
        nk_##math_type##_t forward_sums[8] = {0, 0, 0, 0, 0, 0, 0, 0}, forward_maxima[8] = {0, 0, 0, 0, 0, 0, 0, 0};   \
        nk_##math_type##_t backward_sums[8] = {0, 0, 0, 0, 0, 0, 0, 0}, backward_maxima[8] = {0, 0, 0, 0, 0, 0, 0, 0}; \
        nearest(row_axes, row_stride, rows, column_axes, column_stride, columns, forward_sums, forward_maxima);        \
        nearest(column_axes, column_stride, columns, row_axes, row_stride, rows, backward_sums, backward_maxima);      \
        nk_##math_type##_t const row_count = (nk_##math_type##_t)rows, column_count = (nk_##math_type##_t)columns;     \
        for (int flip = 0; flip != 8; ++flip) {                                                                        \
            chamfers[flip] = (forward_sums[flip] / row_count + backward_sums[flip] / column_count) / 2;                \
            hausdorffs[flip] = forward_maxima[flip] > backward_maxima[flip] ? forward_maxima[flip]                     \
                                                                            : backward_maxima[flip];                   \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_shapes_chamfers_packed_##input_type##_##suffix(                                         \
        void const *const *a_packed, nk_size_t rows, void const *const *b_packed, nk_size_t columns,                   \
        nk_##math_type##_t *results, nk_size_t results_stride, nk_f32_t *rotations, nk_stream_t stream) {              \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        if (!rows || !columns) return nk_success_k;                                                                    \
        nk_##math_type##_t chamfers[8], hausdorffs[8];                                                                 \
        for (nk_size_t row = 0; row != rows; ++row) {                                                                  \
            nk_##math_type##_t *results_row = (nk_##math_type##_t *)((char *)results + row * results_stride);          \
            for (nk_size_t column = 0; column != columns; ++column) {                                                  \
                nk_shapes_flips_##input_type##_##suffix##_(a_packed[row], b_packed[column], chamfers, hausdorffs);     \
                nk_shapes_best_flip_##math_type##_serial_(                                                             \
                    chamfers, a_packed[row], b_packed[column], results_row + column,                                   \
                    rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);                              \
            }                                                                                                          \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_shapes_hausdorffs_packed_##input_type##_##suffix(                                       \
        void const *const *a_packed, nk_size_t rows, void const *const *b_packed, nk_size_t columns,                   \
        nk_##math_type##_t *results, nk_size_t results_stride, nk_f32_t *rotations, nk_stream_t stream) {              \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        if (!rows || !columns) return nk_success_k;                                                                    \
        nk_##math_type##_t chamfers[8], hausdorffs[8];                                                                 \
        for (nk_size_t row = 0; row != rows; ++row) {                                                                  \
            nk_##math_type##_t *results_row = (nk_##math_type##_t *)((char *)results + row * results_stride);          \
            for (nk_size_t column = 0; column != columns; ++column) {                                                  \
                nk_shapes_flips_##input_type##_##suffix##_(a_packed[row], b_packed[column], chamfers, hausdorffs);     \
                nk_shapes_best_flip_##math_type##_serial_(                                                             \
                    hausdorffs, a_packed[row], b_packed[column], results_row + column,                                 \
                    rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);                              \
            }                                                                                                          \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }

/** The Wasserstein-Procrustes and Gromov-Wasserstein kernels of one input type for the capability
 *  @p suffix, around its instantiations of the shared solvers. */
#define nk_define_shapes_transport_kernels_(input_type, math_type, suffix, compute_sqrt, infinity)                   \
    NUMKONG_API nk_status_t nk_shapes_wassersteinprocrustes_packed_##input_type##_##suffix(                          \
        void const *const *a_packed, nk_size_t rows, void const *const *b_packed, nk_size_t columns,                 \
        nk_##math_type##_t *results, nk_size_t results_stride, nk_f32_t *rotations, void *workspace,                 \
        nk_size_t workspace_bytes, nk_stream_t stream) {                                                             \
        nk_assert_(stream == NUMKONG_NULL);                                                                          \
        if (!rows || !columns) return nk_success_k;                                                                  \
        if ((uintptr_t)workspace & (nk_default_alignment_k - 1)) return nk_misaligned_k;                             \
        nk_size_t const row_points = nk_shapes_largest_cloud_##math_type##_serial_(a_packed, rows);                  \
        nk_size_t const column_points = nk_shapes_largest_cloud_##math_type##_serial_(b_packed, columns);            \
        nk_size_t const points = row_points > column_points ? row_points : column_points;                            \
        nk_shapes_workspace_##math_type##_serial_t views;                                                            \
        if (workspace_bytes < nk_shapes_workspace_##math_type##_serial_(points, NUMKONG_NULL, &views))               \
            return nk_unexpected_dimensions_k;                                                                       \
        nk_shapes_workspace_##math_type##_serial_(points, (nk_u8_t *)workspace, &views);                             \
        for (nk_size_t row = 0; row != rows; ++row) {                                                                \
            nk_##math_type##_t *results_row = (nk_##math_type##_t *)((char *)results + row * results_stride);        \
            for (nk_size_t column = 0; column != columns; ++column) {                                                \
                int const swapped = nk_shapes_column_first_##input_type##_serial_(a_packed[row], b_packed[column]);  \
                void const *row_packed = swapped ? b_packed[column] : a_packed[row];                                 \
                void const *column_packed = swapped ? a_packed[row] : b_packed[column];                              \
                nk_shapes_header_##math_type##_t const *row_header =                                                 \
                    (nk_shapes_header_##math_type##_t const *)row_packed;                                            \
                nk_shapes_header_##math_type##_t const *column_header =                                              \
                    (nk_shapes_header_##math_type##_t const *)column_packed;                                         \
                nk_size_t const solved_rows = (nk_size_t)row_header->points;                                         \
                nk_size_t const solved_columns = (nk_size_t)column_header->points;                                   \
                nk_##math_type##_t inner[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1},                                           \
                                   pair_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};                                   \
                if (!solved_rows || !solved_columns)                                                                 \
                    results_row[column] = solved_rows == solved_columns ? 0 : infinity;                              \
                else {                                                                                               \
                    nk_shapes_widen_##input_type##_serial_(row_packed, views.row_points, views.points_stride);       \
                    nk_shapes_widen_##input_type##_serial_(column_packed, views.column_points, views.points_stride); \
                    nk_##math_type##_t const radii = row_header->max_radius + column_header->max_radius;             \
                    results_row[column] = compute_sqrt(nk_shapes_wasserstein_procrustes_##math_type##_##suffix##_(   \
                        &views, solved_rows, solved_columns, radii * radii, inner));                                 \
                    nk_shapes_rotation_##math_type##_serial_(row_header->frame, inner, column_header->frame,         \
                                                             pair_rotation);                                         \
                }                                                                                                    \
                if (!rotations) continue;                                                                            \
                nk_f32_t *pair_output = rotations + (row * columns + column) * 9;                                    \
                for (int cell = 0; cell != 9; ++cell)                                                                \
                    pair_output[cell] = (nk_f32_t)pair_rotation[swapped ? cell % 3 * 3 + cell / 3 : cell];           \
            }                                                                                                        \
        }                                                                                                            \
        return nk_success_k;                                                                                         \
    }                                                                                                                \
    NUMKONG_API nk_status_t nk_shapes_gromovwassersteins_packed_##input_type##_##suffix(                             \
        void const *const *a_packed, nk_size_t rows, void const *const *b_packed, nk_size_t columns,                 \
        nk_##math_type##_t *results, nk_size_t results_stride, nk_f32_t *rotations, void *workspace,                 \
        nk_size_t workspace_bytes, nk_stream_t stream) {                                                             \
        nk_assert_(stream == NUMKONG_NULL);                                                                          \
        if (!rows || !columns) return nk_success_k;                                                                  \
        if ((uintptr_t)workspace & (nk_default_alignment_k - 1)) return nk_misaligned_k;                             \
        nk_size_t const row_points = nk_shapes_largest_cloud_##math_type##_serial_(a_packed, rows);                  \
        nk_size_t const column_points = nk_shapes_largest_cloud_##math_type##_serial_(b_packed, columns);            \
        nk_size_t const points = row_points > column_points ? row_points : column_points;                            \
        nk_shapes_workspace_##math_type##_serial_t views;                                                            \
        nk_size_t const transport_bytes = nk_shapes_workspace_##math_type##_serial_(points, NUMKONG_NULL, &views);   \
        if (workspace_bytes < transport_bytes + 5 * views.matrix_stride * sizeof(nk_f64_t))                          \
            return nk_unexpected_dimensions_k;                                                                       \
        nk_shapes_workspace_##math_type##_serial_(points, (nk_u8_t *)workspace, &views);                             \
        nk_f64_t *cached_distances = views.matrices, *pair_distances = views.matrices + views.matrix_stride;         \
        for (nk_size_t row = 0; row != rows; ++row) {                                                                \
            nk_##math_type##_t *results_row = (nk_##math_type##_t *)((char *)results + row * results_stride);        \
            nk_size_t const cached_points =                                                                          \
                (nk_size_t)((nk_shapes_header_##math_type##_t const *)a_packed[row])->points;                        \
            nk_shapes_widen_##input_type##_serial_(a_packed[row], views.row_points, views.points_stride);            \
            nk_shapes_distances_##math_type##_##suffix##_(views.row_points, cached_points, views.points_stride,      \
                                                          cached_distances);                                         \
            for (nk_size_t column = 0; column != columns; ++column) {                                                \
                int const swapped = nk_shapes_column_first_##input_type##_serial_(a_packed[row], b_packed[column]);  \
                void const *row_packed = swapped ? b_packed[column] : a_packed[row];                                 \
                void const *column_packed = swapped ? a_packed[row] : b_packed[column];                              \
                nk_shapes_header_##math_type##_t const *row_header =                                                 \
                    (nk_shapes_header_##math_type##_t const *)row_packed;                                            \
                nk_shapes_header_##math_type##_t const *column_header =                                              \
                    (nk_shapes_header_##math_type##_t const *)column_packed;                                         \
                nk_size_t const solved_rows = (nk_size_t)row_header->points;                                         \
                nk_size_t const solved_columns = (nk_size_t)column_header->points;                                   \
                nk_##math_type##_t inner[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1},                                           \
                                   pair_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};                                   \
                if (!solved_rows || !solved_columns)                                                                 \
                    results_row[column] = solved_rows == solved_columns ? 0 : infinity;                              \
                else {                                                                                               \
                    nk_shapes_widen_##input_type##_serial_(row_packed, views.row_points, views.points_stride);       \
                    nk_shapes_widen_##input_type##_serial_(column_packed, views.column_points, views.points_stride); \
                    /* The cached matrix is this row's cloud, on whichever side it is solved */                      \
                    nk_shapes_distances_##math_type##_##suffix##_(swapped ? views.row_points : views.column_points,  \
                                                                  swapped ? solved_rows : solved_columns,            \
                                                                  views.points_stride, pair_distances);              \
                    nk_##math_type##_t const radii = row_header->max_radius + column_header->max_radius;             \
                    nk_f64_t const loss = nk_shapes_gromov_wasserstein_##math_type##_##suffix##_(                    \
                        &views, solved_rows, solved_columns, radii * radii,                                          \
                        swapped ? pair_distances : cached_distances, swapped ? cached_distances : pair_distances,    \
                        inner);                                                                                      \
                    results_row[column] = (nk_##math_type##_t)(nk_sqrt_f64_serial_(loss) / 2);                       \
                    nk_shapes_rotation_##math_type##_serial_(row_header->frame, inner, column_header->frame,         \
                                                             pair_rotation);                                         \
                }                                                                                                    \
                if (!rotations) continue;                                                                            \
                nk_f32_t *pair_output = rotations + (row * columns + column) * 9;                                    \
                for (int cell = 0; cell != 9; ++cell)                                                                \
                    pair_output[cell] = (nk_f32_t)pair_rotation[swapped ? cell % 3 * 3 + cell / 3 : cell];           \
            }                                                                                                        \
        }                                                                                                            \
        return nk_success_k;                                                                                         \
    }

#if NUMKONG_TARGET_SERIAL
/*  Keep the serial instantiations below actually scalar, regardless of build type. Without this,
 *  -O3 + LTO can vectorize or clone the serial kernels under AVX-512 callers in dispatch_*.c, which
 *  wastes binary and breaks the nk_*_serial-as-scalar-oracle contract that tests and
 *  numerical-stability docs rely on. See dots/serial.h. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

nk_define_shapes_(f16, f32)
nk_define_shapes_(f32, f64)
nk_define_shapes_flip_kernels_(f16, f32, serial, nk_shapes_nearest_f16_serial_, NUMKONG_F32_INF)
nk_define_shapes_flip_kernels_(f32, f64, serial, nk_shapes_nearest_f32_serial_, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, serial, nk_sqrt_f32_serial_, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f32, f64, serial, nk_sqrt_f64_serial_, NUMKONG_F64_INF)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_SERIAL

/*  The solver and kernel macros stay defined, as each SIMD backend instantiates them. */
#undef NUMKONG_F32_SVD_GAMMA_
#undef NUMKONG_F32_SVD_CSTAR_
#undef NUMKONG_F32_SVD_SSTAR_
#undef NUMKONG_F32_SVD_EPSILON_
#undef NUMKONG_F64_SVD_GAMMA_
#undef NUMKONG_F64_SVD_CSTAR_
#undef NUMKONG_F64_SVD_SSTAR_
#undef NUMKONG_F64_SVD_EPSILON_
#undef nk_define_cond_swap_
#undef nk_define_conditional_negating_swap_
#undef nk_define_sum_two_products_
#undef nk_define_approximate_givens_quaternion_
#undef nk_define_jacobi_conjugation_
#undef nk_define_quaternion_to_mat3x3_
#undef nk_define_jacobi_eigenanalysis_
#undef nk_define_qr_givens_quaternion_
#undef nk_define_sort_singular_values_
#undef nk_define_qr_orthogonal_factor_
#undef nk_define_svd3x3_
#undef nk_define_det3x3_
#undef nk_define_shapes_pack_
#undef nk_define_shapes_headers_
#undef nk_define_shapes_nearest_
#undef nk_define_shapes_workspace_
#undef nk_define_shapes_pricing_
#undef nk_define_shapes_transport_
#undef nk_define_shapes_

#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("", on)
#elif defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif

#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_MESHES_SERIAL_H
