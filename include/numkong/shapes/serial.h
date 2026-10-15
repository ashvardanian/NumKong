/**
 *  @file include/numkong/shapes/serial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief Many-to-many shape distances between 3D point clouds for SIMD-free CPUs.
 *
 *  @sa include/numkong/shapes.h
 *
 *  Every capability packs through the serial routine here and solves exact transport through its
 *  own network simplex: a northwest-corner tree rooted at the first row point, block pricing, and
 *  the leaving arc by Cunningham's rule, which keeps the tree strongly feasible so that degenerate
 *  pivots never cycle.
 */
#ifndef NUMKONG_SHAPES_SERIAL_H
#define NUMKONG_SHAPES_SERIAL_H

#include "numkong/types.h"
#include "numkong/shapes.h"        // `nk_shapes_pack_f16_serial`, `nk_shapes_unpack_f16_serial`
#include "numkong/cast/serial.h"   // `nk_f16_to_f32_serial_`, `nk_f32_to_f16_serial_`
#include "numkong/scalar/serial.h" // `nk_sqrt_f32_serial_`, `nk_rsqrt_f64_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

/** Byte boundary every axis of a pack and every array of a transport workspace starts on, within a
 *  buffer on an @c nk_default_alignment_k boundary. */
enum { nk_shapes_alignment_k_ = 64 };

/** Opens every pack of @c f16 points; the x, y and z arrays of @c f16 follow. */
typedef struct nk_shapes_packed_header_f32_t {

    /** Number of 3D points. */
    nk_u32_t points;

    /** Mean input point. */
    nk_f32_t centroid[3];

    /** Unit quaternion (x, y, z, w) of 𝑄, rotating centered inputs onto the principal axes. */
    nk_f32_t frame[4];

    /** What the rotated inputs were divided by: their RMS distance from the centroid, or 1. */
    nk_f32_t scale;

    /** Descending eigenvalues of the packed points' second moments. */
    nk_f32_t spreads[3];

    /** Largest packed point norm. */
    nk_f32_t max_radius;
} nk_shapes_packed_header_f32_t;

/** Opens every pack of @c f32 points; the x, y and z arrays of @c f32 follow. */
typedef struct nk_shapes_packed_header_f64_t {

    /** Number of 3D points. */
    nk_u64_t points;

    /** Mean input point. */
    nk_f64_t centroid[3];

    /** Unit quaternion (x, y, z, w) of 𝑄, rotating centered inputs onto the principal axes. */
    nk_f64_t frame[4];

    /** What the rotated inputs were divided by: their RMS distance from the centroid, or 1. */
    nk_f64_t scale;

    /** Descending eigenvalues of the packed points' second moments. */
    nk_f64_t spreads[3];

    /** Largest packed point norm. */
    nk_f64_t max_radius;
} nk_shapes_packed_header_f64_t;

/** The x array of a pack of @c f16 points, with the elements from each axis to the next. */
NUMKONG_INLINE nk_f16_t const *nk_shapes_packed_axes_f16_serial_(void const *packed, nk_size_t *axis_stride) {
    nk_size_t const points = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f32_t),
                                                                 nk_shapes_alignment_k_);
    *axis_stride = nk_size_round_up_to_multiple_(points * sizeof(nk_f16_t), nk_shapes_alignment_k_) / sizeof(nk_f16_t);
    return (nk_f16_t const *)((char const *)packed + header_bytes);
}

/** The x array of a pack of @c f32 points, with the elements from each axis to the next. */
NUMKONG_INLINE nk_f32_t const *nk_shapes_packed_axes_f32_serial_(void const *packed, nk_size_t *axis_stride) {
    nk_size_t const points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)packed)->points;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f64_t),
                                                                 nk_shapes_alignment_k_);
    *axis_stride = nk_size_round_up_to_multiple_(points * sizeof(nk_f32_t), nk_shapes_alignment_k_) / sizeof(nk_f32_t);
    return (nk_f32_t const *)((char const *)packed + header_bytes);
}

/** The x array of a pack of @c f64 points, with the elements from each axis to the next. */
NUMKONG_INLINE nk_f64_t const *nk_shapes_packed_axes_f64_serial_(void const *packed, nk_size_t *axis_stride) {
    nk_size_t const points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)packed)->points;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f64_t),
                                                                 nk_shapes_alignment_k_);
    *axis_stride = nk_size_round_up_to_multiple_(points * sizeof(nk_f64_t), nk_shapes_alignment_k_) / sizeof(nk_f64_t);
    return (nk_f64_t const *)((char const *)packed + header_bytes);
}

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
 *  coincides, as the rounded centroid of coincident points still leaves a spread. Every product
 *  sits in a statement of its own, as Clang contracts only within one and GCC's ISO modes not at
 *  all, so the arithmetic rounds alike on machines with and without FMA units. */
NUMKONG_INLINE nk_f32_t nk_shapes_moments_f16_serial_(nk_f16_t const *points, nk_size_t count, nk_f32_t *centroid,
                                                      nk_f32_t *covariance) {
    nk_f32_t x, y, z, product, sum_x = 0, sum_y = 0, sum_z = 0, first[3];
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    nk_f16_to_f32_serial_(points + 0, &first[0]), nk_f16_to_f32_serial_(points + 1, &first[1]);
    nk_f16_to_f32_serial_(points + 2, &first[2]);
    int coincident = 1;
    for (nk_size_t index = 0; index != count; ++index) {
        nk_f16_to_f32_serial_(points + index * 3 + 0, &x), nk_f16_to_f32_serial_(points + index * 3 + 1, &y);
        nk_f16_to_f32_serial_(points + index * 3 + 2, &z);
        sum_x += x, sum_y += y, sum_z += z;
        coincident &= x == first[0] && y == first[1] && z == first[2];
    }
    nk_f32_t const points_count = (nk_f32_t)count;
    centroid[0] = sum_x / points_count, centroid[1] = sum_y / points_count, centroid[2] = sum_z / points_count;
    for (nk_size_t index = 0; index != count; ++index) {
        nk_f16_to_f32_serial_(points + index * 3 + 0, &x), nk_f16_to_f32_serial_(points + index * 3 + 1, &y);
        nk_f16_to_f32_serial_(points + index * 3 + 2, &z);
        x -= centroid[0], y -= centroid[1], z -= centroid[2];
        product = x * x, covariance[0] += product, product = y * x, covariance[1] += product;
        product = y * y, covariance[2] += product, product = z * x, covariance[3] += product;
        product = z * y, covariance[4] += product, product = z * z, covariance[5] += product;
    }
    for (int entry = 0; entry != 6; ++entry) covariance[entry] /= points_count;
    return coincident ? 0 : nk_sqrt_f32_serial_(covariance[0] + covariance[2] + covariance[5]);
}

/** @copydoc nk_shapes_moments_f16_serial_ */
NUMKONG_INLINE nk_f64_t nk_shapes_moments_f32_serial_(nk_f32_t const *points, nk_size_t count, nk_f64_t *centroid,
                                                      nk_f64_t *covariance) {
    nk_f64_t x, y, z, product, sum_x = 0, sum_y = 0, sum_z = 0;
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    nk_f64_t const first[3] = {points[0], points[1], points[2]};
    int coincident = 1;
    for (nk_size_t index = 0; index != count; ++index) {
        x = points[index * 3 + 0], y = points[index * 3 + 1], z = points[index * 3 + 2];
        sum_x += x, sum_y += y, sum_z += z;
        coincident &= x == first[0] && y == first[1] && z == first[2];
    }
    nk_f64_t const points_count = (nk_f64_t)count;
    centroid[0] = sum_x / points_count, centroid[1] = sum_y / points_count, centroid[2] = sum_z / points_count;
    for (nk_size_t index = 0; index != count; ++index) {
        x = points[index * 3 + 0], y = points[index * 3 + 1], z = points[index * 3 + 2];
        x -= centroid[0], y -= centroid[1], z -= centroid[2];
        product = x * x, covariance[0] += product, product = y * x, covariance[1] += product;
        product = y * y, covariance[2] += product, product = z * x, covariance[3] += product;
        product = z * y, covariance[4] += product, product = z * z, covariance[5] += product;
    }
    for (int entry = 0; entry != 6; ++entry) covariance[entry] /= points_count;
    return coincident ? 0 : nk_sqrt_f64_serial_(covariance[0] + covariance[2] + covariance[5]);
}

/** The moments of a cloud of @c f64 points, like the @c f32 ones with the centroid and the
 *  covariance summed by TwoSum and Dot2. */
NUMKONG_INLINE nk_f64_t nk_shapes_moments_f64_serial_(nk_f64_t const *points, nk_size_t count, nk_f64_t *centroid,
                                                      nk_f64_t *covariance) {
    nk_f64_t x, y, z, sums[3] = {0, 0, 0}, sum_compensations[3] = {0, 0, 0};
    nk_f64_t compensations[6] = {0, 0, 0, 0, 0, 0};
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    int coincident = 1;
    for (nk_size_t index = 0; index != count; ++index) {
        // A Dot2 step by one is a TwoSum
        for (int axis = 0; axis != 3; ++axis)
            nk_dot2_f64_serial_(&sums[axis], &sum_compensations[axis], points[index * 3 + axis], 1);
        coincident &= points[index * 3 + 0] == points[0] && points[index * 3 + 1] == points[1] &&
                      points[index * 3 + 2] == points[2];
    }
    nk_f64_t const points_count = (nk_f64_t)count;
    for (int axis = 0; axis != 3; ++axis) centroid[axis] = (sums[axis] + sum_compensations[axis]) / points_count;
    for (nk_size_t index = 0; index != count; ++index) {
        x = points[index * 3 + 0] - centroid[0], y = points[index * 3 + 1] - centroid[1];
        z = points[index * 3 + 2] - centroid[2];
        nk_dot2_f64_serial_(&covariance[0], &compensations[0], x, x);
        nk_dot2_f64_serial_(&covariance[1], &compensations[1], y, x);
        nk_dot2_f64_serial_(&covariance[2], &compensations[2], y, y);
        nk_dot2_f64_serial_(&covariance[3], &compensations[3], z, x);
        nk_dot2_f64_serial_(&covariance[4], &compensations[4], z, y);
        nk_dot2_f64_serial_(&covariance[5], &compensations[5], z, z);
    }
    for (int entry = 0; entry != 6; ++entry)
        covariance[entry] = (covariance[entry] + compensations[entry]) / points_count;
    return coincident ? 0 : nk_sqrt_f64_serial_(covariance[0] + covariance[2] + covariance[5]);
}

/** The 6 unique second moments of a pack's stored points in the order Jacobi takes them, xx, yx,
 *  yy, zx, zy and zz, each divided by the count, returning the largest squared norm. */
NUMKONG_INLINE nk_f32_t nk_shapes_packed_moments_f16_serial_(nk_f16_t const *packed_x, nk_size_t axis_stride,
                                                             nk_size_t count, nk_f32_t *moments) {
    nk_f32_t x, y, z, product, max_norm_squared = 0;
    for (int entry = 0; entry != 6; ++entry) moments[entry] = 0;
    for (nk_size_t index = 0; index != count; ++index) {
        nk_f16_to_f32_serial_(packed_x + index, &x), nk_f16_to_f32_serial_(packed_x + axis_stride + index, &y);
        nk_f16_to_f32_serial_(packed_x + 2 * axis_stride + index, &z);
        nk_f32_t const xx = x * x, yy = y * y, zz = z * z, norm_squared = xx + yy + zz;
        moments[0] += xx, moments[2] += yy, moments[5] += zz;
        product = y * x, moments[1] += product, product = z * x, moments[3] += product;
        product = z * y, moments[4] += product;
        max_norm_squared = norm_squared > max_norm_squared ? norm_squared : max_norm_squared;
    }
    for (int entry = 0; entry != 6; ++entry) moments[entry] /= (nk_f32_t)count;
    return max_norm_squared;
}

/** @copydoc nk_shapes_packed_moments_f16_serial_ */
NUMKONG_INLINE nk_f64_t nk_shapes_packed_moments_f32_serial_(nk_f32_t const *packed_x, nk_size_t axis_stride,
                                                             nk_size_t count, nk_f64_t *moments) {
    nk_f64_t x, y, z, product, max_norm_squared = 0;
    for (int entry = 0; entry != 6; ++entry) moments[entry] = 0;
    for (nk_size_t index = 0; index != count; ++index) {
        x = packed_x[index], y = packed_x[axis_stride + index], z = packed_x[2 * axis_stride + index];
        nk_f64_t const xx = x * x, yy = y * y, zz = z * z, norm_squared = xx + yy + zz;
        moments[0] += xx, moments[2] += yy, moments[5] += zz;
        product = y * x, moments[1] += product, product = z * x, moments[3] += product;
        product = z * y, moments[4] += product;
        max_norm_squared = norm_squared > max_norm_squared ? norm_squared : max_norm_squared;
    }
    for (int entry = 0; entry != 6; ++entry) moments[entry] /= (nk_f64_t)count;
    return max_norm_squared;
}

/** The second moments of a pack of @c f64 points, like the @c f32 ones summed by Dot2. */
NUMKONG_INLINE nk_f64_t nk_shapes_packed_moments_f64_serial_(nk_f64_t const *packed_x, nk_size_t axis_stride,
                                                             nk_size_t count, nk_f64_t *moments) {
    nk_f64_t compensations[6] = {0, 0, 0, 0, 0, 0}, max_norm_squared = 0;
    for (int entry = 0; entry != 6; ++entry) moments[entry] = 0;
    for (nk_size_t index = 0; index != count; ++index) {
        nk_f64_t const x = packed_x[index], y = packed_x[axis_stride + index], z = packed_x[2 * axis_stride + index];
        nk_f64_t const xx = x * x, yy = y * y, zz = z * z, norm_squared = xx + yy + zz;
        nk_dot2_f64_serial_(&moments[0], &compensations[0], x, x);
        nk_dot2_f64_serial_(&moments[1], &compensations[1], y, x);
        nk_dot2_f64_serial_(&moments[2], &compensations[2], y, y);
        nk_dot2_f64_serial_(&moments[3], &compensations[3], z, x);
        nk_dot2_f64_serial_(&moments[4], &compensations[4], z, y);
        nk_dot2_f64_serial_(&moments[5], &compensations[5], z, z);
        max_norm_squared = norm_squared > max_norm_squared ? norm_squared : max_norm_squared;
    }
    for (int entry = 0; entry != 6; ++entry) moments[entry] = (moments[entry] + compensations[entry]) / (nk_f64_t)count;
    return max_norm_squared;
}

/** The pack of one cloud from its moments: the header, then the x, y and z arrays of its points
 *  centered, rotated onto the principal axes and divided by @p scale. Then its inverse, which
 *  restores every packed point as scale × 𝑄ᵀ × x′ + centroid. Every product sits in a statement of
 *  its own, as the moments' do. */
#define nk_define_shapes_pack_(input_type, math_type, points_type, load_and_convert, convert_and_store, compute_sqrt,  \
                               compute_rsqrt, tie_tolerance)                                                           \
    NUMKONG_INLINE void nk_shapes_centered_##input_type##_serial_(nk_##input_type##_t const *points, nk_size_t index,  \
                                                                  nk_##math_type##_t const *centroid,                  \
                                                                  nk_##math_type##_t *centered) {                      \
        load_and_convert(points + index * 3 + 0, &centered[0]),                                                        \
            load_and_convert(points + index * 3 + 1, &centered[1]);                                                    \
        load_and_convert(points + index * 3 + 2, &centered[2]);                                                        \
        centered[0] -= centroid[0], centered[1] -= centroid[1], centered[2] -= centroid[2];                            \
    }                                                                                                                  \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_dot_##input_type##_serial_(nk_##math_type##_t const *left,             \
                                                                           nk_##math_type##_t const *right) {          \
        nk_##math_type##_t sum = left[0] * right[0], product;                                                          \
        product = left[1] * right[1], sum += product;                                                                  \
        product = left[2] * right[2], sum += product;                                                                  \
        return sum;                                                                                                    \
    }                                                                                                                  \
    /* The unit complex number at 1 / order of the angle of (real, imaginary) in (−π, π], as the                   \
     * product of the halvings of that angle that the binary digits of 1 / order pick */                               \
    NUMKONG_INLINE void nk_shapes_unit_root_##input_type##_serial_(                                                    \
        nk_##math_type##_t real, nk_##math_type##_t imaginary, int order, nk_##math_type##_t *root) {                  \
        nk_##math_type##_t root_real = 1, root_imaginary = 0, half_real, half_imaginary, product;                      \
        if (order == 1) {                                                                                              \
            root[0] = real, root[1] = imaginary;                                                                       \
            return;                                                                                                    \
        }                                                                                                              \
        int remainder = 1;                                                                                             \
        for (int digit = 0; digit != (int)sizeof(nk_##math_type##_t) * 7; ++digit) {                                   \
            /* Halving through the larger of the half angle's cosine and sine keeps divisors big */                    \
            if (real >= 0) half_real = compute_sqrt((1 + real) / 2), half_imaginary = imaginary / (2 * half_real);     \
            else                                                                                                       \
                half_imaginary = (imaginary < 0 ? -1 : 1) * compute_sqrt((1 - real) / 2),                              \
                half_real = imaginary / (2 * half_imaginary);                                                          \
            real = half_real, imaginary = half_imaginary, remainder *= 2;                                              \
            if (remainder < order) continue;                                                                           \
            remainder -= order;                                                                                        \
            half_real = root_real * real, product = root_imaginary * imaginary, half_real -= product;                  \
            root_imaginary *= real, product = root_real * imaginary, root_imaginary += product, root_real = half_real; \
        }                                                                                                              \
        product = root_real * root_real, half_real = product, product = root_imaginary * root_imaginary;               \
        half_real += product, half_real = compute_rsqrt(half_real);                                                    \
        root[0] = root_real * half_real, root[1] = root_imaginary * half_real;                                         \
    }                                                                                                                  \
    /* Turns basis rows a and b about row c until the first harmonic above the tolerance, a sum                        \
     * of wᵐ over w = x · a + i x · b, or of h × wᵐ over h = x · c, lies along a, and settles the              \
     * m choices of a by the next harmonic that tells them apart. First orients c by Σh³ */                          \
    NUMKONG_INLINE void nk_shapes_turn_plane_##input_type##_serial_(                                                   \
        nk_##input_type##_t const *points, nk_size_t count, nk_##math_type##_t const *centroid,                        \
        nk_##math_type##_t *basis) {                                                                                   \
        nk_##math_type##_t centered[3], height, distance, product, cubes = 0, cube_magnitudes = 0;                     \
        for (nk_size_t index = 0; index != count; ++index) {                                                           \
            nk_shapes_centered_##input_type##_serial_(points, index, centroid, centered);                              \
            height = nk_shapes_dot_##input_type##_serial_(basis + 6, centered);                                        \
            product = height * height, product *= height, cubes += product;                                            \
            distance = compute_sqrt(nk_shapes_dot_##input_type##_serial_(centered, centered));                         \
            product = distance * distance, product *= distance, cube_magnitudes += product;                            \
        }                                                                                                              \
        int const oriented = (cubes < 0 ? -cubes : cubes) > tie_tolerance * cube_magnitudes;                           \
        if (oriented && cubes < 0)                                                                                     \
            for (int cell = 3; cell != 9; ++cell) basis[cell] = -basis[cell];                                          \
        /* Even entries sum wᵐ and odd ones h × wᵐ for m = 1 to 8, beside sums of |w|ᵐ and |x|ᵐ⁺¹ */                   \
        nk_##math_type##_t harmonics[16][2], magnitudes[16];                                                           \
        for (int entry = 0; entry != 16; ++entry)                                                                      \
            harmonics[entry][0] = 0, harmonics[entry][1] = 0, magnitudes[entry] = 0;                                   \
        for (nk_size_t index = 0; index != count; ++index) {                                                           \
            nk_shapes_centered_##input_type##_serial_(points, index, centroid, centered);                              \
            nk_##math_type##_t const along = nk_shapes_dot_##input_type##_serial_(basis, centered),                    \
                                     across = nk_shapes_dot_##input_type##_serial_(basis + 3, centered);               \
            height = nk_shapes_dot_##input_type##_serial_(basis + 6, centered);                                        \
            distance = compute_sqrt(nk_shapes_dot_##input_type##_serial_(centered, centered));                         \
            nk_##math_type##_t radius = along * along, power_real = 1, power_imaginary = 0, power_radius = 1;          \
            nk_##math_type##_t power_distance = distance, next;                                                        \
            product = across * across, radius += product, radius = compute_sqrt(radius);                               \
            for (int entry = 0; entry != 16; entry += 2) {                                                             \
                next = power_real * along, product = power_imaginary * across, next -= product;                        \
                power_imaginary *= along, product = power_real * across, power_imaginary += product;                   \
                power_real = next, power_radius *= radius, power_distance *= distance;                                 \
                harmonics[entry][0] += power_real, harmonics[entry][1] += power_imaginary;                             \
                magnitudes[entry] += power_radius, magnitudes[entry + 1] += power_distance;                            \
                product = height * power_real, harmonics[entry + 1][0] += product;                                     \
                product = height * power_imaginary, harmonics[entry + 1][1] += product;                                \
            }                                                                                                          \
        }                                                                                                              \
        /* Unless c is oriented, flipping it conjugates every wᵐ sum and negates its twin, so a                      \
           twin only chooses at odd m, and only harmonics flipping like the chosen root settle */                      \
        int chosen = -1, settling = -1;                                                                                \
        for (int entry = 0; entry != 16 && settling < 0; ++entry) {                                                    \
            int const twin = entry & 1, odd = (entry / 2) % 2 == 0;                                                    \
            if (!oriented && (chosen < 0 ? twin && !odd : twin != ((chosen & 1) && odd))) continue;                    \
            if (chosen >= 0 && (entry / 2 + 1) % (chosen / 2 + 1) == 0) continue;                                      \
            nk_##math_type##_t strength = harmonics[entry][0] * harmonics[entry][0],                                   \
                               threshold = tie_tolerance * magnitudes[entry];                                          \
            product = harmonics[entry][1] * harmonics[entry][1], strength += product, threshold *= threshold;          \
            if (!(strength > threshold)) continue;                                                                     \
            if (chosen < 0) chosen = entry;                                                                            \
            else settling = entry;                                                                                     \
        }                                                                                                              \
        if (chosen < 0) return;                                                                                        \
        int const order = chosen / 2 + 1;                                                                              \
        nk_##math_type##_t root[2], turn[2], candidate[2], power[2], next, score, best_score = 0;                      \
        product = harmonics[chosen][0] * harmonics[chosen][0], next = product;                                         \
        product = harmonics[chosen][1] * harmonics[chosen][1], next += product, next = compute_rsqrt(next);            \
        nk_shapes_unit_root_##input_type##_serial_(harmonics[chosen][0] * next, harmonics[chosen][1] * next, order,    \
                                                   root);                                                              \
        if (settling >= 0) {                                                                                           \
            /* The m candidates for a step by e^(2πi / m), the square of the m-th root of −1 */                        \
            nk_shapes_unit_root_##input_type##_serial_(-1, 0, order, turn);                                            \
            next = turn[0] * turn[0], product = turn[1] * turn[1], next -= product;                                    \
            turn[1] *= turn[0], turn[1] *= 2, turn[0] = next;                                                          \
            candidate[0] = root[0], candidate[1] = root[1];                                                            \
            for (int choice = 0; choice != order; ++choice) {                                                          \
                /* The settling harmonic as the candidate would leave it, × conj(candidate)ⁿ */                        \
                power[0] = harmonics[settling][0], power[1] = harmonics[settling][1];                                  \
                for (int step = 0; step != settling / 2 + 1; ++step) {                                                 \
                    next = power[0] * candidate[0], product = power[1] * candidate[1], next += product;                \
                    power[1] *= candidate[0], product = power[0] * candidate[1], power[1] -= product, power[0] = next; \
                }                                                                                                      \
                score = power[0];                                                                                      \
                if (choice == 0 || score > best_score)                                                                 \
                    best_score = score, root[0] = candidate[0], root[1] = candidate[1];                                \
                next = candidate[0] * turn[0], product = candidate[1] * turn[1], next -= product;                      \
                candidate[1] *= turn[0], product = candidate[0] * turn[1], candidate[1] += product,                    \
                                         candidate[0] = next;                                                          \
            }                                                                                                          \
        }                                                                                                              \
        /* w × conj(root) lays the chosen harmonic along a */                                                          \
        for (int axis = 0; axis != 3; ++axis) {                                                                        \
            nk_##math_type##_t const along = basis[axis], across = basis[3 + axis];                                    \
            nk_##math_type##_t turned = root[0] * along;                                                               \
            product = root[1] * across, turned += product, basis[axis] = turned;                                       \
            turned = root[0] * across, product = root[1] * along, turned -= product, basis[3 + axis] = turned;         \
        }                                                                                                              \
    }                                                                                                                  \
    /* Points basis row c at the point whose direction u maximizes Σ(x · u)³, or Σ(x · u)⁴ when                 \
     * no cubic clears the tolerance, completes a proper basis and turns the plane about c */                          \
    NUMKONG_INLINE void nk_shapes_pick_axis_##input_type##_serial_(nk_##input_type##_t const *points, nk_size_t count, \
                                                                   nk_##math_type##_t const *centroid,                 \
                                                                   nk_##math_type##_t *basis) {                        \
        nk_##math_type##_t centered[3], candidate[3], product, cube_magnitudes = 0, best_cubes = 0, best_quartics = 0; \
        nk_size_t cubes_index = count, quartics_index = count;                                                         \
        for (nk_size_t candidate_index = 0; candidate_index != count; ++candidate_index) {                             \
            nk_shapes_centered_##input_type##_serial_(points, candidate_index, centroid, candidate);                   \
            nk_##math_type##_t const length_squared = nk_shapes_dot_##input_type##_serial_(candidate, candidate),      \
                                     length = compute_sqrt(length_squared);                                            \
            product = length_squared * length, cube_magnitudes += product;                                             \
            if (!(length_squared > 0)) continue;                                                                       \
            nk_##math_type##_t cubes = 0, quartics = 0;                                                                \
            for (nk_size_t index = 0; index != count; ++index) {                                                       \
                nk_shapes_centered_##input_type##_serial_(points, index, centroid, centered);                          \
                nk_##math_type##_t const projection = nk_shapes_dot_##input_type##_serial_(centered, candidate),       \
                                         square = projection * projection;                                             \
                product = square * projection, cubes += product, product = square * square, quartics += product;       \
            }                                                                                                          \
            product = length_squared * length, cubes /= product, product = length_squared * length_squared;            \
            quartics /= product;                                                                                       \
            if (cubes > best_cubes) best_cubes = cubes, cubes_index = candidate_index;                                 \
            if (quartics > best_quartics) best_quartics = quartics, quartics_index = candidate_index;                  \
        }                                                                                                              \
        nk_size_t const chosen = best_cubes > tie_tolerance * cube_magnitudes ? cubes_index : quartics_index;          \
        if (chosen == count) return;                                                                                   \
        nk_##math_type##_t axis[3], least_alignment = 2, alignment, inverse;                                           \
        nk_shapes_centered_##input_type##_serial_(points, chosen, centroid, axis);                                     \
        inverse = compute_rsqrt(nk_shapes_dot_##input_type##_serial_(axis, axis));                                     \
        axis[0] *= inverse, axis[1] *= inverse, axis[2] *= inverse;                                                    \
        /* a starts from the row least aligned with c, which keeps most of its length */                               \
        int start = 0;                                                                                                 \
        for (int row = 0; row != 3; ++row) {                                                                           \
            alignment = nk_shapes_dot_##input_type##_serial_(basis + row * 3, axis);                                   \
            alignment = alignment < 0 ? -alignment : alignment;                                                        \
            if (alignment < least_alignment) least_alignment = alignment, start = row;                                 \
        }                                                                                                              \
        alignment = nk_shapes_dot_##input_type##_serial_(basis + start * 3, axis);                                     \
        for (int cell = 0; cell != 3; ++cell)                                                                          \
            product = alignment * axis[cell], candidate[cell] = basis[start * 3 + cell] - product;                     \
        inverse = compute_rsqrt(nk_shapes_dot_##input_type##_serial_(candidate, candidate));                           \
        for (int cell = 0; cell != 3; ++cell) basis[cell] = candidate[cell] * inverse, basis[6 + cell] = axis[cell];   \
        /* b = c × a completes a proper basis */                                                                       \
        for (int cell = 0; cell != 3; ++cell) {                                                                        \
            int const next = (cell + 1) % 3, after = (cell + 2) % 3;                                                   \
            nk_##math_type##_t cross = axis[next] * basis[after];                                                      \
            product = axis[after] * basis[next], cross -= product, basis[3 + cell] = cross;                            \
        }                                                                                                              \
        nk_shapes_turn_plane_##input_type##_serial_(points, count, centroid, basis);                                   \
    }                                                                                                                  \
    NUMKONG_INLINE void nk_shapes_pack_##input_type##_serial_(                                                         \
        nk_##input_type##_t const *points, nk_size_t count, nk_##math_type##_t const *centroid,                        \
        nk_##math_type##_t const *covariance, nk_##math_type##_t scale, void *packed) {                                \
        nk_shapes_packed_header_##math_type##_t *header = (nk_shapes_packed_header_##math_type##_t *)packed;           \
        nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_##math_type##_t),  \
                                                                     nk_shapes_alignment_k_);                          \
        nk_size_t const axis_bytes = nk_size_round_up_to_multiple_(count * sizeof(nk_##input_type##_t),                \
                                                                   nk_shapes_alignment_k_);                            \
        nk_u8_t *packed_bytes = (nk_u8_t *)packed;                                                                     \
        for (nk_size_t byte_index = 0; byte_index != header_bytes + 3 * axis_bytes; ++byte_index)                      \
            packed_bytes[byte_index] = 0;                                                                              \
        header->frame[3] = 1, header->scale = scale;                                                                   \
        if (!count) return;                                                                                            \
        nk_size_t const axis_stride = axis_bytes / sizeof(nk_##input_type##_t);                                        \
        nk_##input_type##_t *packed_x = (nk_##input_type##_t *)(packed_bytes + header_bytes);                          \
        nk_##input_type##_t *packed_y = packed_x + axis_stride, *packed_z = packed_y + axis_stride;                    \
        nk_##math_type##_t x, y, z, product;                                                                           \
        nk_##math_type##_t covariance_xx = covariance[0], covariance_yx = covariance[1];                               \
        nk_##math_type##_t covariance_yy = covariance[2], covariance_zx = covariance[3];                               \
        nk_##math_type##_t covariance_zy = covariance[4], covariance_zz = covariance[5];                               \
        nk_##math_type##_t quaternion[4], axes[9];                                                                     \
        nk_jacobi_eigenanalysis_##math_type##_serial_(&covariance_xx, &covariance_yx, &covariance_yy, &covariance_zx,  \
                                                      &covariance_zy, &covariance_zz, quaternion);                     \
        nk_quaternion_to_mat3x3_##math_type##_serial_(quaternion, axes);                                               \
        nk_##math_type##_t eigenvalue_0 = covariance_xx, eigenvalue_1 = covariance_yy, eigenvalue_2 = covariance_zz;   \
        /* Each swap moves one eigenvector and negates the other, so the axes stay a rotation */                       \
        int should_swap = eigenvalue_0 < eigenvalue_1;                                                                 \
        for (int row = 0; row != 3; ++row)                                                                             \
            nk_conditional_negating_swap_##math_type##_serial_(should_swap, &axes[row * 3 + 0], &axes[row * 3 + 1]);   \
        nk_cond_swap_##math_type##_serial_(should_swap, &eigenvalue_0, &eigenvalue_1);                                 \
        should_swap = eigenvalue_0 < eigenvalue_2;                                                                     \
        for (int row = 0; row != 3; ++row)                                                                             \
            nk_conditional_negating_swap_##math_type##_serial_(should_swap, &axes[row * 3 + 0], &axes[row * 3 + 2]);   \
        nk_cond_swap_##math_type##_serial_(should_swap, &eigenvalue_0, &eigenvalue_2);                                 \
        should_swap = eigenvalue_1 < eigenvalue_2;                                                                     \
        for (int row = 0; row != 3; ++row)                                                                             \
            nk_conditional_negating_swap_##math_type##_serial_(should_swap, &axes[row * 3 + 1], &axes[row * 3 + 2]);   \
        nk_cond_swap_##math_type##_serial_(should_swap, &eigenvalue_1, &eigenvalue_2);                                 \
        /* Tied eigenvalues leave their eigenvectors to rounding, so higher moments settle them,                       \
           with rows a and b spanning the tied plane and c the third axis, in a proper order */                        \
        nk_##math_type##_t const tie = tie_tolerance * eigenvalue_0;                                                   \
        int const upper_tied = eigenvalue_0 - eigenvalue_1 <= tie, lower_tied = eigenvalue_1 - eigenvalue_2 <= tie;    \
        if (upper_tied || lower_tied) {                                                                                \
            int const first = upper_tied ? 0 : 1;                                                                      \
            nk_##math_type##_t basis[9];                                                                               \
            for (int row = 0; row != 3; ++row)                                                                         \
                for (int cell = 0; cell != 3; ++cell) basis[row * 3 + cell] = axes[cell * 3 + (first + row) % 3];      \
            if (upper_tied && lower_tied) nk_shapes_pick_axis_##input_type##_serial_(points, count, centroid, basis);  \
            else nk_shapes_turn_plane_##input_type##_serial_(points, count, centroid, basis);                          \
            for (int row = 0; row != 3; ++row)                                                                         \
                for (int cell = 0; cell != 3; ++cell) axes[cell * 3 + (first + row) % 3] = basis[row * 3 + cell];      \
        }                                                                                                              \
        /* Shepperd's method divides by the largest of 4w², 4x², 4y² and 4z² */                                        \
        nk_##math_type##_t const trace = axes[0] + axes[4] + axes[8];                                                  \
        nk_##math_type##_t divisor;                                                                                    \
        if (trace > 0) {                                                                                               \
            divisor = 2 * compute_sqrt(trace + 1);                                                                     \
            quaternion[0] = (axes[7] - axes[5]) / divisor, quaternion[1] = (axes[2] - axes[6]) / divisor;              \
            quaternion[2] = (axes[3] - axes[1]) / divisor, quaternion[3] = divisor / 4;                                \
        }                                                                                                              \
        else if (axes[0] > axes[4] && axes[0] > axes[8]) {                                                             \
            divisor = 2 * compute_sqrt(1 + axes[0] - axes[4] - axes[8]);                                               \
            quaternion[0] = divisor / 4, quaternion[1] = (axes[1] + axes[3]) / divisor;                                \
            quaternion[2] = (axes[2] + axes[6]) / divisor, quaternion[3] = (axes[7] - axes[5]) / divisor;              \
        }                                                                                                              \
        else if (axes[4] > axes[8]) {                                                                                  \
            divisor = 2 * compute_sqrt(1 + axes[4] - axes[0] - axes[8]);                                               \
            quaternion[0] = (axes[1] + axes[3]) / divisor, quaternion[1] = divisor / 4;                                \
            quaternion[2] = (axes[5] + axes[7]) / divisor, quaternion[3] = (axes[2] - axes[6]) / divisor;              \
        }                                                                                                              \
        else {                                                                                                         \
            divisor = 2 * compute_sqrt(1 + axes[8] - axes[0] - axes[4]);                                               \
            quaternion[0] = (axes[2] + axes[6]) / divisor, quaternion[1] = (axes[5] + axes[7]) / divisor;              \
            quaternion[2] = divisor / 4, quaternion[3] = (axes[3] - axes[1]) / divisor;                                \
        }                                                                                                              \
        nk_##math_type##_t const inverse_norm = compute_rsqrt(                                                         \
            nk_sum_two_products_##math_type##_serial_(quaternion[0], quaternion[0], quaternion[1], quaternion[1]) +    \
            nk_sum_two_products_##math_type##_serial_(quaternion[2], quaternion[2], quaternion[3], quaternion[3]));    \
        /* The header holds the frame 𝑄 = Vᵀ, whose quaternion is the conjugate of V's */                              \
        header->frame[0] = -quaternion[0] * inverse_norm, header->frame[1] = -quaternion[1] * inverse_norm;            \
        header->frame[2] = -quaternion[2] * inverse_norm, header->frame[3] = quaternion[3] * inverse_norm;             \
        nk_quaternion_to_mat3x3_##math_type##_serial_(header->frame, axes);                                            \
        /* Dividing 𝑄 by the scale divides every rotated point, and dividing by 1 changes no bit */                    \
        for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;                                                     \
        nk_##math_type##_t rotated[3], moments[6];                                                                     \
        for (nk_size_t index = 0; index != count; ++index) {                                                           \
            load_and_convert(points + index * 3 + 0, &x), load_and_convert(points + index * 3 + 1, &y);                \
            load_and_convert(points + index * 3 + 2, &z);                                                              \
            x -= centroid[0], y -= centroid[1], z -= centroid[2];                                                      \
            for (int axis = 0; axis != 3; ++axis) {                                                                    \
                rotated[axis] = axes[axis * 3 + 0] * x;                                                                \
                product = axes[axis * 3 + 1] * y, rotated[axis] += product;                                            \
                product = axes[axis * 3 + 2] * z, rotated[axis] += product;                                            \
            }                                                                                                          \
            convert_and_store(&rotated[0], packed_x + index), convert_and_store(&rotated[1], packed_y + index);        \
            convert_and_store(&rotated[2], packed_z + index);                                                          \
        }                                                                                                              \
        nk_##math_type##_t const max_norm_squared = nk_shapes_packed_moments_##input_type##_serial_(                   \
            packed_x, axis_stride, count, moments);                                                                    \
        nk_jacobi_eigenanalysis_##math_type##_serial_(&moments[0], &moments[1], &moments[2], &moments[3], &moments[4], \
                                                      &moments[5], quaternion);                                        \
        nk_cond_swap_##math_type##_serial_(moments[0] < moments[2], &moments[0], &moments[2]);                         \
        nk_cond_swap_##math_type##_serial_(moments[0] < moments[5], &moments[0], &moments[5]);                         \
        nk_cond_swap_##math_type##_serial_(moments[2] < moments[5], &moments[2], &moments[5]);                         \
        header->points = (nk_##points_type##_t)count;                                                                  \
        header->centroid[0] = centroid[0], header->centroid[1] = centroid[1], header->centroid[2] = centroid[2];       \
        header->spreads[0] = moments[0], header->spreads[1] = moments[2], header->spreads[2] = moments[5];             \
        header->max_radius = compute_sqrt(max_norm_squared);                                                           \
    }                                                                                                                  \
    NUMKONG_INLINE void nk_shapes_unpack_##input_type##_serial_(void const *packed, nk_##input_type##_t *points) {     \
        nk_shapes_packed_header_##math_type##_t const *header =                                                        \
            (nk_shapes_packed_header_##math_type##_t const *)packed;                                                   \
        nk_size_t axis_stride;                                                                                         \
        nk_##input_type##_t const *packed_x = nk_shapes_packed_axes_##input_type##_serial_(packed, &axis_stride);      \
        nk_size_t const count = (nk_size_t)header->points;                                                             \
        nk_##math_type##_t axes[9], packed_point[3], restored, product;                                                \
        nk_quaternion_to_mat3x3_##math_type##_serial_(header->frame, axes);                                            \
        /* Scaling 𝑄 scales every restored offset, and scaling by 1 changes no bit */                                  \
        for (int cell = 0; cell != 9; ++cell) axes[cell] *= header->scale;                                             \
        for (nk_size_t index = 0; index != count; ++index) {                                                           \
            for (int axis = 0; axis != 3; ++axis)                                                                      \
                load_and_convert(packed_x + axis * axis_stride + index, &packed_point[axis]);                          \
            for (int axis = 0; axis != 3; ++axis) {                                                                    \
                restored = axes[axis] * packed_point[0];                                                               \
                product = axes[3 + axis] * packed_point[1], restored += product;                                       \
                product = axes[6 + axis] * packed_point[2], restored += product;                                       \
                restored += header->centroid[axis];                                                                    \
                convert_and_store(&restored, points + index * 3 + axis);                                               \
            }                                                                                                          \
        }                                                                                                              \
    }

nk_define_shapes_pack_(f16, f32, u32, nk_f16_to_f32_serial_, nk_f32_to_f16_serial_, nk_sqrt_f32_serial_,
                       nk_rsqrt_f32_serial_, 0x1p-7f)
nk_define_shapes_pack_(f32, f64, u64, nk_assign_from_to_, nk_assign_from_to_, nk_sqrt_f64_serial_, nk_rsqrt_f64_serial_,
                       0x1p-16)
nk_define_shapes_pack_(f64, f64, u64, nk_assign_from_to_, nk_assign_from_to_, nk_sqrt_f64_serial_, nk_rsqrt_f64_serial_,
                       0x1p-16)

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
        nk_shapes_packed_header_##math_type##_t const *row_header =                                                   \
            (nk_shapes_packed_header_##math_type##_t const *)row_packed;                                              \
        nk_shapes_packed_header_##math_type##_t const *column_header =                                                \
            (nk_shapes_packed_header_##math_type##_t const *)column_packed;                                           \
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
            nk_shapes_packed_header_##math_type##_t const *header =                                                   \
                (nk_shapes_packed_header_##math_type##_t const *)packs[index];                                        \
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

/** One directed pass between @c f64 packs, like the @c f32 one with each flip's sum of roots
 *  compensated, which keeps a Chamfer distance a few roundings off at any cloud size. */
NUMKONG_INLINE void nk_shapes_nearest_f64_serial_(nk_f64_t const *outer_axes, nk_size_t outer_stride,
                                                  nk_size_t outer_count, nk_f64_t const *inner_axes,
                                                  nk_size_t inner_stride, nk_size_t inner_count, nk_f64_t *sums,
                                                  nk_f64_t *maxima) {
    nk_f64_t compensations[8] = {0, 0, 0, 0, 0, 0, 0, 0}, nearest[8];
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        nk_f64_t const outer_x = outer_axes[outer_index], outer_y = outer_axes[outer_stride + outer_index];
        nk_f64_t const outer_z = outer_axes[2 * outer_stride + outer_index];
        for (int flip = 0; flip != 8; ++flip) nearest[flip] = NUMKONG_F64_INF;
        for (nk_size_t inner_index = 0; inner_index != inner_count; ++inner_index) {
            nk_f64_t const inner_x = inner_axes[inner_index], inner_y = inner_axes[inner_stride + inner_index];
            nk_f64_t const inner_z = inner_axes[2 * inner_stride + inner_index];
            // A flip negates an inner axis, turning its difference into a sum
            nk_f64_t const minus_x = outer_x - inner_x, plus_x = outer_x + inner_x;
            nk_f64_t const minus_y = outer_y - inner_y, plus_y = outer_y + inner_y;
            nk_f64_t const minus_z = outer_z - inner_z, plus_z = outer_z + inner_z;
            nk_f64_t const minus_xx = minus_x * minus_x, plus_xx = plus_x * plus_x;
            nk_f64_t const minus_yy = minus_y * minus_y, plus_yy = plus_y * plus_y;
            nk_f64_t const minus_zz = minus_z * minus_z, plus_zz = plus_z * plus_z;
            nk_f64_t const planar[4] = {minus_xx + minus_yy, plus_xx + minus_yy, minus_xx + plus_yy, plus_xx + plus_yy};
            for (int flip = 0; flip != 8; ++flip) {
                nk_f64_t const distance_squared = planar[flip & 3] + ((flip & 4) ? plus_zz : minus_zz);
                nearest[flip] = distance_squared < nearest[flip] ? distance_squared : nearest[flip];
            }
        }
        for (int flip = 0; flip != 8; ++flip) {
            nk_f64_t const distance = nk_sqrt_f64_serial_(nearest[flip]);
            nk_dot2_f64_serial_(&sums[flip], &compensations[flip], distance, 1);
            maxima[flip] = distance > maxima[flip] ? distance : maxima[flip];
        }
    }
    for (int flip = 0; flip != 8; ++flip) sums[flip] += compensations[flip];
}

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

/** The transport workspace for clouds of up to @p row_points and @p column_points: the widened row
 *  and column points and the rotated column points, each as x, y and z arrays @c points_stride
 *  apart; the tree's parent, flow, depth, first child, next and previous sibling arrays,
 *  @c tree_stride apart, with their f64 potentials; and the best proper and improper plans as row,
 *  column and flow arrays, @c plan_stride apart. The f64 Gromov-Wasserstein matrices follow, in
 *  @c gromov_bytes: the dense plan, its gradient and a vertex's gradient, @c matrix_stride apart
 *  for @p row_points × @p column_points, then one block of @p row_points² + @p column_points² for
 *  the distances within each cloud. Its layout function carves one when given memory, leaving null
 *  views without it, and returns the bytes up to the matrices either way. */
#define nk_define_shapes_workspace_(math_type)                                                                       \
    typedef struct nk_shapes_workspace_##math_type##_serial_t {                                                      \
        nk_##math_type##_t *row_points, *column_points, *rotated_points;                                             \
        nk_f64_t *matrices, *potential;                                                                              \
        nk_u32_t *tree, *proper_plan, *improper_plan;                                                                \
        nk_size_t points_stride, tree_stride, plan_stride, matrix_stride, gromov_bytes;                              \
    } nk_shapes_workspace_##math_type##_serial_t;                                                                    \
    NUMKONG_INLINE nk_size_t nk_shapes_workspace_##math_type##_serial_(                                              \
        nk_size_t row_points, nk_size_t column_points, nk_u8_t *workspace,                                           \
        nk_shapes_workspace_##math_type##_serial_t *views) {                                                         \
        nk_size_t const points = row_points > column_points ? row_points : column_points;                            \
        nk_size_t const points_bytes = nk_size_round_up_to_multiple_(3 * points * sizeof(nk_##math_type##_t),        \
                                                                     nk_shapes_alignment_k_);                        \
        nk_size_t const nodes_bytes = nk_size_round_up_to_multiple_(2 * points * sizeof(nk_u32_t),                   \
                                                                    nk_shapes_alignment_k_);                         \
        nk_size_t const potential_bytes = nk_size_round_up_to_multiple_(2 * points * sizeof(nk_f64_t),               \
                                                                        nk_shapes_alignment_k_);                     \
        nk_size_t const plan_bytes = nk_size_round_up_to_multiple_((points ? 2 * points - 1 : 0) * sizeof(nk_u32_t), \
                                                                   nk_shapes_alignment_k_);                          \
        nk_size_t const matrix_bytes = nk_size_round_up_to_multiple_(row_points * column_points * sizeof(nk_f64_t),  \
                                                                     nk_shapes_alignment_k_);                        \
        nk_size_t const distances_bytes = nk_size_round_up_to_multiple_(                                             \
            (row_points * row_points + column_points * column_points) * sizeof(nk_f64_t), nk_shapes_alignment_k_);   \
        nk_size_t const transport_bytes = 3 * points_bytes + 6 * nodes_bytes + potential_bytes + 6 * plan_bytes;     \
        views->points_stride = points, views->tree_stride = nodes_bytes / sizeof(nk_u32_t);                          \
        views->plan_stride = plan_bytes / sizeof(nk_u32_t), views->matrix_stride = matrix_bytes / sizeof(nk_f64_t);  \
        views->gromov_bytes = 3 * matrix_bytes + distances_bytes;                                                    \
        views->row_points = views->column_points = views->rotated_points = NUMKONG_NULL;                             \
        views->matrices = views->potential = NUMKONG_NULL;                                                           \
        views->tree = views->proper_plan = views->improper_plan = NUMKONG_NULL;                                      \
        if (!workspace) return transport_bytes;                                                                      \
        /* The order matters, as price-block tail lanes read past rotated_points, potential and G */                 \
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
#define nk_define_shapes_wasserstein_procrustes_(input_type, math_type, suffix, infinity)                            \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_wasserstein_procrustes_##input_type##_##suffix##_(                   \
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

nk_define_shapes_wasserstein_procrustes_(f16, f32, serial, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f32, f64, serial, NUMKONG_F64_INF)

/** Wasserstein-Procrustes of one canonically ordered pair of @c f64 clouds, like the @c f32 one
 *  with each plan's cost and cross-covariance compensated. */
NUMKONG_INLINE nk_f64_t nk_shapes_wasserstein_procrustes_f64_serial_(nk_shapes_workspace_f64_serial_t const *views,
                                                                     nk_size_t rows, nk_size_t columns,
                                                                     nk_f64_t cost_scale, nk_f64_t *rotation) {
    nk_size_t const stride = views->points_stride;
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f64_t const mass = (nk_f64_t)(rows * columns);
    nk_f64_t best = NUMKONG_F64_INF, best_proper = NUMKONG_F64_INF, best_improper = NUMKONG_F64_INF;
    for (int flip = 0; flip != 8; ++flip) {
        nk_f64_t current[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, start_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        current[0] = (flip & 1) ? -1 : 1, current[4] = (flip & 2) ? -1 : 1, current[8] = (flip & 4) ? -1 : 1;
        nk_f64_t const flip_determinant = current[0] * current[4] * current[8];
        nk_u32_t *parity_plan = flip_determinant > 0 ? views->proper_plan : views->improper_plan;
        nk_f64_t *parity_best = flip_determinant > 0 ? &best_proper : &best_improper;
        nk_f64_t start_best = NUMKONG_F64_INF;
        nk_shapes_northwest_corner_serial_(rows, columns, views->tree, views->tree_stride);
        for (;;) {
            for (nk_size_t column = 0; column != columns; ++column) {
                nk_f64_t const x = views->column_points[column], y = views->column_points[stride + column];
                nk_f64_t const z = views->column_points[2 * stride + column];
                for (int axis = 0; axis != 3; ++axis)
                    views->rotated_points[axis * stride + column] = current[axis * 3] * x + current[axis * 3 + 1] * y +
                                                                    current[axis * 3 + 2] * z;
            }
            nk_shapes_network_simplex_f64_serial_(views, NUMKONG_NULL, rows, columns, cost_scale);
            nk_f64_t cost = 0, cost_compensation = 0;
            for (nk_size_t node = 1; node != rows + columns; ++node) {
                nk_size_t const row = node < rows ? node : parent[node];
                nk_size_t const column = (node < rows ? parent[node] : node) - rows;
                nk_dot2_f64_serial_(&cost, &cost_compensation, (nk_f64_t)flow[node],
                                    nk_shapes_point_cost_f64_serial_(views, row, column));
            }
            cost = (cost + cost_compensation) / mass;
            if (!(cost < start_best)) break;
            start_best = cost;
            for (int cell = 0; cell != 9; ++cell) start_rotation[cell] = current[cell];
            if (cost < *parity_best) {
                *parity_best = cost;
                nk_shapes_tree_arcs_serial_(rows, columns, views->tree, views->tree_stride, parity_plan,
                                            views->plan_stride);
            }
            nk_f64_t sums[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, compensations[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (nk_size_t node = 1; node != rows + columns; ++node) {
                if (!flow[node]) continue;
                nk_size_t const row = node < rows ? node : parent[node];
                nk_size_t const column = (node < rows ? parent[node] : node) - rows;
                nk_f64_t const weight = (nk_f64_t)flow[node] / mass;
                for (int column_axis = 0; column_axis != 3; ++column_axis) {
                    nk_f64_t const weighted = weight * views->column_points[column_axis * stride + column];
                    for (int row_axis = 0; row_axis != 3; ++row_axis)
                        nk_dot2_f64_serial_(&sums[column_axis * 3 + row_axis],
                                            &compensations[column_axis * 3 + row_axis], weighted,
                                            views->row_points[row_axis * stride + row]);
                }
            }
            nk_f64_t cross_covariance[9], svd_left[9], svd_diagonal[9], svd_right[9];
            for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = sums[cell] + compensations[cell];
            nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, current);
            // Each start stays among the rotations, or among the reflections, it began with
            if (nk_det3x3_f64_serial_(current) * flip_determinant < 0) {
                svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
                nk_rotation_from_svd_f64_serial_(svd_left, svd_right, current);
            }
        }
        if (start_best < best) {
            best = start_best;
            for (int cell = 0; cell != 9; ++cell) rotation[cell] = start_rotation[cell];
        }
    }
    return best;
}

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

/** Dx × V × Dy like @c nk_shapes_distance_gradient_f64_serial_, with each cell's sum over the arcs
 *  compensated, for @c f64 clouds. */
NUMKONG_INLINE void nk_shapes_distance_gradient_compensated_f64_serial_(nk_u32_t const *plan, nk_size_t plan_stride,
                                                                        nk_size_t arcs, nk_f64_t mass,
                                                                        nk_f64_t const *row_distances, nk_size_t rows,
                                                                        nk_f64_t const *column_distances,
                                                                        nk_size_t columns, nk_f64_t *gradient) {
    for (nk_size_t row = 0; row != rows; ++row)
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_f64_t sum = 0, compensation = 0;
            for (nk_size_t arc = 0; arc != arcs; ++arc) {
                if (!plan[2 * plan_stride + arc]) continue;
                nk_f64_t const weighted = (nk_f64_t)plan[2 * plan_stride + arc] / mass *
                                          row_distances[plan[arc] * rows + row];
                nk_dot2_f64_serial_(&sum, &compensation, weighted,
                                    column_distances[plan[plan_stride + arc] * columns + column]);
            }
            gradient[row * columns + column] = sum + compensation;
        }
}

/** ⟨G, P⟩ like @c nk_shapes_plan_score_f64_serial_, compensated, for @c f64 clouds. */
NUMKONG_INLINE nk_f64_t nk_shapes_plan_score_compensated_f64_serial_(nk_f64_t const *gradient, nk_f64_t const *plan,
                                                                     nk_size_t cells, nk_f64_t *gradient_scale) {
    nk_f64_t score = 0, compensation = 0, scale = 0;
    for (nk_size_t cell = 0; cell != cells; ++cell) {
        nk_dot2_f64_serial_(&score, &compensation, gradient[cell], plan[cell]);
        nk_f64_t const magnitude = gradient[cell] < 0 ? -gradient[cell] : gradient[cell];
        scale = magnitude > scale ? magnitude : scale;
    }
    *gradient_scale = scale;
    return score + compensation;
}

/** Gromov-Wasserstein of one canonically ordered pair over plain distances within each cloud:
 *  Frank-Wolfe with exact transport steps from the product plan and the best proper and improper
 *  Wasserstein-Procrustes plans. Returns the least square loss and the orthogonal Procrustes fit of
 *  its plan. Only the coordinates come in the math type: the distances, plans, gradients, line
 *  search, loss and fit all stay in f64, as a non-convex search follows f32 rounding near plan ties
 *  into other optima. Each capability runs its own network simplex around the serial passes over
 *  the dense matrices. */
#define nk_define_shapes_gromov_wasserstein_(input_type, math_type, suffix)                                            \
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
    NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_##input_type##_##suffix##_(                                   \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t rows, nk_size_t columns,                    \
        nk_##math_type##_t cost_scale, nk_f64_t const *row_distances, nk_f64_t const *column_distances,                \
        nk_##math_type##_t *rotation) {                                                                                \
        nk_size_t const cells = rows * columns, arcs = rows + columns - 1, stride = views->points_stride;              \
        nk_size_t const plan_stride = views->plan_stride;                                                              \
        nk_f64_t *plan = views->matrices, *gradient = views->matrices + views->matrix_stride;                          \
        nk_f64_t *vertex_gradient = views->matrices + 2 * views->matrix_stride;                                        \
        nk_f64_t const mass = (nk_f64_t)cells;                                                                         \
        nk_shapes_wasserstein_procrustes_##input_type##_##suffix##_(views, rows, columns, cost_scale, rotation);       \
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
                nk_shapes_distance_gradient_f64_serial_(start_plan, plan_stride, arcs, mass, row_distances, rows,      \
                                                        column_distances, columns, gradient);                          \
            }                                                                                                          \
            nk_f64_t gradient_scale;                                                                                   \
            nk_f64_t plan_score = nk_shapes_plan_score_f64_serial_(gradient, plan, cells, &gradient_scale);            \
            for (int step_index = 0; step_index != 32; ++step_index) {                                                 \
                if (!step_index) nk_shapes_northwest_corner_serial_(rows, columns, views->tree, views->tree_stride);   \
                nk_shapes_network_simplex_##math_type##_##suffix##_(views, gradient, rows, columns, gradient_scale);   \
                nk_shapes_tree_arcs_serial_(rows, columns, views->tree, views->tree_stride, vertex, plan_stride);      \
                nk_shapes_distance_gradient_f64_serial_(vertex, plan_stride, arcs, mass, row_distances, rows,          \
                                                        column_distances, columns, vertex_gradient);                   \
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
                nk_shapes_plan_step_f64_serial_(plan, gradient, vertex_gradient, cells, 1 - step, step);               \
                for (nk_size_t arc = 0; arc != arcs; ++arc)                                                            \
                    plan[vertex[arc] * columns + vertex[plan_stride + arc]] +=                                         \
                        step * ((nk_f64_t)vertex[2 * plan_stride + arc] / mass);                                       \
                plan_score = nk_shapes_plan_score_f64_serial_(gradient, plan, cells, &gradient_scale);                 \
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

nk_define_shapes_gromov_wasserstein_(f16, f32, serial)
nk_define_shapes_gromov_wasserstein_(f32, f64, serial)

/** Gromov-Wasserstein of one canonically ordered pair of @c f64 clouds, like the @c f32 one with
 *  every sum over cells and arcs compensated. */
NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f64_serial_(nk_shapes_workspace_f64_serial_t const *views,
                                                                 nk_size_t rows, nk_size_t columns, nk_f64_t cost_scale,
                                                                 nk_f64_t const *row_distances,
                                                                 nk_f64_t const *column_distances, nk_f64_t *rotation) {
    nk_size_t const cells = rows * columns, arcs = rows + columns - 1, stride = views->points_stride;
    nk_size_t const plan_stride = views->plan_stride;
    nk_f64_t *plan = views->matrices, *gradient = views->matrices + views->matrix_stride;
    nk_f64_t *vertex_gradient = views->matrices + 2 * views->matrix_stride;
    nk_f64_t const mass = (nk_f64_t)cells;
    nk_shapes_wasserstein_procrustes_f64_serial_(views, rows, columns, cost_scale, rotation);
    nk_f64_t row_spread = 0, row_spread_compensation = 0, column_spread = 0, column_spread_compensation = 0;
    for (nk_size_t cell = 0; cell != rows * rows; ++cell)
        nk_dot2_f64_serial_(&row_spread, &row_spread_compensation, row_distances[cell], row_distances[cell]);
    for (nk_size_t cell = 0; cell != columns * columns; ++cell)
        nk_dot2_f64_serial_(&column_spread, &column_spread_compensation, column_distances[cell],
                            column_distances[cell]);
    row_spread = (row_spread + row_spread_compensation) / (nk_f64_t)(rows * rows);
    column_spread = (column_spread + column_spread_compensation) / (nk_f64_t)(columns * columns);
    // Once its start spreads into the dense plan, the proper plan holds each step's vertex
    nk_u32_t *vertex = views->proper_plan;
    nk_f64_t best = NUMKONG_F64_INF;
    for (int start = 0; start != 3; ++start) {
        if (start == 2) {
            // The product plan's gradient is the outer product of the two matrices' row sums
            for (nk_size_t column = 0; column != columns; ++column) {
                nk_f64_t column_sum = 0, column_sum_compensation = 0;
                for (nk_size_t other = 0; other != columns; ++other)
                    nk_dot2_f64_serial_(&column_sum, &column_sum_compensation,
                                        column_distances[other * columns + column], 1);
                vertex_gradient[column] = column_sum + column_sum_compensation;
            }
            for (nk_size_t row = 0; row != rows; ++row) {
                nk_f64_t row_sum = 0, row_sum_compensation = 0;
                for (nk_size_t other = 0; other != rows; ++other)
                    nk_dot2_f64_serial_(&row_sum, &row_sum_compensation, row_distances[row * rows + other], 1);
                row_sum += row_sum_compensation;
                for (nk_size_t column = 0; column != columns; ++column) {
                    plan[row * columns + column] = 1 / mass;
                    gradient[row * columns + column] = row_sum * vertex_gradient[column] / mass;
                }
            }
        }
        else {
            nk_u32_t const *start_plan = start ? views->improper_plan : views->proper_plan;
            for (nk_size_t cell = 0; cell != cells; ++cell) plan[cell] = 0;
            for (nk_size_t arc = 0; arc != arcs; ++arc)
                plan[start_plan[arc] * columns + start_plan[plan_stride + arc]] =
                    (nk_f64_t)start_plan[2 * plan_stride + arc] / mass;
            nk_shapes_distance_gradient_compensated_f64_serial_(start_plan, plan_stride, arcs, mass, row_distances,
                                                                rows, column_distances, columns, gradient);
        }
        nk_f64_t gradient_scale;
        nk_f64_t plan_score = nk_shapes_plan_score_compensated_f64_serial_(gradient, plan, cells, &gradient_scale);
        for (int step_index = 0; step_index != 32; ++step_index) {
            if (!step_index) nk_shapes_northwest_corner_serial_(rows, columns, views->tree, views->tree_stride);
            nk_shapes_network_simplex_f64_serial_(views, gradient, rows, columns, gradient_scale);
            nk_shapes_tree_arcs_serial_(rows, columns, views->tree, views->tree_stride, vertex, plan_stride);
            nk_shapes_distance_gradient_compensated_f64_serial_(vertex, plan_stride, arcs, mass, row_distances, rows,
                                                                column_distances, columns, vertex_gradient);
            nk_f64_t vertex_score = 0, vertex_score_compensation = 0;
            nk_f64_t vertex_curvature = 0, vertex_curvature_compensation = 0;
            for (nk_size_t arc = 0; arc != arcs; ++arc) {
                nk_size_t const cell = vertex[arc] * columns + vertex[plan_stride + arc];
                nk_f64_t const weight = (nk_f64_t)vertex[2 * plan_stride + arc] / mass;
                nk_dot2_f64_serial_(&vertex_score, &vertex_score_compensation, gradient[cell], weight);
                nk_dot2_f64_serial_(&vertex_curvature, &vertex_curvature_compensation, vertex_gradient[cell], weight);
            }
            vertex_score += vertex_score_compensation, vertex_curvature += vertex_curvature_compensation;
            nk_f64_t const slope = vertex_score - plan_score;
            nk_f64_t const curvature = vertex_curvature - 2 * vertex_score + plan_score;
            nk_f64_t const quadratic = -2 * curvature, linear = -4 * slope;
            nk_f64_t step = quadratic + linear < 0 ? 1 : 0;
            if (quadratic > 0) {
                step = -linear / (2 * quadratic);
                step = step < 0 ? 0 : step > 1 ? 1 : step;
            }
            if (step == 0) break;
            nk_shapes_plan_step_f64_serial_(plan, gradient, vertex_gradient, cells, 1 - step, step);
            for (nk_size_t arc = 0; arc != arcs; ++arc)
                plan[vertex[arc] * columns + vertex[plan_stride + arc]] +=
                    step * ((nk_f64_t)vertex[2 * plan_stride + arc] / mass);
            plan_score = nk_shapes_plan_score_compensated_f64_serial_(gradient, plan, cells, &gradient_scale);
        }
        nk_f64_t const loss = row_spread + column_spread - 2 * plan_score;
        if (!(loss < best)) continue;
        best = loss;
        nk_f64_t sums[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, compensations[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (nk_size_t row = 0; row != rows; ++row)
            for (nk_size_t column = 0; column != columns; ++column)
                for (int column_axis = 0; column_axis != 3; ++column_axis) {
                    nk_f64_t const weighted = plan[row * columns + column] *
                                              views->column_points[column_axis * stride + column];
                    for (int row_axis = 0; row_axis != 3; ++row_axis)
                        nk_dot2_f64_serial_(&sums[column_axis * 3 + row_axis],
                                            &compensations[column_axis * 3 + row_axis], weighted,
                                            views->row_points[row_axis * stride + row]);
                }
        nk_f64_t cross_covariance[9], svd_left[9], svd_diagonal[9], svd_right[9];
        for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = sums[cell] + compensations[cell];
        nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);
        // The McAdams factors are proper, so det H's sign moves into V to fit over O(3)
        if (nk_det3x3_f64_serial_(cross_covariance) < 0)
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, rotation);
    }
    return best;
}

/** The canonical order of a transport pair, by points, spreads, radius and then packed bytes, so
 *  swapping the sides solves the same problem; and the widening of a pack into workspace points. */
#define nk_define_shapes_transport_(input_type, math_type, load_and_convert)                                       \
    NUMKONG_INLINE int nk_shapes_column_first_##input_type##_serial_(void const *row_packed,                       \
                                                                     void const *column_packed) {                  \
        nk_shapes_packed_header_##math_type##_t const *row_header =                                                \
            (nk_shapes_packed_header_##math_type##_t const *)row_packed;                                           \
        nk_shapes_packed_header_##math_type##_t const *column_header =                                             \
            (nk_shapes_packed_header_##math_type##_t const *)column_packed;                                        \
        if (row_header->points != column_header->points) return column_header->points < row_header->points;        \
        for (int axis = 0; axis != 3; ++axis)                                                                      \
            if (row_header->spreads[axis] != column_header->spreads[axis])                                         \
                return column_header->spreads[axis] < row_header->spreads[axis];                                   \
        if (row_header->max_radius != column_header->max_radius)                                                   \
            return column_header->max_radius < row_header->max_radius;                                             \
        nk_size_t axis_stride;                                                                                     \
        nk_u8_t const *row_bytes = (nk_u8_t const *)nk_shapes_packed_axes_##input_type##_serial_(row_packed,       \
                                                                                                 &axis_stride);    \
        nk_u8_t const *column_bytes = (nk_u8_t const *)nk_shapes_packed_axes_##input_type##_serial_(column_packed, \
                                                                                                    &axis_stride); \
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
        nk_##input_type##_t const *axes = nk_shapes_packed_axes_##input_type##_serial_(packed, &axis_stride);      \
        nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_##math_type##_t const *)packed)->points;      \
        for (int axis = 0; axis != 3; ++axis)                                                                      \
            for (nk_size_t index = 0; index != count; ++index)                                                     \
                load_and_convert(axes + axis * axis_stride + index, points + axis * points_stride + index);        \
    }

nk_define_shapes_transport_(f16, f32, nk_f16_to_f32_serial_)
nk_define_shapes_transport_(f32, f64, nk_assign_from_to_)
nk_define_shapes_transport_(f64, f64, nk_assign_from_to_)

/** The @c _serial pack, pack-reading, unpack and workspace-size kernels of one input type, which
 *  every capability shares. */
#define nk_define_shapes_(input_type, math_type)                                                                      \
    NUMKONG_API nk_status_t nk_shapes_pack_size_##input_type##_serial(nk_size_t points, nk_size_t *bytes) {           \
        *bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_##math_type##_t),                       \
                                               nk_shapes_alignment_k_) +                                              \
                 3 * nk_size_round_up_to_multiple_(points * sizeof(nk_##input_type##_t), nk_shapes_alignment_k_);     \
        return nk_success_k;                                                                                          \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_shapes_pack_##input_type##_serial(nk_##input_type##_t const *points, nk_size_t count,  \
                                                                 void *packed, nk_stream_t stream) {                  \
        nk_assert_(stream == NUMKONG_NULL);                                                                           \
        nk_##math_type##_t centroid[3], covariance[6];                                                                \
        nk_shapes_moments_##input_type##_serial_(points, count, centroid, covariance);                                \
        nk_shapes_pack_##input_type##_serial_(points, count, centroid, covariance, 1, packed);                        \
        return nk_success_k;                                                                                          \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_shapes_pack_normalized_##input_type##_serial(                                          \
        nk_##input_type##_t const *points, nk_size_t count, void *packed, nk_stream_t stream) {                       \
        nk_assert_(stream == NUMKONG_NULL);                                                                           \
        nk_##math_type##_t centroid[3], covariance[6];                                                                \
        nk_##math_type##_t const radius = nk_shapes_moments_##input_type##_serial_(points, count, centroid,           \
                                                                                   covariance);                       \
        nk_shapes_pack_##input_type##_serial_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);  \
        return nk_success_k;                                                                                          \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_shapes_packed_shape_##input_type##_serial(                                             \
        void const *packed, nk_size_t *points, nk_##math_type##_t *centroid, nk_##math_type##_t *frame,               \
        nk_##math_type##_t *scale, nk_##math_type##_t *spreads, nk_##math_type##_t *max_radius, nk_stream_t stream) { \
        nk_assert_(stream == NUMKONG_NULL);                                                                           \
        nk_shapes_packed_header_##math_type##_t const *header =                                                       \
            (nk_shapes_packed_header_##math_type##_t const *)packed;                                                  \
        if (points) *points = (nk_size_t)header->points;                                                              \
        for (int axis = 0; axis != 3; ++axis) {                                                                       \
            if (centroid) centroid[axis] = header->centroid[axis];                                                    \
            if (spreads) spreads[axis] = header->spreads[axis];                                                       \
        }                                                                                                             \
        for (int component = 0; component != 4; ++component)                                                          \
            if (frame) frame[component] = header->frame[component];                                                   \
        if (scale) *scale = header->scale;                                                                            \
        if (max_radius) *max_radius = header->max_radius;                                                             \
        return nk_success_k;                                                                                          \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_shapes_unpack_##input_type##_serial(void const *packed, nk_##input_type##_t *points,   \
                                                                   nk_stream_t stream) {                              \
        nk_assert_(stream == NUMKONG_NULL);                                                                           \
        nk_shapes_unpack_##input_type##_serial_(packed, points);                                                      \
        return nk_success_k;                                                                                          \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_##input_type##_serial(                            \
        nk_size_t row_points, nk_size_t column_points, nk_size_t *bytes) {                                            \
        nk_shapes_workspace_##math_type##_serial_t views;                                                             \
        *bytes = nk_shapes_workspace_##math_type##_serial_(row_points, column_points, NUMKONG_NULL, &views);          \
        return nk_success_k;                                                                                          \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_##input_type##_serial(                               \
        nk_size_t row_points, nk_size_t column_points, nk_size_t *bytes) {                                            \
        nk_shapes_workspace_##math_type##_serial_t views;                                                             \
        *bytes = nk_shapes_workspace_##math_type##_serial_(row_points, column_points, NUMKONG_NULL, &views) +         \
                 views.gromov_bytes;                                                                                  \
        return nk_success_k;                                                                                          \
    }

/** The Chamfer and Hausdorff kernels of one input type for the capability @p suffix, whose
 *  @p nearest pass runs once each way per pair. */
#define nk_define_shapes_flip_kernels_(input_type, math_type, suffix, nearest, infinity)                               \
    NUMKONG_INLINE void nk_shapes_flips_##input_type##_##suffix##_(void const *row_packed, void const *column_packed,  \
                                                                   nk_##math_type##_t *chamfers,                       \
                                                                   nk_##math_type##_t *hausdorffs) {                   \
        nk_size_t const rows = (nk_size_t)((nk_shapes_packed_header_##math_type##_t const *)row_packed)->points;       \
        nk_size_t const columns = (nk_size_t)((nk_shapes_packed_header_##math_type##_t const *)column_packed)->points; \
        if (!rows || !columns) {                                                                                       \
            nk_##math_type##_t const distance = rows == columns ? 0 : infinity;                                        \
            for (int flip = 0; flip != 8; ++flip) chamfers[flip] = distance, hausdorffs[flip] = distance;              \
            return;                                                                                                    \
        }                                                                                                              \
        nk_size_t row_stride, column_stride;                                                                           \
        nk_##input_type##_t const *row_axes = nk_shapes_packed_axes_##input_type##_serial_(row_packed, &row_stride);   \
        nk_##input_type##_t const *column_axes = nk_shapes_packed_axes_##input_type##_serial_(column_packed,           \
                                                                                              &column_stride);         \
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
    NUMKONG_API nk_status_t nk_chamfers_packed_##input_type##_##suffix(                                                \
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
    NUMKONG_API nk_status_t nk_hausdorffs_packed_##input_type##_##suffix(                                              \
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
 *  @p suffix, around that type's pair solvers. */
#define nk_define_shapes_transport_kernels_(input_type, math_type, suffix, infinity)                                 \
    NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_##input_type##_##suffix(                                 \
        void const *const *a_packed, nk_size_t rows, void const *const *b_packed, nk_size_t columns,                 \
        nk_##math_type##_t *results, nk_size_t results_stride, nk_f32_t *rotations, void *workspace,                 \
        nk_size_t workspace_bytes, nk_stream_t stream) {                                                             \
        nk_assert_(stream == NUMKONG_NULL);                                                                          \
        if (!rows || !columns) return nk_success_k;                                                                  \
        if ((uintptr_t)workspace & (nk_default_alignment_k - 1)) return nk_misaligned_k;                             \
        nk_size_t const row_points = nk_shapes_largest_cloud_##math_type##_serial_(a_packed, rows);                  \
        nk_size_t const column_points = nk_shapes_largest_cloud_##math_type##_serial_(b_packed, columns);            \
        nk_shapes_workspace_##math_type##_serial_t views;                                                            \
        if (workspace_bytes <                                                                                        \
            nk_shapes_workspace_##math_type##_serial_(row_points, column_points, NUMKONG_NULL, &views))              \
            return nk_unexpected_dimensions_k;                                                                       \
        nk_shapes_workspace_##math_type##_serial_(row_points, column_points, (nk_u8_t *)workspace, &views);          \
        for (nk_size_t row = 0; row != rows; ++row) {                                                                \
            nk_##math_type##_t *results_row = (nk_##math_type##_t *)((char *)results + row * results_stride);        \
            for (nk_size_t column = 0; column != columns; ++column) {                                                \
                int const swapped = nk_shapes_column_first_##input_type##_serial_(a_packed[row], b_packed[column]);  \
                void const *row_packed = swapped ? b_packed[column] : a_packed[row];                                 \
                void const *column_packed = swapped ? a_packed[row] : b_packed[column];                              \
                nk_shapes_packed_header_##math_type##_t const *row_header =                                          \
                    (nk_shapes_packed_header_##math_type##_t const *)row_packed;                                     \
                nk_shapes_packed_header_##math_type##_t const *column_header =                                       \
                    (nk_shapes_packed_header_##math_type##_t const *)column_packed;                                  \
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
                    results_row[column] = nk_sqrt_##math_type##_serial_(                                             \
                        nk_shapes_wasserstein_procrustes_##input_type##_##suffix##_(                                 \
                            &views, solved_rows, solved_columns, radii * radii, inner));                             \
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
    NUMKONG_API nk_status_t nk_gromovwassersteins_packed_##input_type##_##suffix(                                    \
        void const *const *a_packed, nk_size_t rows, void const *const *b_packed, nk_size_t columns,                 \
        nk_##math_type##_t *results, nk_size_t results_stride, nk_f32_t *rotations, void *workspace,                 \
        nk_size_t workspace_bytes, nk_stream_t stream) {                                                             \
        nk_assert_(stream == NUMKONG_NULL);                                                                          \
        if (!rows || !columns) return nk_success_k;                                                                  \
        if ((uintptr_t)workspace & (nk_default_alignment_k - 1)) return nk_misaligned_k;                             \
        nk_size_t const row_points = nk_shapes_largest_cloud_##math_type##_serial_(a_packed, rows);                  \
        nk_size_t const column_points = nk_shapes_largest_cloud_##math_type##_serial_(b_packed, columns);            \
        nk_shapes_workspace_##math_type##_serial_t views;                                                            \
        nk_size_t const transport_bytes = nk_shapes_workspace_##math_type##_serial_(row_points, column_points,       \
                                                                                    NUMKONG_NULL, &views);           \
        if (workspace_bytes < transport_bytes + views.gromov_bytes) return nk_unexpected_dimensions_k;               \
        nk_shapes_workspace_##math_type##_serial_(row_points, column_points, (nk_u8_t *)workspace, &views);          \
        nk_f64_t *cached_distances = views.matrices + 3 * views.matrix_stride;                                       \
        for (nk_size_t row = 0; row != rows; ++row) {                                                                \
            nk_##math_type##_t *results_row = (nk_##math_type##_t *)((char *)results + row * results_stride);        \
            nk_size_t const cached_points =                                                                          \
                (nk_size_t)((nk_shapes_packed_header_##math_type##_t const *)a_packed[row])->points;                 \
            nk_f64_t *pair_distances = cached_distances + cached_points * cached_points;                             \
            nk_shapes_widen_##input_type##_serial_(a_packed[row], views.row_points, views.points_stride);            \
            nk_shapes_distances_##math_type##_##suffix##_(views.row_points, cached_points, views.points_stride,      \
                                                          cached_distances);                                         \
            for (nk_size_t column = 0; column != columns; ++column) {                                                \
                int const swapped = nk_shapes_column_first_##input_type##_serial_(a_packed[row], b_packed[column]);  \
                void const *row_packed = swapped ? b_packed[column] : a_packed[row];                                 \
                void const *column_packed = swapped ? a_packed[row] : b_packed[column];                              \
                nk_shapes_packed_header_##math_type##_t const *row_header =                                          \
                    (nk_shapes_packed_header_##math_type##_t const *)row_packed;                                     \
                nk_shapes_packed_header_##math_type##_t const *column_header =                                       \
                    (nk_shapes_packed_header_##math_type##_t const *)column_packed;                                  \
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
                    nk_f64_t const loss = nk_shapes_gromov_wasserstein_##input_type##_##suffix##_(                   \
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

/**
 *  @brief Σ‖aᵢ − R × bᵢ‖² at the best rotation R, folded into Ga + Gb − 2λ and clamped at zero.
 *
 *  λ is the largest eigenvalue of Theobald's quaternion key matrix built from M = Σ aᵢ × bᵢᵀ, row
 *  major, found by Newton's method on its characteristic polynomial from (Ga + Gb) / 2, an upper
 *  bound it falls from monotonically. M is divided by that bound first, so the quartic terms stay
 *  in range. The fold cancels to √ε of @c f32, which stays below the rounding of @c f16 inputs.
 *
 *  Thin frames nearly repeat the largest root, where Newton stops √ε short, so two Rayleigh
 *  quotients of adjugate columns of K − λI refine λ, each squaring the error of its direction.
 *  Linear frames repeat it exactly, leaving the adjugate rounding noise, and fold over the rotation
 *  of the Kabsch SVD instead, the rule of the one-to-one kernels.
 */
NUMKONG_INLINE nk_f32_t nk_rmsds_folded_f32_serial_(nk_f32_t const *cross_covariance, nk_f32_t norms_squared) {
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
    nk_f32_t const frobenius = nk_sqrt_f32_serial_(-quadratic / 2),
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
        nk_svd3x3_f32_serial_(transposed, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f32_serial_(svd_left, svd_right, rotation);
        if (nk_det3x3_f32_serial_(rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f32_serial_(svd_left, svd_right, rotation);
        }
        for (int cell = 0; cell != 9; ++cell) trace += rotation[cell] * cross_covariance[cell];
        residual = norms_squared - 2 * trace;
    }
    return residual > 0 ? residual : 0;
}

/** The proper rotation R minimizing Σ‖aᵢ − R × bᵢ‖², from the SVD of H = Σ bᵢ × aᵢᵀ, row major. */
NUMKONG_INLINE void nk_rmsds_rotation_f64_serial_(nk_f64_t const *cross_covariance, nk_f64_t *rotation) {
    nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
    nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);
    nk_rotation_from_svd_f64_serial_(svd_left, svd_right, rotation);
    if (nk_det3x3_f64_serial_(rotation) < 0) {
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, rotation);
    }
}

/** Frames whose norms the RMSD kernels hold at once, one pass per frame and tile of pairs. */
enum { nk_rmsds_tile_frames_k_ = 64 };

/** Whether every pack of @c f16 points in @p packs holds @p points points, as RMSD pairs need. */
NUMKONG_INLINE int nk_rmsds_same_points_f16_serial_(void const *const *packs, nk_size_t count, nk_size_t points) {
    for (nk_size_t index = 0; index != count; ++index)
        if (((nk_shapes_packed_header_f32_t const *)packs[index])->points != points) return 0;
    return 1;
}

/** Whether every pack of @c f32 points in @p packs holds @p points points, as RMSD pairs need. */
NUMKONG_INLINE int nk_rmsds_same_points_f32_serial_(void const *const *packs, nk_size_t count, nk_size_t points) {
    for (nk_size_t index = 0; index != count; ++index)
        if (((nk_shapes_packed_header_f64_t const *)packs[index])->points != points) return 0;
    return 1;
}

/** Whether every pack of @c f64 points in @p packs holds @p points points, as RMSD pairs need. */
NUMKONG_INLINE int nk_rmsds_same_points_f64_serial_(void const *const *packs, nk_size_t count, nk_size_t points) {
    for (nk_size_t index = 0; index != count; ++index)
        if (((nk_shapes_packed_header_f64_t const *)packs[index])->points != points) return 0;
    return 1;
}

/** Σ‖xᵢ′‖² of a pack of @c f16 points, summed in @c f32 over the stored points. */
NUMKONG_INLINE nk_f32_t nk_rmsds_norm_f16_serial_(void const *packed) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t axis_stride;
    nk_f16_t const *axes = nk_shapes_packed_axes_f16_serial_(packed, &axis_stride);
    nk_f32_t norm = 0, value, square;
    for (int axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index != count; ++index)
            nk_f16_to_f32_serial_(axes + axis * axis_stride + index, &value), square = value * value, norm += square;
    return norm;
}

/** Superposed RMSD of one pair of @c f16 packs with Ga + Gb in @p norms_squared, M in @c f32. */
NUMKONG_INLINE nk_f32_t nk_rmsds_pair_f16_serial_(void const *row_packed, void const *column_packed,
                                                  nk_f32_t norms_squared) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_serial_(row_packed, &axis_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_serial_(column_packed, &axis_stride);
    nk_f32_t cross_covariance[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, a[3], b[3], product;
    for (nk_size_t index = 0; index != count; ++index) {
        for (int axis = 0; axis != 3; ++axis)
            nk_f16_to_f32_serial_(a_axes + axis * axis_stride + index, &a[axis]),
                nk_f16_to_f32_serial_(b_axes + axis * axis_stride + index, &b[axis]);
        for (int row = 0; row != 3; ++row)
            for (int column = 0; column != 3; ++column)
                product = a[row] * b[column], cross_covariance[row * 3 + column] += product;
    }
    nk_f32_t const residual = nk_rmsds_folded_f32_serial_(cross_covariance, norms_squared);
    return nk_sqrt_f32_serial_(residual / (nk_f32_t)count);
}

/** Superposed RMSD of one pair of @c f32 packs: H in @c f64, its Kabsch rotation, then a second
 *  pass summing the residuals directly, compensated, as the fold would cancel to √ε of @c f64 once
 *  the frames align. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f32_serial_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f32_t const *a_axes = nk_shapes_packed_axes_f32_serial_(row_packed, &axis_stride);
    nk_f32_t const *b_axes = nk_shapes_packed_axes_f32_serial_(column_packed, &axis_stride);
    nk_f64_t cross_covariance[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, product;
    for (nk_size_t index = 0; index != count; ++index)
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                product = (nk_f64_t)b_axes[column * axis_stride + index] * (nk_f64_t)a_axes[row * axis_stride + index],
                cross_covariance[column * 3 + row] += product;
    nk_f64_t rotation[9];
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t index = 0; index != count; ++index)
        for (int axis = 0; axis != 3; ++axis) {
            nk_f64_t residual = a_axes[axis * axis_stride + index];
            for (int column = 0; column != 3; ++column)
                product = rotation[axis * 3 + column] * (nk_f64_t)b_axes[column * axis_stride + index],
                residual -= product;
            nk_dot2_f64_serial_(&sum, &compensation, residual, residual);
        }
    return nk_sqrt_f64_serial_((sum + compensation) / (nk_f64_t)count);
}

/** Superposed RMSD of one pair of @c f64 packs, like the @c f32 pair with H compensated too. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f64_serial_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f64_t const *a_axes = nk_shapes_packed_axes_f64_serial_(row_packed, &axis_stride);
    nk_f64_t const *b_axes = nk_shapes_packed_axes_f64_serial_(column_packed, &axis_stride);
    nk_f64_t sums[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, compensations[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (nk_size_t index = 0; index != count; ++index)
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                nk_dot2_f64_serial_(&sums[column * 3 + row], &compensations[column * 3 + row],
                                    b_axes[column * axis_stride + index], a_axes[row * axis_stride + index]);
    nk_f64_t cross_covariance[9], rotation[9], product;
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = sums[cell] + compensations[cell];
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t index = 0; index != count; ++index)
        for (int axis = 0; axis != 3; ++axis) {
            nk_f64_t residual = a_axes[axis * axis_stride + index];
            for (int column = 0; column != 3; ++column)
                product = rotation[axis * 3 + column] * b_axes[column * axis_stride + index], residual -= product;
            nk_dot2_f64_serial_(&sum, &compensation, residual, residual);
        }
    return nk_sqrt_f64_serial_((sum + compensation) / (nk_f64_t)count);
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

nk_define_shapes_(f64, f64)
nk_define_shapes_(f32, f64)
nk_define_shapes_(f16, f32)
nk_define_shapes_flip_kernels_(f64, f64, serial, nk_shapes_nearest_f64_serial_, NUMKONG_F64_INF)
nk_define_shapes_flip_kernels_(f32, f64, serial, nk_shapes_nearest_f32_serial_, NUMKONG_F64_INF)
nk_define_shapes_flip_kernels_(f16, f32, serial, nk_shapes_nearest_f16_serial_, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f64, f64, serial, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f32, f64, serial, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, serial, NUMKONG_F32_INF)

NUMKONG_API nk_status_t nk_rmsds_packed_f64_serial(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                   nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f64_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f64_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column)
            results_row[column] = nk_rmsds_pair_f64_serial_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f32_serial(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                   nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f32_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f32_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column)
            results_row[column] = nk_rmsds_pair_f32_serial_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_serial(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                   nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = ((nk_shapes_packed_header_f32_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f16_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f16_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    nk_f32_t row_norms[nk_rmsds_tile_frames_k_], column_norms[nk_rmsds_tile_frames_k_];
    for (nk_size_t row_begin = 0; row_begin < rows; row_begin += nk_rmsds_tile_frames_k_) {
        nk_size_t const row_end = row_begin + nk_rmsds_tile_frames_k_ < rows ? row_begin + nk_rmsds_tile_frames_k_
                                                                             : rows;
        for (nk_size_t row = row_begin; row != row_end; ++row)
            row_norms[row - row_begin] = nk_rmsds_norm_f16_serial_(a_packed[row]);
        for (nk_size_t column_begin = 0; column_begin < columns; column_begin += nk_rmsds_tile_frames_k_) {
            nk_size_t const column_end = column_begin + nk_rmsds_tile_frames_k_ < columns
                                             ? column_begin + nk_rmsds_tile_frames_k_
                                             : columns;
            for (nk_size_t column = column_begin; column != column_end; ++column)
                column_norms[column - column_begin] = nk_rmsds_norm_f16_serial_(b_packed[column]);
            for (nk_size_t row = row_begin; row != row_end; ++row) {
                nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
                for (nk_size_t column = column_begin; column != column_end; ++column)
                    results_row[column] = nk_rmsds_pair_f16_serial_(
                        a_packed[row], b_packed[column],
                        row_norms[row - row_begin] + column_norms[column - column_begin]);
            }
        }
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_SERIAL

/*  The solver and kernel macros stay defined, as each capability instantiates them. */
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

#endif // NUMKONG_SHAPES_SERIAL_H
