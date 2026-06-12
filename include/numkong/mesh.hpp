/**
 *  @brief C++ bindings for mesh-distance kernels.
 *  @file include/numkong/mesh.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_MESH_HPP
#define NK_MESH_HPP

#include <cstdint>
#include <type_traits>
#include <utility>

#include "numkong/mesh.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

#pragma region SVD Helpers for Scalar Fallbacks

/** @brief 3x3 matrix determinant. */
template <typename scalar_type_>
scalar_type_ det3x3_(scalar_type_ const *m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/** @brief Conditional swap helper. */
template <typename scalar_type_>
void conditional_swap_(bool c, scalar_type_ *x, scalar_type_ *y) {
    scalar_type_ temp = *x;
    *x = c ? *y : *x;
    *y = c ? temp : *y;
}

/** @brief Conditional negating swap helper. */
template <typename scalar_type_>
void conditional_negating_swap_(bool c, scalar_type_ *x, scalar_type_ *y) {
    scalar_type_ neg_x = scalar_type_(0.0) - *x;
    *x = c ? *y : *x;
    *y = c ? neg_x : *y;
}

/** @brief Approximate Givens quaternion for Jacobi eigenanalysis. */
template <typename scalar_type_>
void approximate_givens_quaternion_(scalar_type_ a11, scalar_type_ a12, scalar_type_ a22, scalar_type_ *cos_half,
                                    scalar_type_ *sin_half) {
    constexpr scalar_type_ gamma_k = scalar_type_(5.828427124746190);  // gamma = (sqrt8 + 3)^2 / 4
    constexpr scalar_type_ cstar_k = scalar_type_(0.9238795325112867); // cos(pi/8)
    constexpr scalar_type_ sstar_k = scalar_type_(0.3826834323650898); // sin(pi/8)

    *cos_half = scalar_type_(2.0) * (a11 - a22);
    *sin_half = a12;
    bool use_givens = gamma_k * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);
    scalar_type_ w = ((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half)).rsqrt();
    *cos_half = use_givens ? w * (*cos_half) : cstar_k;
    *sin_half = use_givens ? w * (*sin_half) : sstar_k;
}

/** @brief Jacobi conjugation step for eigenanalysis. */
template <typename scalar_type_>
void jacobi_conjugation_(int idx_x, int idx_y, int idx_z, scalar_type_ *s11, scalar_type_ *s21, scalar_type_ *s22,
                         scalar_type_ *s31, scalar_type_ *s32, scalar_type_ *s33, scalar_type_ *quat) {

    scalar_type_ cos_half, sin_half;
    approximate_givens_quaternion_(*s11, *s21, *s22, &cos_half, &sin_half);
    scalar_type_ scale = cos_half * cos_half + sin_half * sin_half;
    scalar_type_ cos_theta = (cos_half * cos_half - sin_half * sin_half) / scale;
    scalar_type_ sin_theta = (scalar_type_(2.0) * sin_half * cos_half) / scale;
    scalar_type_ s11_old = *s11, s21_old = *s21, s22_old = *s22;
    scalar_type_ s31_old = *s31, s32_old = *s32, s33_old = *s33;

    *s11 = cos_theta * (cos_theta * s11_old + sin_theta * s21_old) +
           sin_theta * (cos_theta * s21_old + sin_theta * s22_old);
    *s21 = cos_theta * ((scalar_type_(0.0) - sin_theta) * s11_old + cos_theta * s21_old) +
           sin_theta * ((scalar_type_(0.0) - sin_theta) * s21_old + cos_theta * s22_old);
    *s22 = (scalar_type_(0.0) - sin_theta) * ((scalar_type_(0.0) - sin_theta) * s11_old + cos_theta * s21_old) +
           cos_theta * ((scalar_type_(0.0) - sin_theta) * s21_old + cos_theta * s22_old);
    *s31 = cos_theta * s31_old + sin_theta * s32_old;
    *s32 = (scalar_type_(0.0) - sin_theta) * s31_old + cos_theta * s32_old;
    *s33 = s33_old;

    // Update quaternion accumulator
    scalar_type_ quat_temp[3];
    quat_temp[0] = quat[0] * sin_half;
    quat_temp[1] = quat[1] * sin_half;
    quat_temp[2] = quat[2] * sin_half;
    sin_half = sin_half * quat[3];
    quat[0] = quat[0] * cos_half;
    quat[1] = quat[1] * cos_half;
    quat[2] = quat[2] * cos_half;
    quat[3] = quat[3] * cos_half;
    quat[idx_z] = quat[idx_z] + sin_half;
    quat[3] = quat[3] - quat_temp[idx_z];
    quat[idx_x] = quat[idx_x] + quat_temp[idx_y];
    quat[idx_y] = quat[idx_y] - quat_temp[idx_x];
    // Cyclic permutation of matrix elements
    s11_old = *s22, s21_old = *s32, s22_old = *s33, s31_old = *s21, s32_old = *s31, s33_old = *s11;
    *s11 = s11_old, *s21 = s21_old, *s22 = s22_old, *s31 = s31_old, *s32 = s32_old, *s33 = s33_old;
}

/** @brief Convert quaternion to 3x3 rotation matrix. */
template <typename scalar_type_>
void quaternion_to_mat3x3_(scalar_type_ const *quat, scalar_type_ *matrix) {
    scalar_type_ w = quat[3], x = quat[0], y = quat[1], z = quat[2];
    scalar_type_ q_xx = x * x, q_yy = y * y, q_zz = z * z;
    scalar_type_ q_xz = x * z, q_xy = x * y, q_yz = y * z;
    scalar_type_ q_wx = w * x, q_wy = w * y, q_wz = w * z;
    matrix[0] = scalar_type_(1.0) - scalar_type_(2.0) * (q_yy + q_zz);
    matrix[1] = scalar_type_(2.0) * (q_xy - q_wz);
    matrix[2] = scalar_type_(2.0) * (q_xz + q_wy);
    matrix[3] = scalar_type_(2.0) * (q_xy + q_wz);
    matrix[4] = scalar_type_(1.0) - scalar_type_(2.0) * (q_xx + q_zz);
    matrix[5] = scalar_type_(2.0) * (q_yz - q_wx);
    matrix[6] = scalar_type_(2.0) * (q_xz - q_wy);
    matrix[7] = scalar_type_(2.0) * (q_yz + q_wx);
    matrix[8] = scalar_type_(1.0) - scalar_type_(2.0) * (q_xx + q_yy);
}

/** @brief Jacobi eigenanalysis for symmetric 3x3 matrix. */
template <typename scalar_type_>
void jacobi_eigenanalysis_(scalar_type_ *s11, scalar_type_ *s21, scalar_type_ *s22, scalar_type_ *s31,
                           scalar_type_ *s32, scalar_type_ *s33, scalar_type_ *quat) {
    quat[0] = scalar_type_(0.0);
    quat[1] = scalar_type_(0.0);
    quat[2] = scalar_type_(0.0);
    quat[3] = scalar_type_(1.0);
    // 16 iterations for better convergence
    for (unsigned int iter = 0; iter < 16; iter++) {
        jacobi_conjugation_(0, 1, 2, s11, s21, s22, s31, s32, s33, quat);
        jacobi_conjugation_(1, 2, 0, s11, s21, s22, s31, s32, s33, quat);
        jacobi_conjugation_(2, 0, 1, s11, s21, s22, s31, s32, s33, quat);
    }
    scalar_type_ norm = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).rsqrt();
    quat[0] = quat[0] * norm;
    quat[1] = quat[1] * norm;
    quat[2] = quat[2] * norm;
    quat[3] = quat[3] * norm;
}

/** @brief QR Givens quaternion for QR decomposition. */
template <typename scalar_type_>
void qr_givens_quaternion_(scalar_type_ a1, scalar_type_ a2, scalar_type_ *cos_half, scalar_type_ *sin_half) {
    constexpr scalar_type_ epsilon_k = scalar_type_(1e-12);

    scalar_type_ a1_sq_plus_a2_sq = a1 * a1 + a2 * a2;
    scalar_type_ rho = a1_sq_plus_a2_sq * a1_sq_plus_a2_sq.rsqrt();
    rho = a1_sq_plus_a2_sq > epsilon_k ? rho : scalar_type_(0.0);
    *sin_half = rho > epsilon_k ? a2 : scalar_type_(0.0);
    scalar_type_ abs_a1 = a1 < scalar_type_(0.0) ? (scalar_type_(0.0) - a1) : a1;
    scalar_type_ max_rho = rho > epsilon_k ? rho : epsilon_k;
    *cos_half = abs_a1 + max_rho;
    bool should_swap = a1 < scalar_type_(0.0);
    conditional_swap_(should_swap, sin_half, cos_half);
    scalar_type_ w = ((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half)).rsqrt();
    *cos_half = (*cos_half) * w;
    *sin_half = (*sin_half) * w;
}

/** @brief Sort singular values in descending order. */
template <typename scalar_type_>
void sort_singular_values_(scalar_type_ *b, scalar_type_ *v) {
    scalar_type_ rho1 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];
    scalar_type_ rho2 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];
    scalar_type_ rho3 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];
    bool should_swap;
    // Sort columns by descending singular value magnitude
    should_swap = rho1 < rho2;
    conditional_negating_swap_(should_swap, &b[0], &b[1]);
    conditional_negating_swap_(should_swap, &v[0], &v[1]);
    conditional_negating_swap_(should_swap, &b[3], &b[4]);
    conditional_negating_swap_(should_swap, &v[3], &v[4]);
    conditional_negating_swap_(should_swap, &b[6], &b[7]);
    conditional_negating_swap_(should_swap, &v[6], &v[7]);
    conditional_swap_(should_swap, &rho1, &rho2);
    should_swap = rho1 < rho3;
    conditional_negating_swap_(should_swap, &b[0], &b[2]);
    conditional_negating_swap_(should_swap, &v[0], &v[2]);
    conditional_negating_swap_(should_swap, &b[3], &b[5]);
    conditional_negating_swap_(should_swap, &v[3], &v[5]);
    conditional_negating_swap_(should_swap, &b[6], &b[8]);
    conditional_negating_swap_(should_swap, &v[6], &v[8]);
    conditional_swap_(should_swap, &rho1, &rho3);
    should_swap = rho2 < rho3;
    conditional_negating_swap_(should_swap, &b[1], &b[2]);
    conditional_negating_swap_(should_swap, &v[1], &v[2]);
    conditional_negating_swap_(should_swap, &b[4], &b[5]);
    conditional_negating_swap_(should_swap, &v[4], &v[5]);
    conditional_negating_swap_(should_swap, &b[7], &b[8]);
    conditional_negating_swap_(should_swap, &v[7], &v[8]);
}

/** @brief QR decomposition of 3x3 matrix. */
template <typename scalar_type_>
void qr_decomposition_(scalar_type_ const *input, scalar_type_ *q, scalar_type_ *r) {
    scalar_type_ cos_half_1, sin_half_1;
    scalar_type_ cos_half_2, sin_half_2;
    scalar_type_ cos_half_3, sin_half_3;
    scalar_type_ cos_theta, sin_theta;
    scalar_type_ rotation_temp[9], matrix_temp[9];
    // First Givens rotation (zero input[3])
    qr_givens_quaternion_(input[0], input[3], &cos_half_1, &sin_half_1);
    cos_theta = scalar_type_(1.0) - scalar_type_(2.0) * sin_half_1 * sin_half_1;
    sin_theta = scalar_type_(2.0) * cos_half_1 * sin_half_1;
    rotation_temp[0] = cos_theta * input[0] + sin_theta * input[3];
    rotation_temp[1] = cos_theta * input[1] + sin_theta * input[4];
    rotation_temp[2] = cos_theta * input[2] + sin_theta * input[5];
    rotation_temp[3] = (scalar_type_(0.0) - sin_theta) * input[0] + cos_theta * input[3];
    rotation_temp[4] = (scalar_type_(0.0) - sin_theta) * input[1] + cos_theta * input[4];
    rotation_temp[5] = (scalar_type_(0.0) - sin_theta) * input[2] + cos_theta * input[5];
    rotation_temp[6] = input[6];
    rotation_temp[7] = input[7];
    rotation_temp[8] = input[8];
    // Second Givens rotation (zero rotation_temp[6])
    qr_givens_quaternion_(rotation_temp[0], rotation_temp[6], &cos_half_2, &sin_half_2);
    cos_theta = scalar_type_(1.0) - scalar_type_(2.0) * sin_half_2 * sin_half_2;
    sin_theta = scalar_type_(2.0) * cos_half_2 * sin_half_2;
    matrix_temp[0] = cos_theta * rotation_temp[0] + sin_theta * rotation_temp[6];
    matrix_temp[1] = cos_theta * rotation_temp[1] + sin_theta * rotation_temp[7];
    matrix_temp[2] = cos_theta * rotation_temp[2] + sin_theta * rotation_temp[8];
    matrix_temp[3] = rotation_temp[3];
    matrix_temp[4] = rotation_temp[4];
    matrix_temp[5] = rotation_temp[5];
    matrix_temp[6] = (scalar_type_(0.0) - sin_theta) * rotation_temp[0] + cos_theta * rotation_temp[6];
    matrix_temp[7] = (scalar_type_(0.0) - sin_theta) * rotation_temp[1] + cos_theta * rotation_temp[7];
    matrix_temp[8] = (scalar_type_(0.0) - sin_theta) * rotation_temp[2] + cos_theta * rotation_temp[8];
    // Third Givens rotation (zero matrix_temp[7])
    qr_givens_quaternion_(matrix_temp[4], matrix_temp[7], &cos_half_3, &sin_half_3);
    cos_theta = scalar_type_(1.0) - scalar_type_(2.0) * sin_half_3 * sin_half_3;
    sin_theta = scalar_type_(2.0) * cos_half_3 * sin_half_3;
    r[0] = matrix_temp[0];
    r[1] = matrix_temp[1];
    r[2] = matrix_temp[2];
    r[3] = cos_theta * matrix_temp[3] + sin_theta * matrix_temp[6];
    r[4] = cos_theta * matrix_temp[4] + sin_theta * matrix_temp[7];
    r[5] = cos_theta * matrix_temp[5] + sin_theta * matrix_temp[8];
    r[6] = (scalar_type_(0.0) - sin_theta) * matrix_temp[3] + cos_theta * matrix_temp[6];
    r[7] = (scalar_type_(0.0) - sin_theta) * matrix_temp[4] + cos_theta * matrix_temp[7];
    r[8] = (scalar_type_(0.0) - sin_theta) * matrix_temp[5] + cos_theta * matrix_temp[8];
    // Construct Q = Q1 * Q2 * Q3 (closed-form expressions)
    scalar_type_ sin_half_1_sq = sin_half_1 * sin_half_1;
    scalar_type_ sin_half_2_sq = sin_half_2 * sin_half_2;
    scalar_type_ sin_half_3_sq = sin_half_3 * sin_half_3;
    q[0] = (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_1_sq) *
           (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_2_sq);
    q[1] = scalar_type_(4.0) * cos_half_2 * cos_half_3 * (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_1_sq) *
               sin_half_2 * sin_half_3 +
           scalar_type_(2.0) * cos_half_1 * sin_half_1 * (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq);
    q[2] = scalar_type_(4.0) * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -
           scalar_type_(2.0) * cos_half_2 * (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_1_sq) * sin_half_2 *
               (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq);
    q[3] = scalar_type_(2.0) * cos_half_1 * sin_half_1 * (scalar_type_(1.0) - scalar_type_(2.0) * sin_half_2_sq);
    q[4] = scalar_type_(-8.0) * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +
           (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_1_sq) *
               (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq);
    q[5] = scalar_type_(-2.0) * cos_half_3 * sin_half_3 +
           scalar_type_(4.0) * sin_half_1 *
               (cos_half_3 * sin_half_1 * sin_half_3 +
                cos_half_1 * cos_half_2 * sin_half_2 * (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq));
    q[6] = scalar_type_(2.0) * cos_half_2 * sin_half_2;
    q[7] = scalar_type_(2.0) * cos_half_3 * (scalar_type_(1.0) - scalar_type_(2.0) * sin_half_2_sq) * sin_half_3;
    q[8] = (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_2_sq) *
           (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq);
}

/** @brief 3x3 SVD: A = U * S * Vt using McAdams algorithm. */
template <typename scalar_type_>
void svd3x3_(scalar_type_ const *a, scalar_type_ *svd_u, scalar_type_ *svd_s, scalar_type_ *svd_v) {
    // Compute At * A (symmetric)
    scalar_type_ ata[9];
    ata[0] = a[0] * a[0] + a[3] * a[3] + a[6] * a[6];
    ata[1] = a[0] * a[1] + a[3] * a[4] + a[6] * a[7];
    ata[2] = a[0] * a[2] + a[3] * a[5] + a[6] * a[8];
    ata[3] = ata[1];
    ata[4] = a[1] * a[1] + a[4] * a[4] + a[7] * a[7];
    ata[5] = a[1] * a[2] + a[4] * a[5] + a[7] * a[8];
    ata[6] = ata[2];
    ata[7] = ata[5];
    ata[8] = a[2] * a[2] + a[5] * a[5] + a[8] * a[8];
    // Jacobi eigenanalysis of At * A
    scalar_type_ quat[4];
    jacobi_eigenanalysis_(&ata[0], &ata[1], &ata[4], &ata[2], &ata[5], &ata[8], quat);
    quaternion_to_mat3x3_(quat, svd_v);
    // B = A * V
    scalar_type_ product[9];
    product[0] = a[0] * svd_v[0] + a[1] * svd_v[3] + a[2] * svd_v[6];
    product[1] = a[0] * svd_v[1] + a[1] * svd_v[4] + a[2] * svd_v[7];
    product[2] = a[0] * svd_v[2] + a[1] * svd_v[5] + a[2] * svd_v[8];
    product[3] = a[3] * svd_v[0] + a[4] * svd_v[3] + a[5] * svd_v[6];
    product[4] = a[3] * svd_v[1] + a[4] * svd_v[4] + a[5] * svd_v[7];
    product[5] = a[3] * svd_v[2] + a[4] * svd_v[5] + a[5] * svd_v[8];
    product[6] = a[6] * svd_v[0] + a[7] * svd_v[3] + a[8] * svd_v[6];
    product[7] = a[6] * svd_v[1] + a[7] * svd_v[4] + a[8] * svd_v[7];
    product[8] = a[6] * svd_v[2] + a[7] * svd_v[5] + a[8] * svd_v[8];
    // Sort singular values and update V
    sort_singular_values_(product, svd_v);
    // Compute singular values from column norms of sorted B
    scalar_type_ s1_sq = product[0] * product[0] + product[3] * product[3] + product[6] * product[6];
    scalar_type_ s2_sq = product[1] * product[1] + product[4] * product[4] + product[7] * product[7];
    scalar_type_ s3_sq = product[2] * product[2] + product[5] * product[5] + product[8] * product[8];
    // QR decomposition: B = U * R
    scalar_type_ qr_r[9];
    qr_decomposition_(product, svd_u, qr_r);
    // Store singular values in diagonal of svd_s
    svd_s[0] = s1_sq.sqrt();
    svd_s[1] = scalar_type_(0.0);
    svd_s[2] = scalar_type_(0.0);
    svd_s[3] = scalar_type_(0.0);
    svd_s[4] = s2_sq.sqrt();
    svd_s[5] = scalar_type_(0.0);
    svd_s[6] = scalar_type_(0.0);
    svd_s[7] = scalar_type_(0.0);
    svd_s[8] = s3_sq.sqrt();
}

#pragma endregion SVD Helpers for Scalar Fallbacks

#pragma region Mesh Alignment Kernels

/**
 *  @brief Root Mean Square Deviation between two 3D point clouds (no alignment)
 *  @param[in] a,b Point clouds [d x 3] interleaved (x0,y0,z0, x1,y1,z1, ...)
 *  @param[in] d Number of 3D points
 *  @param[out] a_centroid,b_centroid Centroids (3 values each), can be nullptr
 *  @param[out] rotation 3x3 rotation matrix (9 values), always identity, can be nullptr
 *  @param[out] scale Scale factor, always 1.0, can be nullptr
 *  @param[out] metric Output RMSD value
 *
 *  @tparam in_type_ Input point type (f32_t, f64_t, f16_t, bf16_t)
 *  @tparam transform_type_ Type of centroids, rotation, and scale outputs
 *  @tparam metric_type_ Type of the scalar fit metric output
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename transform_type_ = typename in_type_::mesh_transform_t,
          typename metric_type_ = typename in_type_::mesh_metric_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void rmsd(                                               //
    in_type_ const *a, in_type_ const *b, std::size_t n, //
    transform_type_ *a_centroid, transform_type_ *b_centroid, transform_type_ *rotation, transform_type_ *scale,
    metric_type_ *metric) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<transform_type_, typename in_type_::mesh_transform_t> &&
                          std::is_same_v<metric_type_, typename in_type_::mesh_metric_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_rmsd_f64(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                    &metric->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_rmsd_f32(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                    &metric->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_rmsd_f16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                    &metric->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_rmsd_bf16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_,
                     scale ? &scale->raw_ : nullptr, &metric->raw_);
    // Scalar fallback: raw √(Σ‖aᵢ − bᵢ‖² / n), no centering
    else {
        if (a_centroid)
            a_centroid[0] = transform_type_(0.0), a_centroid[1] = transform_type_(0.0),
            a_centroid[2] = transform_type_(0.0);
        if (b_centroid)
            b_centroid[0] = transform_type_(0.0), b_centroid[1] = transform_type_(0.0),
            b_centroid[2] = transform_type_(0.0);
        if (rotation) {
            rotation[0] = transform_type_(1.0), rotation[1] = transform_type_(0.0), rotation[2] = transform_type_(0.0);
            rotation[3] = transform_type_(0.0), rotation[4] = transform_type_(1.0), rotation[5] = transform_type_(0.0);
            rotation[6] = transform_type_(0.0), rotation[7] = transform_type_(0.0), rotation[8] = transform_type_(1.0);
        }
        if (scale) *scale = transform_type_(1.0);

        metric_type_ sum_squared {};
        for (std::size_t i = 0; i < n; i++) {
            metric_type_ dx = metric_type_(a[i * 3 + 0]) - metric_type_(b[i * 3 + 0]);
            metric_type_ dy = metric_type_(a[i * 3 + 1]) - metric_type_(b[i * 3 + 1]);
            metric_type_ dz = metric_type_(a[i * 3 + 2]) - metric_type_(b[i * 3 + 2]);
            sum_squared = sum_squared + dx * dx + dy * dy + dz * dz;
        }

        *metric = (sum_squared / metric_type_(static_cast<double>(n))).sqrt();
    }
}

/**
 *  @brief Kabsch algorithm: min ‖P − R × Q‖² over rotation R ∈ SO(3)
 *  @param[in] a,b Point clouds [n x 3] interleaved (source and target)
 *  @param[in] n Number of 3D points
 *  @param[out] a_centroid,b_centroid Centroids (3 values each), can be nullptr
 *  @param[out] rotation 3x3 rotation matrix (9 values, row-major), can be nullptr
 *  @param[out] scale Scale factor, always 1.0 for Kabsch, can be nullptr
 *  @param[out] metric Output RMSD after optimal rotation
 *
 *  @tparam in_type_ Input point type (f32_t, f64_t, f16_t, bf16_t)
 *  @tparam transform_type_ Type of centroids, rotation, and scale outputs
 *  @tparam metric_type_ Type of the scalar fit metric output
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename transform_type_ = typename in_type_::mesh_transform_t,
          typename metric_type_ = typename in_type_::mesh_metric_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void kabsch(                                             //
    in_type_ const *a, in_type_ const *b, std::size_t n, //
    transform_type_ *a_centroid, transform_type_ *b_centroid, transform_type_ *rotation, transform_type_ *scale,
    metric_type_ *metric) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<transform_type_, typename in_type_::mesh_transform_t> &&
                          std::is_same_v<metric_type_, typename in_type_::mesh_metric_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_kabsch_f64(&a->raw_, &b->raw_, n, a_centroid ? &a_centroid->raw_ : nullptr, &b_centroid->raw_,
                      &rotation->raw_, &scale->raw_, &metric->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_kabsch_f32(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                      &metric->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_kabsch_f16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                      &metric->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_kabsch_bf16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                       &metric->raw_);
    // Scalar fallback
    else {
        // Compute centroids
        metric_type_ sum_a_x {}, sum_a_y {}, sum_a_z {};
        metric_type_ sum_b_x {}, sum_b_y {}, sum_b_z {};
        metric_type_ val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;

        for (std::size_t i = 0; i < n; i++) {
            val_a_x = metric_type_(a[i * 3 + 0]), val_a_y = metric_type_(a[i * 3 + 1]),
            val_a_z = metric_type_(a[i * 3 + 2]);
            val_b_x = metric_type_(b[i * 3 + 0]), val_b_y = metric_type_(b[i * 3 + 1]),
            val_b_z = metric_type_(b[i * 3 + 2]);
            sum_a_x = sum_a_x + val_a_x, sum_a_y = sum_a_y + val_a_y, sum_a_z = sum_a_z + val_a_z;
            sum_b_x = sum_b_x + val_b_x, sum_b_y = sum_b_y + val_b_y, sum_b_z = sum_b_z + val_b_z;
        }

        metric_type_ inv_n = metric_type_(1.0) / metric_type_(static_cast<double>(n));
        metric_type_ centroid_a_x = sum_a_x * inv_n;
        metric_type_ centroid_a_y = sum_a_y * inv_n;
        metric_type_ centroid_a_z = sum_a_z * inv_n;
        metric_type_ centroid_b_x = sum_b_x * inv_n;
        metric_type_ centroid_b_y = sum_b_y * inv_n;
        metric_type_ centroid_b_z = sum_b_z * inv_n;

        if (a_centroid)
            a_centroid[0] = transform_type_(centroid_a_x), a_centroid[1] = transform_type_(centroid_a_y),
            a_centroid[2] = transform_type_(centroid_a_z);

        if (b_centroid)
            b_centroid[0] = transform_type_(centroid_b_x), b_centroid[1] = transform_type_(centroid_b_y),
            b_centroid[2] = transform_type_(centroid_b_z);

        // Build 3x3 covariance matrix H = (A - A_bar)^T x (B - B_bar)
        metric_type_ cross_covariance[9] = {};
        for (std::size_t i = 0; i < n; i++) {
            val_a_x = metric_type_(a[i * 3 + 0]) - centroid_a_x, val_a_y = metric_type_(a[i * 3 + 1]) - centroid_a_y,
            val_a_z = metric_type_(a[i * 3 + 2]) - centroid_a_z;
            val_b_x = metric_type_(b[i * 3 + 0]) - centroid_b_x, val_b_y = metric_type_(b[i * 3 + 1]) - centroid_b_y,
            val_b_z = metric_type_(b[i * 3 + 2]) - centroid_b_z;
            cross_covariance[0] = cross_covariance[0] + val_a_x * val_b_x;
            cross_covariance[1] = cross_covariance[1] + val_a_x * val_b_y;
            cross_covariance[2] = cross_covariance[2] + val_a_x * val_b_z;
            cross_covariance[3] = cross_covariance[3] + val_a_y * val_b_x;
            cross_covariance[4] = cross_covariance[4] + val_a_y * val_b_y;
            cross_covariance[5] = cross_covariance[5] + val_a_y * val_b_z;
            cross_covariance[6] = cross_covariance[6] + val_a_z * val_b_x;
            cross_covariance[7] = cross_covariance[7] + val_a_z * val_b_y;
            cross_covariance[8] = cross_covariance[8] + val_a_z * val_b_z;
        }

        // SVD of H = U * S * Vt
        metric_type_ svd_u[9], svd_s[9], svd_v[9];
        svd3x3_(cross_covariance, svd_u, svd_s, svd_v);

        // R = V * Ut
        metric_type_ rotation_matrix[9];
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

        // Handle reflection: if det(R) < 0, negate third column of V and recompute R
        metric_type_ rotation_det = det3x3_(rotation_matrix);
        if (rotation_det < metric_type_(0.0)) {
            svd_v[2] = metric_type_(0.0) - svd_v[2];
            svd_v[5] = metric_type_(0.0) - svd_v[5];
            svd_v[8] = metric_type_(0.0) - svd_v[8];
            rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
            rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
            rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
            rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
            rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
            rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
            rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
            rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
            rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
        }

        // Output rotation matrix and scale=1.0
        if (rotation) {
            for (unsigned int j = 0; j < 9; j++) rotation[j] = transform_type_(rotation_matrix[j]);
        }
        if (scale) *scale = transform_type_(1.0);

        // Compute RMSD after rotation
        metric_type_ sum_squared {};
        for (std::size_t i = 0; i < n; i++) {
            metric_type_ point_a[3], point_b[3], rotated_point_a[3];
            point_a[0] = metric_type_(a[i * 3 + 0]) - centroid_a_x,
            point_a[1] = metric_type_(a[i * 3 + 1]) - centroid_a_y,
            point_a[2] = metric_type_(a[i * 3 + 2]) - centroid_a_z;
            point_b[0] = metric_type_(b[i * 3 + 0]) - centroid_b_x,
            point_b[1] = metric_type_(b[i * 3 + 1]) - centroid_b_y,
            point_b[2] = metric_type_(b[i * 3 + 2]) - centroid_b_z;
            rotated_point_a[0] = rotation_matrix[0] * point_a[0] + rotation_matrix[1] * point_a[1] +
                                 rotation_matrix[2] * point_a[2];
            rotated_point_a[1] = rotation_matrix[3] * point_a[0] + rotation_matrix[4] * point_a[1] +
                                 rotation_matrix[5] * point_a[2];
            rotated_point_a[2] = rotation_matrix[6] * point_a[0] + rotation_matrix[7] * point_a[1] +
                                 rotation_matrix[8] * point_a[2];
            metric_type_ dx = rotated_point_a[0] - point_b[0];
            metric_type_ dy = rotated_point_a[1] - point_b[1];
            metric_type_ dz = rotated_point_a[2] - point_b[2];
            sum_squared = sum_squared + dx * dx + dy * dy + dz * dz;
        }

        *metric = (sum_squared * inv_n).sqrt();
    }
}

/**
 *  @brief Umeyama algorithm: min ‖P − s × R × Q‖² over R ∈ SO(3), s ∈ ℝ⁺
 *  @param[in] a,b Point clouds [n x 3] interleaved (source and target)
 *  @param[in] n Number of 3D points
 *  @param[out] a_centroid,b_centroid Centroids (3 values each), can be nullptr
 *  @param[out] rotation 3x3 rotation matrix (9 values, row-major), can be nullptr
 *  @param[out] scale Uniform scale factor, can be nullptr
 *  @param[out] metric Output RMSD after optimal transformation
 *
 *  @tparam in_type_ Input point type (f32_t, f64_t, f16_t, bf16_t)
 *  @tparam transform_type_ Type of centroids, rotation, and scale outputs
 *  @tparam metric_type_ Type of the scalar fit metric output
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename transform_type_ = typename in_type_::mesh_transform_t,
          typename metric_type_ = typename in_type_::mesh_metric_t, allow_simd_t allow_simd_ = prefer_simd_k>
void umeyama(in_type_ const *a, in_type_ const *b, std::size_t n, transform_type_ *a_centroid,
             transform_type_ *b_centroid, transform_type_ *rotation, transform_type_ *scale,
             metric_type_ *metric) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<transform_type_, typename in_type_::mesh_transform_t> &&
                          std::is_same_v<metric_type_, typename in_type_::mesh_metric_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_umeyama_f64(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                       &metric->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_umeyama_f32(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                       &metric->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_umeyama_f16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                       &metric->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_umeyama_bf16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                        &metric->raw_);
    // Scalar fallback
    else {
        // Compute centroids
        metric_type_ sum_a_x {}, sum_a_y {}, sum_a_z {};
        metric_type_ sum_b_x {}, sum_b_y {}, sum_b_z {};
        metric_type_ val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;

        for (std::size_t i = 0; i < n; i++) {
            val_a_x = metric_type_(a[i * 3 + 0]), val_a_y = metric_type_(a[i * 3 + 1]),
            val_a_z = metric_type_(a[i * 3 + 2]);
            val_b_x = metric_type_(b[i * 3 + 0]), val_b_y = metric_type_(b[i * 3 + 1]),
            val_b_z = metric_type_(b[i * 3 + 2]);
            sum_a_x = sum_a_x + val_a_x, sum_a_y = sum_a_y + val_a_y, sum_a_z = sum_a_z + val_a_z;
            sum_b_x = sum_b_x + val_b_x, sum_b_y = sum_b_y + val_b_y, sum_b_z = sum_b_z + val_b_z;
        }

        metric_type_ inv_n = metric_type_(1.0) / metric_type_(static_cast<double>(n));
        metric_type_ centroid_a_x = sum_a_x * inv_n;
        metric_type_ centroid_a_y = sum_a_y * inv_n;
        metric_type_ centroid_a_z = sum_a_z * inv_n;
        metric_type_ centroid_b_x = sum_b_x * inv_n;
        metric_type_ centroid_b_y = sum_b_y * inv_n;
        metric_type_ centroid_b_z = sum_b_z * inv_n;

        if (a_centroid)
            a_centroid[0] = transform_type_(centroid_a_x), a_centroid[1] = transform_type_(centroid_a_y),
            a_centroid[2] = transform_type_(centroid_a_z);

        if (b_centroid)
            b_centroid[0] = transform_type_(centroid_b_x), b_centroid[1] = transform_type_(centroid_b_y),
            b_centroid[2] = transform_type_(centroid_b_z);

        // Build covariance matrix H and compute variance of A
        metric_type_ cross_covariance[9] = {};
        metric_type_ variance_a {};
        for (std::size_t i = 0; i < n; i++) {
            val_a_x = metric_type_(a[i * 3 + 0]) - centroid_a_x;
            val_a_y = metric_type_(a[i * 3 + 1]) - centroid_a_y;
            val_a_z = metric_type_(a[i * 3 + 2]) - centroid_a_z;
            val_b_x = metric_type_(b[i * 3 + 0]) - centroid_b_x;
            val_b_y = metric_type_(b[i * 3 + 1]) - centroid_b_y;
            val_b_z = metric_type_(b[i * 3 + 2]) - centroid_b_z;
            variance_a = variance_a + val_a_x * val_a_x + val_a_y * val_a_y + val_a_z * val_a_z;
            cross_covariance[0] = cross_covariance[0] + val_a_x * val_b_x;
            cross_covariance[1] = cross_covariance[1] + val_a_x * val_b_y;
            cross_covariance[2] = cross_covariance[2] + val_a_x * val_b_z;
            cross_covariance[3] = cross_covariance[3] + val_a_y * val_b_x;
            cross_covariance[4] = cross_covariance[4] + val_a_y * val_b_y;
            cross_covariance[5] = cross_covariance[5] + val_a_y * val_b_z;
            cross_covariance[6] = cross_covariance[6] + val_a_z * val_b_x;
            cross_covariance[7] = cross_covariance[7] + val_a_z * val_b_y;
            cross_covariance[8] = cross_covariance[8] + val_a_z * val_b_z;
        }
        variance_a = variance_a * inv_n;

        // SVD of H = U * S * Vt
        metric_type_ svd_u[9], svd_s[9], svd_v[9];
        svd3x3_(cross_covariance, svd_u, svd_s, svd_v);

        // R = V * Ut
        metric_type_ rotation_matrix[9];
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

        // Handle reflection and compute scale: c = trace(D*S) / variance_a
        // D = diag(1, 1, det(R)), svd_s contains singular values on diagonal
        metric_type_ rotation_det = det3x3_(rotation_matrix);
        metric_type_ sign_det = rotation_det < metric_type_(0.0) ? metric_type_(-1.0) : metric_type_(1.0);
        metric_type_ trace_scaled_s = svd_s[0] + svd_s[4] + sign_det * svd_s[8];
        metric_type_ scale_factor = trace_scaled_s / (metric_type_(static_cast<double>(n)) * variance_a);

        if (scale) *scale = transform_type_(scale_factor);

        if (rotation_det < metric_type_(0.0)) {
            svd_v[2] = metric_type_(0.0) - svd_v[2];
            svd_v[5] = metric_type_(0.0) - svd_v[5];
            svd_v[8] = metric_type_(0.0) - svd_v[8];
            rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
            rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
            rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
            rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
            rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
            rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
            rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
            rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
            rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
        }

        // Output rotation matrix
        if (rotation) {
            for (unsigned int j = 0; j < 9; j++) rotation[j] = transform_type_(rotation_matrix[j]);
        }

        // Compute RMSD after similarity transform: ||c * R * a - b||
        metric_type_ sum_squared {};
        for (std::size_t i = 0; i < n; i++) {
            metric_type_ point_a[3], point_b[3], rotated_point_a[3];
            point_a[0] = metric_type_(a[i * 3 + 0]) - centroid_a_x,
            point_a[1] = metric_type_(a[i * 3 + 1]) - centroid_a_y,
            point_a[2] = metric_type_(a[i * 3 + 2]) - centroid_a_z;
            point_b[0] = metric_type_(b[i * 3 + 0]) - centroid_b_x,
            point_b[1] = metric_type_(b[i * 3 + 1]) - centroid_b_y,
            point_b[2] = metric_type_(b[i * 3 + 2]) - centroid_b_z;
            rotated_point_a[0] = scale_factor * (rotation_matrix[0] * point_a[0] + rotation_matrix[1] * point_a[1] +
                                                 rotation_matrix[2] * point_a[2]);
            rotated_point_a[1] = scale_factor * (rotation_matrix[3] * point_a[0] + rotation_matrix[4] * point_a[1] +
                                                 rotation_matrix[5] * point_a[2]);
            rotated_point_a[2] = scale_factor * (rotation_matrix[6] * point_a[0] + rotation_matrix[7] * point_a[1] +
                                                 rotation_matrix[8] * point_a[2]);
            metric_type_ dx = rotated_point_a[0] - point_b[0];
            metric_type_ dy = rotated_point_a[1] - point_b[1];
            metric_type_ dz = rotated_point_a[2] - point_b[2];
            sum_squared = sum_squared + dx * dx + dy * dy + dz * dz;
        }

        *metric = (sum_squared * inv_n).sqrt();
    }
}

#pragma endregion Mesh Alignment Kernels

} // namespace ashvardanian::numkong

#endif // NK_MESH_HPP
