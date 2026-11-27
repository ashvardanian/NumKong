/**
 *  @file test/shapes.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief Reference shape distances, the oracle of the host and device shape tests.
 *
 *  Each reference runs the algorithm of @c shapes/serial.h in a distance type of the caller's
 *  choosing, which the tests instantiate in @c f118_t and @c f64_t.
 */
#pragma once
#ifndef NUMKONG_TEST_SHAPES_HPP
#define NUMKONG_TEST_SHAPES_HPP

#include <cmath>   // `std::sqrt`, `std::ldexp`
#include <cstdint> // `std::uint32_t`, `std::uintptr_t`
#include <cstring> // `std::memcpy`

#include <limits>      // `std::numeric_limits`
#include <span>        // `std::span`
#include <type_traits> // `std::conditional_t`
#include <utility>     // `std::swap`
#include <vector>      // `std::vector`

#include "numkong/shapes.hpp" // `nk::a_shapes_dtype`

#include "harness.hpp" // `half_ulp`

namespace ashvardanian::numkong::test {

/** The type the transport duals and the Gromov-Wasserstein line search run in: at least @c f64. */
template <typename math_type_>
using shapes_wide_ = std::conditional_t<(sizeof(math_type_) < sizeof(f64_t)), f64_t, math_type_>;

template <typename scalar_type_>
scalar_type_ det3x3_(scalar_type_ const *m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

template <typename scalar_type_>
void conditional_swap_(bool should_swap, scalar_type_ *x, scalar_type_ *y) {
    scalar_type_ original_x = *x;
    *x = should_swap ? *y : *x;
    *y = should_swap ? original_x : *y;
}

template <typename scalar_type_>
void conditional_negating_swap_(bool should_swap, scalar_type_ *x, scalar_type_ *y) {
    scalar_type_ negated_x = narrow<scalar_type_>(0.0) - *x;
    *x = should_swap ? *y : *x;
    *y = should_swap ? negated_x : *y;
}

/** Whether @p scalar_type_ takes the single-precision constants of the C kernels' 3 × 3 solvers. */
template <typename scalar_type_>
inline constexpr bool shapes_single_precision_ = sizeof(scalar_type_) == sizeof(nk_f32_t);

template <typename scalar_type_>
void approximate_givens_quaternion_(scalar_type_ a11, scalar_type_ a12, scalar_type_ a22, scalar_type_ *cos_half,
                                    scalar_type_ *sin_half) {
    constexpr bool single_k = shapes_single_precision_<scalar_type_>;
    // γ = (√8 + 3)² / 4, with cos(π/8) and sin(π/8)
    scalar_type_ const gamma = single_k ? narrow<scalar_type_>(5.828427124f) : narrow<scalar_type_>(5.828427124746190);
    scalar_type_ const cstar = single_k ? narrow<scalar_type_>(0.923879532f) : narrow<scalar_type_>(0.9238795325112867);
    scalar_type_ const sstar = single_k ? narrow<scalar_type_>(0.3826834323f)
                                        : narrow<scalar_type_>(0.3826834323650898);

    *cos_half = narrow<scalar_type_>(2.0) * (a11 - a22);
    *sin_half = a12;
    bool use_givens = gamma * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);
    scalar_type_ w = reciprocal_square_root_reference((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));
    *cos_half = use_givens ? w * (*cos_half) : cstar;
    *sin_half = use_givens ? w * (*sin_half) : sstar;
}

template <typename scalar_type_>
void jacobi_conjugation_(int idx_x, int idx_y, int idx_z, scalar_type_ *s11, scalar_type_ *s21, scalar_type_ *s22,
                         scalar_type_ *s31, scalar_type_ *s32, scalar_type_ *s33, scalar_type_ *quat) {

    scalar_type_ cos_half, sin_half;
    approximate_givens_quaternion_(*s11, *s21, *s22, &cos_half, &sin_half);
    scalar_type_ scale = cos_half * cos_half + sin_half * sin_half;
    scalar_type_ cos_theta = (cos_half * cos_half - sin_half * sin_half) / scale;
    scalar_type_ sin_theta = (narrow<scalar_type_>(2.0) * sin_half * cos_half) / scale;
    scalar_type_ s11_old = *s11, s21_old = *s21, s22_old = *s22;
    scalar_type_ s31_old = *s31, s32_old = *s32, s33_old = *s33;

    *s11 = cos_theta * (cos_theta * s11_old + sin_theta * s21_old) +
           sin_theta * (cos_theta * s21_old + sin_theta * s22_old);
    *s21 = cos_theta * ((narrow<scalar_type_>(0.0) - sin_theta) * s11_old + cos_theta * s21_old) +
           sin_theta * ((narrow<scalar_type_>(0.0) - sin_theta) * s21_old + cos_theta * s22_old);
    *s22 = (narrow<scalar_type_>(0.0) - sin_theta) *
               ((narrow<scalar_type_>(0.0) - sin_theta) * s11_old + cos_theta * s21_old) +
           cos_theta * ((narrow<scalar_type_>(0.0) - sin_theta) * s21_old + cos_theta * s22_old);
    *s31 = cos_theta * s31_old + sin_theta * s32_old;
    *s32 = (narrow<scalar_type_>(0.0) - sin_theta) * s31_old + cos_theta * s32_old;
    *s33 = s33_old;

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
    s11_old = *s22, s21_old = *s32, s22_old = *s33, s31_old = *s21, s32_old = *s31, s33_old = *s11;
    *s11 = s11_old, *s21 = s21_old, *s22 = s22_old, *s31 = s31_old, *s32 = s32_old, *s33 = s33_old;
}

/** The row-major rotation of the unit quaternion (x, y, z, w). */
template <typename scalar_type_>
void quaternion_to_mat3x3_(scalar_type_ const *quat, scalar_type_ *matrix) {
    scalar_type_ w = quat[3], x = quat[0], y = quat[1], z = quat[2];
    scalar_type_ q_xx = x * x, q_yy = y * y, q_zz = z * z;
    scalar_type_ q_xz = x * z, q_xy = x * y, q_yz = y * z;
    scalar_type_ q_wx = w * x, q_wy = w * y, q_wz = w * z;
    matrix[0] = narrow<scalar_type_>(1.0) - narrow<scalar_type_>(2.0) * (q_yy + q_zz);
    matrix[1] = narrow<scalar_type_>(2.0) * (q_xy - q_wz);
    matrix[2] = narrow<scalar_type_>(2.0) * (q_xz + q_wy);
    matrix[3] = narrow<scalar_type_>(2.0) * (q_xy + q_wz);
    matrix[4] = narrow<scalar_type_>(1.0) - narrow<scalar_type_>(2.0) * (q_xx + q_zz);
    matrix[5] = narrow<scalar_type_>(2.0) * (q_yz - q_wx);
    matrix[6] = narrow<scalar_type_>(2.0) * (q_xz - q_wy);
    matrix[7] = narrow<scalar_type_>(2.0) * (q_yz + q_wx);
    matrix[8] = narrow<scalar_type_>(1.0) - narrow<scalar_type_>(2.0) * (q_xx + q_yy);
}

/** The unit quaternion (x, y, z, w) of a row-major rotation, by Shepperd's method. */
template <typename scalar_type_>
void mat3x3_to_quaternion_(scalar_type_ const *m, scalar_type_ *quat) {
    scalar_type_ const one(1.0), two(2.0), quarter(0.25);
    scalar_type_ const trace = m[0] + m[4] + m[8];
    if (trace > narrow<scalar_type_>(0.0)) {
        scalar_type_ const s = two * square_root_reference(trace + one);
        quat[3] = s * quarter, quat[0] = (m[7] - m[5]) / s, quat[1] = (m[2] - m[6]) / s, quat[2] = (m[3] - m[1]) / s;
    }
    else if (m[0] > m[4] && m[0] > m[8]) {
        scalar_type_ const s = two * square_root_reference(one + m[0] - m[4] - m[8]);
        quat[3] = (m[7] - m[5]) / s, quat[0] = s * quarter, quat[1] = (m[1] + m[3]) / s, quat[2] = (m[2] + m[6]) / s;
    }
    else if (m[4] > m[8]) {
        scalar_type_ const s = two * square_root_reference(one + m[4] - m[0] - m[8]);
        quat[3] = (m[2] - m[6]) / s, quat[0] = (m[1] + m[3]) / s, quat[1] = s * quarter, quat[2] = (m[5] + m[7]) / s;
    }
    else {
        scalar_type_ const s = two * square_root_reference(one + m[8] - m[0] - m[4]);
        quat[3] = (m[3] - m[1]) / s, quat[0] = (m[2] + m[6]) / s, quat[1] = (m[5] + m[7]) / s, quat[2] = s * quarter;
    }
    scalar_type_ const norm = reciprocal_square_root_reference(quat[0] * quat[0] + quat[1] * quat[1] +
                                                               quat[2] * quat[2] + quat[3] * quat[3]);
    for (int i = 0; i != 4; ++i) quat[i] = quat[i] * norm;
}

/** Diagonalizes the symmetric 3 × 3 matrix in place, leaving its eigenvalues on the diagonal and
 *  the quaternion of the eigenvector columns in @p quat. */
template <typename scalar_type_>
void jacobi_eigenanalysis_(scalar_type_ *s11, scalar_type_ *s21, scalar_type_ *s22, scalar_type_ *s31,
                           scalar_type_ *s32, scalar_type_ *s33, scalar_type_ *quat) {
    quat[0] = narrow<scalar_type_>(0.0);
    quat[1] = narrow<scalar_type_>(0.0);
    quat[2] = narrow<scalar_type_>(0.0);
    quat[3] = narrow<scalar_type_>(1.0);
    for (unsigned int iter = 0; iter < 16; iter++) {
        jacobi_conjugation_(0, 1, 2, s11, s21, s22, s31, s32, s33, quat);
        jacobi_conjugation_(1, 2, 0, s11, s21, s22, s31, s32, s33, quat);
        jacobi_conjugation_(2, 0, 1, s11, s21, s22, s31, s32, s33, quat);
    }
    scalar_type_ norm = reciprocal_square_root_reference(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] +
                                                         quat[3] * quat[3]);
    quat[0] = quat[0] * norm;
    quat[1] = quat[1] * norm;
    quat[2] = quat[2] * norm;
    quat[3] = quat[3] * norm;
}

/** Sorts three eigenvalues descending, swapping their eigenvector columns and negating one of each
 *  swapped pair, so the determinant stays. */
template <typename scalar_type_>
void sort_eigenvectors_(scalar_type_ *values, scalar_type_ *vectors) {
    auto const step = [&](int first, int second) {
        bool const should_swap = values[first] < values[second];
        conditional_swap_(should_swap, &values[first], &values[second]);
        for (int row = 0; row != 3; ++row)
            conditional_negating_swap_(should_swap, &vectors[row * 3 + first], &vectors[row * 3 + second]);
    };
    step(0, 1), step(0, 2), step(1, 2);
}

template <typename scalar_type_>
void qr_givens_quaternion_(scalar_type_ a1, scalar_type_ a2, scalar_type_ *cos_half, scalar_type_ *sin_half) {
    scalar_type_ const epsilon = shapes_single_precision_<scalar_type_> ? narrow<scalar_type_>(1e-6f)
                                                                        : narrow<scalar_type_>(1e-12);

    scalar_type_ a1_sq_plus_a2_sq = a1 * a1 + a2 * a2;
    scalar_type_ rho = a1_sq_plus_a2_sq * reciprocal_square_root_reference(a1_sq_plus_a2_sq);
    rho = a1_sq_plus_a2_sq > epsilon ? rho : narrow<scalar_type_>(0.0);
    *sin_half = rho > epsilon ? a2 : narrow<scalar_type_>(0.0);
    scalar_type_ abs_a1 = a1 < narrow<scalar_type_>(0.0) ? (narrow<scalar_type_>(0.0) - a1) : a1;
    scalar_type_ max_rho = rho > epsilon ? rho : epsilon;
    *cos_half = abs_a1 + max_rho;
    bool should_swap = a1 < narrow<scalar_type_>(0.0);
    conditional_swap_(should_swap, sin_half, cos_half);
    scalar_type_ w = reciprocal_square_root_reference((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));
    *cos_half = (*cos_half) * w;
    *sin_half = (*sin_half) * w;
}

template <typename scalar_type_>
void sort_singular_values_(scalar_type_ *b, scalar_type_ *v) {
    scalar_type_ rho1 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];
    scalar_type_ rho2 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];
    scalar_type_ rho3 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];
    bool should_swap;
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

/** The orthogonal factor of a 3 × 3 QR decomposition by three Givens rotations. */
template <typename scalar_type_>
void qr_orthogonal_factor_(scalar_type_ const *input, scalar_type_ *q) {
    scalar_type_ const one(1.0), two(2.0), four(4.0), eight(8.0);
    scalar_type_ cos_half_1, sin_half_1, cos_half_2, sin_half_2, cos_half_3, sin_half_3;
    qr_givens_quaternion_(input[0], input[3], &cos_half_1, &sin_half_1);
    scalar_type_ cos_theta = one - two * sin_half_1 * sin_half_1;
    scalar_type_ sin_theta = two * cos_half_1 * sin_half_1;
    scalar_type_ const rotated_1 = cos_theta * input[1] + sin_theta * input[4];
    scalar_type_ const rotated_4 = -sin_theta * input[1] + cos_theta * input[4];
    qr_givens_quaternion_(input[0] * cos_theta + sin_theta * input[3], input[6], &cos_half_2, &sin_half_2);
    cos_theta = one - two * sin_half_2 * sin_half_2;
    sin_theta = two * cos_half_2 * sin_half_2;
    qr_givens_quaternion_(rotated_4, -sin_theta * rotated_1 + cos_theta * input[7], &cos_half_3, &sin_half_3);
    scalar_type_ const sin_half_1_sq = sin_half_1 * sin_half_1;
    scalar_type_ const sin_half_2_sq = sin_half_2 * sin_half_2;
    scalar_type_ const sin_half_3_sq = sin_half_3 * sin_half_3;
    q[0] = (-one + two * sin_half_1_sq) * (-one + two * sin_half_2_sq);
    q[1] = four * cos_half_2 * cos_half_3 * (-one + two * sin_half_1_sq) * sin_half_2 * sin_half_3 +
           two * cos_half_1 * sin_half_1 * (-one + two * sin_half_3_sq);
    q[2] = four * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -
           two * cos_half_2 * (-one + two * sin_half_1_sq) * sin_half_2 * (-one + two * sin_half_3_sq);
    q[3] = two * cos_half_1 * sin_half_1 * (one - two * sin_half_2_sq);
    q[4] = -eight * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +
           (-one + two * sin_half_1_sq) * (-one + two * sin_half_3_sq);
    q[5] = -two * cos_half_3 * sin_half_3 + four * sin_half_1 *
                                                (cos_half_3 * sin_half_1 * sin_half_3 +
                                                 cos_half_1 * cos_half_2 * sin_half_2 * (-one + two * sin_half_3_sq));
    q[6] = two * cos_half_2 * sin_half_2;
    q[7] = two * cos_half_3 * (one - two * sin_half_2_sq) * sin_half_3;
    q[8] = (-one + two * sin_half_2_sq) * (-one + two * sin_half_3_sq);
}

/** The 3 × 3 SVD A = U × Σ × Vᵀ by McAdams' method, keeping only the orthogonal factors. */
template <typename scalar_type_>
void svd3x3_(scalar_type_ const *a, scalar_type_ *svd_u, scalar_type_ *svd_v) {
    scalar_type_ ata[9];
    ata[0] = a[0] * a[0] + a[3] * a[3] + a[6] * a[6];
    ata[1] = a[0] * a[1] + a[3] * a[4] + a[6] * a[7];
    ata[2] = a[0] * a[2] + a[3] * a[5] + a[6] * a[8];
    ata[4] = a[1] * a[1] + a[4] * a[4] + a[7] * a[7];
    ata[5] = a[1] * a[2] + a[4] * a[5] + a[7] * a[8];
    ata[8] = a[2] * a[2] + a[5] * a[5] + a[8] * a[8];
    scalar_type_ quat[4];
    jacobi_eigenanalysis_(&ata[0], &ata[1], &ata[4], &ata[2], &ata[5], &ata[8], quat);
    quaternion_to_mat3x3_(quat, svd_v);
    scalar_type_ product[9];
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            product[row * 3 + column] = a[row * 3 + 0] * svd_v[0 * 3 + column] +
                                        a[row * 3 + 1] * svd_v[1 * 3 + column] + a[row * 3 + 2] * svd_v[2 * 3 + column];
    sort_singular_values_(product, svd_v);
    qr_orthogonal_factor_(product, svd_u);
}

/** R = V × Uᵀ, the rotation an SVD of a cross-covariance yields. */
template <typename scalar_type_>
void rotation_from_svd_(scalar_type_ const *svd_u, scalar_type_ const *svd_v, scalar_type_ *rotation) {
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            rotation[row * 3 + column] = svd_v[row * 3 + 0] * svd_u[column * 3 + 0] +
                                         svd_v[row * 3 + 1] * svd_u[column * 3 + 1] +
                                         svd_v[row * 3 + 2] * svd_u[column * 3 + 2];
}

/** Converts @p value of the math type into an output, skipping @c double when both types match. */
template <typename output_type_, typename math_type_>
output_type_ shapes_output_(math_type_ value) noexcept {
    if constexpr (std::is_same_v<output_type_, math_type_>) return value;
    else return output_type_(to_double(value));
}

/** Writes the rotation V_a × core × V_bᵀ. */
template <typename math_type_, typename rotation_type_>
void shapes_write_rotation_(math_type_ const *a_frame, math_type_ const *core, math_type_ const *b_frame,
                            rotation_type_ *rotation) noexcept {
    math_type_ right[9];
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            right[row * 3 + column] = core[row * 3 + 0] * b_frame[column * 3 + 0] +
                                      core[row * 3 + 1] * b_frame[column * 3 + 1] +
                                      core[row * 3 + 2] * b_frame[column * 3 + 2];
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column) {
            math_type_ const value = a_frame[row * 3 + 0] * right[0 * 3 + column] +
                                     a_frame[row * 3 + 1] * right[1 * 3 + column] +
                                     a_frame[row * 3 + 2] * right[2 * 3 + column];
            rotation[row * 3 + column] = shapes_output_<rotation_type_>(value);
        }
}

template <typename rotation_type_>
void shapes_write_identity_(rotation_type_ *rotation) noexcept {
    for (int i = 0; i != 9; ++i) rotation[i] = rotation_type_(i % 4 == 0 ? 1.0 : 0.0);
}

/** The diagonal of flip @p flip: bit k negates axis k. */
template <typename math_type_>
void shapes_flip_(unsigned flip, math_type_ *matrix) noexcept {
    for (int i = 0; i != 9; ++i) matrix[i] = math_type_(0.0);
    for (unsigned axis = 0; axis != 3; ++axis) matrix[axis * 4] = math_type_((flip >> axis) & 1 ? -1.0 : 1.0);
}

/** The unit complex number at 1 / @p order of the angle of @p real + i @p imaginary in (−π, π],
 *  as the product of the halvings of that angle that the binary digits of 1 / @p order pick. */
template <typename math_type_>
void shapes_unit_root_(math_type_ real, math_type_ imaginary, int order, math_type_ *root) noexcept {
    using math_t = math_type_;
    math_t const zero(0.0), one(1.0), two(2.0);
    if (order == 1) {
        root[0] = real, root[1] = imaginary;
        return;
    }
    math_t root_real = one, root_imaginary = zero;
    int remainder = 1;
    for (int digit = 0; digit != int(sizeof(math_t)) * 7; ++digit) {
        // Halving through the larger of the half angle's cosine and sine keeps divisors big
        math_t half_real, half_imaginary;
        if (real >= zero)
            half_real = square_root_reference((one + real) / two), half_imaginary = imaginary / (two * half_real);
        else
            half_imaginary = (imaginary < zero ? zero - one : one) * square_root_reference((one - real) / two),
            half_real = imaginary / (two * half_imaginary);
        real = half_real, imaginary = half_imaginary, remainder *= 2;
        if (remainder < order) continue;
        remainder -= order;
        math_t const next_real = root_real * real - root_imaginary * imaginary;
        root_imaginary = root_imaginary * real + root_real * imaginary, root_real = next_real;
    }
    math_t const inverse = reciprocal_square_root_reference(root_real * root_real + root_imaginary * root_imaginary);
    root[0] = root_real * inverse, root[1] = root_imaginary * inverse;
}

/** The centered point @p index of a cloud of @p in_type_ in @p math_type_ . */
template <typename in_type_, typename math_type_>
void shapes_centered_(typename in_type_::raw_t const *points, std::size_t index, math_type_ const *centroid,
                      math_type_ *centered) noexcept {
    for (std::size_t axis = 0; axis != 3; ++axis)
        centered[axis] = math_type_(to_double(in_type_::from_raw(points[index * 3 + axis]))) - centroid[axis];
}

/** The dot product of two 3D vectors. */
template <typename math_type_>
math_type_ shapes_dot_(math_type_ const *left, math_type_ const *right) noexcept {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

/** Turns basis rows a and b about row c until the first harmonic above @p tie_tolerance , a sum
 *  of wᵐ over w = x · a + i x · b, or of h × wᵐ over h = x · c, lies along a, and settles the m
 *  choices of a by the next harmonic that tells them apart, as @c shapes/serial.h settles a tied
 *  plane. First orients c by Σh³. */
template <typename in_type_, typename math_type_>
void shapes_turn_plane_(typename in_type_::raw_t const *points, std::size_t count, math_type_ const *centroid,
                        math_type_ tie_tolerance, math_type_ *basis) noexcept {
    using math_t = math_type_;
    math_t const zero(0.0), one(1.0);
    math_t centered[3], cubes = zero, cube_magnitudes = zero;
    for (std::size_t index = 0; index != count; ++index) {
        shapes_centered_<in_type_>(points, index, centroid, centered);
        math_t const height = shapes_dot_(basis + 6, centered),
                     distance = square_root_reference(shapes_dot_(centered, centered));
        cubes = cubes + height * height * height, cube_magnitudes = cube_magnitudes + distance * distance * distance;
    }
    bool const oriented = cubes.abs() > tie_tolerance * cube_magnitudes;
    if (oriented && cubes < zero)
        for (int cell = 3; cell != 9; ++cell) basis[cell] = zero - basis[cell];
    // Even entries sum wᵐ and odd ones h × wᵐ for m = 1 to 8, beside sums of |w|ᵐ and |x|ᵐ⁺¹
    math_t harmonics[16][2], magnitudes[16];
    for (int entry = 0; entry != 16; ++entry) harmonics[entry][0] = harmonics[entry][1] = magnitudes[entry] = zero;
    for (std::size_t index = 0; index != count; ++index) {
        shapes_centered_<in_type_>(points, index, centroid, centered);
        math_t const along = shapes_dot_(basis, centered), across = shapes_dot_(basis + 3, centered);
        math_t const height = shapes_dot_(basis + 6, centered),
                     distance = square_root_reference(shapes_dot_(centered, centered));
        math_t const radius = square_root_reference(along * along + across * across);
        math_t power_real = one, power_imaginary = zero, power_radius = one, power_distance = distance;
        for (int entry = 0; entry != 16; entry += 2) {
            math_t const next = power_real * along - power_imaginary * across;
            power_imaginary = power_imaginary * along + power_real * across, power_real = next;
            power_radius = power_radius * radius, power_distance = power_distance * distance;
            harmonics[entry][0] = harmonics[entry][0] + power_real;
            harmonics[entry][1] = harmonics[entry][1] + power_imaginary;
            magnitudes[entry] = magnitudes[entry] + power_radius;
            magnitudes[entry + 1] = magnitudes[entry + 1] + power_distance;
            harmonics[entry + 1][0] = harmonics[entry + 1][0] + height * power_real;
            harmonics[entry + 1][1] = harmonics[entry + 1][1] + height * power_imaginary;
        }
    }
    // Unless c is oriented, flipping it conjugates every wᵐ sum and negates its twin, so a twin
    // only chooses at odd m, and only harmonics flipping like the chosen root settle
    int chosen = -1, settling = -1;
    for (int entry = 0; entry != 16 && settling < 0; ++entry) {
        int const twin = entry & 1, odd = (entry / 2) % 2 == 0;
        if (!oriented && (chosen < 0 ? twin && !odd : twin != ((chosen & 1) && odd))) continue;
        if (chosen >= 0 && (entry / 2 + 1) % (chosen / 2 + 1) == 0) continue;
        math_t const strength = harmonics[entry][0] * harmonics[entry][0] + harmonics[entry][1] * harmonics[entry][1];
        math_t const threshold = tie_tolerance * magnitudes[entry];
        if (!(strength > threshold * threshold)) continue;
        if (chosen < 0) chosen = entry;
        else settling = entry;
    }
    if (chosen < 0) return;
    int const order = chosen / 2 + 1;
    math_t root[2];
    math_t const inverse = reciprocal_square_root_reference(harmonics[chosen][0] * harmonics[chosen][0] +
                                                            harmonics[chosen][1] * harmonics[chosen][1]);
    shapes_unit_root_(harmonics[chosen][0] * inverse, harmonics[chosen][1] * inverse, order, root);
    if (settling >= 0) {
        // The m candidates for a step by e^(2πi / m), the square of the m-th root of −1
        math_t turn[2];
        shapes_unit_root_(zero - one, zero, order, turn);
        math_t const turn_real = turn[0] * turn[0] - turn[1] * turn[1],
                     turn_imaginary = math_t(2.0) * turn[0] * turn[1];
        math_t candidate[2] = {root[0], root[1]}, best_score = zero;
        for (int choice = 0; choice != order; ++choice) {
            // The settling harmonic as the candidate would leave it, × conj(candidate)ⁿ
            math_t power[2] = {harmonics[settling][0], harmonics[settling][1]};
            for (int step = 0; step != settling / 2 + 1; ++step) {
                math_t const next = power[0] * candidate[0] + power[1] * candidate[1];
                power[1] = power[1] * candidate[0] - power[0] * candidate[1], power[0] = next;
            }
            if (choice == 0 || power[0] > best_score)
                best_score = power[0], root[0] = candidate[0], root[1] = candidate[1];
            math_t const next = candidate[0] * turn_real - candidate[1] * turn_imaginary;
            candidate[1] = candidate[1] * turn_real + candidate[0] * turn_imaginary, candidate[0] = next;
        }
    }
    // w × conj(root) lays the chosen harmonic along a
    for (int axis = 0; axis != 3; ++axis) {
        math_t const along = basis[axis], across = basis[3 + axis];
        basis[axis] = root[0] * along + root[1] * across, basis[3 + axis] = root[0] * across - root[1] * along;
    }
}

/** Points basis row c at the point whose direction u maximizes Σ(x · u)³, or Σ(x · u)⁴ when no
 *  cubic clears @p tie_tolerance , completes a proper basis and turns the plane about c, as
 *  @c shapes/serial.h settles three tied eigenvalues. */
template <typename in_type_, typename math_type_>
void shapes_pick_axis_(typename in_type_::raw_t const *points, std::size_t count, math_type_ const *centroid,
                       math_type_ tie_tolerance, math_type_ *basis) noexcept {
    using math_t = math_type_;
    math_t const zero(0.0);
    math_t centered[3], candidate[3], cube_magnitudes = zero, best_cubes = zero, best_quartics = zero;
    std::size_t cubes_index = count, quartics_index = count;
    for (std::size_t candidate_index = 0; candidate_index != count; ++candidate_index) {
        shapes_centered_<in_type_>(points, candidate_index, centroid, candidate);
        math_t const length_squared = shapes_dot_(candidate, candidate), length = square_root_reference(length_squared);
        cube_magnitudes = cube_magnitudes + length_squared * length;
        if (!(length_squared > zero)) continue;
        math_t cubes = zero, quartics = zero;
        for (std::size_t index = 0; index != count; ++index) {
            shapes_centered_<in_type_>(points, index, centroid, centered);
            math_t const projection = shapes_dot_(centered, candidate), square = projection * projection;
            cubes = cubes + square * projection, quartics = quartics + square * square;
        }
        cubes = cubes / (length_squared * length), quartics = quartics / (length_squared * length_squared);
        if (cubes > best_cubes) best_cubes = cubes, cubes_index = candidate_index;
        if (quartics > best_quartics) best_quartics = quartics, quartics_index = candidate_index;
    }
    std::size_t const chosen = best_cubes > tie_tolerance * cube_magnitudes ? cubes_index : quartics_index;
    if (chosen == count) return;
    math_t axis[3], least_alignment(2.0);
    shapes_centered_<in_type_>(points, chosen, centroid, axis);
    math_t const axis_inverse = reciprocal_square_root_reference(shapes_dot_(axis, axis));
    for (math_t &value : axis) value = value * axis_inverse;
    // a starts from the row least aligned with c, which keeps most of its length
    int start = 0;
    for (int row = 0; row != 3; ++row)
        if (math_t const alignment = shapes_dot_(basis + row * 3, axis).abs(); alignment < least_alignment)
            least_alignment = alignment, start = row;
    math_t const alignment = shapes_dot_(basis + start * 3, axis);
    for (int cell = 0; cell != 3; ++cell) candidate[cell] = basis[start * 3 + cell] - alignment * axis[cell];
    math_t const candidate_inverse = reciprocal_square_root_reference(shapes_dot_(candidate, candidate));
    for (int cell = 0; cell != 3; ++cell)
        basis[cell] = candidate[cell] * candidate_inverse, basis[6 + cell] = axis[cell];
    // b = c × a completes a proper basis
    for (int cell = 0; cell != 3; ++cell)
        basis[3 + cell] = axis[(cell + 1) % 3] * basis[(cell + 2) % 3] - axis[(cell + 2) % 3] * basis[(cell + 1) % 3];
    shapes_turn_plane_<in_type_>(points, count, centroid, tie_tolerance, basis);
}

/** Settles the eigenvector columns of @p frame , sorted by the descending @p eigenvalues , where
 *  neighbors tie within the serial tolerance: rounding alone picks tied eigenvectors, so higher
 *  moments settle them as @c shapes/serial.h does. */
template <typename in_type_, typename math_type_>
void shapes_settle_ties_(typename in_type_::raw_t const *points, std::size_t count, math_type_ const *centroid,
                         math_type_ const *eigenvalues, math_type_ *frame) noexcept {
    math_type_ const tie_tolerance(std::is_same_v<in_type_, f16_t> ? 0x1p-7 : 0x1p-16);
    bool const upper_tied = eigenvalues[0] - eigenvalues[1] <= tie_tolerance * eigenvalues[0];
    bool const lower_tied = eigenvalues[1] - eigenvalues[2] <= tie_tolerance * eigenvalues[0];
    if (!upper_tied && !lower_tied) return;
    // Rows a and b span the tied plane and c the third axis, in a proper order
    std::size_t const first = upper_tied ? 0 : 1;
    math_type_ basis[9];
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t cell = 0; cell != 3; ++cell) basis[row * 3 + cell] = frame[cell * 3 + (first + row) % 3];
    if (upper_tied && lower_tied) shapes_pick_axis_<in_type_>(points, count, centroid, tie_tolerance, basis);
    else shapes_turn_plane_<in_type_>(points, count, centroid, tie_tolerance, basis);
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t cell = 0; cell != 3; ++cell) frame[cell * 3 + (first + row) % 3] = basis[row * 3 + cell];
}

/** One cloud as the reference packs it: its points centered, turned onto the principal axes and
 *  divided by the scale, each rounded once into the input type as a pack stores it and read in
 *  @p math_type_, beside the constants a pack keeps. */
template <typename math_type_>
struct shapes_reference_cloud_ {
    std::size_t points = 0;
    math_type_ centroid[3] {}, spreads[3] {}, scale {1.0}, max_radius {};

    /** V, whose columns are the principal axes: a packed point is Vᵀ × (x − centroid) / scale. */
    math_type_ principal_axes[9] {};

    /** The packed points, x, y and z of each in turn. */
    std::vector<math_type_> coordinates;

    math_type_ coordinate(std::size_t point, std::size_t axis) const noexcept { return coordinates[point * 3 + axis]; }
    void frame(math_type_ *matrix) const noexcept {
        for (std::size_t i = 0; i != 9; ++i) matrix[i] = principal_axes[i];
    }
};

/** Packs one cloud by the algorithm of @c shapes.h in @p math_type_, divided by its RMS distance
 *  from the centroid when @p kind_ is @c nk_kernel_shapes_pack_normalized_k. */
template <typename in_type_, typename math_type_, nk_kernel_kind_t kind_>
shapes_reference_cloud_<math_type_> shapes_pack_reference(typename in_type_::raw_t const *points, std::size_t count) {
    using raw_t = typename in_type_::raw_t;
    using math_t = math_type_;
    shapes_reference_cloud_<math_t> cloud;
    cloud.points = count;
    for (std::size_t axis = 0; axis != 3; ++axis) cloud.principal_axes[axis * 4] = math_t(1.0);
    if (!count) return cloud;
    auto const widen = [](raw_t value) noexcept { return math_t(to_double(in_type_::from_raw(value))); };
    math_t const count_math = math_t(to_double(count));
    math_t centroid[3] {};
    for (std::size_t i = 0; i != count; ++i)
        for (std::size_t axis = 0; axis != 3; ++axis) centroid[axis] = centroid[axis] + widen(points[i * 3 + axis]);
    for (math_t &value : centroid) value = value / count_math;

    // The 6 unique entries in the order Jacobi takes them: c00, c10, c11, c20, c21, c22
    math_t covariance[6] {};
    for (std::size_t i = 0; i != count; ++i) {
        math_t const x = widen(points[i * 3 + 0]) - centroid[0], y = widen(points[i * 3 + 1]) - centroid[1],
                     z = widen(points[i * 3 + 2]) - centroid[2];
        covariance[0] = covariance[0] + x * x, covariance[1] = covariance[1] + y * x;
        covariance[2] = covariance[2] + y * y, covariance[3] = covariance[3] + z * x;
        covariance[4] = covariance[4] + z * y, covariance[5] = covariance[5] + z * z;
    }
    for (math_t &value : covariance) value = value / count_math;
    // Coincident points keep a scale of 1, however their rounded centroid spreads them
    math_t scale(1.0);
    if constexpr (kind_ == nk_kernel_shapes_pack_normalized_k) {
        bool coincident = true;
        for (std::size_t i = 0; i != count * 3; ++i) coincident &= widen(points[i]) == widen(points[i % 3]);
        math_t const spread = covariance[0] + covariance[2] + covariance[5];
        if (!coincident && math_t(0.0) < spread) scale = square_root_reference(spread);
    }
    math_t quaternion[4], frame[9];
    jacobi_eigenanalysis_(&covariance[0], &covariance[1], &covariance[2], &covariance[3], &covariance[4],
                          &covariance[5], quaternion);
    quaternion_to_mat3x3_(quaternion, frame);
    math_t eigenvalues[3] = {covariance[0], covariance[2], covariance[5]};
    sort_eigenvectors_(eigenvalues, frame);
    shapes_settle_ties_<in_type_>(points, count, centroid, eigenvalues, frame);
    mat3x3_to_quaternion_(frame, quaternion);
    quaternion_to_mat3x3_(quaternion, frame);

    cloud.coordinates.resize(count * 3);
    math_t moments[6] {}, max_norm {};
    for (std::size_t i = 0; i != count; ++i) {
        math_t const centered[3] = {widen(points[i * 3 + 0]) - centroid[0], widen(points[i * 3 + 1]) - centroid[1],
                                    widen(points[i * 3 + 2]) - centroid[2]};
        math_t *packed = &cloud.coordinates[i * 3];
        for (std::size_t axis = 0; axis != 3; ++axis)
            packed[axis] = widen(
                narrow<in_type_>(to_double((frame[0 * 3 + axis] * centered[0] + frame[1 * 3 + axis] * centered[1] +
                                            frame[2 * 3 + axis] * centered[2]) /
                                           scale))
                    .raw_);
        moments[0] = moments[0] + packed[0] * packed[0], moments[1] = moments[1] + packed[1] * packed[0];
        moments[2] = moments[2] + packed[1] * packed[1], moments[3] = moments[3] + packed[2] * packed[0];
        moments[4] = moments[4] + packed[2] * packed[1], moments[5] = moments[5] + packed[2] * packed[2];
        math_t const norm = packed[0] * packed[0] + packed[1] * packed[1] + packed[2] * packed[2];
        if (max_norm < norm) max_norm = norm;
    }
    for (math_t &value : moments) value = value / count_math;
    math_t moments_frame[4];
    jacobi_eigenanalysis_(&moments[0], &moments[1], &moments[2], &moments[3], &moments[4], &moments[5], moments_frame);
    math_t spreads[3] = {moments[0], moments[2], moments[5]};
    if (spreads[0] < spreads[1]) std::swap(spreads[0], spreads[1]);
    if (spreads[0] < spreads[2]) std::swap(spreads[0], spreads[2]);
    if (spreads[1] < spreads[2]) std::swap(spreads[1], spreads[2]);

    for (std::size_t axis = 0; axis != 3; ++axis)
        cloud.centroid[axis] = centroid[axis], cloud.spreads[axis] = spreads[axis];
    for (std::size_t i = 0; i != 9; ++i) cloud.principal_axes[i] = frame[i];
    cloud.scale = scale;
    cloud.max_radius = square_root_reference(max_norm);
    return cloud;
}

/** Packs every cloud of @p clouds as @c shapes_pack_reference does, plainly or normalized by
 *  @p kind_. */
template <typename in_type_, typename math_type_, nk_kernel_kind_t kind_>
std::vector<shapes_reference_cloud_<math_type_>> shapes_pack_references_(
    std::span<std::span<in_type_ const> const> clouds) {
    std::vector<shapes_reference_cloud_<math_type_>> packs;
    packs.reserve(clouds.size());
    for (std::span<in_type_ const> cloud : clouds)
        packs.push_back(shapes_pack_reference<in_type_, math_type_, kind_>(
            reinterpret_cast<typename in_type_::raw_t const *>(cloud.data()), cloud.size() / 3));
    return packs;
}

/** Chamfer or Hausdorff of every pair of clouds in @p distance_type_, by the 8-flip passes of
 *  @c shapes.h over the packs the reference makes of them. */
template <typename in_type_, nk_kernel_kind_t kind_, nk_kernel_kind_t pack_kind_ = nk_kernel_shapes_pack_k,
          typename distance_type_, typename rotation_type_>
void shapes_nearest_reference(std::span<std::span<in_type_ const> const> a_clouds,
                              std::span<std::span<in_type_ const> const> b_clouds, distance_type_ *results,
                              std::size_t results_stride, rotation_type_ *rotations) {
    using math_t = distance_type_;
    std::size_t const rows = a_clouds.size(), columns = b_clouds.size();
    std::vector<shapes_reference_cloud_<math_t>> const a_packs = shapes_pack_references_<in_type_, math_t, pack_kind_>(
        a_clouds);
    std::vector<shapes_reference_cloud_<math_t>> const b_packs = shapes_pack_references_<in_type_, math_t, pack_kind_>(
        b_clouds);
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            shapes_reference_cloud_<math_t> const &a = a_packs[row], &b = b_packs[column];
            distance_type_ &result = reinterpret_cast<distance_type_ *>(reinterpret_cast<char *>(results) +
                                                                        row * results_stride)[column];
            rotation_type_ *rotation = rotations ? rotations + (row * columns + column) * 9 : nullptr;
            if (!a.points || !b.points) {
                result = a.points == b.points ? math_t(0.0) : math_t::positive_infinity();
                if (rotation) shapes_write_identity_(rotation);
                continue;
            }
            auto const squared = [&](std::size_t i, std::size_t j, unsigned flip) noexcept {
                math_t sum {};
                for (std::size_t axis = 0; axis != 3; ++axis) {
                    math_t const ai = a.coordinate(i, axis);
                    math_t const bj = b.coordinate(j, axis);
                    math_t const difference = (flip >> axis) & 1 ? ai + bj : ai - bj;
                    sum = sum + difference * difference;
                }
                return sum;
            };
            math_t best = math_t::positive_infinity();
            unsigned best_flip = 0;
            for (unsigned flip = 0; flip != 8; ++flip) {
                math_t forward_sum {}, forward_max {}, backward_sum {}, backward_max {};
                for (std::size_t i = 0; i != a.points; ++i) {
                    math_t nearest = math_t::positive_infinity();
                    for (std::size_t j = 0; j != b.points; ++j)
                        if (math_t const value = squared(i, j, flip); value < nearest) nearest = value;
                    math_t const root = square_root_reference(nearest);
                    forward_sum = forward_sum + root;
                    if (forward_max < root) forward_max = root;
                }
                for (std::size_t j = 0; j != b.points; ++j) {
                    math_t nearest = math_t::positive_infinity();
                    for (std::size_t i = 0; i != a.points; ++i)
                        if (math_t const value = squared(i, j, flip); value < nearest) nearest = value;
                    math_t const root = square_root_reference(nearest);
                    backward_sum = backward_sum + root;
                    if (backward_max < root) backward_max = root;
                }
                math_t const score = kind_ == nk_kernel_chamfers_packed_k
                                         ? math_t(0.5) * (forward_sum / math_t(to_double(a.points)) +
                                                          backward_sum / math_t(to_double(b.points)))
                                         : (forward_max < backward_max ? backward_max : forward_max);
                if (score < best) best = score, best_flip = flip;
            }
            result = best;
            if (!rotation) continue;
            math_t a_frame[9], b_frame[9], flip_matrix[9];
            a.frame(a_frame), b.frame(b_frame), shapes_flip_(best_flip, flip_matrix);
            shapes_write_rotation_(a_frame, flip_matrix, b_frame, rotation);
        }
}

/** Superposed RMSD of every pair in @p distance_type_: the proper rotation the SVD of the
 *  cross-covariance H = Σ bᵢ × aᵢᵀ yields, then Σ‖aᵢ − R × bᵢ‖² summed directly. Every cloud must
 *  hold as many points as the first, and empty ones score zero. */
template <typename in_type_, typename distance_type_>
status_t shapes_rmsds_reference(std::span<std::span<in_type_ const> const> a_clouds,
                                std::span<std::span<in_type_ const> const> b_clouds, distance_type_ *results,
                                std::size_t results_stride) {
    using math_t = distance_type_;
    std::size_t const rows = a_clouds.size(), columns = b_clouds.size();
    if (!rows || !columns) return status_t::success_k;
    std::size_t const points = a_clouds[0].size() / 3;
    for (std::span<in_type_ const> cloud : a_clouds)
        if (cloud.size() / 3 != points) return status_t::unexpected_dimensions_k;
    for (std::span<in_type_ const> cloud : b_clouds)
        if (cloud.size() / 3 != points) return status_t::unexpected_dimensions_k;
    std::vector<shapes_reference_cloud_<math_t>> const a_packs =
        shapes_pack_references_<in_type_, math_t, nk_kernel_shapes_pack_k>(a_clouds);
    std::vector<shapes_reference_cloud_<math_t>> const b_packs =
        shapes_pack_references_<in_type_, math_t, nk_kernel_shapes_pack_k>(b_clouds);
    // One pass per pair, its row and column read off the pair's index
    for (std::size_t pair = 0; pair != rows * columns; ++pair) {
        std::size_t const row = pair / columns, column = pair % columns;
        shapes_reference_cloud_<math_t> const &a = a_packs[row], &b = b_packs[column];
        distance_type_ &result = reinterpret_cast<distance_type_ *>(reinterpret_cast<char *>(results) +
                                                                    row * results_stride)[column];
        if (!points) {
            result = math_t(0.0);
            continue;
        }
        // Entry 3 × j + k of H sums b_j × a_k over the points
        math_t cross[9] {};
        for (std::size_t i = 0; i != points; ++i)
            for (std::size_t entry = 0; entry != 9; ++entry)
                cross[entry] = cross[entry] + b.coordinate(i, entry / 3) * a.coordinate(i, entry % 3);
        math_t svd_u[9], svd_v[9], rotation[9];
        svd3x3_(cross, svd_u, svd_v);
        rotation_from_svd_(svd_u, svd_v, rotation);
        // A reflection flips the axis of the smallest singular value back into a rotation
        if (det3x3_(rotation) < math_t(0.0)) {
            for (std::size_t axis = 0; axis != 3; ++axis) svd_v[axis * 3 + 2] = math_t(0.0) - svd_v[axis * 3 + 2];
            rotation_from_svd_(svd_u, svd_v, rotation);
        }
        math_t sum {};
        for (std::size_t i = 0; i != points; ++i)
            for (std::size_t axis = 0; axis != 3; ++axis) {
                math_t const residual = a.coordinate(i, axis) - rotation[axis * 3 + 0] * b.coordinate(i, 0) -
                                        rotation[axis * 3 + 1] * b.coordinate(i, 1) -
                                        rotation[axis * 3 + 2] * b.coordinate(i, 2);
                sum = sum + residual * residual;
            }
        result = (sum / math_t(to_double(points))).sqrt();
    }
    return status_t::success_k;
}

/** Orders two packs by points, spreads, max_radius, and then their coordinates. */
template <typename math_type_>
int shapes_compare_(shapes_reference_cloud_<math_type_> const &a,
                    shapes_reference_cloud_<math_type_> const &b) noexcept {
    if (a.points != b.points) return a.points < b.points ? -1 : 1;
    for (std::size_t axis = 0; axis != 3; ++axis)
        if (a.spreads[axis] != b.spreads[axis]) return a.spreads[axis] < b.spreads[axis] ? -1 : 1;
    if (a.max_radius != b.max_radius) return a.max_radius < b.max_radius ? -1 : 1;
    for (std::size_t axis = 0; axis != 3; ++axis)
        for (std::size_t point = 0; point != a.points; ++point)
            if (a.coordinate(point, axis) != b.coordinate(point, axis))
                return a.coordinate(point, axis) < b.coordinate(point, axis) ? -1 : 1;
    return 0;
}

/** The stopping tolerance of the network simplex in every math type, relative to its cost scale. */
inline constexpr double shapes_transport_epsilon_k = 1e-12;

/**
 *  @brief How far a kernel's packing may move each point of @p cloud from where the reference packs
 *      it, in packed units.
 *
 *  The kernel and the reference both round each packed coordinate into @p scalar_type_ from values
 *  their arithmetic leaves a little apart, so the two may land one step apart at the radius, on
 *  each axis. The centroid takes the distance type's roundings of n inputs, or 8 for compensated
 *  @c f64 sums, over the largest input, and the projection a few more over the radius.
 *  The flip kinds also measure in the cloud's own frame, which those roundings of the covariance
 *  turn by their share of the smallest gap between the spreads.
 */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename math_type_>
nk_f64_t shapes_storage_bound(shapes_reference_cloud_<math_type_> const &cloud) noexcept {
    nk_f64_t const roundoff = half_ulp<typename scalar_type_::shapes_distance_t>(1.0);
    nk_f64_t const terms = std::is_same_v<scalar_type_, f64_t> ? 8 : to_double(cloud.points);
    nk_f64_t const radius = to_double(cloud.max_radius), scale = to_double(cloud.scale);
    nk_f64_t largest_input = 0;
    for (std::size_t axis = 0; axis != 3; ++axis)
        largest_input = nk::max_of(largest_input, std::fabs(to_double(cloud.centroid[axis])));
    largest_input += scale * radius;
    nk_f64_t displacement = 2 * std::sqrt(3.0) * half_ulp<scalar_type_>(radius) +
                            (terms + 2) * roundoff * largest_input / scale + 64 * roundoff * radius;
    if constexpr (kind_ == nk_kernel_chamfers_packed_k || kind_ == nk_kernel_hausdorffs_packed_k) {
        nk_f64_t const spreads[3] = {to_double(cloud.spreads[0]), to_double(cloud.spreads[1]),
                                     to_double(cloud.spreads[2])};
        nk_f64_t const gap = nk::min_of(spreads[0] - spreads[1], spreads[1] - spreads[2]);
        displacement += (terms + 16) * roundoff * radius * radius * radius / gap;
    }
    return displacement;
}

/**
 *  @brief Bounds a kernel's distance error by the arithmetic of its math type, plus
 *      @p displacement, how far @ref shapes_storage_bound lets both clouds' packing move their
 *      points, or zero between two kernels on the same packs.
 *
 *  Chamfer sums the n + m nearest-partner roots, each a few roundings off, and Hausdorff keeps one.
 *  The transport kinds add a plan value over up to n + m − 1 arcs costing at most (r_a + r_b)²,
 *  which the simplex may also leave suboptimal by its tolerance, and Gromov-Wasserstein sums
 *  distance products up to four times that scale. Square roots then shrink a value's error by twice
 *  the result, or leave its root. Compensated @c f64 sums keep at most 8 roundings of their total,
 *  whatever their length. Every distance moves by at most the points' displacement, and
 *  Gromov-Wasserstein by twice it, as each moves both ends of a distance within a cloud.
 */
template <typename scalar_type_, nk_kernel_kind_t kind_>
nk_f64_t shapes_distance_bound(std::size_t a_points, std::size_t b_points, nk_f64_t a_radius, nk_f64_t b_radius,
                               nk_f64_t reference, nk_f64_t displacement) noexcept {
    using distance_t = typename scalar_type_::shapes_distance_t;
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0);
    nk_f64_t const terms = std::is_same_v<scalar_type_, f64_t> ? 8 : to_double(a_points + b_points);
    nk_f64_t const scale = (a_radius + b_radius) * (a_radius + b_radius);
    nk_f64_t const tolerance = shapes_transport_epsilon_k * scale;
    if constexpr (kind_ == nk_kernel_chamfers_packed_k) return (terms + 8) * roundoff * reference + displacement;
    else if constexpr (kind_ == nk_kernel_hausdorffs_packed_k) return 8 * roundoff * reference + displacement;
    else if constexpr (kind_ == nk_kernel_wassersteinprocrustes_packed_k) {
        nk_f64_t const squared = (terms + 32) * roundoff * scale + tolerance;
        return nk::min_of(std::sqrt(squared), squared / reference) + displacement;
    }
    else {
        nk_f64_t const loss = 4 * ((terms + 64) * roundoff * scale + tolerance);
        return nk::min_of(0.5 * std::sqrt(loss), loss / (8 * reference)) + 2 * displacement;
    }
}

/** Bounds a rotation entry's error: its rounding into @c f32, plus the math type's quaternion
 *  products for the flip kinds, or its 3 × 3 SVD for the transport kinds, plus @p turn, the angle
 *  @ref shapes_storage_bound turns both frames by, each cloud's displacement over its radius, or
 *  zero between two kernels on the same packs. */
template <typename scalar_type_, nk_kernel_kind_t kind_>
nk_f64_t shapes_rotation_bound(nk_f64_t turn) noexcept {
    nk_f64_t const roundoff = half_ulp<typename scalar_type_::shapes_distance_t>(1.0);
    bool const flips = kind_ == nk_kernel_chamfers_packed_k || kind_ == nk_kernel_hausdorffs_packed_k;
    return std::ldexp(1.0, -24) + (flips ? 32 : 4096) * roundoff + turn;
}

/**
 *  @brief Network simplex over the uniform transport between @c rows sources and @c columns sinks.
 *
 *  Rows are nodes 0 … rows − 1 and supply @c columns units each, columns are the following nodes
 *  and demand @c rows units each. The spanning tree hangs from row 0, every non-root node holding
 *  the flow of the arc to its parent, which always runs from the row to the column. Costs come in
 *  the math type, and the duals widen them to at least @c f64, so they solve the LP of those
 *  rounded costs exactly.
 */
template <typename math_type_>
struct shapes_simplex_ {
    using dual_t = shapes_wide_<math_type_>;
    static constexpr std::uint32_t no_node = 0xFFFFFFFFu;

    std::size_t rows = 0, columns = 0;
    std::uint32_t *parent = nullptr, *flow = nullptr, *depth = nullptr;
    std::uint32_t *first_child = nullptr, *next_sibling = nullptr, *previous_sibling = nullptr;
    dual_t *potential = nullptr;

    void attach_(std::uint32_t node, std::uint32_t above) noexcept {
        parent[node] = above, previous_sibling[node] = no_node, next_sibling[node] = first_child[above];
        if (first_child[above] != no_node) previous_sibling[first_child[above]] = node;
        first_child[above] = node;
    }

    void detach_(std::uint32_t node) noexcept {
        if (previous_sibling[node] != no_node) next_sibling[previous_sibling[node]] = next_sibling[node];
        else first_child[parent[node]] = next_sibling[node];
        if (next_sibling[node] != no_node) previous_sibling[next_sibling[node]] = previous_sibling[node];
    }

    /** Visits every descendant of @p root in preorder. */
    template <typename visitor_type_>
    void descendants_(std::uint32_t root, visitor_type_ &&visit) noexcept {
        std::uint32_t node = first_child[root];
        while (node != no_node) {
            visit(node);
            if (first_child[node] != no_node) {
                node = first_child[node];
                continue;
            }
            while (node != root && next_sibling[node] == no_node) node = parent[node];
            node = node == root ? no_node : next_sibling[node];
        }
    }

    template <typename cost_type_>
    dual_t parent_cost_(std::uint32_t node, cost_type_ const &cost) const noexcept {
        return shapes_output_<dual_t>(node < rows ? cost(node, parent[node] - rows) : cost(parent[node], node - rows));
    }

    template <typename cost_type_>
    void potential_from_parent_(std::uint32_t node, cost_type_ const &cost) noexcept {
        dual_t const arc_cost = parent_cost_(node, cost);
        potential[node] = node < rows ? potential[parent[node]] + arc_cost : potential[parent[node]] - arc_cost;
    }

    /** Starts from the northwest-corner staircase, with potentials in insertion order. */
    template <typename cost_type_>
    void northwest_corner(cost_type_ const &cost) noexcept {
        std::size_t const nodes = rows + columns;
        for (std::size_t node = 0; node != nodes; ++node)
            parent[node] = first_child[node] = next_sibling[node] = previous_sibling[node] = no_node,
            flow[node] = depth[node] = 0;
        potential[0] = dual_t(0.0);
        std::size_t row = 0, column = 0, supply = columns, demand = rows;
        std::uint32_t node = static_cast<std::uint32_t>(rows), above = 0;
        for (;;) {
            attach_(node, above);
            std::size_t const moved = supply < demand ? supply : demand;
            flow[node] = static_cast<std::uint32_t>(moved), supply -= moved, demand -= moved;
            depth[node] = depth[above] + 1;
            potential_from_parent_(node, cost);
            if (row + 1 == rows && column + 1 == columns) break;
            if (supply == 0 && row + 1 < rows)
                ++row, supply = columns, node = static_cast<std::uint32_t>(row),
                       above = static_cast<std::uint32_t>(rows + column);
            else
                ++column, demand = rows, node = static_cast<std::uint32_t>(rows + column),
                          above = static_cast<std::uint32_t>(row);
        }
    }

    /** Keeps the tree as a warm start under new costs, recomputing potentials from the root. */
    template <typename cost_type_>
    void recompute_potentials(cost_type_ const &cost) noexcept {
        potential[0] = dual_t(0.0);
        descendants_(0, [&](std::uint32_t node) noexcept { potential_from_parent_(node, cost); });
    }

    /** Pivots by block pricing from a cursor restarting at arc 0, until no arc's reduced cost
     *  falls below @c -tolerance. The block holding such an arc enters its first arc within
     *  @c tolerance of the block's least, as every capability does, so no rounding orders ties. */
    template <typename cost_type_>
    void solve(cost_type_ const &cost, dual_t tolerance) noexcept {
        std::size_t const arcs = rows * columns;
        std::size_t block = 10;
        while (block * block < arcs) ++block;
        dual_t const threshold = dual_t(0.0) - tolerance;
        auto const reduced = [&](std::size_t arc) noexcept {
            std::size_t const row = arc / columns, column = arc % columns;
            return shapes_output_<dual_t>(cost(row, column)) - potential[row] + potential[rows + column];
        };
        std::size_t cursor = 0;
        for (;;) {
            dual_t entering_cost {};
            std::size_t entering = arcs, arc = cursor, block_start = cursor, in_block = 0;
            bool found = false;
            for (std::size_t scanned = 0; scanned != arcs; ++scanned) {
                if (dual_t const value = reduced(arc); value < entering_cost) entering_cost = value, entering = arc;
                arc = arc + 1 == arcs ? 0 : arc + 1;
                if (++in_block == block || scanned + 1 == arcs) {
                    if (entering_cost < threshold) {
                        found = true;
                        break;
                    }
                    in_block = 0, block_start = arc;
                }
            }
            if (!found) return;
            dual_t const tie_cost = entering_cost + tolerance;
            for (std::size_t tied = block_start; tied != entering; tied = tied + 1 == arcs ? 0 : tied + 1)
                if (dual_t const value = reduced(tied); value < tie_cost) {
                    entering_cost = value, entering = tied;
                    break;
                }
            cursor = arc;
            pivot_(entering / columns, entering % columns, entering_cost);
        }
    }

    void pivot_(std::size_t row, std::size_t column, dual_t reduced) noexcept {
        std::uint32_t const tail = static_cast<std::uint32_t>(row), head = static_cast<std::uint32_t>(rows + column);
        std::uint32_t first = tail, second = head;
        while (first != second) {
            if (depth[first] > depth[second]) first = parent[first];
            else if (depth[second] > depth[first]) second = parent[second];
            else first = parent[first], second = parent[second];
        }
        std::uint32_t const join = first;

        // Cunningham's rule: the last blocking arc in the cycle order join → tail → head → join
        std::uint64_t tail_delta = ~std::uint64_t(0), head_delta = ~std::uint64_t(0);
        std::uint32_t tail_leaving = no_node, head_leaving = no_node;
        for (std::uint32_t node = tail; node != join; node = parent[node])
            if (node < rows && flow[node] < tail_delta) tail_delta = flow[node], tail_leaving = node;
        for (std::uint32_t node = head; node != join; node = parent[node])
            if (node >= rows && flow[node] <= head_delta) head_delta = flow[node], head_leaving = node;
        bool const leaves_head_side = head_delta <= tail_delta;
        std::uint32_t const delta = static_cast<std::uint32_t>(leaves_head_side ? head_delta : tail_delta);
        for (std::uint32_t node = tail; node != join; node = parent[node])
            flow[node] = node < rows ? flow[node] - delta : flow[node] + delta;
        for (std::uint32_t node = head; node != join; node = parent[node])
            flow[node] = node < rows ? flow[node] + delta : flow[node] - delta;

        std::uint32_t const moved_root = leaves_head_side ? head : tail;
        std::uint32_t const leaving = leaves_head_side ? head_leaving : tail_leaving;
        std::uint32_t above = leaves_head_side ? tail : head, carried = delta;
        for (std::uint32_t node = moved_root;;) {
            std::uint32_t const next = parent[node], kept = flow[node];
            detach_(node), attach_(node, above), flow[node] = carried;
            if (node == leaving) break;
            above = node, carried = kept, node = next;
        }
        dual_t const shift = leaves_head_side ? dual_t(0.0) - reduced : reduced;
        depth[moved_root] = depth[parent[moved_root]] + 1, potential[moved_root] = potential[moved_root] + shift;
        descendants_(moved_root, [&](std::uint32_t node) noexcept {
            depth[node] = depth[parent[node]] + 1, potential[node] = potential[node] + shift;
        });
    }

    /** Visits every tree arc as (row, column, flow). */
    template <typename visitor_type_>
    void arcs(visitor_type_ &&visit) const noexcept {
        for (std::size_t node = 0; node != rows + columns; ++node) {
            if (parent[node] == no_node) continue;
            if (node < rows) visit(node, parent[node] - rows, flow[node]);
            else visit(static_cast<std::size_t>(parent[node]), node - rows, flow[node]);
        }
    }

    template <typename cost_type_>
    math_type_ plan_value(cost_type_ const &cost) const noexcept {
        math_type_ sum {};
        arcs([&](std::size_t row, std::size_t column, std::uint32_t moved) noexcept {
            sum = sum + math_type_(to_double(moved)) * cost(row, column);
        });
        return sum / math_type_(to_double(rows * columns));
    }
};

/** The transport references' workspace arrays, each on an @c nk_default_alignment_k boundary. */
template <typename math_type_>
struct shapes_transport_workspace_ {
    math_type_ *row_points = nullptr, *column_points = nullptr, *rotated_columns = nullptr;
    shapes_simplex_<math_type_> simplex;

    /** The best proper and improper plans, as (row, column, flow) triples. */
    std::uint32_t *proper_plan = nullptr, *improper_plan = nullptr;
    std::size_t proper_arcs = 0, improper_arcs = 0;

    /** The Gromov-Wasserstein plan, its gradient and a vertex's gradient, then one block of the
     *  distances within the row cloud followed by those within the column cloud. */
    shapes_wide_<math_type_> *plan = nullptr, *gradient = nullptr, *vertex_gradient = nullptr, *distances = nullptr;
};

template <typename value_type_>
value_type_ *shapes_carve_(char *base, std::size_t &offset, std::size_t count) noexcept {
    value_type_ *carved = base ? reinterpret_cast<value_type_ *>(base + offset) : nullptr;
    offset += nk_size_round_up_to_multiple_(count * sizeof(value_type_), nk_default_alignment_k);
    return carved;
}

/** Carves the workspace for clouds of up to @p row_points and @p column_points from @p base, or
 *  sizes it without one: the transport arrays for the larger side, then the Gromov-Wasserstein
 *  matrices of @p row_points × @p column_points cells and @p row_points² + @p column_points²
 *  distances, which fit either canonical order of a pair. */
template <typename math_type_, nk_kernel_kind_t kind_>
std::size_t shapes_transport_layout_(shapes_transport_workspace_<math_type_> &workspace, char *base,
                                     std::size_t row_points, std::size_t column_points) noexcept {
    std::size_t offset = 0;
    std::size_t const points = row_points < column_points ? column_points : row_points;
    std::size_t const nodes = 2 * points;
    std::size_t const plan_stride = nk_size_round_up_to_multiple_((points ? 2 * points - 1 : 0) * sizeof(std::uint32_t),
                                                                  nk_default_alignment_k) /
                                    sizeof(std::uint32_t);
    std::size_t const plan_values = 3 * plan_stride;
    workspace.row_points = shapes_carve_<math_type_>(base, offset, 3 * points);
    workspace.column_points = shapes_carve_<math_type_>(base, offset, 3 * points);
    workspace.rotated_columns = shapes_carve_<math_type_>(base, offset, 3 * points);
    workspace.simplex.parent = shapes_carve_<std::uint32_t>(base, offset, nodes);
    workspace.simplex.flow = shapes_carve_<std::uint32_t>(base, offset, nodes);
    workspace.simplex.depth = shapes_carve_<std::uint32_t>(base, offset, nodes);
    workspace.simplex.first_child = shapes_carve_<std::uint32_t>(base, offset, nodes);
    workspace.simplex.next_sibling = shapes_carve_<std::uint32_t>(base, offset, nodes);
    workspace.simplex.previous_sibling = shapes_carve_<std::uint32_t>(base, offset, nodes);
    workspace.simplex.potential = shapes_carve_<typename shapes_simplex_<math_type_>::dual_t>(base, offset, nodes);
    workspace.proper_plan = shapes_carve_<std::uint32_t>(base, offset, plan_values);
    workspace.improper_plan = shapes_carve_<std::uint32_t>(base, offset, plan_values);
    if constexpr (kind_ == nk_kernel_gromovwassersteins_packed_k) {
        using wide_t = shapes_wide_<math_type_>;
        std::size_t const cells = row_points * column_points;
        workspace.plan = shapes_carve_<wide_t>(base, offset, cells);
        workspace.gradient = shapes_carve_<wide_t>(base, offset, cells);
        workspace.vertex_gradient = shapes_carve_<wide_t>(base, offset, cells);
        workspace.distances = shapes_carve_<wide_t>(base, offset,
                                                    row_points * row_points + column_points * column_points);
    }
    return offset;
}

/**
 *  @brief Wasserstein-Procrustes of one ordered pair, the best squared plan value of 8 flip starts.
 *  @param[out] rotation The best start's rotation in the packed frames, taking @p b onto @p a.
 *
 *  It leaves the widened clouds and the best proper and improper plans in @p workspace, where
 *  Gromov-Wasserstein reads them.
 */
template <typename math_type_>
math_type_ shapes_procrustes_pair_(shapes_reference_cloud_<math_type_> const &a,
                                   shapes_reference_cloud_<math_type_> const &b,
                                   shapes_transport_workspace_<math_type_> &workspace, math_type_ *rotation) noexcept {
    using math_t = math_type_;
    std::size_t const n = a.points, m = b.points;
    math_t *const a_points = workspace.row_points, *const b_points = workspace.column_points;
    math_t *const rotated = workspace.rotated_columns;
    for (std::size_t i = 0; i != n; ++i)
        for (std::size_t axis = 0; axis != 3; ++axis) a_points[i * 3 + axis] = a.coordinate(i, axis);
    for (std::size_t j = 0; j != m; ++j)
        for (std::size_t axis = 0; axis != 3; ++axis) b_points[j * 3 + axis] = b.coordinate(j, axis);

    shapes_simplex_<math_t> &simplex = workspace.simplex;
    simplex.rows = n, simplex.columns = m;
    math_t const mass = math_t(to_double(n * m));
    math_t const radii = a.max_radius + b.max_radius;
    using dual_t = typename shapes_simplex_<math_t>::dual_t;
    dual_t const tolerance = shapes_output_<dual_t>(radii * radii) * dual_t(shapes_transport_epsilon_k);
    auto const cost = [&](std::size_t i, std::size_t j) noexcept {
        math_t squared {};
        for (std::size_t axis = 0; axis != 3; ++axis) {
            math_t const difference = a_points[i * 3 + axis] - rotated[j * 3 + axis];
            squared = squared + difference * difference;
        }
        return squared;
    };
    auto const save_plan = [&](std::uint32_t *plan, std::size_t &arcs) noexcept {
        arcs = 0;
        simplex.arcs([&](std::size_t row, std::size_t column, std::uint32_t moved) noexcept {
            plan[arcs * 3 + 0] = static_cast<std::uint32_t>(row),
                            plan[arcs * 3 + 1] = static_cast<std::uint32_t>(column);
            plan[arcs * 3 + 2] = moved, ++arcs;
        });
    };

    math_t best = math_t::positive_infinity();
    math_t best_proper = math_t::positive_infinity(), best_improper = math_t::positive_infinity();
    for (unsigned flip = 0; flip != 8; ++flip) {
        bool const proper = (((flip >> 0) ^ (flip >> 1) ^ (flip >> 2)) & 1) == 0;
        math_t current[9], start_rotation[9];
        shapes_flip_(flip, current);
        math_t start_best = math_t::positive_infinity();
        for (std::size_t round = 0;; ++round) {
            for (std::size_t j = 0; j != m; ++j)
                for (std::size_t axis = 0; axis != 3; ++axis)
                    rotated[j * 3 + axis] = current[axis * 3 + 0] * b_points[j * 3 + 0] +
                                            current[axis * 3 + 1] * b_points[j * 3 + 1] +
                                            current[axis * 3 + 2] * b_points[j * 3 + 2];
            if (round == 0) simplex.northwest_corner(cost);
            else simplex.recompute_potentials(cost);
            simplex.solve(cost, tolerance);
            math_t const value = simplex.plan_value(cost);
            if (!(value < start_best)) break;
            start_best = value;
            for (int i = 0; i != 9; ++i) start_rotation[i] = current[i];
            if (proper && value < best_proper)
                best_proper = value, save_plan(workspace.proper_plan, workspace.proper_arcs);
            if (!proper && value < best_improper)
                best_improper = value, save_plan(workspace.improper_plan, workspace.improper_arcs);

            math_t cross[9] {};
            simplex.arcs([&](std::size_t i, std::size_t j, std::uint32_t moved) noexcept {
                math_t const weight = math_t(to_double(moved)) / mass;
                for (std::size_t b_axis = 0; b_axis != 3; ++b_axis)
                    for (std::size_t a_axis = 0; a_axis != 3; ++a_axis)
                        cross[b_axis * 3 + a_axis] = cross[b_axis * 3 + a_axis] +
                                                     weight * b_points[j * 3 + b_axis] * a_points[i * 3 + a_axis];
            });
            math_t svd_u[9], svd_v[9];
            svd3x3_(cross, svd_u, svd_v);
            rotation_from_svd_(svd_u, svd_v, current);
            // Each start stays proper or improper with its flip
            if (det3x3_(current) * math_t(proper ? 1.0 : -1.0) < math_t(0.0)) {
                svd_v[2] = math_t(0.0) - svd_v[2], svd_v[5] = math_t(0.0) - svd_v[5], svd_v[8] = math_t(0.0) - svd_v[8];
                rotation_from_svd_(svd_u, svd_v, current);
            }
        }
        if (start_best < best) {
            best = start_best;
            for (int i = 0; i != 9; ++i) rotation[i] = start_rotation[i];
        }
    }
    return best;
}

/**
 *  @brief Gromov-Wasserstein of one ordered pair: Frank-Wolfe from the best proper and improper
 *      Wasserstein-Procrustes plans and the product plan, returning the least square loss.
 *  @param[out] rotation The orthogonal Procrustes fit of the best plan in the packed frames,
 *      proper or improper, taking @p b onto @p a.
 *
 *  Only the coordinates come in the math type: the distances, plans, gradients, line search and
 *  loss all stay at least @c f64, as a non-convex search would otherwise follow narrower rounding
 *  into other optima.
 */
template <typename math_type_>
shapes_wide_<math_type_> shapes_gromov_pair_(shapes_reference_cloud_<math_type_> const &a,
                                             shapes_reference_cloud_<math_type_> const &b,
                                             shapes_transport_workspace_<math_type_> &workspace,
                                             math_type_ *rotation) noexcept {
    using math_t = math_type_;
    using wide_t = shapes_wide_<math_t>;
    std::size_t const n = a.points, m = b.points;
    math_t procrustes_rotation[9];
    shapes_procrustes_pair_(a, b, workspace, procrustes_rotation);

    math_t const *const a_points = workspace.row_points, *const b_points = workspace.column_points;
    wide_t *const row_distances = workspace.distances, *const column_distances = workspace.distances + n * n;
    wide_t *const plan = workspace.plan, *const gradient = workspace.gradient;
    wide_t *const vertex_gradient = workspace.vertex_gradient;
    auto const distances = [](math_t const *points, std::size_t count, wide_t *matrix) noexcept {
        wide_t sum {};
        for (std::size_t i = 0; i != count; ++i)
            for (std::size_t k = 0; k != count; ++k) {
                wide_t squared {};
                for (std::size_t axis = 0; axis != 3; ++axis) {
                    wide_t const difference = shapes_output_<wide_t>(points[i * 3 + axis]) -
                                              shapes_output_<wide_t>(points[k * 3 + axis]);
                    squared = squared + difference * difference;
                }
                matrix[i * count + k] = square_root_reference(squared);
                sum = sum + matrix[i * count + k] * matrix[i * count + k];
            }
        return sum / wide_t(to_double(count * count));
    };
    wide_t const row_mean_square = distances(a_points, n, row_distances);
    wide_t const column_mean_square = distances(b_points, m, column_distances);

    shapes_simplex_<math_t> &simplex = workspace.simplex;
    wide_t const mass = wide_t(to_double(n * m));
    auto const cost = [&](std::size_t i, std::size_t j) noexcept { return wide_t(0.0) - gradient[i * m + j]; };
    // Dx × V × Dy for a plan V of weighted arcs, written into `target`
    auto const sparse_product = [&](auto &&for_each_arc, wide_t *target) noexcept {
        for (std::size_t cell = 0; cell != n * m; ++cell) target[cell] = wide_t(0.0);
        for_each_arc([&](std::size_t k, std::size_t l, wide_t weight) noexcept {
            for (std::size_t i = 0; i != n; ++i) {
                wide_t const left = weight * row_distances[i * n + k];
                for (std::size_t j = 0; j != m; ++j)
                    target[i * m + j] = target[i * m + j] + left * column_distances[l * m + j];
            }
        });
    };
    auto const inner = [&](wide_t const *left, wide_t const *right) noexcept {
        wide_t sum {};
        for (std::size_t cell = 0; cell != n * m; ++cell) sum = sum + left[cell] * right[cell];
        return sum;
    };

    wide_t best_loss = wide_t::positive_infinity(), best_cross[9] {};
    for (unsigned start = 0; start != 3; ++start) {
        std::uint32_t const *saved = start == 0 ? workspace.proper_plan : workspace.improper_plan;
        std::size_t const saved_arcs = start == 0 ? workspace.proper_arcs : workspace.improper_arcs;
        auto const saved_arc_weights = [&](auto &&visit) noexcept {
            for (std::size_t arc = 0; arc != saved_arcs; ++arc)
                if (saved[arc * 3 + 2])
                    visit(saved[arc * 3 + 0], saved[arc * 3 + 1], wide_t(to_double(saved[arc * 3 + 2])) / mass);
        };
        if (start < 2) {
            for (std::size_t cell = 0; cell != n * m; ++cell) plan[cell] = wide_t(0.0);
            saved_arc_weights([&](std::size_t k, std::size_t l, wide_t weight) noexcept { plan[k * m + l] = weight; });
            sparse_product(saved_arc_weights, gradient);
        }
        else {
            wide_t const uniform = wide_t(1.0) / mass;
            for (std::size_t cell = 0; cell != n * m; ++cell) plan[cell] = uniform;
            for (std::size_t i = 0; i != n; ++i) {
                wide_t row_sum {};
                for (std::size_t k = 0; k != n; ++k) row_sum = row_sum + row_distances[i * n + k];
                for (std::size_t j = 0; j != m; ++j) {
                    wide_t column_sum {};
                    for (std::size_t l = 0; l != m; ++l) column_sum = column_sum + column_distances[l * m + j];
                    gradient[i * m + j] = row_sum * column_sum / mass;
                }
            }
        }
        wide_t gradient_plan = inner(gradient, plan);
        for (std::size_t step = 0; step != 32; ++step) {
            wide_t largest {};
            for (std::size_t cell = 0; cell != n * m; ++cell)
                if (largest < gradient[cell].abs()) largest = gradient[cell].abs();
            if (step == 0) simplex.northwest_corner(cost);
            else simplex.recompute_potentials(cost);
            simplex.solve(cost, largest * wide_t(shapes_transport_epsilon_k));
            auto const vertex_arc_weights = [&](auto &&visit) noexcept {
                simplex.arcs([&](std::size_t k, std::size_t l, std::uint32_t moved) noexcept {
                    if (moved) visit(k, l, wide_t(to_double(moved)) / mass);
                });
            };
            sparse_product(vertex_arc_weights, vertex_gradient);
            wide_t gradient_vertex {}, vertex_vertex {};
            vertex_arc_weights([&](std::size_t k, std::size_t l, wide_t weight) noexcept {
                gradient_vertex = gradient_vertex + gradient[k * m + l] * weight;
                vertex_vertex = vertex_vertex + vertex_gradient[k * m + l] * weight;
            });
            wide_t const linear = gradient_vertex - gradient_plan;
            wide_t const quadratic = vertex_vertex - wide_t(2.0) * gradient_vertex + gradient_plan;
            wide_t const curvature = wide_t(-2.0) * quadratic, slope = wide_t(-4.0) * linear;
            wide_t gamma = curvature + slope < wide_t(0.0) ? wide_t(1.0) : wide_t(0.0);
            if (wide_t(0.0) < curvature) {
                gamma = (wide_t(0.0) - slope) / (wide_t(2.0) * curvature);
                if (gamma < wide_t(0.0)) gamma = wide_t(0.0);
                if (wide_t(1.0) < gamma) gamma = wide_t(1.0);
            }
            if (gamma == wide_t(0.0)) break;
            wide_t const keep = wide_t(1.0) - gamma;
            for (std::size_t cell = 0; cell != n * m; ++cell)
                plan[cell] = keep * plan[cell], gradient[cell] = keep * gradient[cell] + gamma * vertex_gradient[cell];
            vertex_arc_weights([&](std::size_t k, std::size_t l, wide_t weight) noexcept {
                plan[k * m + l] = plan[k * m + l] + gamma * weight;
            });
            gradient_plan = inner(gradient, plan);
        }
        wide_t const loss = row_mean_square + column_mean_square - wide_t(2.0) * gradient_plan;
        if (!(loss < best_loss)) continue;
        best_loss = loss;
        for (wide_t &value : best_cross) value = wide_t(0.0);
        for (std::size_t i = 0; i != n; ++i)
            for (std::size_t j = 0; j != m; ++j)
                for (std::size_t b_axis = 0; b_axis != 3; ++b_axis)
                    for (std::size_t a_axis = 0; a_axis != 3; ++a_axis)
                        best_cross[b_axis * 3 + a_axis] = best_cross[b_axis * 3 + a_axis] +
                                                          plan[i * m + j] *
                                                              shapes_output_<wide_t>(b_points[j * 3 + b_axis]) *
                                                              shapes_output_<wide_t>(a_points[i * 3 + a_axis]);
    }
    // The O(3) Procrustes fit: McAdams keeps U and V proper, folding the sign of det(H′) into σ₃
    wide_t svd_u[9], svd_v[9], fit[9];
    svd3x3_(best_cross, svd_u, svd_v);
    if (det3x3_(best_cross) < wide_t(0.0))
        svd_v[2] = wide_t(0.0) - svd_v[2], svd_v[5] = wide_t(0.0) - svd_v[5], svd_v[8] = wide_t(0.0) - svd_v[8];
    rotation_from_svd_(svd_u, svd_v, fit);
    for (int i = 0; i != 9; ++i) rotation[i] = shapes_output_<math_t>(fit[i]);
    return best_loss;
}

/** The transport kinds between every pair, in canonical order, by the templates above. */
template <typename in_type_, nk_kernel_kind_t kind_, nk_kernel_kind_t pack_kind_ = nk_kernel_shapes_pack_k,
          typename distance_type_, typename rotation_type_>
status_t shapes_transport_reference(std::span<std::span<in_type_ const> const> a_clouds,
                                    std::span<std::span<in_type_ const> const> b_clouds, distance_type_ *results,
                                    std::size_t results_stride, rotation_type_ *rotations) {
    using math_t = distance_type_;
    std::size_t const rows = a_clouds.size(), columns = b_clouds.size();
    if (!rows || !columns) return status_t::success_k;
    std::vector<shapes_reference_cloud_<math_t>> const a_packs = shapes_pack_references_<in_type_, math_t, pack_kind_>(
        a_clouds);
    std::vector<shapes_reference_cloud_<math_t>> const b_packs = shapes_pack_references_<in_type_, math_t, pack_kind_>(
        b_clouds);
    std::size_t row_points = 0, column_points = 0;
    for (shapes_reference_cloud_<math_t> const &pack : a_packs) row_points = nk::max_of(row_points, pack.points);
    for (shapes_reference_cloud_<math_t> const &pack : b_packs) column_points = nk::max_of(column_points, pack.points);
    shapes_transport_workspace_<math_t> arrays;
    std::size_t const workspace_bytes = shapes_transport_layout_<math_t, kind_>(arrays, nullptr, row_points,
                                                                                column_points);
    auto [workspace, workspace_status] = nk::vector<std::byte>::zeros(workspace_bytes + 1);
    if (failed(workspace_status)) return workspace_status;
    shapes_transport_layout_<math_t, kind_>(arrays, reinterpret_cast<char *>(workspace.raw_values_data()), row_points,
                                            column_points);

    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            shapes_reference_cloud_<math_t> const *a = &a_packs[row], *b = &b_packs[column];
            distance_type_ &result = reinterpret_cast<distance_type_ *>(reinterpret_cast<char *>(results) +
                                                                        row * results_stride)[column];
            rotation_type_ *rotation = rotations ? rotations + (row * columns + column) * 9 : nullptr;
            if (!a->points || !b->points) {
                result = a->points == b->points ? math_t(0.0) : math_t::positive_infinity();
                if (rotation) shapes_write_identity_(rotation);
                continue;
            }
            bool const swapped = shapes_compare_(*a, *b) > 0;
            if (swapped) std::swap(a, b);
            math_t core[9];
            if constexpr (kind_ == nk_kernel_wassersteinprocrustes_packed_k)
                result = square_root_reference(shapes_procrustes_pair_(*a, *b, arrays, core));
            else {
                // ½ · √loss in the wide type, rounded once into the distance type
                shapes_wide_<math_t> const loss = shapes_gromov_pair_(*a, *b, arrays, core);
                result = shapes_output_<distance_type_>(
                    square_root_reference(loss < shapes_wide_<math_t>(0.0) ? shapes_wide_<math_t>(0.0) : loss) *
                    shapes_wide_<math_t>(0.5));
            }
            if (!rotation) continue;
            math_t a_frame[9], b_frame[9];
            a->frame(a_frame), b->frame(b_frame);
            shapes_write_rotation_(a_frame, core, b_frame, rotation);
            // A pair solved with its sides swapped took the row cloud onto the column cloud
            if (swapped) {
                std::swap(rotation[1], rotation[3]);
                std::swap(rotation[2], rotation[6]);
                std::swap(rotation[5], rotation[7]);
            }
        }
    return status_t::success_k;
}

/** The largest packed point norm of @p packed, read through the public verb, or NaN when it
 *  fails. */
template <typename scalar_type_>
nk_f64_t shapes_packed_radius(void const *packed) noexcept {
    typename scalar_type_::shapes_distance_t max_radius {};
    nk::status_t const status = nk::shapes_packed_shape<scalar_type_>(packed, nullptr, nullptr, nullptr, nullptr,
                                                                      nullptr, &max_radius, nk::default_capabilities());
    return nk::succeeded(status) ? to_double(max_radius) : std::numeric_limits<nk_f64_t>::quiet_NaN();
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_SHAPES_HPP
