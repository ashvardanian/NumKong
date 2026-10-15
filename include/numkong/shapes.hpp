/**
 *  @file include/numkong/shapes.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief C++ bindings for many-to-many shape distances.
 *
 *  Every wrapper dispatches to the C kernels over a nonzero capability mask, and otherwise runs a
 *  C++ template of the same algorithm in its distance type, which tests instantiate in @c f118_t
 *  and @c f64_t as the reference the kernels are checked against.
 *
 *  @code{.cpp}
 *  std::span<nk::f32_t const> clouds[] = {first_cloud, second_cloud};
 *  auto [packs, status] = nk::shapes_packed<nk::f32_t>::make(clouds);
 *  nk::f64_t distances[4];
 *  status = nk::chamfers_packed<nk::f32_t>(packs.data(), 2, packs.data(), 2, distances, 2 * sizeof(nk::f64_t),
 *                                          nullptr);
 *  @endcode
 */
#ifndef NUMKONG_SHAPES_HPP
#define NUMKONG_SHAPES_HPP

#include <cstdint> // `std::uint32_t`, `std::uintptr_t`
#include <cstring> // `std::memcpy`, `std::memset`, `std::memcmp`

#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span`
#include <type_traits> // `std::conditional_t`
#include <utility>     // `std::exchange`, `std::swap`

#include "numkong/shapes.h"
#include "numkong/shapes/serial.h" // `nk_shapes_packed_header_f32_t`, `nk_shapes_packed_axes_f16_serial_`
#include "numkong/memory.hpp"      // `aligned_allocator`
#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/** The point types of shape packs: @c f64_t and @c f32_t compute in @c f64_t, and @c f16_t in
 *  @c f32_t. */
template <typename in_type_>
concept shapes_dtype = std::is_same_v<in_type_, f64_t> || std::is_same_v<in_type_, f32_t> ||
                       std::is_same_v<in_type_, f16_t>;

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
    scalar_type_ negated_x = scalar_type_(0.0) - *x;
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
    scalar_type_ const gamma = single_k ? scalar_type_(5.828427124f) : scalar_type_(5.828427124746190);
    scalar_type_ const cstar = single_k ? scalar_type_(0.923879532f) : scalar_type_(0.9238795325112867);
    scalar_type_ const sstar = single_k ? scalar_type_(0.3826834323f) : scalar_type_(0.3826834323650898);

    *cos_half = scalar_type_(2.0) * (a11 - a22);
    *sin_half = a12;
    bool use_givens = gamma * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);
    scalar_type_ w = ((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half)).rsqrt();
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

/** The unit quaternion (x, y, z, w) of a row-major rotation, by Shepperd's method. */
template <typename scalar_type_>
void mat3x3_to_quaternion_(scalar_type_ const *m, scalar_type_ *quat) {
    scalar_type_ const one(1.0), two(2.0), quarter(0.25);
    scalar_type_ const trace = m[0] + m[4] + m[8];
    if (trace > scalar_type_(0.0)) {
        scalar_type_ const s = two * (trace + one).sqrt();
        quat[3] = s * quarter, quat[0] = (m[7] - m[5]) / s, quat[1] = (m[2] - m[6]) / s, quat[2] = (m[3] - m[1]) / s;
    }
    else if (m[0] > m[4] && m[0] > m[8]) {
        scalar_type_ const s = two * (one + m[0] - m[4] - m[8]).sqrt();
        quat[3] = (m[7] - m[5]) / s, quat[0] = s * quarter, quat[1] = (m[1] + m[3]) / s, quat[2] = (m[2] + m[6]) / s;
    }
    else if (m[4] > m[8]) {
        scalar_type_ const s = two * (one + m[4] - m[0] - m[8]).sqrt();
        quat[3] = (m[2] - m[6]) / s, quat[0] = (m[1] + m[3]) / s, quat[1] = s * quarter, quat[2] = (m[5] + m[7]) / s;
    }
    else {
        scalar_type_ const s = two * (one + m[8] - m[0] - m[4]).sqrt();
        quat[3] = (m[3] - m[1]) / s, quat[0] = (m[2] + m[6]) / s, quat[1] = (m[5] + m[7]) / s, quat[2] = s * quarter;
    }
    scalar_type_ const norm = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).rsqrt();
    for (int i = 0; i != 4; ++i) quat[i] = quat[i] * norm;
}

/** Diagonalizes the symmetric 3 × 3 matrix in place, leaving its eigenvalues on the diagonal and
 *  the quaternion of the eigenvector columns in @p quat. */
template <typename scalar_type_>
void jacobi_eigenanalysis_(scalar_type_ *s11, scalar_type_ *s21, scalar_type_ *s22, scalar_type_ *s31,
                           scalar_type_ *s32, scalar_type_ *s33, scalar_type_ *quat) {
    quat[0] = scalar_type_(0.0);
    quat[1] = scalar_type_(0.0);
    quat[2] = scalar_type_(0.0);
    quat[3] = scalar_type_(1.0);
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
    scalar_type_ const epsilon = shapes_single_precision_<scalar_type_> ? scalar_type_(1e-6f) : scalar_type_(1e-12);

    scalar_type_ a1_sq_plus_a2_sq = a1 * a1 + a2 * a2;
    scalar_type_ rho = a1_sq_plus_a2_sq * a1_sq_plus_a2_sq.rsqrt();
    rho = a1_sq_plus_a2_sq > epsilon ? rho : scalar_type_(0.0);
    *sin_half = rho > epsilon ? a2 : scalar_type_(0.0);
    scalar_type_ abs_a1 = a1 < scalar_type_(0.0) ? (scalar_type_(0.0) - a1) : a1;
    scalar_type_ max_rho = rho > epsilon ? rho : epsilon;
    *cos_half = abs_a1 + max_rho;
    bool should_swap = a1 < scalar_type_(0.0);
    conditional_swap_(should_swap, sin_half, cos_half);
    scalar_type_ w = ((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half)).rsqrt();
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

/** The header of packs of @p in_type_ points. */
template <typename in_type_>
using shapes_header_ =
    std::conditional_t<std::is_same_v<in_type_, f16_t>, nk_shapes_packed_header_f32_t, nk_shapes_packed_header_f64_t>;

/** One pack, read through its header and its three axes. */
template <typename in_type_>
struct shapes_cloud_ {
    using raw_t = typename in_type_::raw_t;

    shapes_header_<in_type_> const *header = nullptr;
    raw_t const *axes = nullptr;
    std::size_t axis_stride = 0;
    std::size_t points = 0;

    explicit shapes_cloud_(void const *packed) noexcept
        : header(static_cast<shapes_header_<in_type_> const *>(packed)),
          points(static_cast<std::size_t>(header->points)) {
        nk_size_t stride = 0;
        if constexpr (std::is_same_v<in_type_, f64_t>) axes = nk_shapes_packed_axes_f64_serial_(packed, &stride);
        else if constexpr (std::is_same_v<in_type_, f32_t>) axes = nk_shapes_packed_axes_f32_serial_(packed, &stride);
        else axes = nk_shapes_packed_axes_f16_serial_(packed, &stride);
        axis_stride = stride;
    }

    template <typename math_type_>
    math_type_ coordinate(std::size_t point, std::size_t axis) const noexcept {
        return math_type_(static_cast<double>(in_type_::from_raw(axes[axis * axis_stride + point])));
    }

    /** V, the transpose of the header's frame 𝑄, whose columns are the principal axes. */
    template <typename math_type_>
    void frame(math_type_ *matrix) const noexcept {
        math_type_ quaternion[4], rotation[9];
        for (int i = 0; i != 4; ++i) quaternion[i] = math_type_(static_cast<double>(header->frame[i]));
        quaternion_to_mat3x3_(quaternion, rotation);
        for (int row = 0; row != 3; ++row)
            for (int column = 0; column != 3; ++column) matrix[row * 3 + column] = rotation[column * 3 + row];
    }

    template <typename math_type_>
    math_type_ max_radius() const noexcept {
        return math_type_(static_cast<double>(header->max_radius));
    }
};

/** Bytes a pack of @p points takes, by the layout of @c shapes/serial.h. */
template <typename in_type_>
constexpr std::size_t shapes_pack_bytes_(std::size_t points) noexcept {
    return nk_size_round_up_to_multiple_(sizeof(shapes_header_<in_type_>), nk_shapes_alignment_k_) +
           3 * nk_size_round_up_to_multiple_(points * sizeof(typename in_type_::raw_t), nk_shapes_alignment_k_);
}

/** The type the transport duals and the Gromov-Wasserstein line search run in: at least @c f64. */
template <typename math_type_>
using shapes_wide_ = std::conditional_t<(sizeof(math_type_) < sizeof(f64_t)), f64_t, math_type_>;

/** Converts @p value of the math type into an output, skipping @c double when both types match. */
template <typename output_type_, typename math_type_>
output_type_ shapes_output_(math_type_ value) noexcept {
    if constexpr (std::is_same_v<output_type_, math_type_>) return value;
    else return output_type_(static_cast<double>(value));
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

/** Packs one cloud by the algorithm of @c shapes.h in @p math_type_, divided by its RMS distance
 *  from the centroid when @p kind_ is @c nk_kernel_shapes_pack_normalized_k. */
template <typename in_type_, typename math_type_, nk_kernel_kind_t kind_>
void shapes_pack_reference_(typename in_type_::raw_t const *points, std::size_t count, void *packed) noexcept {
    using raw_t = typename in_type_::raw_t;
    using header_t = shapes_header_<in_type_>;
    using math_t = math_type_;
    using constant_t = std::remove_cvref_t<decltype(header_t {}.max_radius)>;

    std::memset(packed, 0, shapes_pack_bytes_<in_type_>(count));
    header_t header {};
    header.points = static_cast<decltype(header.points)>(count);
    header.frame[3] = 1, header.scale = 1;
    if (count) {
        auto const widen = [](raw_t value) noexcept { return math_t(static_cast<double>(in_type_::from_raw(value))); };
        math_t const count_math = math_t(static_cast<double>(count));
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
            if (!coincident && math_t(0.0) < spread) scale = spread.sqrt();
        }
        math_t quaternion[4], frame[9];
        jacobi_eigenanalysis_(&covariance[0], &covariance[1], &covariance[2], &covariance[3], &covariance[4],
                              &covariance[5], quaternion);
        quaternion_to_mat3x3_(quaternion, frame);
        math_t eigenvalues[3] = {covariance[0], covariance[2], covariance[5]};
        sort_eigenvectors_(eigenvalues, frame);
        mat3x3_to_quaternion_(frame, quaternion);
        quaternion_to_mat3x3_(quaternion, frame);

        std::size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(header_t), nk_shapes_alignment_k_);
        std::size_t const axis_stride = nk_size_round_up_to_multiple_(count * sizeof(raw_t), nk_shapes_alignment_k_) /
                                        sizeof(raw_t);
        raw_t *axes = reinterpret_cast<raw_t *>(static_cast<char *>(packed) + header_bytes);
        math_t moments[6] {}, max_norm {};
        for (std::size_t i = 0; i != count; ++i) {
            math_t const centered[3] = {widen(points[i * 3 + 0]) - centroid[0], widen(points[i * 3 + 1]) - centroid[1],
                                        widen(points[i * 3 + 2]) - centroid[2]};
            math_t stored[3];
            for (std::size_t axis = 0; axis != 3; ++axis) {
                math_t const projected = (frame[0 * 3 + axis] * centered[0] + frame[1 * 3 + axis] * centered[1] +
                                          frame[2 * 3 + axis] * centered[2]) /
                                         scale;
                axes[axis * axis_stride + i] = in_type_(static_cast<double>(projected)).raw_;
                stored[axis] = widen(axes[axis * axis_stride + i]);
            }
            moments[0] = moments[0] + stored[0] * stored[0], moments[1] = moments[1] + stored[1] * stored[0];
            moments[2] = moments[2] + stored[1] * stored[1], moments[3] = moments[3] + stored[2] * stored[0];
            moments[4] = moments[4] + stored[2] * stored[1], moments[5] = moments[5] + stored[2] * stored[2];
            math_t const norm = stored[0] * stored[0] + stored[1] * stored[1] + stored[2] * stored[2];
            if (max_norm < norm) max_norm = norm;
        }
        for (math_t &value : moments) value = value / count_math;
        math_t moments_frame[4];
        jacobi_eigenanalysis_(&moments[0], &moments[1], &moments[2], &moments[3], &moments[4], &moments[5],
                              moments_frame);
        math_t spreads[3] = {moments[0], moments[2], moments[5]};
        if (spreads[0] < spreads[1]) std::swap(spreads[0], spreads[1]);
        if (spreads[0] < spreads[2]) std::swap(spreads[0], spreads[2]);
        if (spreads[1] < spreads[2]) std::swap(spreads[1], spreads[2]);

        for (std::size_t axis = 0; axis != 3; ++axis)
            header.centroid[axis] = static_cast<constant_t>(static_cast<double>(centroid[axis])),
            header.spreads[axis] = static_cast<constant_t>(static_cast<double>(spreads[axis]));
        // The header keeps 𝑄 = Vᵀ, whose quaternion is the conjugate of V's
        for (std::size_t i = 0; i != 3; ++i)
            header.frame[i] = static_cast<constant_t>(-static_cast<double>(quaternion[i]));
        header.frame[3] = static_cast<constant_t>(static_cast<double>(quaternion[3]));
        header.scale = static_cast<constant_t>(static_cast<double>(scale));
        header.max_radius = static_cast<constant_t>(static_cast<double>(max_norm.sqrt()));
    }
    std::memcpy(packed, &header, sizeof(header));
}

/** Chamfer or Hausdorff of every pair in @p distance_type_, by the 8-flip passes of @c shapes.h. */
template <typename in_type_, nk_kernel_kind_t kind_, typename distance_type_, typename rotation_type_>
void shapes_nearest_reference_(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                               std::size_t columns, distance_type_ *results, std::size_t results_stride,
                               rotation_type_ *rotations) noexcept {
    using math_t = distance_type_;
    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            shapes_cloud_<in_type_> const a(a_packed[row]), b(b_packed[column]);
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
                    math_t const ai = a.template coordinate<math_t>(i, axis);
                    math_t const bj = b.template coordinate<math_t>(j, axis);
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
                    math_t const root = nearest.sqrt();
                    forward_sum = forward_sum + root;
                    if (forward_max < root) forward_max = root;
                }
                for (std::size_t j = 0; j != b.points; ++j) {
                    math_t nearest = math_t::positive_infinity();
                    for (std::size_t i = 0; i != a.points; ++i)
                        if (math_t const value = squared(i, j, flip); value < nearest) nearest = value;
                    math_t const root = nearest.sqrt();
                    backward_sum = backward_sum + root;
                    if (backward_max < root) backward_max = root;
                }
                math_t const score = kind_ == nk_kernel_chamfers_packed_k
                                         ? math_t(0.5) * (forward_sum / math_t(static_cast<double>(a.points)) +
                                                          backward_sum / math_t(static_cast<double>(b.points)))
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

/** Orders two packs by points, spreads, max_radius, and then their coordinate bytes. */
template <typename in_type_>
int shapes_compare_(shapes_cloud_<in_type_> const &a, shapes_cloud_<in_type_> const &b) noexcept {
    if (a.points != b.points) return a.points < b.points ? -1 : 1;
    for (std::size_t axis = 0; axis != 3; ++axis)
        if (a.header->spreads[axis] != b.header->spreads[axis])
            return a.header->spreads[axis] < b.header->spreads[axis] ? -1 : 1;
    if (a.header->max_radius != b.header->max_radius) return a.header->max_radius < b.header->max_radius ? -1 : 1;
    for (std::size_t axis = 0; axis != 3; ++axis)
        if (int const order = std::memcmp(a.axes + axis * a.axis_stride, b.axes + axis * b.axis_stride,
                                          a.points * sizeof(typename in_type_::raw_t)))
            return order;
    return 0;
}

/** The stopping tolerance of the network simplex in every math type, relative to its cost scale. */
inline constexpr double shapes_transport_epsilon_k = 1e-12;

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
     *  falls below `-tolerance`. */
    template <typename cost_type_>
    void solve(cost_type_ const &cost, dual_t tolerance) noexcept {
        std::size_t const arcs = rows * columns;
        std::size_t block = 10;
        while (block * block < arcs) ++block;
        dual_t const threshold = dual_t(0.0) - tolerance;
        std::size_t cursor = 0;
        for (;;) {
            dual_t entering_cost {};
            std::size_t entering = arcs, arc = cursor, in_block = 0;
            bool found = false;
            for (std::size_t scanned = 0; scanned != arcs; ++scanned) {
                std::size_t const row = arc / columns, column = arc % columns;
                dual_t const reduced = shapes_output_<dual_t>(cost(row, column)) - potential[row] +
                                       potential[rows + column];
                if (reduced < entering_cost) entering_cost = reduced, entering = arc;
                arc = arc + 1 == arcs ? 0 : arc + 1;
                if (++in_block == block || scanned + 1 == arcs) {
                    if (entering_cost < threshold) {
                        found = true;
                        break;
                    }
                    in_block = 0;
                }
            }
            if (!found) return;
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
            sum = sum + math_type_(static_cast<double>(moved)) * cost(row, column);
        });
        return sum / math_type_(static_cast<double>(rows * columns));
    }
};

/** The transport templates' workspace arrays, each on an @c nk_shapes_alignment_k_ boundary. */
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
    offset += nk_size_round_up_to_multiple_(count * sizeof(value_type_), nk_shapes_alignment_k_);
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
    std::size_t const nodes = 2 * points, plan_values = 3 * nodes;
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
template <typename in_type_, typename math_type_>
math_type_ shapes_procrustes_pair_(shapes_cloud_<in_type_> const &a, shapes_cloud_<in_type_> const &b,
                                   shapes_transport_workspace_<math_type_> &workspace, math_type_ *rotation) noexcept {
    using math_t = math_type_;
    std::size_t const n = a.points, m = b.points;
    math_t *const a_points = workspace.row_points, *const b_points = workspace.column_points;
    math_t *const rotated = workspace.rotated_columns;
    for (std::size_t i = 0; i != n; ++i)
        for (std::size_t axis = 0; axis != 3; ++axis) a_points[i * 3 + axis] = a.template coordinate<math_t>(i, axis);
    for (std::size_t j = 0; j != m; ++j)
        for (std::size_t axis = 0; axis != 3; ++axis) b_points[j * 3 + axis] = b.template coordinate<math_t>(j, axis);

    shapes_simplex_<math_t> &simplex = workspace.simplex;
    simplex.rows = n, simplex.columns = m;
    math_t const mass = math_t(static_cast<double>(n * m));
    math_t const radii = a.template max_radius<math_t>() + b.template max_radius<math_t>();
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
                math_t const weight = math_t(static_cast<double>(moved)) / mass;
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
template <typename in_type_, typename math_type_>
shapes_wide_<math_type_> shapes_gromov_pair_(shapes_cloud_<in_type_> const &a, shapes_cloud_<in_type_> const &b,
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
                matrix[i * count + k] = squared.sqrt();
                sum = sum + matrix[i * count + k] * matrix[i * count + k];
            }
        return sum / wide_t(static_cast<double>(count * count));
    };
    wide_t const row_mean_square = distances(a_points, n, row_distances);
    wide_t const column_mean_square = distances(b_points, m, column_distances);

    shapes_simplex_<math_t> &simplex = workspace.simplex;
    wide_t const mass = wide_t(static_cast<double>(n * m));
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
                    visit(saved[arc * 3 + 0], saved[arc * 3 + 1],
                          wide_t(static_cast<double>(saved[arc * 3 + 2])) / mass);
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
                    if (moved) visit(k, l, wide_t(static_cast<double>(moved)) / mass);
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
template <typename in_type_, nk_kernel_kind_t kind_, typename distance_type_, typename rotation_type_>
status_t shapes_transport_reference_(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                                     std::size_t columns, distance_type_ *results, std::size_t results_stride,
                                     rotation_type_ *rotations, std::span<std::byte> workspace) noexcept {
    using math_t = distance_type_;
    if (!rows || !columns) return status_t::success_k;
    if (reinterpret_cast<std::uintptr_t>(workspace.data()) % nk_default_alignment_k) return status_t::misaligned_k;
    std::size_t row_points = 0, column_points = 0;
    for (std::size_t row = 0; row != rows; ++row)
        if (std::size_t const points = shapes_cloud_<in_type_>(a_packed[row]).points; row_points < points)
            row_points = points;
    for (std::size_t column = 0; column != columns; ++column)
        if (std::size_t const points = shapes_cloud_<in_type_>(b_packed[column]).points; column_points < points)
            column_points = points;
    shapes_transport_workspace_<math_t> arrays;
    if (shapes_transport_layout_<math_t, kind_>(arrays, nullptr, row_points, column_points) > workspace.size())
        return status_t::unexpected_dimensions_k;
    shapes_transport_layout_<math_t, kind_>(arrays, reinterpret_cast<char *>(workspace.data()), row_points,
                                            column_points);

    for (std::size_t row = 0; row != rows; ++row)
        for (std::size_t column = 0; column != columns; ++column) {
            shapes_cloud_<in_type_> a(a_packed[row]), b(b_packed[column]);
            distance_type_ &result = reinterpret_cast<distance_type_ *>(reinterpret_cast<char *>(results) +
                                                                        row * results_stride)[column];
            rotation_type_ *rotation = rotations ? rotations + (row * columns + column) * 9 : nullptr;
            if (!a.points || !b.points) {
                result = a.points == b.points ? math_t(0.0) : math_t::positive_infinity();
                if (rotation) shapes_write_identity_(rotation);
                continue;
            }
            bool const swapped = shapes_compare_(a, b) > 0;
            if (swapped) std::swap(a, b);
            math_t core[9];
            if constexpr (kind_ == nk_kernel_wassersteinprocrustes_packed_k)
                result = shapes_procrustes_pair_(a, b, arrays, core).sqrt();
            else {
                // ½ · √loss in the wide type, rounded once into the distance type
                shapes_wide_<math_t> const loss = shapes_gromov_pair_(a, b, arrays, core);
                result = shapes_output_<distance_type_>(
                    (loss < shapes_wide_<math_t>(0.0) ? shapes_wide_<math_t>(0.0) : loss).sqrt() *
                    shapes_wide_<math_t>(0.5));
            }
            if (!rotation) continue;
            math_t a_frame[9], b_frame[9];
            a.frame(a_frame), b.frame(b_frame);
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

/**
 *  @brief Returns the bytes a pack of @p points takes.
 *  @param[in] capabilities Capabilities to pick from, or zero for the layout of @c shapes/serial.h.
 *  @tparam in_type_ Point type, @c f64_t, @c f32_t or @c f16_t.
 */
template <shapes_dtype in_type_>
expected<std::size_t> shapes_pack_size(std::size_t points,
                                       nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = shapes_pack_bytes_<in_type_>(points);
    nk_status_t status = nk_success_k;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t>)
            status = nk_shapes_pack_size_f64_best(points, capabilities, &bytes);
        else if constexpr (std::is_same_v<in_type_, f32_t>)
            status = nk_shapes_pack_size_f32_best(points, capabilities, &bytes);
        else status = nk_shapes_pack_size_f16_best(points, capabilities, &bytes);
    }
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs one cloud: centers it, rotates it onto its principal axes, and keeps its constants.
 *  @param[in] points The cloud, @p count × 3 interleaved values.
 *  @param[out] packed At least @c shapes_pack_size bytes, aligned to @c nk_default_alignment_k.
 *  @param[in] capabilities Capabilities to pick from, or zero for the template in @p math_type_.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *  @tparam math_type_ The template's arithmetic; kernels use @c in_type_::shapes_distance_t.
 */
template <shapes_dtype in_type_, typename math_type_ = typename in_type_::shapes_distance_t>
status_t shapes_pack(typename in_type_::raw_t const *points, std::size_t count, void *packed,
                     nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<math_type_, typename in_type_::shapes_distance_t>;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_shapes_pack_f64_best(points, count, packed, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_shapes_pack_f32_best(points, count, packed, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_shapes_pack_f16_best(points, count, packed, capabilities, stream));
    }
    shapes_pack_reference_<in_type_, math_type_, nk_kernel_shapes_pack_k>(points, count, packed);
    return status_t::success_k;
}

/** Packs one cloud like @c shapes_pack, also dividing it by its RMS distance from the centroid,
 *  which the pack keeps as its scale, or by 1 when the cloud has no spread. */
template <shapes_dtype in_type_, typename math_type_ = typename in_type_::shapes_distance_t>
status_t shapes_pack_normalized(typename in_type_::raw_t const *points, std::size_t count, void *packed,
                                nk_capability_t capabilities = default_capabilities(),
                                nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<math_type_, typename in_type_::shapes_distance_t>;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_shapes_pack_normalized_f64_best(points, count, packed, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_shapes_pack_normalized_f32_best(points, count, packed, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_shapes_pack_normalized_f16_best(points, count, packed, capabilities, stream));
    }
    shapes_pack_reference_<in_type_, math_type_, nk_kernel_shapes_pack_normalized_k>(points, count, packed);
    return status_t::success_k;
}

/**
 *  @brief Reads a pack's points and constants, skipping every null output.
 *  @param[out] centroid Null, or the 3 coordinates of the mean input point.
 *  @param[out] frame Null, or the unit quaternion (x, y, z, w) of 𝑄.
 *  @param[out] spreads Null, or the 3 descending eigenvalues of the packed points' second moments.
 *  @param[in] capabilities Capabilities to pick from, or zero for the layout of @c shapes/serial.h.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
template <shapes_dtype in_type_>
status_t shapes_packed_shape(void const *packed, std::size_t *points, typename in_type_::shapes_distance_t *centroid,
                             typename in_type_::shapes_distance_t *frame, typename in_type_::shapes_distance_t *scale,
                             typename in_type_::shapes_distance_t *spreads,
                             typename in_type_::shapes_distance_t *max_radius,
                             nk_capability_t capabilities = default_capabilities(),
                             nk_stream_t stream = nullptr) noexcept {
    using constant_t = typename in_type_::shapes_distance_t;
    if (capabilities) {
        auto const raw = [](constant_t *values) noexcept {
            return reinterpret_cast<typename constant_t::raw_t *>(values);
        };
        nk_size_t count = 0;
        nk_status_t status;
        if constexpr (std::is_same_v<in_type_, f64_t>)
            status = nk_shapes_packed_shape_f64_best(packed, &count, raw(centroid), raw(frame), raw(scale),
                                                     raw(spreads), raw(max_radius), capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, f32_t>)
            status = nk_shapes_packed_shape_f32_best(packed, &count, raw(centroid), raw(frame), raw(scale),
                                                     raw(spreads), raw(max_radius), capabilities, stream);
        else
            status = nk_shapes_packed_shape_f16_best(packed, &count, raw(centroid), raw(frame), raw(scale),
                                                     raw(spreads), raw(max_radius), capabilities, stream);
        if (points) *points = static_cast<std::size_t>(count);
        return static_cast<status_t>(status);
    }
    shapes_cloud_<in_type_> const cloud(packed);
    if (points) *points = cloud.points;
    for (std::size_t axis = 0; axis != 3; ++axis) {
        if (centroid) centroid[axis] = constant_t::from_raw(cloud.header->centroid[axis]);
        if (spreads) spreads[axis] = constant_t::from_raw(cloud.header->spreads[axis]);
    }
    for (std::size_t component = 0; component != 4; ++component)
        if (frame) frame[component] = constant_t::from_raw(cloud.header->frame[component]);
    if (scale) *scale = constant_t::from_raw(cloud.header->scale);
    if (max_radius) *max_radius = constant_t::from_raw(cloud.header->max_radius);
    return status_t::success_k;
}

/**
 *  @brief Restores a pack's cloud as x = scale × 𝑄ᵀ × x′ + centroid, rounded once into @p in_type_.
 *  @param[out] points The cloud, as many × 3 interleaved values as the pack holds points.
 *  @param[in] capabilities Capabilities to pick from, or zero for the template in @p math_type_.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *  @tparam math_type_ The template's arithmetic; kernels use @c in_type_::shapes_distance_t.
 */
template <shapes_dtype in_type_, typename math_type_ = typename in_type_::shapes_distance_t>
status_t shapes_unpack(void const *packed, typename in_type_::raw_t *points,
                       nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<math_type_, typename in_type_::shapes_distance_t>;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_shapes_unpack_f64_best(packed, points, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_shapes_unpack_f32_best(packed, points, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_shapes_unpack_f16_best(packed, points, capabilities, stream));
    }
    using math_t = math_type_;
    shapes_cloud_<in_type_> const cloud(packed);
    math_t frame[9];
    cloud.frame(frame);
    math_t const scale(static_cast<double>(cloud.header->scale));
    for (std::size_t index = 0; index != cloud.points; ++index)
        for (std::size_t axis = 0; axis != 3; ++axis) {
            math_t restored {};
            for (std::size_t column = 0; column != 3; ++column)
                restored = restored + frame[axis * 3 + column] * cloud.template coordinate<math_t>(index, column);
            restored = scale * restored + math_t(static_cast<double>(cloud.header->centroid[axis]));
            points[index * 3 + axis] = in_type_(static_cast<double>(restored)).raw_;
        }
    return status_t::success_k;
}

/**
 *  @brief Chamfer distances between every row pack and every column pack, the best of 8 axis flips.
 *  @param[out] results The @b [rows,columns] distances, @p results_stride bytes apart.
 *  @param[out] rotations Null, or the @b [rows,columns,3,3] rotations taking each column cloud onto
 *      each row cloud through the centroids.
 *  @param[in] capabilities Capabilities to pick from, or 0 for the template in @p distance_type_.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 */
template <shapes_dtype in_type_, typename distance_type_ = typename in_type_::shapes_distance_t,
          typename rotation_type_ = f32_t>
status_t chamfers_packed(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                         std::size_t columns, distance_type_ *results, std::size_t results_stride,
                         std::type_identity_t<rotation_type_> *rotations,
                         nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<distance_type_, typename in_type_::shapes_distance_t> &&
                              std::is_same_v<rotation_type_, f32_t>;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_chamfers_packed_f64_best(a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                                            reinterpret_cast<nk_f32_t *>(rotations), capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_chamfers_packed_f32_best(a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                                            reinterpret_cast<nk_f32_t *>(rotations), capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_chamfers_packed_f16_best(a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                                            reinterpret_cast<nk_f32_t *>(rotations), capabilities, stream));
    }
    shapes_nearest_reference_<in_type_, nk_kernel_chamfers_packed_k>(a_packed, rows, b_packed, columns, results,
                                                                     results_stride, rotations);
    return status_t::success_k;
}

/** Hausdorff distances between every row and column pack, taking what @c chamfers_packed takes. */
template <shapes_dtype in_type_, typename distance_type_ = typename in_type_::shapes_distance_t,
          typename rotation_type_ = f32_t>
status_t hausdorffs_packed(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                           std::size_t columns, distance_type_ *results, std::size_t results_stride,
                           std::type_identity_t<rotation_type_> *rotations,
                           nk_capability_t capabilities = default_capabilities(),
                           nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<distance_type_, typename in_type_::shapes_distance_t> &&
                              std::is_same_v<rotation_type_, f32_t>;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(
                nk_hausdorffs_packed_f64_best(a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                                              reinterpret_cast<nk_f32_t *>(rotations), capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(
                nk_hausdorffs_packed_f32_best(a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                                              reinterpret_cast<nk_f32_t *>(rotations), capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(
                nk_hausdorffs_packed_f16_best(a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                                              reinterpret_cast<nk_f32_t *>(rotations), capabilities, stream));
    }
    shapes_nearest_reference_<in_type_, nk_kernel_hausdorffs_packed_k>(a_packed, rows, b_packed, columns, results,
                                                                       results_stride, rotations);
    return status_t::success_k;
}

/**
 *  @brief Returns the workspace bytes of @c wassersteinprocrustes_packed between clouds of up to
 *      @p row_points and @p column_points points.
 *  @param[in] capabilities Capabilities to pick from, or 0 for the template in @p distance_type_.
 */
template <shapes_dtype in_type_, typename distance_type_ = typename in_type_::shapes_distance_t>
expected<std::size_t> wassersteinprocrustes_workspace_size(
    std::size_t row_points, std::size_t column_points, nk_capability_t capabilities = default_capabilities()) noexcept {
    constexpr bool dispatch = std::is_same_v<distance_type_, typename in_type_::shapes_distance_t>;
    if (capabilities) {
        nk_size_t bytes = 0;
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch) {
            nk_status_t const status = nk_wassersteinprocrustes_workspace_size_f64_best(row_points, column_points,
                                                                                        capabilities, &bytes);
            return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
        }
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch) {
            nk_status_t const status = nk_wassersteinprocrustes_workspace_size_f32_best(row_points, column_points,
                                                                                        capabilities, &bytes);
            return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
        }
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch) {
            nk_status_t const status = nk_wassersteinprocrustes_workspace_size_f16_best(row_points, column_points,
                                                                                        capabilities, &bytes);
            return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
        }
    }
    shapes_transport_workspace_<distance_type_> layout;
    return {shapes_transport_layout_<distance_type_, nk_kernel_wassersteinprocrustes_packed_k>(
                layout, nullptr, row_points, column_points),
            status_t::success_k};
}

/** The workspace bytes of @c gromovwassersteins_packed, queried like the Procrustes workspace. */
template <shapes_dtype in_type_, typename distance_type_ = typename in_type_::shapes_distance_t>
expected<std::size_t> gromovwassersteins_workspace_size(
    std::size_t row_points, std::size_t column_points, nk_capability_t capabilities = default_capabilities()) noexcept {
    constexpr bool dispatch = std::is_same_v<distance_type_, typename in_type_::shapes_distance_t>;
    if (capabilities) {
        nk_size_t bytes = 0;
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch) {
            nk_status_t const status = nk_gromovwassersteins_workspace_size_f64_best(row_points, column_points,
                                                                                     capabilities, &bytes);
            return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
        }
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch) {
            nk_status_t const status = nk_gromovwassersteins_workspace_size_f32_best(row_points, column_points,
                                                                                     capabilities, &bytes);
            return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
        }
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch) {
            nk_status_t const status = nk_gromovwassersteins_workspace_size_f16_best(row_points, column_points,
                                                                                     capabilities, &bytes);
            return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
        }
    }
    shapes_transport_workspace_<distance_type_> layout;
    return {shapes_transport_layout_<distance_type_, nk_kernel_gromovwassersteins_packed_k>(layout, nullptr, row_points,
                                                                                            column_points),
            status_t::success_k};
}

/**
 *  @brief Wasserstein-Procrustes distances between every row pack and every column pack.
 *  @param[in] workspace Scratch aligned to @c nk_default_alignment_k, of what
 *      @c wassersteinprocrustes_workspace_size reports for the largest clouds, with the same
 *      capabilities and @p distance_type_, or more.
 *  @return @c success_k, @c misaligned_k, @c unexpected_dimensions_k for a short workspace, or
 *      @c missing_kernel_k. The other parameters match @c chamfers_packed.
 */
template <shapes_dtype in_type_, typename distance_type_ = typename in_type_::shapes_distance_t,
          typename rotation_type_ = f32_t>
status_t wassersteinprocrustes_packed(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                                      std::size_t columns, distance_type_ *results, std::size_t results_stride,
                                      std::type_identity_t<rotation_type_> *rotations, std::span<std::byte> workspace,
                                      nk_capability_t capabilities = default_capabilities(),
                                      nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<distance_type_, typename in_type_::shapes_distance_t> &&
                              std::is_same_v<rotation_type_, f32_t>;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_wassersteinprocrustes_packed_f64_best(
                a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                reinterpret_cast<nk_f32_t *>(rotations), workspace.data(), workspace.size(), capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_wassersteinprocrustes_packed_f32_best(
                a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                reinterpret_cast<nk_f32_t *>(rotations), workspace.data(), workspace.size(), capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_wassersteinprocrustes_packed_f16_best(
                a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                reinterpret_cast<nk_f32_t *>(rotations), workspace.data(), workspace.size(), capabilities, stream));
    }
    return shapes_transport_reference_<in_type_, nk_kernel_wassersteinprocrustes_packed_k>(
        a_packed, rows, b_packed, columns, results, results_stride, rotations, workspace);
}

/** Gromov-Wasserstein distances between every row and column pack, with the parameters of
 *  @c wassersteinprocrustes_packed and the workspace of @c gromovwassersteins_workspace_size. */
template <shapes_dtype in_type_, typename distance_type_ = typename in_type_::shapes_distance_t,
          typename rotation_type_ = f32_t>
status_t gromovwassersteins_packed(void const *const *a_packed, std::size_t rows, void const *const *b_packed,
                                   std::size_t columns, distance_type_ *results, std::size_t results_stride,
                                   std::type_identity_t<rotation_type_> *rotations, std::span<std::byte> workspace,
                                   nk_capability_t capabilities = default_capabilities(),
                                   nk_stream_t stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<distance_type_, typename in_type_::shapes_distance_t> &&
                              std::is_same_v<rotation_type_, f32_t>;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_gromovwassersteins_packed_f64_best(
                a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                reinterpret_cast<nk_f32_t *>(rotations), workspace.data(), workspace.size(), capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_gromovwassersteins_packed_f32_best(
                a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                reinterpret_cast<nk_f32_t *>(rotations), workspace.data(), workspace.size(), capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_gromovwassersteins_packed_f16_best(
                a_packed, rows, b_packed, columns, &results->raw_, results_stride,
                reinterpret_cast<nk_f32_t *>(rotations), workspace.data(), workspace.size(), capabilities, stream));
    }
    return shapes_transport_reference_<in_type_, nk_kernel_gromovwassersteins_packed_k>(
        a_packed, rows, b_packed, columns, results, results_stride, rotations, workspace);
}

/**
 *  @brief Owning, move-only set of packed clouds, the operand every many-to-many kernel takes.
 *
 *  One allocation holds the table of pack pointers, then the packs back to back, each on an
 *  @c nk_default_alignment_k boundary, so @c data() passes straight to every shape kernel.
 */
template <shapes_dtype value_type_, typename allocator_type_ = aligned_allocator<char>>
class shapes_packed {
    using alloc_traits = std::allocator_traits<allocator_type_>;

    char *data_ = nullptr;
    std::size_t size_bytes_ = 0;
    std::size_t count_ = 0;
    std::size_t max_points_ = 0;
    [[no_unique_address]] allocator_type_ alloc_;

    std::size_t table_bytes_() const noexcept {
        return nk_size_round_up_to_multiple_(count_ * sizeof(void const *), nk_default_alignment_k);
    }

  public:
    /** The allocator of the packs' one block, which uses-allocator construction looks up. */
    using allocator_type = allocator_type_;

    shapes_packed() noexcept = default;
    explicit shapes_packed(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    ~shapes_packed() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
    }

    shapes_packed(shapes_packed &&o) noexcept
        : data_(std::exchange(o.data_, nullptr)), size_bytes_(std::exchange(o.size_bytes_, 0)),
          count_(std::exchange(o.count_, 0)), max_points_(std::exchange(o.max_points_, 0)),
          alloc_(std::move(o.alloc_)) {}

    shapes_packed &operator=(shapes_packed &&o) noexcept {
        if (this != &o) {
            if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(o.alloc_);
            data_ = std::exchange(o.data_, nullptr);
            size_bytes_ = std::exchange(o.size_bytes_, 0);
            count_ = std::exchange(o.count_, 0);
            max_points_ = std::exchange(o.max_points_, 0);
        }
        return *this;
    }

    shapes_packed(shapes_packed const &) = delete;
    shapes_packed &operator=(shapes_packed const &) = delete;

    /**
     *  @brief Packs every cloud of @p clouds, each @b [points,3] interleaved.
     *  @param[in] capabilities Capabilities to pack with; measure with the same ones.
     *  @return The packs; @c unexpected_dimensions_k when a cloud's length is not a multiple of 3,
     *      @c missing_kernel_k, @c bad_alloc_k, or the packing kernel's failure.
     */
    static expected<shapes_packed> make(std::span<std::span<value_type_ const> const> clouds,
                                        allocator_type_ alloc = {},
                                        nk_capability_t capabilities = default_capabilities()) noexcept {
        return make_<nk_kernel_shapes_pack_k>(clouds, alloc, capabilities);
    }

    /** Packs every cloud like @c make, each divided by its RMS distance from the centroid. */
    static expected<shapes_packed> make_normalized(std::span<std::span<value_type_ const> const> clouds,
                                                   allocator_type_ alloc = {},
                                                   nk_capability_t capabilities = default_capabilities()) noexcept {
        return make_<nk_kernel_shapes_pack_normalized_k>(clouds, alloc, capabilities);
    }

    /** One backend's @c nk_shapes_pack_size_* kernel. */
    using pack_size_kernel_t = nk_status_t (*)(nk_size_t, nk_size_t *);

    /** One backend's plain or normalized @c nk_shapes_pack kernel. */
    using pack_kernel_t = nk_status_t (*)(typename value_type_::raw_t const *, nk_size_t, void *, nk_stream_t);

    /**
     *  @brief Packs every cloud like @c make, through one backend's kernels instead of a capability
     *      mask, such as @c nk_shapes_pack_size_f32_serial and @c nk_shapes_pack_f32_serial.
     *  @param[in] pack_size Measures one pack; pair it with @p pack.
     *  @param[in] pack Packs one cloud on the CPU, with a null stream; a normalized one packs like
     *      @c make_normalized.
     */
    static expected<shapes_packed> make(std::span<std::span<value_type_ const> const> clouds, allocator_type_ alloc,
                                        pack_size_kernel_t pack_size, pack_kernel_t pack) noexcept {
        return pack_clouds_(clouds, alloc, pack_size, pack);
    }

    /** The table of pack pointers the kernels take. */
    void const *const *data() const noexcept { return reinterpret_cast<void const *const *>(data_); }

    /** The pack of cloud @p index. */
    void const *operator[](std::size_t index) const noexcept { return data()[index]; }

    /** The number of packed clouds. */
    std::size_t size() const noexcept { return count_; }

    /** The points in cloud @p index. */
    std::size_t points(std::size_t index) const noexcept {
        std::size_t count = 0;
        status_t const status = shapes_packed_shape<value_type_>(data()[index], &count, nullptr, nullptr, nullptr,
                                                                 nullptr, nullptr);
        return succeeded(status) ? count : 0;
    }

    /** The largest cloud, which sizes the transport workspaces. */
    std::size_t max_points() const noexcept { return max_points_; }

    bool empty() const noexcept { return data_ == nullptr; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }

    /** The allocator the packs came from, so a replacement can be packed from the same one. */
    allocator_type get_allocator() const noexcept { return alloc_; }

  private:
    template <nk_kernel_kind_t kind_>
    static expected<shapes_packed> make_(std::span<std::span<value_type_ const> const> clouds, allocator_type_ alloc,
                                         nk_capability_t capabilities) noexcept {
        auto const pack_size = [capabilities](nk_size_t points, nk_size_t *bytes) noexcept {
            auto const size = shapes_pack_size<value_type_>(points, capabilities);
            *bytes = size.value;
            return size.status;
        };
        auto const pack = [capabilities](typename value_type_::raw_t const *cloud, nk_size_t points, void *packed,
                                         nk_stream_t) noexcept {
            return kind_ == nk_kernel_shapes_pack_k
                       ? shapes_pack<value_type_>(cloud, points, packed, capabilities)
                       : shapes_pack_normalized<value_type_>(cloud, points, packed, capabilities);
        };
        return pack_clouds_(clouds, alloc, pack_size, pack);
    }

    /** Packs every cloud through @p pack, sized by @p pack_size, both in the C kernels' order. */
    template <typename pack_size_type_, typename pack_type_>
    static expected<shapes_packed> pack_clouds_(std::span<std::span<value_type_ const> const> clouds,
                                                allocator_type_ alloc, pack_size_type_ pack_size,
                                                pack_type_ pack) noexcept {
        shapes_packed packs(alloc);
        packs.count_ = clouds.size();
        std::size_t total = packs.table_bytes_();
        for (std::span<value_type_ const> cloud : clouds) {
            if (cloud.size() % 3) return {shapes_packed(alloc), status_t::unexpected_dimensions_k};
            nk_size_t bytes = 0;
            if (status_t const sized = static_cast<status_t>(pack_size(cloud.size() / 3, &bytes)); failed(sized))
                return {shapes_packed(alloc), sized};
            total += nk_size_round_up_to_multiple_(bytes, nk_default_alignment_k);
            if (packs.max_points_ < cloud.size() / 3) packs.max_points_ = cloud.size() / 3;
        }
        if (!total) return {std::move(packs), status_t::success_k};
        packs.size_bytes_ = total;
        packs.data_ = alloc_traits::allocate(packs.alloc_, total);
        if (!packs.data_) return {shapes_packed(alloc), status_t::bad_alloc_k};

        void const **table = reinterpret_cast<void const **>(packs.data_);
        std::size_t offset = packs.table_bytes_();
        for (std::size_t index = 0; index != clouds.size(); ++index) {
            std::size_t const points = clouds[index].size() / 3;
            auto const *cloud = reinterpret_cast<typename value_type_::raw_t const *>(clouds[index].data());
            table[index] = packs.data_ + offset;
            if (status_t const packed = static_cast<status_t>(pack(cloud, points, packs.data_ + offset, nullptr));
                failed(packed))
                return {shapes_packed(alloc), packed};
            nk_size_t bytes = 0;
            static_cast<void>(pack_size(points, &bytes));
            offset += nk_size_round_up_to_multiple_(bytes, nk_default_alignment_k);
        }
        return {std::move(packs), status_t::success_k};
    }
};

} // namespace ashvardanian::numkong

#endif // NUMKONG_SHAPES_HPP
