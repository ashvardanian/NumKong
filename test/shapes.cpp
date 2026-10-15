/**
 *  @file test/shapes.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief Many-to-many Chamfer, Hausdorff, Wasserstein-Procrustes and Gromov-Wasserstein tests.
 */

#include <numeric> // `std::iota`

#include "harness.hpp"
#include "numkong/shapes.hpp" // `nk::chamfers_packed`, `nk::shapes_pack`

namespace ashvardanian::numkong::test {

constexpr nk_kernel_kind_t chamfers_k = nk_kernel_chamfers_packed_k;
constexpr nk_kernel_kind_t hausdorffs_k = nk_kernel_hausdorffs_packed_k;
constexpr nk_kernel_kind_t wassersteinprocrustes_k = nk_kernel_wassersteinprocrustes_packed_k;
constexpr nk_kernel_kind_t gromovwassersteins_k = nk_kernel_gromovwassersteins_packed_k;
constexpr nk_kernel_kind_t pack_k = nk_kernel_shapes_pack_k;
constexpr nk_kernel_kind_t pack_normalized_k = nk_kernel_shapes_pack_normalized_k;

/** The serial normalized pack of @p scalar_type_, for the similar clouds every kernel compares. */
template <typename scalar_type_>
typename scalar_type_::shapes_pack_kernel_t shapes_pack_normalized_serial() noexcept {
    if constexpr (std::is_same_v<scalar_type_, f64_t>) return nk_shapes_pack_normalized_f64_serial;
    else if constexpr (std::is_same_v<scalar_type_, f32_t>) return nk_shapes_pack_normalized_f32_serial;
    else return nk_shapes_pack_normalized_f16_serial;
}

/** Clouds packed into their own aligned buffers, with the pointer table the kernels take. */
struct shapes_packs_t {
    std::vector<nk::vector<char>> buffers;
    std::vector<void const *> pointers;
    std::size_t largest = 0;

    std::size_t size() const noexcept { return pointers.size(); }
};

template <typename scalar_type_>
shapes_packs_t pack_clouds(std::vector<nk::vector<scalar_type_>> const &clouds,
                           typename scalar_type_::shapes_pack_size_kernel_t pack_size,
                           typename scalar_type_::shapes_pack_kernel_t pack, error_stats_t &stats) {
    shapes_packs_t packs;
    for (auto const &cloud : clouds) {
        std::size_t const points = cloud.size_values() / 3;
        nk_size_t bytes = 0;
        stats.expect(pack_size(points, &bytes));
        packs.buffers.push_back(make_vector<char>(bytes));
        stats.expect(pack(cloud.raw_values_data(), points, packs.buffers.back().raw_values_data(), nullptr));
        packs.largest = std::max(packs.largest, points);
    }
    for (auto &buffer : packs.buffers) packs.pointers.push_back(buffer.raw_values_data());
    return packs;
}

/** A cloud drawn from the settings' distribution, moved by @p offset along every axis, and kept
 *  within a quarter of the type's range so its centered and rotated points still fit the pack. */
template <typename scalar_type_>
nk::vector<scalar_type_> random_cloud(settings_t const &settings, std::mt19937 &generator, std::size_t points,
                                      nk_f64_t offset) {
    auto cloud = make_vector<scalar_type_>(points * 3);
    fill_random(settings, generator, cloud);
    nk_f64_t const limit = static_cast<nk_f64_t>(scalar_type_::finite_max()) / 4;
    for (std::size_t i = 0; i != cloud.size_values(); ++i)
        cloud.values_data()[i] = scalar_type_(std::clamp(static_cast<nk_f64_t>(cloud.values_data()[i]), -limit, limit) +
                                              offset);
    return cloud;
}

/** A cloud on the 2⁻⁶ grid, stretched 3 : 2 : 1 so its principal axes stand apart, which signed
 *  axis permutations and integer translations map exactly in @c f16 and @c f32 alike. */
template <typename scalar_type_>
nk::vector<scalar_type_> stretched_cloud(std::mt19937 &generator, std::size_t points) {
    auto cloud = make_vector<scalar_type_>(points * 3);
    int const extents[3] = {192, 128, 64};
    for (std::size_t i = 0; i != points; ++i)
        for (std::size_t axis = 0; axis != 3; ++axis) {
            std::uniform_int_distribution<int> steps(-extents[axis], extents[axis]);
            cloud.values_data()[i * 3 + axis] = scalar_type_(steps(generator) / 64.0);
        }
    return cloud;
}

/** A copy of @p cloud with point i at `map(x, y, z)` of point `order[i]`. */
template <typename scalar_type_, typename map_type_>
nk::vector<scalar_type_> mapped_cloud(nk::vector<scalar_type_> const &cloud, std::vector<std::size_t> const &order,
                                      map_type_ map) {
    auto mapped = make_vector<scalar_type_>(cloud.size_values());
    for (std::size_t i = 0; i != order.size(); ++i) {
        scalar_type_ const *point = cloud.values_data() + order[i] * 3;
        std::array<nk_f64_t, 3> const image = map(static_cast<nk_f64_t>(point[0]), static_cast<nk_f64_t>(point[1]),
                                                  static_cast<nk_f64_t>(point[2]));
        for (std::size_t axis = 0; axis != 3; ++axis) mapped.values_data()[i * 3 + axis] = scalar_type_(image[axis]);
    }
    return mapped;
}

/**
 *  @brief Bounds a kernel's distance error by the arithmetic of its math type.
 *
 *  Chamfer sums the n + m nearest-partner roots, each a few roundings off, and Hausdorff keeps one.
 *  The transport kinds add a plan value over up to n + m − 1 arcs costing at most (r_a + r_b)²,
 *  which the simplex may also leave suboptimal by its tolerance, and Gromov-Wasserstein sums
 *  distance products up to four times that scale. Square roots then shrink a value's error by twice
 *  the result, or leave its root. Compensated @c f64 sums keep at most 8 roundings of their total,
 *  whatever their length.
 */
template <typename scalar_type_, nk_kernel_kind_t kind_>
nk_f64_t shapes_distance_bound(std::size_t a_points, std::size_t b_points, nk_f64_t a_radius, nk_f64_t b_radius,
                               nk_f64_t reference) noexcept {
    using distance_t = typename scalar_type_::shapes_distance_t;
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0);
    nk_f64_t const terms = std::is_same_v<scalar_type_, f64_t> ? 8 : static_cast<nk_f64_t>(a_points + b_points);
    nk_f64_t const scale = (a_radius + b_radius) * (a_radius + b_radius);
    nk_f64_t const tolerance = nk::shapes_transport_epsilon_k * scale;
    if constexpr (kind_ == chamfers_k) return (terms + 8) * roundoff * reference;
    else if constexpr (kind_ == hausdorffs_k) return 8 * roundoff * reference;
    else if constexpr (kind_ == wassersteinprocrustes_k) {
        nk_f64_t const squared = (terms + 32) * roundoff * scale + tolerance;
        return std::min(std::sqrt(squared), squared / reference);
    }
    else {
        nk_f64_t const loss = 4 * ((terms + 64) * roundoff * scale + tolerance);
        return std::min(0.5 * std::sqrt(loss), loss / (8 * reference));
    }
}

/** Bounds a rotation entry's error: its rounding into @c f32, plus the math type's quaternion
 *  products for the flip kinds, or its 3 × 3 SVD for the transport kinds. */
template <typename scalar_type_, nk_kernel_kind_t kind_>
nk_f64_t shapes_rotation_bound() noexcept {
    nk_f64_t const roundoff = half_ulp<typename scalar_type_::shapes_distance_t>(1.0);
    return std::ldexp(1.0, -24) + (kind_ == chamfers_k || kind_ == hausdorffs_k ? 32 : 4096) * roundoff;
}

/** How well @p rotation lays the column pack onto the row pack: its Chamfer or Hausdorff distance,
 *  or the Wasserstein distance it leaves, as no rotation moves a Gromov-Wasserstein loss. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename reference_type_, typename rotation_type_>
nk_f64_t rotation_fit(void const *row_pack, void const *column_pack, rotation_type_ const *rotation) {
    using reference_t = reference_type_;
    nk::shapes_cloud_<scalar_type_> const a(row_pack), b(column_pack);
    reference_t a_frame[9], b_frame[9], turn[9] {};
    a.template frame<reference_t>(a_frame), b.template frame<reference_t>(b_frame);
    // The rotation between the packed frames: V_aᵀ × R × V_b
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t column = 0; column != 3; ++column)
            for (std::size_t i = 0; i != 3; ++i)
                for (std::size_t j = 0; j != 3; ++j)
                    turn[row * 3 + column] = turn[row * 3 + column] +
                                             a_frame[i * 3 + row] *
                                                 nk::shapes_output_<reference_t>(rotation[i * 3 + j]) *
                                                 b_frame[j * 3 + column];
    std::vector<reference_t> a_points(a.points * 3), b_points(b.points * 3);
    for (std::size_t i = 0; i != a.points; ++i)
        for (std::size_t axis = 0; axis != 3; ++axis)
            a_points[i * 3 + axis] = a.template coordinate<reference_t>(i, axis);
    for (std::size_t j = 0; j != b.points; ++j)
        for (std::size_t axis = 0; axis != 3; ++axis)
            for (std::size_t k = 0; k != 3; ++k)
                b_points[j * 3 + axis] = b_points[j * 3 + axis] +
                                         turn[axis * 3 + k] * b.template coordinate<reference_t>(j, k);
    auto const squared = [&](std::size_t i, std::size_t j) {
        reference_t sum {};
        for (std::size_t axis = 0; axis != 3; ++axis) {
            reference_t const difference = a_points[i * 3 + axis] - b_points[j * 3 + axis];
            sum = sum + difference * difference;
        }
        return sum;
    };

    if constexpr (kind_ == chamfers_k || kind_ == hausdorffs_k) {
        reference_t means[2] {}, maxima[2] {};
        for (std::size_t side = 0; side != 2; ++side) {
            std::size_t const outer_count = side ? b.points : a.points, inner_count = side ? a.points : b.points;
            for (std::size_t outer = 0; outer != outer_count; ++outer) {
                reference_t least = reference_t::positive_infinity();
                for (std::size_t inner = 0; inner != inner_count; ++inner) {
                    reference_t const value = side ? squared(inner, outer) : squared(outer, inner);
                    if (value < least) least = value;
                }
                reference_t const root = least.sqrt();
                means[side] = means[side] + root;
                if (maxima[side] < root) maxima[side] = root;
            }
            means[side] = means[side] / reference_t(static_cast<nk_f64_t>(outer_count));
        }
        if constexpr (kind_ == hausdorffs_k)
            return static_cast<nk_f64_t>(maxima[0] < maxima[1] ? maxima[1] : maxima[0]);
        else return static_cast<nk_f64_t>(reference_t(0.5) * (means[0] + means[1]));
    }
    else {
        nk::shapes_transport_workspace_<reference_t> arrays;
        auto workspace = make_vector<char>(
            nk::shapes_transport_layout_<reference_t, wassersteinprocrustes_k>(arrays, nullptr, a.points, b.points));
        nk::shapes_transport_layout_<reference_t, wassersteinprocrustes_k>(arrays, workspace.raw_values_data(),
                                                                           a.points, b.points);
        using dual_t = typename nk::shapes_simplex_<reference_t>::dual_t;
        reference_t const radii = a.template max_radius<reference_t>() + b.template max_radius<reference_t>();
        arrays.simplex.rows = a.points, arrays.simplex.columns = b.points;
        arrays.simplex.northwest_corner(squared);
        arrays.simplex.solve(squared,
                             nk::shapes_output_<dual_t>(radii * radii) * dual_t(nk::shapes_transport_epsilon_k));
        return static_cast<nk_f64_t>(arrays.simplex.plan_value(squared).sqrt());
    }
}

/** Runs @p kind_ over the C++ template in @p reference_type_: the oracle for every kernel. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename reference_type_>
nk::status_t shapes_oracle(shapes_packs_t const &rows, shapes_packs_t const &columns, reference_type_ *results,
                           reference_type_ *rotations) {
    void const *const *a = rows.pointers.data(), *const *b = columns.pointers.data();
    std::size_t const stride = columns.size() * sizeof(reference_type_);
    if constexpr (kind_ == chamfers_k)
        return nk::chamfers_packed<scalar_type_, reference_type_, reference_type_>(
            a, rows.size(), b, columns.size(), results, stride, rotations, no_tiers_k);
    else if constexpr (kind_ == hausdorffs_k)
        return nk::hausdorffs_packed<scalar_type_, reference_type_, reference_type_>(
            a, rows.size(), b, columns.size(), results, stride, rotations, no_tiers_k);
    else if constexpr (kind_ == wassersteinprocrustes_k) {
        std::size_t const bytes = nk::wassersteinprocrustes_workspace_size<scalar_type_, reference_type_>(
                                      rows.largest, columns.largest, no_tiers_k)
                                      .value;
        auto workspace = make_vector<char>(std::max<std::size_t>(bytes, 1));
        return nk::wassersteinprocrustes_packed<scalar_type_, reference_type_, reference_type_>(
            a, rows.size(), b, columns.size(), results, stride, rotations,
            std::span<std::byte>(reinterpret_cast<std::byte *>(workspace.raw_values_data()), bytes), no_tiers_k);
    }
    else {
        std::size_t const bytes = nk::gromovwassersteins_workspace_size<scalar_type_, reference_type_>(
                                      rows.largest, columns.largest, no_tiers_k)
                                      .value;
        auto workspace = make_vector<char>(std::max<std::size_t>(bytes, 1));
        return nk::gromovwassersteins_packed<scalar_type_, reference_type_, reference_type_>(
            a, rows.size(), b, columns.size(), results, stride, rotations,
            std::span<std::byte>(reinterpret_cast<std::byte *>(workspace.raw_values_data()), bytes), no_tiers_k);
    }
}

/**
 *  @brief Checks one many-to-many kernel, called through @p call over row and column packs.
 *
 *  Compares an N × M tile of mixed sizes against the oracle, every other trial moved far from the
 *  origin, then checks that swapping the sides transposes the results bit for bit, that a cloud
 *  scores zero against itself, the empty-cloud conventions, and invariance to rotation, reflection,
 *  translation and point order.
 */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename call_type_>
error_stats_t test_shapes_kind(settings_t const &settings, typename scalar_type_::shapes_pack_size_kernel_t pack_size,
                               typename scalar_type_::shapes_pack_kernel_t pack, call_type_ const &call) {
    using scalar_t = scalar_type_;
    using distance_t = typename scalar_t::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    using reference_t = reference_for<scalar_t>;
    using cloud_t = nk::shapes_cloud_<scalar_t>;

    error_stats_t stats(comparison_family_t::bounded_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const n = settings.shape_points;
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0);
    constexpr bool exact_identity_k = kind_ != gromovwassersteins_k;
    constexpr nk_kernel_kind_t fit_kind_k = kind_ == gromovwassersteins_k ? wassersteinprocrustes_k : kind_;
    std::vector<raw_distance_t> results, transposed_results;
    std::vector<nk_f32_t> rotations, transposed_rotations;
    auto const run = [&](shapes_packs_t const &rows, shapes_packs_t const &columns, std::vector<raw_distance_t> &out,
                         std::vector<nk_f32_t> &out_rotations) {
        out.assign(rows.size() * columns.size(), raw_distance_t(-1));
        out_rotations.assign(rows.size() * columns.size() * 9, nk_f32_t(-2));
        return stats.expect(call(rows, columns, out.data(), out_rotations.data()));
    };
    auto const radius = [](void const *packed) { return static_cast<nk_f64_t>(cloud_t(packed).header->max_radius); };
    auto const is_identity = [](nk_f32_t const *rotation) {
        bool identity = true;
        for (std::size_t i = 0; i != 9; ++i) identity &= rotation[i] == (i % 4 == 0 ? 1.0f : 0.0f);
        return identity;
    };
    auto const orthogonality = [](nk_f32_t const *rotation) {
        nk_f64_t worst = 0;
        for (std::size_t i = 0; i != 3; ++i)
            for (std::size_t j = 0; j != 3; ++j) {
                nk_f64_t dot = 0;
                for (std::size_t k = 0; k != 3; ++k)
                    dot += static_cast<nk_f64_t>(rotation[k * 3 + i]) * rotation[k * 3 + j];
                worst = std::max(worst, std::fabs(dot - (i == j)));
            }
        return worst;
    };

    // Empty clouds: zero against each other, infinite against anything else, with identity
    // rotations; an empty grid succeeds without reading a pack or writing a result
    {
        std::vector<nk::vector<scalar_t>> row_clouds, column_clouds;
        row_clouds.push_back(make_vector<scalar_t>(0)),
            row_clouds.push_back(random_cloud<scalar_t>(settings, generator, n, 0));
        column_clouds.push_back(make_vector<scalar_t>(0)),
            column_clouds.push_back(random_cloud<scalar_t>(settings, generator, n, 0));
        shapes_packs_t const rows = pack_clouds(row_clouds, pack_size, pack, stats);
        shapes_packs_t const columns = pack_clouds(column_clouds, pack_size, pack, stats);
        if (run(rows, columns, results, rotations)) {
            nk_f64_t const both = results[0], row_only = results[2], column_only = results[1];
            stats.expect(both == 0 && std::isinf(row_only) && row_only > 0 && std::isinf(column_only) &&
                             column_only > 0 && std::isfinite(static_cast<nk_f64_t>(results[3])),
                         "empty clouds broke the zero and infinity conventions");
            stats.expect(is_identity(&rotations[0]) && is_identity(&rotations[9]) && is_identity(&rotations[18]),
                         "an empty cloud's rotation is not the identity");
        }
        stats.expect(call(shapes_packs_t {}, columns, nullptr, nullptr) == nk_success_k &&
                         call(rows, shapes_packs_t {}, nullptr, nullptr) == nk_success_k,
                     "an empty grid failed");
    }

    // Exact rigid motions and a shuffle of one cloud keep its distance to another
    {
        auto const base = stretched_cloud<scalar_t>(generator, n);
        std::vector<std::size_t> order(n), shuffled(n);
        std::iota(order.begin(), order.end(), 0), std::iota(shuffled.begin(), shuffled.end(), 0);
        std::shuffle(shuffled.begin(), shuffled.end(), generator);
        std::vector<nk::vector<scalar_t>> row_clouds, column_clouds;
        row_clouds.push_back(mapped_cloud(
            base, order, [](nk_f64_t x, nk_f64_t y, nk_f64_t z) { return std::array<nk_f64_t, 3> {x, y, z}; }));
        row_clouds.push_back(mapped_cloud(base, order, [](nk_f64_t x, nk_f64_t y, nk_f64_t z) {
            return std::array<nk_f64_t, 3> {-y + 5, -z - 7, x + 3};
        }));
        row_clouds.push_back(mapped_cloud(base, order, [](nk_f64_t x, nk_f64_t y, nk_f64_t z) {
            return std::array<nk_f64_t, 3> {y - 6, x + 2, z + 4};
        }));
        row_clouds.push_back(mapped_cloud(
            base, shuffled, [](nk_f64_t x, nk_f64_t y, nk_f64_t z) { return std::array<nk_f64_t, 3> {x, y, z}; }));
        column_clouds.push_back(stretched_cloud<scalar_t>(generator, 2 * n / 3 + 1));
        shapes_packs_t const rows = pack_clouds(row_clouds, pack_size, pack, stats);
        shapes_packs_t const columns = pack_clouds(column_clouds, pack_size, pack, stats);
        if (run(rows, columns, results, rotations)) {
            nk_f64_t const a_radius = radius(rows.pointers[0]), b_radius = radius(columns.pointers[0]);
            // Each packed coordinate may round one step apart between the poses
            nk_f64_t const displacement = 2 * std::sqrt(3.0) * half_ulp<scalar_t>(a_radius) +
                                          64 * roundoff * (a_radius + 10);
            nk_f64_t const bound = (kind_ == gromovwassersteins_k ? 2 : 1) * displacement +
                                   2 * shapes_distance_bound<scalar_t, kind_>(n, columns.largest, a_radius, b_radius,
                                                                              results[0]);
            for (std::size_t pose = 1; pose != 4; ++pose)
                stats.expect(
                    std::fabs(static_cast<nk_f64_t>(results[pose]) - results[0]) <= bound,
                    pose == 3 ? "shuffling the points changed the distance" : "a rigid motion changed the distance");
        }
    }

    // Between normalized packs, a cloud scores about zero against its turned, moved and grown copy,
    // though the copy's rounded inputs turn the principal axes by up to twice their displacement
    // over the smallest gap between the spreads
    {
        auto const base = stretched_cloud<scalar_t>(generator, n);
        std::normal_distribution<nk_f64_t> normal(0, 1);
        std::uniform_real_distribution<nk_f64_t> shift(-4, 4);
        nk::f64_t quaternion[4], turn[9], length = 0;
        for (nk::f64_t &value : quaternion) value = normal(generator), length = length + value * value;
        for (nk::f64_t &value : quaternion) value = value / length.sqrt();
        nk::quaternion_to_mat3x3_(quaternion, turn);
        nk_f64_t motion[9];
        for (std::size_t i = 0; i != 9; ++i) motion[i] = 3.7 * static_cast<nk_f64_t>(turn[i]);
        nk_f64_t const translation[3] = {shift(generator), shift(generator), shift(generator)};
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::vector<nk::vector<scalar_t>> row_clouds, column_clouds;
        row_clouds.push_back(mapped_cloud(
            base, order, [](nk_f64_t x, nk_f64_t y, nk_f64_t z) { return std::array<nk_f64_t, 3> {x, y, z}; }));
        column_clouds.push_back(mapped_cloud(base, order, [&](nk_f64_t x, nk_f64_t y, nk_f64_t z) {
            std::array<nk_f64_t, 3> image;
            for (std::size_t axis = 0; axis != 3; ++axis)
                image[axis] = motion[axis * 3 + 0] * x + motion[axis * 3 + 1] * y + motion[axis * 3 + 2] * z +
                              translation[axis];
            return image;
        }));
        nk_f64_t largest_input = 0;
        for (std::size_t i = 0; i != column_clouds[0].size_values(); ++i)
            largest_input = std::max(largest_input,
                                     std::fabs(static_cast<nk_f64_t>(column_clouds[0].values_data()[i])));
        auto const pack_normalized = shapes_pack_normalized_serial<scalar_t>();
        shapes_packs_t const rows = pack_clouds(row_clouds, pack_size, pack_normalized, stats);
        shapes_packs_t const columns = pack_clouds(column_clouds, pack_size, pack_normalized, stats);
        if (run(rows, columns, results, rotations)) {
            cloud_t const a(rows.pointers[0]), b(columns.pointers[0]);
            nk_f64_t const a_radius = a.header->max_radius, b_radius = b.header->max_radius;
            nk_f64_t const b_scale = b.header->scale;
            auto const *spreads = a.header->spreads;
            // The copy's input rounding in normalized units, then each side's packing
            nk_f64_t const displacement = std::sqrt(3.0) * half_ulp<scalar_t>(largest_input) / b_scale +
                                          2 * std::sqrt(3.0) * half_ulp<scalar_t>(a_radius) +
                                          64 * roundoff * (a_radius + 10);
            nk_f64_t const gap = std::min(spreads[0] - spreads[1], spreads[1] - spreads[2]);
            nk_f64_t const turned = kind_ == chamfers_k || kind_ == hausdorffs_k ? 2 * a_radius * displacement / gap
                                                                                 : 0;
            nk_f64_t const rounding = (kind_ == gromovwassersteins_k ? 2 : 1) * displacement +
                                      2 * shapes_distance_bound<scalar_t, kind_>(n, n, a_radius, b_radius, results[0]);
            stats.expect(std::fabs(static_cast<nk_f64_t>(results[0])) <= rounding + turned,
                         "a grown copy scores above zero between normalized packs");
            shapes_packs_t const plain_rows = pack_clouds(row_clouds, pack_size, pack, stats);
            shapes_packs_t const plain_columns = pack_clouds(column_clouds, pack_size, pack, stats);
            if (run(plain_rows, plain_columns, results, rotations))
                stats.expect(static_cast<nk_f64_t>(results[0]) > rounding * b_scale,
                             "a grown copy scores within rounding of zero between plain packs");
        }
    }

    std::size_t const row_points[] = {n, n / 2 + 2};
    std::size_t const column_points[] = {n, 2 * n / 3 + 1, n / 4 + 3};
    std::vector<reference_t> references, reference_rotations;
    bool far_from_origin = false;
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        nk_f64_t const offset = (far_from_origin = !far_from_origin) ? 64 : 0;
        std::vector<nk::vector<scalar_t>> row_clouds, column_clouds;
        for (std::size_t points : row_points)
            row_clouds.push_back(random_cloud<scalar_t>(settings, generator, points, offset));
        for (std::size_t points : column_points)
            column_clouds.push_back(random_cloud<scalar_t>(settings, generator, points, offset));
        shapes_packs_t const rows = pack_clouds(row_clouds, pack_size, pack, stats);
        shapes_packs_t const columns = pack_clouds(column_clouds, pack_size, pack, stats);
        if (!run(rows, columns, results, rotations)) continue;

        references.assign(rows.size() * columns.size(), reference_t(0.0));
        reference_rotations.assign(rows.size() * columns.size() * 9, reference_t(0.0));
        stats.expect(shapes_oracle<scalar_t, kind_>(rows, columns, references.data(), reference_rotations.data()));
        nk_f64_t const entry_bound = shapes_rotation_bound<scalar_t, kind_>();
        for (std::size_t row = 0; row != rows.size(); ++row)
            for (std::size_t column = 0; column != columns.size(); ++column) {
                std::size_t const cell = row * columns.size() + column;
                void const *const row_pack = rows.pointers[row], *const column_pack = columns.pointers[column];
                nk_f64_t const bound = shapes_distance_bound<scalar_t, kind_>(row_points[row], column_points[column],
                                                                              radius(row_pack), radius(column_pack),
                                                                              static_cast<nk_f64_t>(references[cell]));
                stats.accumulate_bounded(distance_t::from_raw(results[cell]), references[cell], bound);

                // Near-tied flips may pick another rotation than the oracle's, so a rotation counts
                // by its fit, which may trail the oracle's by two distance bounds for the pick and
                // by 3 × bound × r for entries a bound off
                nk_f32_t const *rotation = &rotations[cell * 9];
                stats.expect(orthogonality(rotation) <= 4 * entry_bound, "a rotation is not orthogonal");
                nk_f64_t const oracle_fit = rotation_fit<scalar_t, kind_, reference_t>(row_pack, column_pack,
                                                                                       &reference_rotations[cell * 9]);
                nk_f64_t const fit_bound = shapes_distance_bound<scalar_t, fit_kind_k>(
                    row_points[row], column_points[column], radius(row_pack), radius(column_pack), oracle_fit);
                stats.expect(rotation_fit<scalar_t, kind_, reference_t>(row_pack, column_pack, rotation) <=
                                 oracle_fit + 2 * fit_bound + 3 * entry_bound * radius(column_pack),
                             "a rotation fits the clouds worse than the oracle's");
            }

        // Swapping the sides transposes the distances and the rotations bit for bit
        if (run(columns, rows, transposed_results, transposed_rotations)) {
            bool symmetric = true;
            for (std::size_t row = 0; row != rows.size(); ++row)
                for (std::size_t column = 0; column != columns.size(); ++column) {
                    std::size_t const cell = row * columns.size() + column, mirror = column * rows.size() + row;
                    symmetric &= std::memcmp(&results[cell], &transposed_results[mirror], sizeof(raw_distance_t)) == 0;
                    for (std::size_t i = 0; i != 3; ++i)
                        for (std::size_t j = 0; j != 3; ++j)
                            symmetric &= std::memcmp(&rotations[cell * 9 + i * 3 + j],
                                                     &transposed_rotations[mirror * 9 + j * 3 + i],
                                                     sizeof(nk_f32_t)) == 0;
                }
            stats.expect(symmetric, "swapping the sides changed a distance or a rotation's transpose");
        }

        // A cloud scores zero against itself, or within the roundings of the GW loss
        if (run(rows, rows, transposed_results, transposed_rotations))
            for (std::size_t row = 0; row != rows.size(); ++row) {
                nk_f64_t const own_radius = radius(rows.pointers[row]);
                nk_f64_t const bound = exact_identity_k
                                           ? 0
                                           : shapes_distance_bound<scalar_t, kind_>(row_points[row], row_points[row],
                                                                                    own_radius, own_radius, 0);
                stats.expect(std::fabs(static_cast<nk_f64_t>(transposed_results[row * rows.size() + row])) <= bound,
                             "a cloud scores above zero against itself");
            }
    }
    return stats;
}

/** Checks a Chamfer or Hausdorff kernel. */
template <typename scalar_type_, nk_kernel_kind_t kind_>
error_stats_t test_shapes_packed(settings_t const &settings, typename scalar_type_::shapes_pack_size_kernel_t pack_size,
                                 typename scalar_type_::shapes_pack_kernel_t pack,
                                 typename scalar_type_::shapes_packed_kernel_t kernel) {
    using raw_distance_t = typename scalar_type_::shapes_distance_t::raw_t;
    auto const call = [&](shapes_packs_t const &rows, shapes_packs_t const &columns, raw_distance_t *results,
                          nk_f32_t *rotations) {
        return kernel(rows.pointers.data(), rows.size(), columns.pointers.data(), columns.size(), results,
                      columns.size() * sizeof(raw_distance_t), rotations, nullptr);
    };
    return test_shapes_kind<scalar_type_, kind_>(settings, pack_size, pack, call);
}

/** Checks a transport kernel, then that it refuses a workspace one byte short or misaligned. */
template <typename scalar_type_, nk_kernel_kind_t kind_>
error_stats_t test_shapes_packed_workspace(settings_t const &settings,
                                           typename scalar_type_::shapes_pack_size_kernel_t pack_size,
                                           typename scalar_type_::shapes_pack_kernel_t pack,
                                           typename scalar_type_::shapes_workspace_size_kernel_t workspace_size,
                                           typename scalar_type_::shapes_packed_workspace_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_distance_t = typename scalar_t::shapes_distance_t::raw_t;
    auto const call_with = [&](shapes_packs_t const &rows, shapes_packs_t const &columns, raw_distance_t *results,
                               nk_f32_t *rotations, char *workspace, nk_size_t bytes) {
        return kernel(rows.pointers.data(), rows.size(), columns.pointers.data(), columns.size(), results,
                      columns.size() * sizeof(raw_distance_t), rotations, workspace, bytes, nullptr);
    };
    auto const call = [&](shapes_packs_t const &rows, shapes_packs_t const &columns, raw_distance_t *results,
                          nk_f32_t *rotations) {
        nk_size_t bytes = 0;
        if (nk_status_t const status = workspace_size(rows.largest, columns.largest, &bytes); status != nk_success_k)
            return status;
        auto workspace = make_vector<char>(std::max<nk_size_t>(bytes, 1));
        return call_with(rows, columns, results, rotations, workspace.raw_values_data(), bytes);
    };
    error_stats_t stats = test_shapes_kind<scalar_t, kind_>(settings, pack_size, pack, call);

    std::mt19937 generator(settings.seed.value);
    std::vector<nk::vector<scalar_t>> row_clouds, column_clouds;
    row_clouds.push_back(random_cloud<scalar_t>(settings, generator, settings.shape_points, 0));
    column_clouds.push_back(random_cloud<scalar_t>(settings, generator, settings.shape_points / 2 + 2, 0));
    shapes_packs_t const rows = pack_clouds(row_clouds, pack_size, pack, stats);
    shapes_packs_t const columns = pack_clouds(column_clouds, pack_size, pack, stats);
    nk_size_t bytes = 0;
    stats.expect(workspace_size(rows.largest, columns.largest, &bytes));
    auto workspace = make_vector<char>(bytes + nk_default_alignment_k);
    raw_distance_t result;
    nk_f32_t rotation[9];
    stats.expect(call_with(rows, columns, &result, rotation, workspace.raw_values_data(), bytes - 1) ==
                     nk_unexpected_dimensions_k,
                 "a workspace one byte short was not refused");
    stats.expect(call_with(rows, columns, &result, rotation, workspace.raw_values_data() + 8, bytes) == nk_misaligned_k,
                 "a misaligned workspace was not refused");
    return stats;
}

/** Packs of @p kind_ against the oracle pack: the constants @p packed_shape reads, an orthonormal
 *  proper frame, descending spreads, zero padding, and @p unpack restoring the input at its
 *  original scale within one rounding into the storage type. */
template <typename scalar_type_, nk_kernel_kind_t kind_>
error_stats_t test_shapes_pack(settings_t const &settings, typename scalar_type_::shapes_pack_size_kernel_t pack_size,
                               typename scalar_type_::shapes_pack_kernel_t pack,
                               typename scalar_type_::shapes_packed_shape_kernel_t packed_shape,
                               typename scalar_type_::shapes_unpack_kernel_t unpack) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using constant_t = typename scalar_t::shapes_distance_t; // packs keep constants in the distance type
    using raw_constant_t = typename constant_t::raw_t;
    using reference_t = reference_for<scalar_t>;
    using cloud_t = nk::shapes_cloud_<scalar_t>;

    error_stats_t stats(comparison_family_t::bounded_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const n = settings.shape_points;
    nk_f64_t const roundoff = half_ulp<constant_t>(1.0);
    // Compensated f64 sums keep at most 8 roundings of their total, whatever their length
    nk_f64_t const terms = std::is_same_v<scalar_t, f64_t> ? 8 : static_cast<nk_f64_t>(n);
    nk_size_t count = 0;
    raw_constant_t centroid[3], frame[4], scale, spreads[3], max_radius;

    {
        nk_size_t bytes = 0;
        stats.expect(pack_size(0, &bytes));
        auto packed = make_vector<char>(bytes);
        std::memset(packed.raw_values_data(), 0xFF, bytes);
        stats.expect(pack(nullptr, 0, packed.raw_values_data(), nullptr));
        stats.expect(
            packed_shape(packed.raw_values_data(), &count, centroid, frame, &scale, spreads, &max_radius, nullptr));
        bool zeros = count == 0 && max_radius == 0 && frame[3] == 1 && scale == 1;
        for (std::size_t axis = 0; axis != 3; ++axis)
            zeros &= centroid[axis] == 0 && spreads[axis] == 0 && frame[axis] == 0;
        stats.expect(zeros, "an empty cloud packs into more than zeros, the identity frame and a unit scale");
    }

    // The owner packs every cloud as the wrapper does, behind one table of pointers
    {
        auto const first = random_cloud<scalar_t>(settings, generator, n, 0);
        auto const second = random_cloud<scalar_t>(settings, generator, n / 2 + 1, 0);
        std::span<scalar_t const> const spans[] = {{first.values_data(), first.size_values()},
                                                   {second.values_data(), second.size_values()}};
        auto const owned = kind_ == pack_k ? nk::shapes_packed<scalar_t>::make(spans)
                                           : nk::shapes_packed<scalar_t>::make_normalized(spans);
        bool same = owned && owned.value.size() == 2 && owned.value.max_points() == n &&
                    owned.value.points(1) == n / 2 + 1;
        for (std::size_t index = 0; same && index != 2; ++index) {
            std::size_t const points = spans[index].size() / 3, bytes = nk::shapes_pack_size<scalar_t>(points).value;
            raw_t const *cloud = reinterpret_cast<raw_t const *>(spans[index].data());
            auto expected = make_vector<char>(bytes);
            stats.expect(kind_ == pack_k
                             ? nk::shapes_pack<scalar_t>(cloud, points, expected.raw_values_data())
                             : nk::shapes_pack_normalized<scalar_t>(cloud, points, expected.raw_values_data()));
            same = std::memcmp(owned.value[index], expected.raw_values_data(), bytes) == 0 &&
                   reinterpret_cast<std::uintptr_t>(owned.value[index]) % nk_default_alignment_k == 0;
        }
        stats.expect(same, "the owner's packs differ from the wrapper's or miss the default alignment");
        auto const none = kind_ == pack_k ? nk::shapes_packed<scalar_t>::make({})
                                          : nk::shapes_packed<scalar_t>::make_normalized({});
        stats.expect(none && none.value.empty() && none.value.size() == 0, "the owner of no clouds failed");
    }

    bool far_from_origin = false;
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        nk_f64_t const offset = (far_from_origin = !far_from_origin) ? 64 : 0;
        auto const cloud = random_cloud<scalar_t>(settings, generator, n, offset);
        nk_size_t bytes = 0;
        stats.expect(pack_size(n, &bytes));
        stats.expect(bytes == nk::shapes_pack_size<scalar_t>(n, no_tiers_k).value,
                     "the pack size differs from the oracle's");
        auto packed = make_vector<char>(bytes), oracle = make_vector<char>(bytes);
        std::memset(packed.raw_values_data(), 0xFF, bytes);
        if (!stats.expect(pack(cloud.raw_values_data(), n, packed.raw_values_data(), nullptr))) continue;
        stats.expect(kind_ == pack_k ? nk::shapes_pack<scalar_t, reference_t>(cloud.raw_values_data(), n,
                                                                              oracle.raw_values_data(), no_tiers_k)
                                     : nk::shapes_pack_normalized<scalar_t, reference_t>(
                                           cloud.raw_values_data(), n, oracle.raw_values_data(), no_tiers_k));
        cloud_t const kernel_pack(packed.raw_values_data()), oracle_pack(oracle.raw_values_data());
        stats.expect(
            packed_shape(packed.raw_values_data(), &count, centroid, frame, &scale, spreads, &max_radius, nullptr));
        stats.expect(count == n, "the pack counts other points");

        nk_f64_t largest_input = 0, largest_centroid = 0;
        for (std::size_t i = 0; i != cloud.size_values(); ++i)
            largest_input = std::max(largest_input, std::fabs(static_cast<nk_f64_t>(cloud.values_data()[i])));
        for (std::size_t axis = 0; axis != 3; ++axis)
            largest_centroid = std::max<nk_f64_t>(largest_centroid, std::fabs(centroid[axis]));
        nk_f64_t const radius = static_cast<nk_f64_t>(oracle_pack.header->max_radius);
        nk_f64_t const storage = half_ulp<scalar_t>(radius);

        // The centroid sums n inputs; packed points may round a step apart from the oracle's,
        // moving the second moments by twice that times the radius per entry and the largest norm
        // by its length, which the centroid's own error also shifts
        nk_f64_t const centroid_bound = (terms + 2) * roundoff * largest_input;
        for (std::size_t axis = 0; axis != 3; ++axis) {
            stats.accumulate_bounded(constant_t(centroid[axis]), oracle_pack.header->centroid[axis], centroid_bound);
            stats.accumulate_bounded(constant_t(spreads[axis]), oracle_pack.header->spreads[axis],
                                     12 * radius * storage + (terms + 16) * roundoff * radius * radius);
        }
        stats.accumulate_bounded(constant_t(max_radius), oracle_pack.header->max_radius,
                                 2 * std::sqrt(3.0) * (storage + centroid_bound) + 4 * roundoff * radius);

        // The RMS radius sums n squared deviations; a centroid error moves it by at most that error
        if constexpr (kind_ == pack_k) stats.expect(scale == 1, "a plain pack's scale is not 1");
        else
            stats.accumulate_bounded(constant_t(scale), oracle_pack.header->scale,
                                     (terms + 8) * roundoff * scale + 2 * (terms + 2) * roundoff * largest_input);
        stats.expect(spreads[0] >= spreads[1] && spreads[1] >= spreads[2] &&
                         spreads[2] >= -(terms + 16) * roundoff * radius * radius,
                     "the spreads are not descending");

        reference_t quaternion[4], rotation[9];
        nk_f64_t orthogonality = 0, norm = 0;
        for (std::size_t i = 0; i != 4; ++i)
            quaternion[i] = reference_t(static_cast<nk_f64_t>(frame[i])),
            norm += static_cast<nk_f64_t>(frame[i]) * frame[i];
        nk::quaternion_to_mat3x3_(quaternion, rotation);
        for (std::size_t i = 0; i != 3; ++i)
            for (std::size_t j = 0; j != 3; ++j) {
                reference_t dot = rotation[0 * 3 + i] * rotation[0 * 3 + j] +
                                  rotation[1 * 3 + i] * rotation[1 * 3 + j] + rotation[2 * 3 + i] * rotation[2 * 3 + j];
                orthogonality = std::max(orthogonality, std::fabs(static_cast<nk_f64_t>(dot) - (i == j)));
            }
        // Normalizing and storing round each component a few times, leaving |q|² within 8 roundings
        // of 1, and the Gram matrix and determinant of its matrix within four times that
        stats.expect(std::fabs(norm - 1) <= 8 * roundoff && orthogonality <= 32 * roundoff &&
                         std::fabs(static_cast<nk_f64_t>(nk::det3x3_(rotation)) - 1) <= 32 * roundoff,
                     "the frame is not a unit quaternion of a proper rotation");

        // x = scale × 𝑄ᵀ × x′ + centroid lands within the rounding of x′ into the storage type, and
        // rounding it back moves it at most that far again, since the input is a candidate
        nk_f64_t const trip_bound = 2 * (std::sqrt(3.0) * scale * storage +
                                         32 * roundoff * (largest_centroid + scale * radius) +
                                         half_ulp<constant_t>(largest_centroid));
        auto restored = make_vector<scalar_t>(n * 3);
        stats.expect(unpack(packed.raw_values_data(), restored.raw_values_data(), nullptr));
        nk_f64_t trip_error = 0;
        for (std::size_t i = 0; i != cloud.size_values(); ++i)
            trip_error = std::max(trip_error, std::fabs(static_cast<nk_f64_t>(restored.values_data()[i]) -
                                                        static_cast<nk_f64_t>(cloud.values_data()[i])));
        stats.expect(trip_error <= trip_bound, "an unpacked point lands off its input");

        bool padded = true;
        for (std::size_t axis = 0; axis != 3; ++axis)
            for (std::size_t i = n; i != kernel_pack.axis_stride; ++i) {
                raw_t const value = kernel_pack.axes[axis * kernel_pack.axis_stride + i];
                unsigned char bytes_of[sizeof(raw_t)];
                std::memcpy(bytes_of, &value, sizeof(raw_t));
                for (unsigned char byte : bytes_of) padded &= byte == 0;
            }
        stats.expect(padded, "an axis is not zero-padded");
    }
    return stats;
}

/** Kabsch RMSD and rotation of @p b onto @p a, matched by index, in @p reference_type_. */
template <typename reference_type_, typename scalar_type_>
reference_type_ kabsch_reference(nk::vector<scalar_type_> const &a, nk::vector<scalar_type_> const &b,
                                 reference_type_ *rotation) {
    std::size_t const n = a.size_values() / 3;
    reference_type_ const count(static_cast<nk_f64_t>(n));
    reference_type_ a_centroid[3] {}, b_centroid[3] {}, cross[9] {};
    auto const at = [](nk::vector<scalar_type_> const &cloud, std::size_t i, std::size_t axis) {
        return reference_type_(static_cast<nk_f64_t>(cloud.values_data()[i * 3 + axis]));
    };
    for (std::size_t i = 0; i != n; ++i)
        for (std::size_t axis = 0; axis != 3; ++axis)
            a_centroid[axis] = a_centroid[axis] + at(a, i, axis), b_centroid[axis] = b_centroid[axis] + at(b, i, axis);
    for (std::size_t axis = 0; axis != 3; ++axis)
        a_centroid[axis] = a_centroid[axis] / count, b_centroid[axis] = b_centroid[axis] / count;
    for (std::size_t i = 0; i != n; ++i)
        for (std::size_t b_axis = 0; b_axis != 3; ++b_axis)
            for (std::size_t a_axis = 0; a_axis != 3; ++a_axis)
                cross[b_axis * 3 + a_axis] = cross[b_axis * 3 + a_axis] + (at(b, i, b_axis) - b_centroid[b_axis]) *
                                                                              (at(a, i, a_axis) - a_centroid[a_axis]);
    reference_type_ svd_u[9], svd_v[9];
    nk::svd3x3_(cross, svd_u, svd_v);
    nk::rotation_from_svd_(svd_u, svd_v, rotation);
    if (nk::det3x3_(rotation) < reference_type_(0.0)) {
        for (std::size_t row = 0; row != 3; ++row) svd_v[row * 3 + 2] = reference_type_(0.0) - svd_v[row * 3 + 2];
        nk::rotation_from_svd_(svd_u, svd_v, rotation);
    }
    reference_type_ sum {};
    for (std::size_t i = 0; i != n; ++i)
        for (std::size_t row = 0; row != 3; ++row) {
            reference_type_ residual = at(a, i, row) - a_centroid[row];
            for (std::size_t column = 0; column != 3; ++column)
                residual = residual - rotation[row * 3 + column] * (at(b, i, column) - b_centroid[column]);
            sum = sum + residual * residual;
        }
    return (sum / count).sqrt();
}

/** Kabsch survives as the special case of Wasserstein-Procrustes: between a cloud and its rigid
 *  motion, with noise far below its closest pair distance, the identity plan is optimal, so the
 *  result is the Kabsch RMSD and rotation. */
template <typename scalar_type_>
error_stats_t test_shapes_kabsch(settings_t const &settings, typename scalar_type_::shapes_pack_size_kernel_t pack_size,
                                 typename scalar_type_::shapes_pack_kernel_t pack,
                                 typename scalar_type_::shapes_workspace_size_kernel_t workspace_size,
                                 typename scalar_type_::shapes_packed_workspace_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using distance_t = typename scalar_t::shapes_distance_t;
    using reference_t = reference_for<scalar_t>;
    using cloud_t = nk::shapes_cloud_<scalar_t>;

    error_stats_t stats(comparison_family_t::bounded_k);
    std::mt19937 generator(settings.seed.value);
    std::normal_distribution<nk_f64_t> normal(0, 1);
    std::uniform_real_distribution<nk_f64_t> shift(-4, 4);
    std::size_t const n = settings.shape_points;
    // Lattice points may coincide or sit close, where noise lets a swapped plan beat the identity
    auto const closest_pair_distance = [n](nk::vector<scalar_t> const &cloud) {
        nk_f64_t closest = std::numeric_limits<nk_f64_t>::infinity();
        for (std::size_t i = 0; i != n; ++i)
            for (std::size_t k = i + 1; k != n; ++k) {
                nk_f64_t squared = 0;
                for (std::size_t axis = 0; axis != 3; ++axis) {
                    nk_f64_t const difference = static_cast<nk_f64_t>(cloud.values_data()[i * 3 + axis]) -
                                                static_cast<nk_f64_t>(cloud.values_data()[k * 3 + axis]);
                    squared += difference * difference;
                }
                closest = std::min(closest, squared);
            }
        return std::sqrt(closest);
    };
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        std::vector<nk::vector<scalar_t>> clouds;
        clouds.push_back(stretched_cloud<scalar_t>(generator, n));
        nk_f64_t const noise = std::min(std::ldexp(1.0, -7), closest_pair_distance(clouds[0]) / 32);
        nk::f64_t quaternion[4], rotation_matrix[9], length = 0;
        for (nk::f64_t &value : quaternion) value = normal(generator), length = length + value * value;
        for (nk::f64_t &value : quaternion) value = value / length.sqrt();
        nk::quaternion_to_mat3x3_(quaternion, rotation_matrix);
        nk_f64_t motion[9];
        for (std::size_t i = 0; i != 9; ++i) motion[i] = rotation_matrix[i];
        nk_f64_t const translation[3] = {shift(generator), shift(generator), shift(generator)};
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        clouds.push_back(mapped_cloud(clouds[0], order, [&](nk_f64_t x, nk_f64_t y, nk_f64_t z) {
            std::array<nk_f64_t, 3> image;
            for (std::size_t axis = 0; axis != 3; ++axis)
                image[axis] = motion[axis * 3 + 0] * x + motion[axis * 3 + 1] * y + motion[axis * 3 + 2] * z +
                              translation[axis] + noise * normal(generator);
            return image;
        }));
        shapes_packs_t const packs = pack_clouds(clouds, pack_size, pack, stats);
        nk_size_t bytes = 0;
        stats.expect(workspace_size(n, n, &bytes));
        auto workspace = make_vector<char>(bytes);
        typename distance_t::raw_t result;
        nk_f32_t rotation[9];
        if (!stats.expect(kernel(&packs.pointers[0], 1, &packs.pointers[1], 1, &result, sizeof(result), rotation,
                                 workspace.raw_values_data(), bytes, nullptr)))
            continue;

        reference_t reference_rotation[9];
        reference_t const reference = kabsch_reference(clouds[0], clouds[1], reference_rotation);
        cloud_t const a(packs.pointers[0]), b(packs.pointers[1]);
        nk_f64_t const a_radius = a.header->max_radius, b_radius = b.header->max_radius;
        // Packing moves each point by up to its rounding, and the RMSD by as much
        nk_f64_t const displacement = std::sqrt(3.0) * (half_ulp<scalar_t>(a_radius) + half_ulp<scalar_t>(b_radius));
        stats.accumulate_bounded(distance_t::from_raw(result), reference,
                                 displacement + shapes_distance_bound<scalar_t, wassersteinprocrustes_k>(
                                                    n, n, a_radius, b_radius, static_cast<nk_f64_t>(reference)));

        // The rotation turns by about the cross-covariance error over its smaller singular values
        auto const *spreads = a.header->spreads;
        nk_f64_t const rotation_bound = 4 * std::sqrt(spreads[0] + spreads[1] + spreads[2]) * displacement /
                                            (spreads[1] + spreads[2]) +
                                        shapes_rotation_bound<scalar_t, wassersteinprocrustes_k>();
        nk_f64_t rotation_error = 0;
        for (std::size_t i = 0; i != 9; ++i)
            rotation_error = std::max(rotation_error,
                                      std::fabs(rotation[i] - static_cast<nk_f64_t>(reference_rotation[i])));
        stats.expect(rotation_error <= rotation_bound, "the rotation differs from Kabsch's");
    }
    return stats;
}

template <typename scalar_type_>
using rmsds_packed_kernel_t = nk_status_t (*)(void const *const *, nk_size_t, void const *const *, nk_size_t,
                                              typename scalar_type_::shapes_distance_t::raw_t *, nk_size_t,
                                              nk_stream_t);

/** RMS distance of a frame's points from its centroid, the spread its centered moments see. */
template <typename scalar_type_>
nk_f64_t centroid_spread(nk::vector<scalar_type_> const &frame) noexcept {
    std::size_t const n = frame.size_values() / 3;
    if (!n) return 0;
    nk_f64_t centroid[3] = {0, 0, 0}, sum = 0;
    for (std::size_t i = 0; i != n * 3; ++i) centroid[i % 3] += static_cast<nk_f64_t>(frame.values_data()[i]) / n;
    for (std::size_t i = 0; i != n * 3; ++i) {
        nk_f64_t const delta = static_cast<nk_f64_t>(frame.values_data()[i]) - centroid[i % 3];
        sum += delta * delta;
    }
    return std::sqrt(sum / n);
}

/** The largest magnitude of a frame's coordinates. */
template <typename scalar_type_>
nk_f64_t largest_coordinate(nk::vector<scalar_type_> const &frame) noexcept {
    nk_f64_t largest = 0;
    for (std::size_t i = 0; i != frame.size_values(); ++i)
        largest = std::max(largest, std::fabs(static_cast<nk_f64_t>(frame.values_data()[i])));
    return largest;
}

/**
 *  @brief Bounds the error of an RMSD kernel by the arithmetic it performs.
 *
 *  Packing rounds each centered and rotated coordinate once into the input type, moving every point
 *  by at most the unit roundoff of its norm, and so the RMSD by that share of @p spread, the sum of
 *  both frames' RMS distances from their centroids. The @c f16 fold sums the moments in f32, adding
 *  n roundings of their total, n × @p spread², which the square root turns into its own root or its
 *  ratio to the RMSD, as the one-to-one tests bounded it. The @c f32 residual pass runs in f64,
 *  where a centroid n roundings off shifts each frame by up to that share of @p largest, and the
 *  sums add n roundings of the spread.
 */
template <typename scalar_type_>
nk_f64_t rmsd_error_bound(std::size_t n, nk_f64_t rmsd, nk_f64_t spread, nk_f64_t largest) noexcept {
    nk_f64_t const input_roundoff = std::ldexp(1.0, -static_cast<int>(scalar_type_::mantissa_bits()) - 1);
    if constexpr (std::is_same_v<scalar_type_, f16_t>) {
        nk_f64_t const ssd_error = n * nk_accumulation_error_bound(nk_f32_k) * spread * spread;
        nk_f64_t const root_error = rmsd > 0 ? std::min(std::sqrt(ssd_error), ssd_error / rmsd) : std::sqrt(ssd_error);
        return input_roundoff * spread + root_error;
    }
    else {
        nk_f64_t const wide_roundoff = std::ldexp(1.0, -53);
        return input_roundoff * spread + (n + 2) * wide_roundoff * largest + 2 * (n + 16) * wide_roundoff * spread;
    }
}

/**
 *  @brief Checks a superposed RMSD kernel over shape packs against Kabsch in the reference type.
 *
 *  Unrelated frames, every other trial far from the origin, and rigid motions of one frame with
 *  noise compare within @ref rmsd_error_bound. Empty frames score zero, and the kernel refuses
 *  frames of other sizes.
 */
template <typename scalar_type_>
error_stats_t test_rmsds_packed(settings_t const &settings, typename scalar_type_::shapes_pack_size_kernel_t pack_size,
                                typename scalar_type_::shapes_pack_kernel_t pack,
                                rmsds_packed_kernel_t<scalar_type_> kernel) {
    using scalar_t = scalar_type_;
    using distance_t = typename scalar_t::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::bounded_k);
    std::mt19937 generator(settings.seed.value);
    std::normal_distribution<nk_f64_t> normal(0, 1);
    std::uniform_real_distribution<nk_f64_t> shift(-4, 4);
    std::size_t const n = settings.shape_points;
    auto const run = [&](shapes_packs_t const &rows, shapes_packs_t const &columns,
                         std::vector<raw_distance_t> &results) {
        results.assign(rows.size() * columns.size(), raw_distance_t(-1));
        return kernel(rows.pointers.data(), rows.size(), columns.pointers.data(), columns.size(), results.data(),
                      columns.size() * sizeof(raw_distance_t), nullptr);
    };
    auto const check_pair = [&](nk::vector<scalar_t> const &row_frame, nk::vector<scalar_t> const &column_frame,
                                raw_distance_t result) {
        reference_t rotation[9];
        reference_t const reference = kabsch_reference(row_frame, column_frame, rotation);
        nk_f64_t const spread = centroid_spread(row_frame) + centroid_spread(column_frame);
        nk_f64_t const largest = largest_coordinate(row_frame) + largest_coordinate(column_frame);
        std::size_t const points = row_frame.size_values() / 3;
        stats.accumulate_bounded(distance_t::from_raw(result), reference,
                                 rmsd_error_bound<scalar_t>(points, static_cast<nk_f64_t>(reference), spread, largest));
    };
    std::vector<raw_distance_t> results;

    // Empty frames score zero, an empty grid succeeds, and frames of other sizes are refused
    {
        std::vector<nk::vector<scalar_t>> empty, mixed;
        empty.push_back(make_vector<scalar_t>(0)), empty.push_back(make_vector<scalar_t>(0));
        mixed.push_back(random_cloud<scalar_t>(settings, generator, n, 0));
        mixed.push_back(random_cloud<scalar_t>(settings, generator, n + 1, 0));
        shapes_packs_t const empty_packs = pack_clouds(empty, pack_size, pack, stats);
        shapes_packs_t const mixed_packs = pack_clouds(mixed, pack_size, pack, stats);
        if (stats.expect(run(empty_packs, empty_packs, results)))
            stats.expect(std::all_of(results.begin(), results.end(), [](raw_distance_t r) { return r == 0; }),
                         "empty frames score above zero");
        stats.expect(run(shapes_packs_t {}, mixed_packs, results) == nk_success_k, "an empty grid failed");
        stats.expect(run(mixed_packs, mixed_packs, results) == nk_unexpected_dimensions_k,
                     "frames of other sizes were not refused");
    }

    // Two-point and collinear frames repeat the largest root of the quaternion polynomial exactly:
    // a line against the same steps along another line, and against fresh steps
    for (std::size_t points : {std::size_t(2), n}) {
        std::uniform_real_distribution<nk_f64_t> along(-8, 8);
        std::vector<nk_f64_t> steps(points), fresh_steps(points);
        for (nk_f64_t &step : steps) step = along(generator);
        for (nk_f64_t &step : fresh_steps) step = along(generator);
        auto const line = [&](nk_f64_t const *direction, nk_f64_t offset, std::vector<nk_f64_t> const &line_steps) {
            auto frame = make_vector<scalar_t>(points * 3);
            for (std::size_t i = 0; i != points; ++i)
                for (std::size_t axis = 0; axis != 3; ++axis)
                    frame.values_data()[i * 3 + axis] = scalar_t(offset + line_steps[i] * direction[axis]);
            return frame;
        };
        nk_f64_t const first[3] = {0.48, 0.6, 0.64}, second[3] = {0, 0.6, 0.8};
        std::vector<nk::vector<scalar_t>> row_lines, column_lines;
        row_lines.push_back(line(first, 0, steps));
        column_lines.push_back(line(second, 64, steps)), column_lines.push_back(line(second, 3, fresh_steps));
        shapes_packs_t const rows = pack_clouds(row_lines, pack_size, pack, stats);
        shapes_packs_t const columns = pack_clouds(column_lines, pack_size, pack, stats);
        if (stats.expect(run(rows, columns, results)))
            for (std::size_t column = 0; column != columns.size(); ++column)
                check_pair(row_lines[0], column_lines[column], results[column]);
    }

    bool far_from_origin = false;
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        nk_f64_t const offset = (far_from_origin = !far_from_origin) ? 64 : 0;
        std::vector<nk::vector<scalar_t>> row_frames, column_frames;
        for (std::size_t i = 0; i != 2; ++i)
            row_frames.push_back(random_cloud<scalar_t>(settings, generator, n, offset));
        for (std::size_t i = 0; i != 3; ++i)
            column_frames.push_back(random_cloud<scalar_t>(settings, generator, n, offset));
        shapes_packs_t const rows = pack_clouds(row_frames, pack_size, pack, stats);
        shapes_packs_t const columns = pack_clouds(column_frames, pack_size, pack, stats);
        if (!stats.expect(run(rows, columns, results))) continue;
        for (std::size_t row = 0; row != rows.size(); ++row)
            for (std::size_t column = 0; column != columns.size(); ++column)
                check_pair(row_frames[row], column_frames[column], results[row * columns.size() + column]);

        // Rigid motions of one frame, with noise well below its spread, align to nearly zero
        std::vector<nk::vector<scalar_t>> bases, moved;
        bases.push_back(random_cloud<scalar_t>(settings, generator, n, offset));
        nk::f64_t quaternion[4], turn[9], length = 0;
        for (nk::f64_t &value : quaternion) value = normal(generator), length = length + value * value;
        for (nk::f64_t &value : quaternion) value = value / length.sqrt();
        nk::quaternion_to_mat3x3_(quaternion, turn);
        nk_f64_t const translation[3] = {shift(generator), shift(generator), shift(generator)};
        nk_f64_t const noise = std::ldexp(centroid_spread(bases[0]), -10);
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        moved.push_back(mapped_cloud(bases[0], order, [&](nk_f64_t x, nk_f64_t y, nk_f64_t z) {
            std::array<nk_f64_t, 3> image;
            for (std::size_t axis = 0; axis != 3; ++axis)
                image[axis] = static_cast<nk_f64_t>(turn[axis * 3 + 0]) * x +
                              static_cast<nk_f64_t>(turn[axis * 3 + 1]) * y +
                              static_cast<nk_f64_t>(turn[axis * 3 + 2]) * z + translation[axis] +
                              noise * normal(generator);
            return image;
        }));
        shapes_packs_t const base_packs = pack_clouds(bases, pack_size, pack, stats);
        shapes_packs_t const moved_packs = pack_clouds(moved, pack_size, pack, stats);
        if (stats.expect(run(base_packs, moved_packs, results))) check_pair(bases[0], moved[0], results[0]);
    }
    return stats;
}

/**
 *  @brief Checks that a copy of a cloud with tied eigenvalues packs into its frame up to 8 flips.
 *
 *  A tetrahedron, methane and a cube tie all three eigenvalues and benzene ties a plane. Random
 *  clouds whitened to a tied sphere or plane join them but for @c f16, whose rounding leaves their
 *  weakest harmonics within reach of the tolerance. Every turned, moved and noisy copy scores
 *  within its displacement under the best flip, which a frame turned within a tied plane misses by
 *  a share of the radius, and its RMSD stays within @ref rmsd_error_bound.
 */
template <typename scalar_type_>
error_stats_t test_shapes_ties(settings_t const &settings, typename scalar_type_::shapes_pack_size_kernel_t pack_size,
                               typename scalar_type_::shapes_pack_kernel_t pack,
                               rmsds_packed_kernel_t<scalar_type_> rmsds) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using distance_t = typename scalar_t::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::bounded_k);
    std::mt19937 generator(settings.seed.value);
    std::normal_distribution<nk_f64_t> normal(0, 1);
    std::uniform_real_distribution<nk_f64_t> shift(-3, 3), noise(-1e-8, 1e-8);
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0);

    auto const cloud_of = [](std::vector<nk_f64_t> const &coordinates) {
        auto cloud = make_vector<scalar_t>(coordinates.size());
        for (std::size_t i = 0; i != coordinates.size(); ++i) cloud.values_data()[i] = scalar_t(coordinates[i]);
        return cloud;
    };
    // Centered columns orthonormalized and scaled back to unit spread make the covariance 𝐼 / 3
    auto const whitened = [&](std::size_t points, nk_f64_t last_axis_stretch) {
        std::vector<nk_f64_t> columns(points * 3);
        for (nk_f64_t &value : columns) value = normal(generator);
        for (std::size_t axis = 0; axis != 3; ++axis) {
            nk_f64_t *column = &columns[axis * points];
            nk_f64_t mean = std::accumulate(column, column + points, 0.0) / points;
            for (std::size_t i = 0; i != points; ++i) column[i] -= mean;
            for (std::size_t previous = 0; previous != axis; ++previous) {
                nk_f64_t const *other = &columns[previous * points];
                nk_f64_t const overlap = std::inner_product(column, column + points, other, 0.0);
                for (std::size_t i = 0; i != points; ++i) column[i] -= overlap * other[i];
            }
            nk_f64_t const length = std::sqrt(std::inner_product(column, column + points, column, 0.0));
            for (std::size_t i = 0; i != points; ++i) column[i] *= std::sqrt(points / 3.0) / length;
        }
        std::vector<nk_f64_t> coordinates(points * 3);
        for (std::size_t i = 0; i != points; ++i)
            for (std::size_t axis = 0; axis != 3; ++axis)
                coordinates[i * 3 + axis] = columns[axis * points + i] * (axis == 2 ? last_axis_stretch : 1.0);
        return cloud_of(coordinates);
    };
    nk_f64_t const methane = 1.09 / std::sqrt(3.0);
    std::vector<nk::vector<scalar_t>> clouds;
    clouds.push_back(cloud_of({1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1}));
    clouds.push_back(cloud_of({0, 0, 0, methane, methane, methane, methane, -methane, -methane, -methane, methane,
                               -methane, -methane, -methane, methane}));
    clouds.push_back(cloud_of({-1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1}));
    {
        std::vector<nk_f64_t> benzene;
        for (std::size_t atom = 0; atom != 6; ++atom)
            for (nk_f64_t radius : {1.39, 2.48}) {
                nk_f64_t const angle = atom * std::acos(-1.0) / 3;
                benzene.insert(benzene.end(), {radius * std::cos(angle), radius * std::sin(angle), 0});
            }
        clouds.push_back(cloud_of(benzene));
    }
    if constexpr (!std::is_same_v<scalar_t, f16_t>)
        clouds.push_back(whitened(settings.shape_points, 1)), clouds.push_back(whitened(settings.shape_points, 0.5));

    auto const packed_point = [](void const *packed, std::size_t point, std::size_t axis) {
        nk_size_t stride = 0;
        raw_t const *axes;
        if constexpr (std::is_same_v<scalar_t, f16_t>) axes = nk_shapes_packed_axes_f16_serial_(packed, &stride);
        else if constexpr (std::is_same_v<scalar_t, f32_t>) axes = nk_shapes_packed_axes_f32_serial_(packed, &stride);
        else axes = nk_shapes_packed_axes_f64_serial_(packed, &stride);
        return static_cast<nk_f64_t>(scalar_t::from_raw(axes[axis * stride + point]));
    };
    // The larger of the Chamfer and Hausdorff distances between two packs, least over the 8 flips
    auto const flip_distance = [&](void const *a, void const *b, std::size_t points) {
        nk_f64_t least = std::numeric_limits<nk_f64_t>::infinity();
        for (unsigned flip = 0; flip != 8; ++flip) {
            nk_f64_t sum = 0, maximum = 0;
            for (void const *outer : {a, b})
                for (std::size_t i = 0; i != points; ++i) {
                    nk_f64_t nearest = std::numeric_limits<nk_f64_t>::infinity();
                    for (std::size_t j = 0; j != points; ++j) {
                        nk_f64_t squared = 0;
                        for (std::size_t axis = 0; axis != 3; ++axis) {
                            nk_f64_t const sign = (flip >> axis) & 1 ? -1 : 1;
                            nk_f64_t const difference = outer == a
                                                            ? packed_point(a, i, axis) - sign * packed_point(b, j, axis)
                                                            : sign * packed_point(b, i, axis) -
                                                                  packed_point(a, j, axis);
                            squared += difference * difference;
                        }
                        nearest = std::min(nearest, squared);
                    }
                    sum += std::sqrt(nearest), maximum = std::max(maximum, std::sqrt(nearest));
                }
            least = std::min(least, std::max(sum / (2 * points), maximum));
        }
        return least;
    };

    std::vector<raw_distance_t> results(1);
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();)
        for (auto const &cloud : clouds) {
            std::size_t const points = cloud.size_values() / 3;
            nk::f64_t quaternion[4], turn[9], length = 0;
            for (nk::f64_t &value : quaternion) value = normal(generator), length = length + value * value;
            for (nk::f64_t &value : quaternion) value = value / length.sqrt();
            nk::quaternion_to_mat3x3_(quaternion, turn);
            nk_f64_t const translation[3] = {shift(generator), shift(generator), shift(generator)};
            std::vector<std::size_t> order(points);
            std::iota(order.begin(), order.end(), 0);
            std::vector<nk::vector<scalar_t>> originals, copies;
            originals.push_back(mapped_cloud(
                cloud, order, [](nk_f64_t x, nk_f64_t y, nk_f64_t z) { return std::array<nk_f64_t, 3> {x, y, z}; }));
            copies.push_back(mapped_cloud(cloud, order, [&](nk_f64_t x, nk_f64_t y, nk_f64_t z) {
                std::array<nk_f64_t, 3> image;
                for (std::size_t axis = 0; axis != 3; ++axis)
                    image[axis] = static_cast<nk_f64_t>(turn[axis * 3 + 0]) * x +
                                  static_cast<nk_f64_t>(turn[axis * 3 + 1]) * y +
                                  static_cast<nk_f64_t>(turn[axis * 3 + 2]) * z + translation[axis] + noise(generator);
                return image;
            }));
            shapes_packs_t const original_packs = pack_clouds(originals, pack_size, pack, stats);
            shapes_packs_t const copy_packs = pack_clouds(copies, pack_size, pack, stats);
            nk_f64_t const radius = largest_coordinate(cloud) * std::sqrt(3.0);
            nk_f64_t const largest_input = largest_coordinate(copies[0]);
            // The copy's noise and input rounding, then each side's packing, a few times over for
            // the harmonics that turn the tied plane
            nk_f64_t const displacement = std::sqrt(3.0) * (1e-8 + half_ulp<scalar_t>(largest_input) +
                                                            2 * half_ulp<scalar_t>(radius)) +
                                          64 * roundoff * (radius + largest_input);
            stats.expect(flip_distance(original_packs.pointers[0], copy_packs.pointers[0], points) <= 64 * displacement,
                         "a copy of a tied cloud packs into another frame");
            if (stats.expect(rmsds(original_packs.pointers.data(), 1, copy_packs.pointers.data(), 1, results.data(),
                                   sizeof(raw_distance_t), nullptr))) {
                reference_t rotation[9];
                reference_t const reference = kabsch_reference(originals[0], copies[0], rotation);
                nk_f64_t const spread = centroid_spread(originals[0]) + centroid_spread(copies[0]);
                nk_f64_t const largest = largest_coordinate(originals[0]) + largest_coordinate(copies[0]);
                stats.accumulate_bounded(
                    distance_t::from_raw(results[0]), reference,
                    rmsd_error_bound<scalar_t>(points, static_cast<nk_f64_t>(reference), spread, largest));
            }
        }
    return stats;
}

void test_shapes(error_stats_section_t &check) {

    check.section("Shapes Serial", nk_cap_serial_k);
    check("shapes_pack_f64_serial", test_shapes_pack<f64_t, pack_k>, nk_shapes_pack_size_f64_serial,
          nk_shapes_pack_f64_serial, nk_shapes_packed_shape_f64_serial, nk_shapes_unpack_f64_serial);
    check("shapes_pack_f32_serial", test_shapes_pack<f32_t, pack_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_shapes_packed_shape_f32_serial, nk_shapes_unpack_f32_serial);
    check("shapes_pack_f16_serial", test_shapes_pack<f16_t, pack_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_shapes_packed_shape_f16_serial, nk_shapes_unpack_f16_serial);
    check("shapes_pack_normalized_f64_serial", test_shapes_pack<f64_t, pack_normalized_k>,
          nk_shapes_pack_size_f64_serial, nk_shapes_pack_normalized_f64_serial, nk_shapes_packed_shape_f64_serial,
          nk_shapes_unpack_f64_serial);
    check("shapes_pack_normalized_f32_serial", test_shapes_pack<f32_t, pack_normalized_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_normalized_f32_serial, nk_shapes_packed_shape_f32_serial,
          nk_shapes_unpack_f32_serial);
    check("shapes_pack_normalized_f16_serial", test_shapes_pack<f16_t, pack_normalized_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_normalized_f16_serial, nk_shapes_packed_shape_f16_serial,
          nk_shapes_unpack_f16_serial);
    check("chamfers_packed_f64_serial", test_shapes_packed<f64_t, chamfers_k>, nk_shapes_pack_size_f64_serial,
          nk_shapes_pack_f64_serial, nk_chamfers_packed_f64_serial);
    check("chamfers_packed_f32_serial", test_shapes_packed<f32_t, chamfers_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_chamfers_packed_f32_serial);
    check("chamfers_packed_f16_serial", test_shapes_packed<f16_t, chamfers_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_chamfers_packed_f16_serial);
    check("hausdorffs_packed_f64_serial", test_shapes_packed<f64_t, hausdorffs_k>, nk_shapes_pack_size_f64_serial,
          nk_shapes_pack_f64_serial, nk_hausdorffs_packed_f64_serial);
    check("hausdorffs_packed_f32_serial", test_shapes_packed<f32_t, hausdorffs_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_hausdorffs_packed_f32_serial);
    check("hausdorffs_packed_f16_serial", test_shapes_packed<f16_t, hausdorffs_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_hausdorffs_packed_f16_serial);
    check("wassersteinprocrustes_packed_f64_serial", test_shapes_packed_workspace<f64_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f64_serial, nk_shapes_pack_f64_serial, nk_wassersteinprocrustes_workspace_size_f64_serial,
          nk_wassersteinprocrustes_packed_f64_serial);
    check("wassersteinprocrustes_packed_f32_serial", test_shapes_packed_workspace<f32_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_serial);
    check("wassersteinprocrustes_packed_f16_serial", test_shapes_packed_workspace<f16_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_serial);
    check("wassersteinprocrustes_kabsch_f64_serial", test_shapes_kabsch<f64_t>, nk_shapes_pack_size_f64_serial,
          nk_shapes_pack_f64_serial, nk_wassersteinprocrustes_workspace_size_f64_serial,
          nk_wassersteinprocrustes_packed_f64_serial);
    check("wassersteinprocrustes_kabsch_f32_serial", test_shapes_kabsch<f32_t>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_serial);
    check("wassersteinprocrustes_kabsch_f16_serial", test_shapes_kabsch<f16_t>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_serial);
    check("gromovwassersteins_packed_f64_serial", test_shapes_packed_workspace<f64_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f64_serial, nk_shapes_pack_f64_serial, nk_gromovwassersteins_workspace_size_f64_serial,
          nk_gromovwassersteins_packed_f64_serial);
    check("gromovwassersteins_packed_f32_serial", test_shapes_packed_workspace<f32_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
          nk_gromovwassersteins_packed_f32_serial);
    check("gromovwassersteins_packed_f16_serial", test_shapes_packed_workspace<f16_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
          nk_gromovwassersteins_packed_f16_serial);
    check("shapes_ties_f64_serial", test_shapes_ties<f64_t>, nk_shapes_pack_size_f64_serial, nk_shapes_pack_f64_serial,
          nk_rmsds_packed_f64_serial);
    check("shapes_ties_f32_serial", test_shapes_ties<f32_t>, nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial,
          nk_rmsds_packed_f32_serial);
    check("shapes_ties_f16_serial", test_shapes_ties<f16_t>, nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial,
          nk_rmsds_packed_f16_serial);
    check("rmsds_packed_f64_serial", test_rmsds_packed<f64_t>, nk_shapes_pack_size_f64_serial,
          nk_shapes_pack_f64_serial, nk_rmsds_packed_f64_serial);
    check("rmsds_packed_f32_serial", test_rmsds_packed<f32_t>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_rmsds_packed_f32_serial);
    check("rmsds_packed_f16_serial", test_rmsds_packed<f16_t>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_rmsds_packed_f16_serial);

#if !NUMKONG_HEADER_ONLY
    check.section("Shapes Runtime Dispatch", nk_cap_serial_k);
    check("shapes_pack_f64", test_shapes_pack<f64_t, pack_k>, cpu_best<nk_shapes_pack_size_f64_best>,
          cpu_best<nk_shapes_pack_f64_best>, cpu_best<nk_shapes_packed_shape_f64_best>,
          cpu_best<nk_shapes_unpack_f64_best>);
    check("shapes_pack_f32", test_shapes_pack<f32_t, pack_k>, cpu_best<nk_shapes_pack_size_f32_best>,
          cpu_best<nk_shapes_pack_f32_best>, cpu_best<nk_shapes_packed_shape_f32_best>,
          cpu_best<nk_shapes_unpack_f32_best>);
    check("shapes_pack_f16", test_shapes_pack<f16_t, pack_k>, cpu_best<nk_shapes_pack_size_f16_best>,
          cpu_best<nk_shapes_pack_f16_best>, cpu_best<nk_shapes_packed_shape_f16_best>,
          cpu_best<nk_shapes_unpack_f16_best>);
    check("shapes_pack_normalized_f64", test_shapes_pack<f64_t, pack_normalized_k>,
          cpu_best<nk_shapes_pack_size_f64_best>, cpu_best<nk_shapes_pack_normalized_f64_best>,
          cpu_best<nk_shapes_packed_shape_f64_best>, cpu_best<nk_shapes_unpack_f64_best>);
    check("shapes_pack_normalized_f32", test_shapes_pack<f32_t, pack_normalized_k>,
          cpu_best<nk_shapes_pack_size_f32_best>, cpu_best<nk_shapes_pack_normalized_f32_best>,
          cpu_best<nk_shapes_packed_shape_f32_best>, cpu_best<nk_shapes_unpack_f32_best>);
    check("shapes_pack_normalized_f16", test_shapes_pack<f16_t, pack_normalized_k>,
          cpu_best<nk_shapes_pack_size_f16_best>, cpu_best<nk_shapes_pack_normalized_f16_best>,
          cpu_best<nk_shapes_packed_shape_f16_best>, cpu_best<nk_shapes_unpack_f16_best>);
    check("chamfers_packed_f64", test_shapes_packed<f64_t, chamfers_k>, cpu_best<nk_shapes_pack_size_f64_best>,
          cpu_best<nk_shapes_pack_f64_best>, cpu_best<nk_chamfers_packed_f64_best>);
    check("chamfers_packed_f32", test_shapes_packed<f32_t, chamfers_k>, cpu_best<nk_shapes_pack_size_f32_best>,
          cpu_best<nk_shapes_pack_f32_best>, cpu_best<nk_chamfers_packed_f32_best>);
    check("chamfers_packed_f16", test_shapes_packed<f16_t, chamfers_k>, cpu_best<nk_shapes_pack_size_f16_best>,
          cpu_best<nk_shapes_pack_f16_best>, cpu_best<nk_chamfers_packed_f16_best>);
    check("hausdorffs_packed_f64", test_shapes_packed<f64_t, hausdorffs_k>, cpu_best<nk_shapes_pack_size_f64_best>,
          cpu_best<nk_shapes_pack_f64_best>, cpu_best<nk_hausdorffs_packed_f64_best>);
    check("hausdorffs_packed_f32", test_shapes_packed<f32_t, hausdorffs_k>, cpu_best<nk_shapes_pack_size_f32_best>,
          cpu_best<nk_shapes_pack_f32_best>, cpu_best<nk_hausdorffs_packed_f32_best>);
    check("hausdorffs_packed_f16", test_shapes_packed<f16_t, hausdorffs_k>, cpu_best<nk_shapes_pack_size_f16_best>,
          cpu_best<nk_shapes_pack_f16_best>, cpu_best<nk_hausdorffs_packed_f16_best>);
    check("wassersteinprocrustes_packed_f64", test_shapes_packed_workspace<f64_t, wassersteinprocrustes_k>,
          cpu_best<nk_shapes_pack_size_f64_best>, cpu_best<nk_shapes_pack_f64_best>,
          cpu_best<nk_wassersteinprocrustes_workspace_size_f64_best>,
          cpu_best<nk_wassersteinprocrustes_packed_f64_best>);
    check("wassersteinprocrustes_packed_f32", test_shapes_packed_workspace<f32_t, wassersteinprocrustes_k>,
          cpu_best<nk_shapes_pack_size_f32_best>, cpu_best<nk_shapes_pack_f32_best>,
          cpu_best<nk_wassersteinprocrustes_workspace_size_f32_best>,
          cpu_best<nk_wassersteinprocrustes_packed_f32_best>);
    check("wassersteinprocrustes_packed_f16", test_shapes_packed_workspace<f16_t, wassersteinprocrustes_k>,
          cpu_best<nk_shapes_pack_size_f16_best>, cpu_best<nk_shapes_pack_f16_best>,
          cpu_best<nk_wassersteinprocrustes_workspace_size_f16_best>,
          cpu_best<nk_wassersteinprocrustes_packed_f16_best>);
    check("wassersteinprocrustes_kabsch_f64", test_shapes_kabsch<f64_t>, cpu_best<nk_shapes_pack_size_f64_best>,
          cpu_best<nk_shapes_pack_f64_best>, cpu_best<nk_wassersteinprocrustes_workspace_size_f64_best>,
          cpu_best<nk_wassersteinprocrustes_packed_f64_best>);
    check("wassersteinprocrustes_kabsch_f32", test_shapes_kabsch<f32_t>, cpu_best<nk_shapes_pack_size_f32_best>,
          cpu_best<nk_shapes_pack_f32_best>, cpu_best<nk_wassersteinprocrustes_workspace_size_f32_best>,
          cpu_best<nk_wassersteinprocrustes_packed_f32_best>);
    check("wassersteinprocrustes_kabsch_f16", test_shapes_kabsch<f16_t>, cpu_best<nk_shapes_pack_size_f16_best>,
          cpu_best<nk_shapes_pack_f16_best>, cpu_best<nk_wassersteinprocrustes_workspace_size_f16_best>,
          cpu_best<nk_wassersteinprocrustes_packed_f16_best>);
    check("gromovwassersteins_packed_f64", test_shapes_packed_workspace<f64_t, gromovwassersteins_k>,
          cpu_best<nk_shapes_pack_size_f64_best>, cpu_best<nk_shapes_pack_f64_best>,
          cpu_best<nk_gromovwassersteins_workspace_size_f64_best>, cpu_best<nk_gromovwassersteins_packed_f64_best>);
    check("gromovwassersteins_packed_f32", test_shapes_packed_workspace<f32_t, gromovwassersteins_k>,
          cpu_best<nk_shapes_pack_size_f32_best>, cpu_best<nk_shapes_pack_f32_best>,
          cpu_best<nk_gromovwassersteins_workspace_size_f32_best>, cpu_best<nk_gromovwassersteins_packed_f32_best>);
    check("gromovwassersteins_packed_f16", test_shapes_packed_workspace<f16_t, gromovwassersteins_k>,
          cpu_best<nk_shapes_pack_size_f16_best>, cpu_best<nk_shapes_pack_f16_best>,
          cpu_best<nk_gromovwassersteins_workspace_size_f16_best>, cpu_best<nk_gromovwassersteins_packed_f16_best>);
    check("rmsds_packed_f64", test_rmsds_packed<f64_t>, cpu_best<nk_shapes_pack_size_f64_best>,
          cpu_best<nk_shapes_pack_f64_best>, cpu_best<nk_rmsds_packed_f64_best>);
    check("rmsds_packed_f32", test_rmsds_packed<f32_t>, cpu_best<nk_shapes_pack_size_f32_best>,
          cpu_best<nk_shapes_pack_f32_best>, cpu_best<nk_rmsds_packed_f32_best>);
    check("rmsds_packed_f16", test_rmsds_packed<f16_t>, cpu_best<nk_shapes_pack_size_f16_best>,
          cpu_best<nk_shapes_pack_f16_best>, cpu_best<nk_rmsds_packed_f16_best>);
#endif

#if NUMKONG_TARGET_NEON
    check.section("Shapes NEON", nk_cap_neon_k);
    check("chamfers_packed_f32_neon", test_shapes_packed<f32_t, chamfers_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_chamfers_packed_f32_neon);
    check("chamfers_packed_f16_neon", test_shapes_packed<f16_t, chamfers_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_chamfers_packed_f16_neon);
    check("hausdorffs_packed_f32_neon", test_shapes_packed<f32_t, hausdorffs_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_hausdorffs_packed_f32_neon);
    check("hausdorffs_packed_f16_neon", test_shapes_packed<f16_t, hausdorffs_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_hausdorffs_packed_f16_neon);
    check("wassersteinprocrustes_packed_f32_neon", test_shapes_packed_workspace<f32_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_neon);
    check("wassersteinprocrustes_packed_f16_neon", test_shapes_packed_workspace<f16_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_neon);
    check("wassersteinprocrustes_kabsch_f32_neon", test_shapes_kabsch<f32_t>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_neon);
    check("wassersteinprocrustes_kabsch_f16_neon", test_shapes_kabsch<f16_t>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_neon);
    check("gromovwassersteins_packed_f32_neon", test_shapes_packed_workspace<f32_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
          nk_gromovwassersteins_packed_f32_neon);
    check("gromovwassersteins_packed_f16_neon", test_shapes_packed_workspace<f16_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
          nk_gromovwassersteins_packed_f16_neon);
#endif

#if NUMKONG_TARGET_NEONFHM
    check.section("Shapes NEON FHM", nk_cap_neonfhm_k);
    check("chamfers_packed_f16_neonfhm", test_shapes_packed<f16_t, chamfers_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_chamfers_packed_f16_neonfhm);
    check("hausdorffs_packed_f16_neonfhm", test_shapes_packed<f16_t, hausdorffs_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_hausdorffs_packed_f16_neonfhm);
#endif

#if NUMKONG_TARGET_HASWELL
    check.section("Shapes Haswell", nk_cap_haswell_k);
    check("chamfers_packed_f32_haswell", test_shapes_packed<f32_t, chamfers_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_chamfers_packed_f32_haswell);
    check("chamfers_packed_f16_haswell", test_shapes_packed<f16_t, chamfers_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_chamfers_packed_f16_haswell);
    check("hausdorffs_packed_f32_haswell", test_shapes_packed<f32_t, hausdorffs_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_hausdorffs_packed_f32_haswell);
    check("hausdorffs_packed_f16_haswell", test_shapes_packed<f16_t, hausdorffs_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_hausdorffs_packed_f16_haswell);
    check("wassersteinprocrustes_packed_f32_haswell", test_shapes_packed_workspace<f32_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_haswell);
    check("wassersteinprocrustes_packed_f16_haswell", test_shapes_packed_workspace<f16_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_haswell);
    check("wassersteinprocrustes_kabsch_f32_haswell", test_shapes_kabsch<f32_t>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_haswell);
    check("wassersteinprocrustes_kabsch_f16_haswell", test_shapes_kabsch<f16_t>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_haswell);
    check("gromovwassersteins_packed_f32_haswell", test_shapes_packed_workspace<f32_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
          nk_gromovwassersteins_packed_f32_haswell);
    check("gromovwassersteins_packed_f16_haswell", test_shapes_packed_workspace<f16_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
          nk_gromovwassersteins_packed_f16_haswell);
#endif

#if NUMKONG_TARGET_SKYLAKE
    check.section("Shapes Skylake", nk_cap_skylake_k);
    check("chamfers_packed_f64_skylake", test_shapes_packed<f64_t, chamfers_k>, nk_shapes_pack_size_f64_serial,
          nk_shapes_pack_f64_serial, nk_chamfers_packed_f64_skylake);
    check("chamfers_packed_f32_skylake", test_shapes_packed<f32_t, chamfers_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_chamfers_packed_f32_skylake);
    check("chamfers_packed_f16_skylake", test_shapes_packed<f16_t, chamfers_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_chamfers_packed_f16_skylake);
    check("hausdorffs_packed_f64_skylake", test_shapes_packed<f64_t, hausdorffs_k>, nk_shapes_pack_size_f64_serial,
          nk_shapes_pack_f64_serial, nk_hausdorffs_packed_f64_skylake);
    check("hausdorffs_packed_f32_skylake", test_shapes_packed<f32_t, hausdorffs_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_hausdorffs_packed_f32_skylake);
    check("hausdorffs_packed_f16_skylake", test_shapes_packed<f16_t, hausdorffs_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_hausdorffs_packed_f16_skylake);
    check("wassersteinprocrustes_packed_f64_skylake", test_shapes_packed_workspace<f64_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f64_serial, nk_shapes_pack_f64_serial, nk_wassersteinprocrustes_workspace_size_f64_serial,
          nk_wassersteinprocrustes_packed_f64_skylake);
    check("wassersteinprocrustes_packed_f32_skylake", test_shapes_packed_workspace<f32_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_skylake);
    check("wassersteinprocrustes_packed_f16_skylake", test_shapes_packed_workspace<f16_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_skylake);
    check("wassersteinprocrustes_kabsch_f64_skylake", test_shapes_kabsch<f64_t>, nk_shapes_pack_size_f64_serial,
          nk_shapes_pack_f64_serial, nk_wassersteinprocrustes_workspace_size_f64_serial,
          nk_wassersteinprocrustes_packed_f64_skylake);
    check("wassersteinprocrustes_kabsch_f32_skylake", test_shapes_kabsch<f32_t>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_skylake);
    check("wassersteinprocrustes_kabsch_f16_skylake", test_shapes_kabsch<f16_t>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_skylake);
    check("gromovwassersteins_packed_f64_skylake", test_shapes_packed_workspace<f64_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f64_serial, nk_shapes_pack_f64_serial, nk_gromovwassersteins_workspace_size_f64_serial,
          nk_gromovwassersteins_packed_f64_skylake);
    check("gromovwassersteins_packed_f32_skylake", test_shapes_packed_workspace<f32_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
          nk_gromovwassersteins_packed_f32_skylake);
    check("gromovwassersteins_packed_f16_skylake", test_shapes_packed_workspace<f16_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
          nk_gromovwassersteins_packed_f16_skylake);
    check("rmsds_packed_f64_skylake", test_rmsds_packed<f64_t>, nk_shapes_pack_size_f64_serial,
          nk_shapes_pack_f64_serial, nk_rmsds_packed_f64_skylake);
    check("rmsds_packed_f32_skylake", test_rmsds_packed<f32_t>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_rmsds_packed_f32_skylake);
    check("rmsds_packed_f16_skylake", test_rmsds_packed<f16_t>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_rmsds_packed_f16_skylake);
#endif

#if NUMKONG_TARGET_RVV
    check.section("Shapes RVV", nk_cap_rvv_k);
    check("chamfers_packed_f32_rvv", test_shapes_packed<f32_t, chamfers_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_chamfers_packed_f32_rvv);
    check("chamfers_packed_f16_rvv", test_shapes_packed<f16_t, chamfers_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_chamfers_packed_f16_rvv);
    check("hausdorffs_packed_f32_rvv", test_shapes_packed<f32_t, hausdorffs_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_hausdorffs_packed_f32_rvv);
    check("hausdorffs_packed_f16_rvv", test_shapes_packed<f16_t, hausdorffs_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_hausdorffs_packed_f16_rvv);
    check("wassersteinprocrustes_packed_f32_rvv", test_shapes_packed_workspace<f32_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_rvv);
    check("wassersteinprocrustes_packed_f16_rvv", test_shapes_packed_workspace<f16_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_rvv);
    check("wassersteinprocrustes_kabsch_f32_rvv", test_shapes_kabsch<f32_t>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_rvv);
    check("wassersteinprocrustes_kabsch_f16_rvv", test_shapes_kabsch<f16_t>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_rvv);
    check("gromovwassersteins_packed_f32_rvv", test_shapes_packed_workspace<f32_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
          nk_gromovwassersteins_packed_f32_rvv);
    check("gromovwassersteins_packed_f16_rvv", test_shapes_packed_workspace<f16_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
          nk_gromovwassersteins_packed_f16_rvv);
#endif

#if NUMKONG_TARGET_V128RELAXED
    check.section("Shapes V128 Relaxed", nk_cap_v128relaxed_k);
    check("chamfers_packed_f32_v128relaxed", test_shapes_packed<f32_t, chamfers_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_chamfers_packed_f32_v128relaxed);
    check("chamfers_packed_f16_v128relaxed", test_shapes_packed<f16_t, chamfers_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_chamfers_packed_f16_v128relaxed);
    check("hausdorffs_packed_f32_v128relaxed", test_shapes_packed<f32_t, hausdorffs_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_hausdorffs_packed_f32_v128relaxed);
    check("hausdorffs_packed_f16_v128relaxed", test_shapes_packed<f16_t, hausdorffs_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_hausdorffs_packed_f16_v128relaxed);
    check("wassersteinprocrustes_packed_f32_v128relaxed", test_shapes_packed_workspace<f32_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_v128relaxed);
    check("wassersteinprocrustes_packed_f16_v128relaxed", test_shapes_packed_workspace<f16_t, wassersteinprocrustes_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_v128relaxed);
    check("wassersteinprocrustes_kabsch_f32_v128relaxed", test_shapes_kabsch<f32_t>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_v128relaxed);
    check("wassersteinprocrustes_kabsch_f16_v128relaxed", test_shapes_kabsch<f16_t>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_v128relaxed);
    check("gromovwassersteins_packed_f32_v128relaxed", test_shapes_packed_workspace<f32_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f32_serial, nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial,
          nk_gromovwassersteins_packed_f32_v128relaxed);
    check("gromovwassersteins_packed_f16_v128relaxed", test_shapes_packed_workspace<f16_t, gromovwassersteins_k>,
          nk_shapes_pack_size_f16_serial, nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial,
          nk_gromovwassersteins_packed_f16_v128relaxed);
#endif
}

} // namespace ashvardanian::numkong::test
