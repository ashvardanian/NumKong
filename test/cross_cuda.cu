/**
 *  @file test/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batch operation tests - CUDA capabilities.
 *
 *  Compiled with every capability some listed architecture runs: the CUDA baseline, Ampere, Hopper,
 *  Blackwell and Blackwell RTX.
 */
#include "cross_device.hpp"
#include "geospatial.hpp"          // `test_geodesics_packed`, `test_terrain_packed`
#include "numkong/shapes/serial.h" // `nk_shapes_packed_header_f32_t`
#include "numkong/shapes.hpp"      // `nk::shapes_transport_epsilon_k`

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_

/** Transport clouds: random, or exact ties of a lattice with power-of-two steps, points repeated
 *  4 times, or a regular polygon around a column of points. */
enum class transport_layout_t { random_k, lattice_k, repeated_k, ring_k };

/** Packs @p count clouds of @p points each of @p scalar_type_ in @p layout with @p pack into device
 *  memory, back to back, and the table of their addresses after them. */
template <typename scalar_type_, typename pack_size_type_, typename pack_type_>
auto device_transport_packs(settings_t const &settings, cuda_backend_t &backend, std::mt19937 &generator,
                            transport_layout_t layout, std::size_t count, std::size_t points, pack_size_type_ pack_size,
                            pack_type_ pack, error_stats_t &stats) {
    auto coordinates = make_vector<scalar_type_>(count * points * 3);
    fill_random(settings, generator, coordinates);
    std::vector<std::span<scalar_type_ const>> clouds(count);
    std::size_t const corners = points > 8 ? points - points / 8 : points;
    for (std::size_t index = 0; index != count; ++index) {
        scalar_type_ *values = coordinates.values_data() + index * points * 3;
        for (std::size_t point = 0; point != points; ++point) {
            float const shift = float(index % 3);
            if (layout == transport_layout_t::lattice_k)
                values[point * 3] = float(point % 4) + shift, values[point * 3 + 1] = 0.5f * float(point / 4 % 4),
                               values[point * 3 + 2] = 0.25f * float(point / 16);
            else if (layout == transport_layout_t::repeated_k)
                for (std::size_t axis = 0; axis != 3; ++axis)
                    values[point * 3 + axis] = values[(point - point % 4) * 3 + axis];
            else if (layout == transport_layout_t::ring_k) {
                double const angle = 6.283185307179586 * double(point) / double(corners);
                bool const corner = point < corners;
                values[point * 3] = float(corner ? std::cos(angle) : 0.0) + shift;
                values[point * 3 + 1] = float(corner ? std::sin(angle) : 0.0);
                values[point * 3 + 2] = corner ? 0.0f : 0.5f * float(point - corners);
            }
        }
        clouds[index] = {values, points * 3};
    }
    using packs_t = nk::shapes_packed<scalar_type_, cuda_backend_t::allocator<char>>;
    auto packs = packs_t::make(clouds, allocator_of<char>(backend), pack_size, pack);
    stats.expect(packs.status);
    return std::move(packs.value);
}

/** The CUDA transport kernels against the serial ones on random and tied clouds of unequal sizes,
 *  within the bound the CPU capabilities meet: the roundings of a plan of up to n + m − 1 arcs,
 *  each costing at most the squared sum of both radii, and the simplex tolerance, under a root. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename pack_size_type_, typename pack_type_,
          typename workspace_size_type_, typename kernel_type_, typename serial_size_type_, typename serial_type_>
error_stats_t test_transport_packed_cuda(settings_t const &settings, pack_size_type_ pack_size, pack_type_ pack,
                                         workspace_size_type_ workspace_size, kernel_type_ kernel,
                                         serial_size_type_ serial_workspace_size, serial_type_ serial) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<cuda_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    std::size_t const points = settings.shape_points, rows = 5, columns = 7, largest = points + 3;
    nk_size_t bytes = 0, serial_bytes = 0;
    nk_unused_(workspace_size(largest, largest, &bytes));
    nk_unused_(serial_workspace_size(largest, largest, &serial_bytes));
    auto workspace = make_vector<std::byte>(backend, bytes);
    auto serial_workspace = make_vector<std::byte>(backend, serial_bytes);
    auto results = make_vector<distance_t>(backend, rows * columns);
    auto rotations = make_vector<f32_t>(backend, rows * columns * 9);
    std::vector<raw_distance_t> expected(rows * columns);
    std::vector<nk_f32_t> expected_rotations(rows * columns * 9);
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0), terms = static_cast<nk_f64_t>(points + largest);
    for (transport_layout_t const layout : {transport_layout_t::random_k, transport_layout_t::lattice_k,
                                            transport_layout_t::repeated_k, transport_layout_t::ring_k}) {
        auto const row_packs = device_transport_packs<scalar_type_>(settings, backend, generator, layout, rows, points,
                                                                    pack_size, pack, stats);
        auto const column_packs = device_transport_packs<scalar_type_>(settings, backend, generator, layout, columns,
                                                                       largest, pack_size, pack, stats);
        backend.call(kernel, row_packs.data(), rows, column_packs.data(), columns, results.raw_values_data(),
                     columns * sizeof(raw_distance_t), rotations.raw_values_data(), workspace.raw_values_data(), bytes);
        if (!stats.expect(backend.synchronize())) return stats;
        stats.expect(serial(row_packs.data(), rows, column_packs.data(), columns, expected.data(),
                            columns * sizeof(raw_distance_t), expected_rotations.data(),
                            serial_workspace.raw_values_data(), serial_bytes, nullptr));
        for (std::size_t row = 0; row != rows; ++row)
            for (std::size_t column = 0; column != columns; ++column) {
                auto const radius = [](void const *packed) {
                    if constexpr (std::is_same_v<scalar_type_, f32_t>)
                        return static_cast<nk_f64_t>(((nk_shapes_packed_header_f64_t const *)packed)->max_radius);
                    else return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
                };
                nk_f64_t const reach = radius(row_packs[row]) + radius(column_packs[column]);
                nk_f64_t const scale = reach * reach, tolerance = nk::shapes_transport_epsilon_k * scale;
                nk_f64_t const reference = expected[row * columns + column];
                nk_f64_t bound;
                if constexpr (kind_ == nk_kernel_wassersteinprocrustes_packed_k) {
                    nk_f64_t const squared = (terms + 32) * roundoff * scale + tolerance;
                    bound = nk::min_of(std::sqrt(squared), squared / reference);
                }
                else {
                    nk_f64_t const loss = 4 * ((terms + 64) * roundoff * scale + tolerance);
                    bound = nk::min_of(0.5 * std::sqrt(loss), loss / (8 * reference));
                }
                stats.accumulate_bounded(distance_t::from_raw(results.raw_values_data()[row * columns + column]),
                                         f64_t(reference), bound);
            }
    }
    return stats;
}

void test_cross_cuda(error_stats_section_t &check) {
    check.section("Cross CUDA", nk_cap_cuda_k);
    check("dots_packed_f64_cuda", test_dots_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_cuda,
          nk_dots_pack_f64_cuda, nk_dots_packed_f64_cuda);
    check("dots_pack_f64_cuda", test_dots_pack_layout<f64_t, cuda_backend_t, nk_dots_pack_size_f64_cuda,
                                                      nk_dots_packed_shape_f64_cuda, nk_dots_pack_f64_cuda>);
    check("dots_contract_f64_cuda", test_dots_launch_contract<f64_t, cuda_backend_t, nk_dots_pack_size_f64_cuda,
                                                              nk_dots_packed_f64_cuda, nk_dots_symmetric_f64_cuda>);
    check("dots_symmetric_f64_cuda", test_dots_symmetric<f64_t, cuda_backend_t>, nk_dots_symmetric_f64_cuda);
    check("dots_packed_f32_cuda", test_dots_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_cuda,
          nk_dots_pack_f32_cuda, nk_dots_packed_f32_cuda);
    check("dots_pack_f32_cuda", test_dots_pack_layout<f32_t, cuda_backend_t, nk_dots_pack_size_f32_cuda,
                                                      nk_dots_packed_shape_f32_cuda, nk_dots_pack_f32_cuda>);
    check("dots_contract_f32_cuda", test_dots_launch_contract<f32_t, cuda_backend_t, nk_dots_pack_size_f32_cuda,
                                                              nk_dots_packed_f32_cuda, nk_dots_symmetric_f32_cuda>);
    check("dots_symmetric_f32_cuda", test_dots_symmetric<f32_t, cuda_backend_t>, nk_dots_symmetric_f32_cuda);
    check("dots_packed_bf16_cuda", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_cuda,
          nk_dots_pack_bf16_cuda, nk_dots_packed_bf16_cuda);
    check("dots_pack_bf16_cuda", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_cuda,
                                                       nk_dots_packed_shape_bf16_cuda, nk_dots_pack_bf16_cuda>);
    check("dots_contract_bf16_cuda", test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_cuda,
                                                               nk_dots_packed_bf16_cuda, nk_dots_symmetric_bf16_cuda>);
    check("dots_symmetric_bf16_cuda", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_cuda);
    check("dots_packed_f16_cuda", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_cuda,
          nk_dots_pack_f16_cuda, nk_dots_packed_f16_cuda);
    check("dots_pack_f16_cuda", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_cuda,
                                                      nk_dots_packed_shape_f16_cuda, nk_dots_pack_f16_cuda>);
    check("dots_contract_f16_cuda", test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_cuda,
                                                              nk_dots_packed_f16_cuda, nk_dots_symmetric_f16_cuda>);
    check("dots_symmetric_f16_cuda", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_cuda);
    check("dots_packed_e5m2_cuda", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_cuda,
          nk_dots_pack_e5m2_cuda, nk_dots_packed_e5m2_cuda);
    check("dots_pack_e5m2_cuda", test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_cuda,
                                                       nk_dots_packed_shape_e5m2_cuda, nk_dots_pack_e5m2_cuda>);
    check("dots_contract_e5m2_cuda", test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_cuda,
                                                               nk_dots_packed_e5m2_cuda, nk_dots_symmetric_e5m2_cuda>);
    check("dots_symmetric_e5m2_cuda", test_dots_symmetric<e5m2_t, cuda_backend_t>, nk_dots_symmetric_e5m2_cuda);
    check("dots_packed_e4m3_cuda", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_cuda,
          nk_dots_pack_e4m3_cuda, nk_dots_packed_e4m3_cuda);
    check("dots_pack_e4m3_cuda", test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_cuda,
                                                       nk_dots_packed_shape_e4m3_cuda, nk_dots_pack_e4m3_cuda>);
    check("dots_contract_e4m3_cuda", test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_cuda,
                                                               nk_dots_packed_e4m3_cuda, nk_dots_symmetric_e4m3_cuda>);
    check("dots_symmetric_e4m3_cuda", test_dots_symmetric<e4m3_t, cuda_backend_t>, nk_dots_symmetric_e4m3_cuda);
    check("dots_packed_e3m2_cuda", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_cuda,
          nk_dots_pack_e3m2_cuda, nk_dots_packed_e3m2_cuda);
    check("dots_pack_e3m2_cuda", test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_cuda,
                                                       nk_dots_packed_shape_e3m2_cuda, nk_dots_pack_e3m2_cuda>);
    check("dots_contract_e3m2_cuda", test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_cuda,
                                                               nk_dots_packed_e3m2_cuda, nk_dots_symmetric_e3m2_cuda>);
    check("dots_symmetric_e3m2_cuda", test_dots_symmetric<e3m2_t, cuda_backend_t>, nk_dots_symmetric_e3m2_cuda);
    check("dots_packed_e2m3_cuda", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_cuda,
          nk_dots_pack_e2m3_cuda, nk_dots_packed_e2m3_cuda);
    check("dots_pack_e2m3_cuda", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_cuda,
                                                       nk_dots_packed_shape_e2m3_cuda, nk_dots_pack_e2m3_cuda>);
    check("dots_contract_e2m3_cuda", test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_cuda,
                                                               nk_dots_packed_e2m3_cuda, nk_dots_symmetric_e2m3_cuda>);
    check("dots_symmetric_e2m3_cuda", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_cuda);
    check("dots_packed_e2m1_cuda", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_cuda,
          nk_dots_pack_e2m1_cuda, nk_dots_packed_e2m1_cuda);
    check("dots_pack_e2m1_cuda", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_cuda,
                                                       nk_dots_packed_shape_e2m1_cuda, nk_dots_pack_e2m1_cuda>);
    check("dots_contract_e2m1_cuda", test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_cuda,
                                                               nk_dots_packed_e2m1_cuda, nk_dots_symmetric_e2m1_cuda>);
    check("dots_symmetric_e2m1_cuda", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_cuda);
    check("dots_packed_nvfp4_cuda", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_cuda,
          nk_dots_pack_nvfp4_cuda, nk_dots_packed_nvfp4_cuda);
    check("dots_pack_nvfp4_cuda", test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_cuda,
                                                        nk_dots_packed_shape_nvfp4_cuda, nk_dots_pack_nvfp4_cuda>);
    check("dots_contract_nvfp4_cuda",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_cuda, nk_dots_packed_nvfp4_cuda,
                                    nk_dots_symmetric_nvfp4_cuda>);
    check("dots_symmetric_nvfp4_cuda", test_dots_symmetric<nvfp4_t, cuda_backend_t>, nk_dots_symmetric_nvfp4_cuda);
    check("dots_packed_mxfp4_cuda", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_cuda,
          nk_dots_pack_mxfp4_cuda, nk_dots_packed_mxfp4_cuda);
    check("dots_pack_mxfp4_cuda", test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_cuda,
                                                        nk_dots_packed_shape_mxfp4_cuda, nk_dots_pack_mxfp4_cuda>);
    check("dots_contract_mxfp4_cuda",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_cuda, nk_dots_packed_mxfp4_cuda,
                                    nk_dots_symmetric_mxfp4_cuda>);
    check("dots_symmetric_mxfp4_cuda", test_dots_symmetric<mxfp4_t, cuda_backend_t>, nk_dots_symmetric_mxfp4_cuda);
    check("dots_packed_mxfp8e4m3_cuda", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>, nk_dots_pack_size_mxfp8e4m3_cuda,
          nk_dots_pack_mxfp8e4m3_cuda, nk_dots_packed_mxfp8e4m3_cuda);
    check("dots_pack_mxfp8e4m3_cuda",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_cuda,
                                nk_dots_packed_shape_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda>);
    check("dots_contract_mxfp8e4m3_cuda",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_cuda,
                                    nk_dots_packed_mxfp8e4m3_cuda, nk_dots_symmetric_mxfp8e4m3_cuda>);
    check("dots_symmetric_mxfp8e4m3_cuda", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_cuda);
    check("dots_packed_mxfp8e5m2_cuda", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>, nk_dots_pack_size_mxfp8e5m2_cuda,
          nk_dots_pack_mxfp8e5m2_cuda, nk_dots_packed_mxfp8e5m2_cuda);
    check("dots_pack_mxfp8e5m2_cuda",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_cuda,
                                nk_dots_packed_shape_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda>);
    check("dots_contract_mxfp8e5m2_cuda",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_cuda,
                                    nk_dots_packed_mxfp8e5m2_cuda, nk_dots_symmetric_mxfp8e5m2_cuda>);
    check("dots_symmetric_mxfp8e5m2_cuda", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_cuda);
    check("dots_packed_i8_cuda", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_cuda,
          nk_dots_pack_i8_cuda, nk_dots_packed_i8_cuda);
    check("dots_pack_i8_cuda", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_cuda,
                                                     nk_dots_packed_shape_i8_cuda, nk_dots_pack_i8_cuda>);
    check("dots_contract_i8_cuda", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_cuda,
                                                             nk_dots_packed_i8_cuda, nk_dots_symmetric_i8_cuda>);
    check("dots_symmetric_i8_cuda", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_cuda);
    check("dots_packed_u8_cuda", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_cuda,
          nk_dots_pack_u8_cuda, nk_dots_packed_u8_cuda);
    check("dots_pack_u8_cuda", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_cuda,
                                                     nk_dots_packed_shape_u8_cuda, nk_dots_pack_u8_cuda>);
    check("dots_contract_u8_cuda", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_cuda,
                                                             nk_dots_packed_u8_cuda, nk_dots_symmetric_u8_cuda>);
    check("dots_symmetric_u8_cuda", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_cuda);
    check("dots_packed_i4_cuda", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_cuda,
          nk_dots_pack_i4_cuda, nk_dots_packed_i4_cuda);
    check("dots_pack_i4_cuda", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_cuda,
                                                     nk_dots_packed_shape_i4_cuda, nk_dots_pack_i4_cuda>);
    check("dots_contract_i4_cuda", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_cuda,
                                                             nk_dots_packed_i4_cuda, nk_dots_symmetric_i4_cuda>);
    check("dots_symmetric_i4_cuda", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_cuda);
    check("dots_packed_u4_cuda", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_cuda,
          nk_dots_pack_u4_cuda, nk_dots_packed_u4_cuda);
    check("dots_pack_u4_cuda", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_cuda,
                                                     nk_dots_packed_shape_u4_cuda, nk_dots_pack_u4_cuda>);
    check("dots_contract_u4_cuda", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_cuda,
                                                             nk_dots_packed_u4_cuda, nk_dots_symmetric_u4_cuda>);
    check("dots_symmetric_u4_cuda", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_cuda);
    check("angulars_packed_f64_cuda", test_angulars_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_cuda,
          nk_dots_pack_f64_cuda, nk_angulars_packed_f64_cuda);
    check("angulars_symmetric_f64_cuda", test_angulars_symmetric<f64_t, cuda_backend_t>,
          nk_angulars_symmetric_f64_cuda);
    check("euclideans_packed_f64_cuda", test_euclideans_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_cuda,
          nk_dots_pack_f64_cuda, nk_euclideans_packed_f64_cuda);
    check("euclideans_symmetric_f64_cuda", test_euclideans_symmetric<f64_t, cuda_backend_t>,
          nk_euclideans_symmetric_f64_cuda);
    check("angulars_packed_f32_cuda", test_angulars_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_cuda,
          nk_dots_pack_f32_cuda, nk_angulars_packed_f32_cuda);
    check("angulars_symmetric_f32_cuda", test_angulars_symmetric<f32_t, cuda_backend_t>,
          nk_angulars_symmetric_f32_cuda);
    check("euclideans_packed_f32_cuda", test_euclideans_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_cuda,
          nk_dots_pack_f32_cuda, nk_euclideans_packed_f32_cuda);
    check("euclideans_symmetric_f32_cuda", test_euclideans_symmetric<f32_t, cuda_backend_t>,
          nk_euclideans_symmetric_f32_cuda);
    check("angulars_packed_bf16_cuda", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_cuda,
          nk_dots_pack_bf16_cuda, nk_angulars_packed_bf16_cuda);
    check("angulars_symmetric_bf16_cuda", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_cuda);
    check("euclideans_packed_bf16_cuda", test_euclideans_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_cuda,
          nk_dots_pack_bf16_cuda, nk_euclideans_packed_bf16_cuda);
    check("euclideans_symmetric_bf16_cuda", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_cuda);
    check("angulars_packed_f16_cuda", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_cuda,
          nk_dots_pack_f16_cuda, nk_angulars_packed_f16_cuda);
    check("angulars_symmetric_f16_cuda", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_cuda);
    check("euclideans_packed_f16_cuda", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_cuda,
          nk_dots_pack_f16_cuda, nk_euclideans_packed_f16_cuda);
    check("euclideans_symmetric_f16_cuda", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_cuda);
    check("angulars_packed_e5m2_cuda", test_angulars_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_cuda,
          nk_dots_pack_e5m2_cuda, nk_angulars_packed_e5m2_cuda);
    check("angulars_symmetric_e5m2_cuda", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_cuda);
    check("euclideans_packed_e5m2_cuda", test_euclideans_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_cuda,
          nk_dots_pack_e5m2_cuda, nk_euclideans_packed_e5m2_cuda);
    check("euclideans_symmetric_e5m2_cuda", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_cuda);
    check("angulars_packed_e4m3_cuda", test_angulars_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_cuda,
          nk_dots_pack_e4m3_cuda, nk_angulars_packed_e4m3_cuda);
    check("angulars_symmetric_e4m3_cuda", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_cuda);
    check("euclideans_packed_e4m3_cuda", test_euclideans_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_cuda,
          nk_dots_pack_e4m3_cuda, nk_euclideans_packed_e4m3_cuda);
    check("euclideans_symmetric_e4m3_cuda", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_cuda);
    check("angulars_packed_e3m2_cuda", test_angulars_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_cuda,
          nk_dots_pack_e3m2_cuda, nk_angulars_packed_e3m2_cuda);
    check("angulars_symmetric_e3m2_cuda", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_cuda);
    check("euclideans_packed_e3m2_cuda", test_euclideans_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_cuda,
          nk_dots_pack_e3m2_cuda, nk_euclideans_packed_e3m2_cuda);
    check("euclideans_symmetric_e3m2_cuda", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_cuda);
    check("angulars_packed_e2m3_cuda", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_cuda,
          nk_dots_pack_e2m3_cuda, nk_angulars_packed_e2m3_cuda);
    check("angulars_symmetric_e2m3_cuda", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_cuda);
    check("euclideans_packed_e2m3_cuda", test_euclideans_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_cuda,
          nk_dots_pack_e2m3_cuda, nk_euclideans_packed_e2m3_cuda);
    check("euclideans_symmetric_e2m3_cuda", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_cuda);
    check("angulars_packed_e2m1_cuda", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_cuda,
          nk_dots_pack_e2m1_cuda, nk_angulars_packed_e2m1_cuda);
    check("angulars_symmetric_e2m1_cuda", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_cuda);
    check("euclideans_packed_e2m1_cuda", test_euclideans_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_cuda,
          nk_dots_pack_e2m1_cuda, nk_euclideans_packed_e2m1_cuda);
    check("euclideans_symmetric_e2m1_cuda", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_cuda);
    check("angulars_packed_nvfp4_cuda", test_angulars_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_cuda,
          nk_dots_pack_nvfp4_cuda, nk_angulars_packed_nvfp4_cuda);
    check("angulars_symmetric_nvfp4_cuda", test_angulars_symmetric<nvfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_nvfp4_cuda);
    check("euclideans_packed_nvfp4_cuda", test_euclideans_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_cuda,
          nk_dots_pack_nvfp4_cuda, nk_euclideans_packed_nvfp4_cuda);
    check("euclideans_symmetric_nvfp4_cuda", test_euclideans_symmetric<nvfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_nvfp4_cuda);
    check("angulars_packed_mxfp4_cuda", test_angulars_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_cuda,
          nk_dots_pack_mxfp4_cuda, nk_angulars_packed_mxfp4_cuda);
    check("angulars_symmetric_mxfp4_cuda", test_angulars_symmetric<mxfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp4_cuda);
    check("euclideans_packed_mxfp4_cuda", test_euclideans_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_cuda,
          nk_dots_pack_mxfp4_cuda, nk_euclideans_packed_mxfp4_cuda);
    check("euclideans_symmetric_mxfp4_cuda", test_euclideans_symmetric<mxfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp4_cuda);
    check("angulars_packed_mxfp8e4m3_cuda", test_angulars_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda, nk_angulars_packed_mxfp8e4m3_cuda);
    check("angulars_symmetric_mxfp8e4m3_cuda", test_angulars_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e4m3_cuda);
    check("euclideans_packed_mxfp8e4m3_cuda", test_euclideans_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda, nk_euclideans_packed_mxfp8e4m3_cuda);
    check("euclideans_symmetric_mxfp8e4m3_cuda", test_euclideans_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e4m3_cuda);
    check("angulars_packed_mxfp8e5m2_cuda", test_angulars_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda, nk_angulars_packed_mxfp8e5m2_cuda);
    check("angulars_symmetric_mxfp8e5m2_cuda", test_angulars_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e5m2_cuda);
    check("euclideans_packed_mxfp8e5m2_cuda", test_euclideans_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda, nk_euclideans_packed_mxfp8e5m2_cuda);
    check("euclideans_symmetric_mxfp8e5m2_cuda", test_euclideans_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e5m2_cuda);
    check("angulars_packed_i8_cuda", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_cuda,
          nk_dots_pack_i8_cuda, nk_angulars_packed_i8_cuda);
    check("angulars_symmetric_i8_cuda", test_angulars_symmetric<i8_t, cuda_backend_t>, nk_angulars_symmetric_i8_cuda);
    check("euclideans_packed_i8_cuda", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_cuda,
          nk_dots_pack_i8_cuda, nk_euclideans_packed_i8_cuda);
    check("euclideans_symmetric_i8_cuda", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_cuda);
    check("angulars_packed_u8_cuda", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_cuda,
          nk_dots_pack_u8_cuda, nk_angulars_packed_u8_cuda);
    check("angulars_symmetric_u8_cuda", test_angulars_symmetric<u8_t, cuda_backend_t>, nk_angulars_symmetric_u8_cuda);
    check("euclideans_packed_u8_cuda", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_cuda,
          nk_dots_pack_u8_cuda, nk_euclideans_packed_u8_cuda);
    check("euclideans_symmetric_u8_cuda", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_cuda);
    check("angulars_packed_i4_cuda", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_cuda,
          nk_dots_pack_i4_cuda, nk_angulars_packed_i4_cuda);
    check("angulars_symmetric_i4_cuda", test_angulars_symmetric<i4x2_t, cuda_backend_t>, nk_angulars_symmetric_i4_cuda);
    check("euclideans_packed_i4_cuda", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_cuda,
          nk_dots_pack_i4_cuda, nk_euclideans_packed_i4_cuda);
    check("euclideans_symmetric_i4_cuda", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_cuda);
    check("angulars_packed_u4_cuda", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_cuda,
          nk_dots_pack_u4_cuda, nk_angulars_packed_u4_cuda);
    check("angulars_symmetric_u4_cuda", test_angulars_symmetric<u4x2_t, cuda_backend_t>, nk_angulars_symmetric_u4_cuda);
    check("euclideans_packed_u4_cuda", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_cuda,
          nk_dots_pack_u4_cuda, nk_euclideans_packed_u4_cuda);
    check("euclideans_symmetric_u4_cuda", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_cuda);
    check("attention_packed_bf16_cuda", test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_cuda, nk_attention_pack_bf16_cuda, nk_attention_packed_bf16_cuda);
    check("attention_packed_f16_cuda", test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_cuda, nk_attention_pack_f16_cuda, nk_attention_packed_f16_cuda);
    check("attention_packed_e4m3_cuda", test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_cuda, nk_attention_pack_e4m3_cuda, nk_attention_packed_e4m3_cuda);
    check("attention_packed_i8_cuda", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_cuda, nk_attention_pack_i8_cuda, nk_attention_packed_i8_cuda);
    check("attention_packed_gradients_bf16_cuda", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_cuda, nk_attention_pack_bf16_cuda, nk_attention_packed_gradients_bf16_cuda);
    check("attention_rope_f32_cuda", test_attention_rope<f32_t, cuda_backend_t>, nk_attention_rope_f32_cuda);
    check("attention_rope_bf16_cuda", test_attention_rope<bf16_t, cuda_backend_t>, nk_attention_rope_bf16_cuda);
    check("attention_rope_e4m3_cuda", test_attention_rope<e4m3_t, cuda_backend_t>, nk_attention_rope_e4m3_cuda);
    check("wassersteinprocrustes_packed_f32_cuda",
          test_transport_packed_cuda<f32_t, nk_kernel_wassersteinprocrustes_packed_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_cuda,
          nk_wassersteinprocrustes_packed_f32_cuda, nk_wassersteinprocrustes_workspace_size_f32_serial,
          nk_wassersteinprocrustes_packed_f32_serial);
    check("wassersteinprocrustes_packed_f16_cuda",
          test_transport_packed_cuda<f16_t, nk_kernel_wassersteinprocrustes_packed_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_cuda,
          nk_wassersteinprocrustes_packed_f16_cuda, nk_wassersteinprocrustes_workspace_size_f16_serial,
          nk_wassersteinprocrustes_packed_f16_serial);
    check("gromovwassersteins_packed_f32_cuda",
          test_transport_packed_cuda<f32_t, nk_kernel_gromovwassersteins_packed_k>, nk_shapes_pack_size_f32_serial,
          nk_shapes_pack_f32_serial, nk_gromovwassersteins_workspace_size_f32_cuda,
          nk_gromovwassersteins_packed_f32_cuda, nk_gromovwassersteins_workspace_size_f32_serial,
          nk_gromovwassersteins_packed_f32_serial);
    check("gromovwassersteins_packed_f16_cuda",
          test_transport_packed_cuda<f16_t, nk_kernel_gromovwassersteins_packed_k>, nk_shapes_pack_size_f16_serial,
          nk_shapes_pack_f16_serial, nk_gromovwassersteins_workspace_size_f16_cuda,
          nk_gromovwassersteins_packed_f16_cuda, nk_gromovwassersteins_workspace_size_f16_serial,
          nk_gromovwassersteins_packed_f16_serial);
    check("haversines_packed_f64_cuda", test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k, cuda_backend_t>,
          nk_haversines_pack_size_f64_cuda, nk_haversines_pack_f64_cuda, nk_haversines_packed_f64_cuda);
    check("vincentys_packed_f64_cuda", test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k, cuda_backend_t>,
          nk_vincentys_pack_size_f64_cuda, nk_vincentys_pack_f64_cuda, nk_vincentys_packed_f64_cuda);
    check("clearances_packed_f32_cuda", test_terrain_packed<f32_t, cuda_backend_t>, nk_terrain_pack_size_f32_cuda,
          nk_terrain_pack_f32_cuda, nk_terrain_heights_packed_f32_cuda, nk_clearances_packed_f32_cuda);

#if NUMKONG_TARGET_AMPERE
    check.section("Cross Ampere", nk_cap_ampere_k);
    check("dots_packed_f64_ampere", test_dots_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, nk_dots_packed_f64_ampere);
    check("dots_pack_f64_ampere", test_dots_pack_layout<f64_t, cuda_backend_t, nk_dots_pack_size_f64_ampere,
                                                        nk_dots_packed_shape_f64_ampere, nk_dots_pack_f64_ampere>);
    check("dots_contract_f64_ampere",
          test_dots_launch_contract<f64_t, cuda_backend_t, nk_dots_pack_size_f64_ampere, nk_dots_packed_f64_ampere,
                                    nk_dots_symmetric_f64_ampere>);
    check("dots_symmetric_f64_ampere", test_dots_symmetric<f64_t, cuda_backend_t>, nk_dots_symmetric_f64_ampere);
    check("dots_packed_f64_ampere_workspace", test_dots_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, with_workspace(nk_dots_packed_f64_ampere, nk_dots_packed_workspace_size_f64_ampere));
    check("dots_symmetric_f64_ampere_workspace", test_dots_symmetric<f64_t, cuda_backend_t>,
          with_workspace(nk_dots_symmetric_f64_ampere, nk_dots_symmetric_workspace_size_f64_ampere));
    check("dots_packed_f32_ampere", test_dots_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_ampere,
          nk_dots_pack_f32_ampere, nk_dots_packed_f32_ampere);
    check("dots_pack_f32_ampere", test_dots_pack_layout<f32_t, cuda_backend_t, nk_dots_pack_size_f32_ampere,
                                                        nk_dots_packed_shape_f32_ampere, nk_dots_pack_f32_ampere>);
    check("dots_contract_f32_ampere",
          test_dots_launch_contract<f32_t, cuda_backend_t, nk_dots_pack_size_f32_ampere, nk_dots_packed_f32_ampere,
                                    nk_dots_symmetric_f32_ampere>);
    check("dots_symmetric_f32_ampere", test_dots_symmetric<f32_t, cuda_backend_t>, nk_dots_symmetric_f32_ampere);
    check("dots_packed_bf16_ampere", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_ampere,
          nk_dots_pack_bf16_ampere, nk_dots_packed_bf16_ampere);
    check("dots_pack_bf16_ampere", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_ampere,
                                                         nk_dots_packed_shape_bf16_ampere, nk_dots_pack_bf16_ampere>);
    check("dots_contract_bf16_ampere",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_ampere, nk_dots_packed_bf16_ampere,
                                    nk_dots_symmetric_bf16_ampere>);
    check("dots_symmetric_bf16_ampere", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_ampere);
    check("dots_packed_f16_ampere", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_dots_packed_f16_ampere);
    check("dots_pack_f16_ampere", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_ampere,
                                                        nk_dots_packed_shape_f16_ampere, nk_dots_pack_f16_ampere>);
    check("dots_contract_f16_ampere",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_ampere, nk_dots_packed_f16_ampere,
                                    nk_dots_symmetric_f16_ampere>);
    check("dots_symmetric_f16_ampere", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_ampere);
    check("dots_packed_e5m2_ampere", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_ampere,
          nk_dots_pack_e5m2_ampere, nk_dots_packed_e5m2_ampere);
    check("dots_pack_e5m2_ampere", test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_ampere,
                                                         nk_dots_packed_shape_e5m2_ampere, nk_dots_pack_e5m2_ampere>);
    check("dots_contract_e5m2_ampere",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_ampere, nk_dots_packed_e5m2_ampere,
                                    nk_dots_symmetric_e5m2_ampere>);
    check("dots_symmetric_e5m2_ampere", test_dots_symmetric<e5m2_t, cuda_backend_t>, nk_dots_symmetric_e5m2_ampere);
    check("dots_packed_e4m3_ampere", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_ampere,
          nk_dots_pack_e4m3_ampere, nk_dots_packed_e4m3_ampere);
    check("dots_pack_e4m3_ampere", test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_ampere,
                                                         nk_dots_packed_shape_e4m3_ampere, nk_dots_pack_e4m3_ampere>);
    check("dots_contract_e4m3_ampere",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_ampere, nk_dots_packed_e4m3_ampere,
                                    nk_dots_symmetric_e4m3_ampere>);
    check("dots_symmetric_e4m3_ampere", test_dots_symmetric<e4m3_t, cuda_backend_t>, nk_dots_symmetric_e4m3_ampere);
    check("dots_packed_e3m2_ampere", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_ampere,
          nk_dots_pack_e3m2_ampere, nk_dots_packed_e3m2_ampere);
    check("dots_pack_e3m2_ampere", test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_ampere,
                                                         nk_dots_packed_shape_e3m2_ampere, nk_dots_pack_e3m2_ampere>);
    check("dots_contract_e3m2_ampere",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_ampere, nk_dots_packed_e3m2_ampere,
                                    nk_dots_symmetric_e3m2_ampere>);
    check("dots_symmetric_e3m2_ampere", test_dots_symmetric<e3m2_t, cuda_backend_t>, nk_dots_symmetric_e3m2_ampere);
    check("dots_packed_e2m3_ampere", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_ampere,
          nk_dots_pack_e2m3_ampere, nk_dots_packed_e2m3_ampere);
    check("dots_pack_e2m3_ampere", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_ampere,
                                                         nk_dots_packed_shape_e2m3_ampere, nk_dots_pack_e2m3_ampere>);
    check("dots_contract_e2m3_ampere",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_ampere, nk_dots_packed_e2m3_ampere,
                                    nk_dots_symmetric_e2m3_ampere>);
    check("dots_symmetric_e2m3_ampere", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_ampere);
    check("dots_packed_e2m1_ampere", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_ampere,
          nk_dots_pack_e2m1_ampere, nk_dots_packed_e2m1_ampere);
    check("dots_pack_e2m1_ampere", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_ampere,
                                                         nk_dots_packed_shape_e2m1_ampere, nk_dots_pack_e2m1_ampere>);
    check("dots_contract_e2m1_ampere",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_ampere, nk_dots_packed_e2m1_ampere,
                                    nk_dots_symmetric_e2m1_ampere>);
    check("dots_symmetric_e2m1_ampere", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_ampere);
    check("dots_packed_nvfp4_ampere", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_ampere,
          nk_dots_pack_nvfp4_ampere, nk_dots_packed_nvfp4_ampere);
    check("dots_pack_nvfp4_ampere",
          test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_ampere,
                                nk_dots_packed_shape_nvfp4_ampere, nk_dots_pack_nvfp4_ampere>);
    check("dots_contract_nvfp4_ampere",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_ampere,
                                    nk_dots_packed_nvfp4_ampere, nk_dots_symmetric_nvfp4_ampere>);
    check("dots_symmetric_nvfp4_ampere", test_dots_symmetric<nvfp4_t, cuda_backend_t>, nk_dots_symmetric_nvfp4_ampere);
    check("dots_packed_mxfp4_ampere", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_ampere,
          nk_dots_pack_mxfp4_ampere, nk_dots_packed_mxfp4_ampere);
    check("dots_pack_mxfp4_ampere",
          test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_ampere,
                                nk_dots_packed_shape_mxfp4_ampere, nk_dots_pack_mxfp4_ampere>);
    check("dots_contract_mxfp4_ampere",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_ampere,
                                    nk_dots_packed_mxfp4_ampere, nk_dots_symmetric_mxfp4_ampere>);
    check("dots_symmetric_mxfp4_ampere", test_dots_symmetric<mxfp4_t, cuda_backend_t>, nk_dots_symmetric_mxfp4_ampere);
    check("dots_packed_mxfp8e4m3_ampere", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_ampere, nk_dots_pack_mxfp8e4m3_ampere, nk_dots_packed_mxfp8e4m3_ampere);
    check("dots_pack_mxfp8e4m3_ampere",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_ampere,
                                nk_dots_packed_shape_mxfp8e4m3_ampere, nk_dots_pack_mxfp8e4m3_ampere>);
    check("dots_contract_mxfp8e4m3_ampere",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_ampere,
                                    nk_dots_packed_mxfp8e4m3_ampere, nk_dots_symmetric_mxfp8e4m3_ampere>);
    check("dots_symmetric_mxfp8e4m3_ampere", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_ampere);
    check("dots_packed_mxfp8e5m2_ampere", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_ampere, nk_dots_pack_mxfp8e5m2_ampere, nk_dots_packed_mxfp8e5m2_ampere);
    check("dots_pack_mxfp8e5m2_ampere",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_ampere,
                                nk_dots_packed_shape_mxfp8e5m2_ampere, nk_dots_pack_mxfp8e5m2_ampere>);
    check("dots_contract_mxfp8e5m2_ampere",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_ampere,
                                    nk_dots_packed_mxfp8e5m2_ampere, nk_dots_symmetric_mxfp8e5m2_ampere>);
    check("dots_symmetric_mxfp8e5m2_ampere", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_ampere);
    check("dots_packed_i8_ampere", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_dots_packed_i8_ampere);
    check("dots_pack_i8_ampere", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_ampere,
                                                       nk_dots_packed_shape_i8_ampere, nk_dots_pack_i8_ampere>);
    check("dots_contract_i8_ampere", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_ampere,
                                                               nk_dots_packed_i8_ampere, nk_dots_symmetric_i8_ampere>);
    check("dots_symmetric_i8_ampere", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_ampere);
    check("dots_packed_u8_ampere", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_dots_packed_u8_ampere);
    check("dots_pack_u8_ampere", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_ampere,
                                                       nk_dots_packed_shape_u8_ampere, nk_dots_pack_u8_ampere>);
    check("dots_contract_u8_ampere", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_ampere,
                                                               nk_dots_packed_u8_ampere, nk_dots_symmetric_u8_ampere>);
    check("dots_symmetric_u8_ampere", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_ampere);
    check("dots_packed_i4_ampere", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_dots_packed_i4_ampere);
    check("dots_pack_i4_ampere", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_ampere,
                                                       nk_dots_packed_shape_i4_ampere, nk_dots_pack_i4_ampere>);
    check("dots_contract_i4_ampere", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_ampere,
                                                               nk_dots_packed_i4_ampere, nk_dots_symmetric_i4_ampere>);
    check("dots_symmetric_i4_ampere", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_ampere);
    check("dots_packed_u4_ampere", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_dots_packed_u4_ampere);
    check("dots_pack_u4_ampere", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_ampere,
                                                       nk_dots_packed_shape_u4_ampere, nk_dots_pack_u4_ampere>);
    check("dots_contract_u4_ampere", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_ampere,
                                                               nk_dots_packed_u4_ampere, nk_dots_symmetric_u4_ampere>);
    check("dots_symmetric_u4_ampere", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_ampere);
    check("angulars_packed_f64_ampere", test_angulars_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, nk_angulars_packed_f64_ampere);
    check("angulars_symmetric_f64_ampere", test_angulars_symmetric<f64_t, cuda_backend_t>,
          nk_angulars_symmetric_f64_ampere);
    check("euclideans_packed_f64_ampere", test_euclideans_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, nk_euclideans_packed_f64_ampere);
    check("euclideans_symmetric_f64_ampere", test_euclideans_symmetric<f64_t, cuda_backend_t>,
          nk_euclideans_symmetric_f64_ampere);
    check("angulars_packed_f64_ampere_workspace", test_angulars_packed<f64_t, cuda_backend_t>,
          nk_dots_pack_size_f64_ampere, nk_dots_pack_f64_ampere,
          with_workspace(nk_angulars_packed_f64_ampere, nk_angulars_packed_workspace_size_f64_ampere));
    check("angulars_symmetric_f64_ampere_workspace", test_angulars_symmetric<f64_t, cuda_backend_t>,
          with_workspace(nk_angulars_symmetric_f64_ampere, nk_angulars_symmetric_workspace_size_f64_ampere));
    check("euclideans_packed_f64_ampere_workspace", test_euclideans_packed<f64_t, cuda_backend_t>,
          nk_dots_pack_size_f64_ampere, nk_dots_pack_f64_ampere,
          with_workspace(nk_euclideans_packed_f64_ampere, nk_euclideans_packed_workspace_size_f64_ampere));
    check("euclideans_symmetric_f64_ampere_workspace", test_euclideans_symmetric<f64_t, cuda_backend_t>,
          with_workspace(nk_euclideans_symmetric_f64_ampere, nk_euclideans_symmetric_workspace_size_f64_ampere));
    check("angulars_packed_f32_ampere", test_angulars_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_ampere,
          nk_dots_pack_f32_ampere, nk_angulars_packed_f32_ampere);
    check("angulars_symmetric_f32_ampere", test_angulars_symmetric<f32_t, cuda_backend_t>,
          nk_angulars_symmetric_f32_ampere);
    check("euclideans_packed_f32_ampere", test_euclideans_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_ampere,
          nk_dots_pack_f32_ampere, nk_euclideans_packed_f32_ampere);
    check("euclideans_symmetric_f32_ampere", test_euclideans_symmetric<f32_t, cuda_backend_t>,
          nk_euclideans_symmetric_f32_ampere);
    check("angulars_packed_bf16_ampere", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_ampere,
          nk_dots_pack_bf16_ampere, nk_angulars_packed_bf16_ampere);
    check("angulars_symmetric_bf16_ampere", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_ampere);
    check("euclideans_packed_bf16_ampere", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_ampere, nk_dots_pack_bf16_ampere, nk_euclideans_packed_bf16_ampere);
    check("euclideans_symmetric_bf16_ampere", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_ampere);
    check("angulars_packed_f16_ampere", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_angulars_packed_f16_ampere);
    check("angulars_symmetric_f16_ampere", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_ampere);
    check("euclideans_packed_f16_ampere", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_euclideans_packed_f16_ampere);
    check("euclideans_symmetric_f16_ampere", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_ampere);
    check("angulars_packed_e5m2_ampere", test_angulars_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_ampere,
          nk_dots_pack_e5m2_ampere, nk_angulars_packed_e5m2_ampere);
    check("angulars_symmetric_e5m2_ampere", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_ampere);
    check("euclideans_packed_e5m2_ampere", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_ampere, nk_dots_pack_e5m2_ampere, nk_euclideans_packed_e5m2_ampere);
    check("euclideans_symmetric_e5m2_ampere", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_ampere);
    check("angulars_packed_e4m3_ampere", test_angulars_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_ampere,
          nk_dots_pack_e4m3_ampere, nk_angulars_packed_e4m3_ampere);
    check("angulars_symmetric_e4m3_ampere", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_ampere);
    check("euclideans_packed_e4m3_ampere", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_ampere, nk_dots_pack_e4m3_ampere, nk_euclideans_packed_e4m3_ampere);
    check("euclideans_symmetric_e4m3_ampere", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_ampere);
    check("angulars_packed_e3m2_ampere", test_angulars_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_ampere,
          nk_dots_pack_e3m2_ampere, nk_angulars_packed_e3m2_ampere);
    check("angulars_symmetric_e3m2_ampere", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_ampere);
    check("euclideans_packed_e3m2_ampere", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_ampere, nk_dots_pack_e3m2_ampere, nk_euclideans_packed_e3m2_ampere);
    check("euclideans_symmetric_e3m2_ampere", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_ampere);
    check("angulars_packed_e2m3_ampere", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_ampere,
          nk_dots_pack_e2m3_ampere, nk_angulars_packed_e2m3_ampere);
    check("angulars_symmetric_e2m3_ampere", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_ampere);
    check("euclideans_packed_e2m3_ampere", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_ampere, nk_dots_pack_e2m3_ampere, nk_euclideans_packed_e2m3_ampere);
    check("euclideans_symmetric_e2m3_ampere", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_ampere);
    check("angulars_packed_e2m1_ampere", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_ampere,
          nk_dots_pack_e2m1_ampere, nk_angulars_packed_e2m1_ampere);
    check("angulars_symmetric_e2m1_ampere", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_ampere);
    check("euclideans_packed_e2m1_ampere", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_ampere, nk_dots_pack_e2m1_ampere, nk_euclideans_packed_e2m1_ampere);
    check("euclideans_symmetric_e2m1_ampere", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_ampere);
    check("angulars_packed_i8_ampere", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_angulars_packed_i8_ampere);
    check("angulars_symmetric_i8_ampere", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_ampere);
    check("euclideans_packed_i8_ampere", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_euclideans_packed_i8_ampere);
    check("euclideans_symmetric_i8_ampere", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_ampere);
    check("angulars_packed_u8_ampere", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_angulars_packed_u8_ampere);
    check("angulars_symmetric_u8_ampere", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_ampere);
    check("euclideans_packed_u8_ampere", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_euclideans_packed_u8_ampere);
    check("euclideans_symmetric_u8_ampere", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_ampere);
    check("angulars_packed_i4_ampere", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_angulars_packed_i4_ampere);
    check("angulars_symmetric_i4_ampere", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_ampere);
    check("euclideans_packed_i4_ampere", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_euclideans_packed_i4_ampere);
    check("euclideans_symmetric_i4_ampere", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_ampere);
    check("angulars_packed_u4_ampere", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_angulars_packed_u4_ampere);
    check("angulars_symmetric_u4_ampere", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_ampere);
    check("euclideans_packed_u4_ampere", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_euclideans_packed_u4_ampere);
    check("euclideans_symmetric_u4_ampere", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_ampere);
    check("attention_packed_bf16_ampere", test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere, nk_attention_packed_bf16_ampere);
    check("attention_packed_f16_ampere", test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_ampere, nk_attention_pack_f16_ampere, nk_attention_packed_f16_ampere);
    check("attention_packed_e4m3_ampere", test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_ampere, nk_attention_pack_e4m3_ampere, nk_attention_packed_e4m3_ampere);
    check("attention_packed_i8_ampere", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_ampere, nk_attention_pack_i8_ampere, nk_attention_packed_i8_ampere);
    check("attention_packed_gradients_bf16_ampere", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere, nk_attention_packed_gradients_bf16_ampere);
#endif // NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_HOPPER
    check.section("Cross Hopper", nk_cap_hopper_k);
    check("dots_packed_bf16_hopper", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_hopper,
          nk_dots_pack_bf16_hopper, nk_dots_packed_bf16_hopper);
    check("dots_pack_bf16_hopper", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_hopper,
                                                         nk_dots_packed_shape_bf16_hopper, nk_dots_pack_bf16_hopper>);
    check("dots_contract_bf16_hopper",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_hopper, nk_dots_packed_bf16_hopper,
                                    nk_dots_symmetric_bf16_hopper>);
    check("dots_symmetric_bf16_hopper", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_hopper);
    check("dots_packed_f16_hopper", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_hopper,
          nk_dots_pack_f16_hopper, nk_dots_packed_f16_hopper);
    check("dots_pack_f16_hopper", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_hopper,
                                                        nk_dots_packed_shape_f16_hopper, nk_dots_pack_f16_hopper>);
    check("dots_contract_f16_hopper",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_hopper, nk_dots_packed_f16_hopper,
                                    nk_dots_symmetric_f16_hopper>);
    check("dots_symmetric_f16_hopper", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_hopper);
    check("dots_packed_e2m3_hopper", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_hopper,
          nk_dots_pack_e2m3_hopper, nk_dots_packed_e2m3_hopper);
    check("dots_pack_e2m3_hopper", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_hopper,
                                                         nk_dots_packed_shape_e2m3_hopper, nk_dots_pack_e2m3_hopper>);
    check("dots_contract_e2m3_hopper",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_hopper, nk_dots_packed_e2m3_hopper,
                                    nk_dots_symmetric_e2m3_hopper>);
    check("dots_symmetric_e2m3_hopper", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_hopper);
    check("dots_packed_e2m1_hopper", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_hopper,
          nk_dots_pack_e2m1_hopper, nk_dots_packed_e2m1_hopper);
    check("dots_pack_e2m1_hopper", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_hopper,
                                                         nk_dots_packed_shape_e2m1_hopper, nk_dots_pack_e2m1_hopper>);
    check("dots_contract_e2m1_hopper",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_hopper, nk_dots_packed_e2m1_hopper,
                                    nk_dots_symmetric_e2m1_hopper>);
    check("dots_symmetric_e2m1_hopper", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_hopper);
    check("dots_packed_nvfp4_hopper", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_hopper,
          nk_dots_pack_nvfp4_hopper, nk_dots_packed_nvfp4_hopper);
    check("dots_pack_nvfp4_hopper",
          test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_hopper,
                                nk_dots_packed_shape_nvfp4_hopper, nk_dots_pack_nvfp4_hopper>);
    check("dots_contract_nvfp4_hopper",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_hopper,
                                    nk_dots_packed_nvfp4_hopper, nk_dots_symmetric_nvfp4_hopper>);
    check("dots_symmetric_nvfp4_hopper", test_dots_symmetric<nvfp4_t, cuda_backend_t>, nk_dots_symmetric_nvfp4_hopper);
    check("dots_packed_mxfp4_hopper", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_hopper,
          nk_dots_pack_mxfp4_hopper, nk_dots_packed_mxfp4_hopper);
    check("dots_pack_mxfp4_hopper",
          test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_hopper,
                                nk_dots_packed_shape_mxfp4_hopper, nk_dots_pack_mxfp4_hopper>);
    check("dots_contract_mxfp4_hopper",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_hopper,
                                    nk_dots_packed_mxfp4_hopper, nk_dots_symmetric_mxfp4_hopper>);
    check("dots_symmetric_mxfp4_hopper", test_dots_symmetric<mxfp4_t, cuda_backend_t>, nk_dots_symmetric_mxfp4_hopper);
    check("dots_packed_mxfp8e4m3_hopper", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_hopper, nk_dots_pack_mxfp8e4m3_hopper, nk_dots_packed_mxfp8e4m3_hopper);
    check("dots_pack_mxfp8e4m3_hopper",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_hopper,
                                nk_dots_packed_shape_mxfp8e4m3_hopper, nk_dots_pack_mxfp8e4m3_hopper>);
    check("dots_contract_mxfp8e4m3_hopper",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_hopper,
                                    nk_dots_packed_mxfp8e4m3_hopper, nk_dots_symmetric_mxfp8e4m3_hopper>);
    check("dots_symmetric_mxfp8e4m3_hopper", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_hopper);
    check("dots_packed_mxfp8e5m2_hopper", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_hopper, nk_dots_pack_mxfp8e5m2_hopper, nk_dots_packed_mxfp8e5m2_hopper);
    check("dots_pack_mxfp8e5m2_hopper",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_hopper,
                                nk_dots_packed_shape_mxfp8e5m2_hopper, nk_dots_pack_mxfp8e5m2_hopper>);
    check("dots_contract_mxfp8e5m2_hopper",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_hopper,
                                    nk_dots_packed_mxfp8e5m2_hopper, nk_dots_symmetric_mxfp8e5m2_hopper>);
    check("dots_symmetric_mxfp8e5m2_hopper", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_hopper);
    check("dots_packed_i8_hopper", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_hopper,
          nk_dots_pack_i8_hopper, nk_dots_packed_i8_hopper);
    check("dots_pack_i8_hopper", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_hopper,
                                                       nk_dots_packed_shape_i8_hopper, nk_dots_pack_i8_hopper>);
    check("dots_contract_i8_hopper", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_hopper,
                                                               nk_dots_packed_i8_hopper, nk_dots_symmetric_i8_hopper>);
    check("dots_symmetric_i8_hopper", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_hopper);
    check("dots_packed_u8_hopper", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_hopper,
          nk_dots_pack_u8_hopper, nk_dots_packed_u8_hopper);
    check("dots_pack_u8_hopper", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_hopper,
                                                       nk_dots_packed_shape_u8_hopper, nk_dots_pack_u8_hopper>);
    check("dots_contract_u8_hopper", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_hopper,
                                                               nk_dots_packed_u8_hopper, nk_dots_symmetric_u8_hopper>);
    check("dots_symmetric_u8_hopper", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_hopper);
    check("dots_packed_i4_hopper", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_hopper,
          nk_dots_pack_i4_hopper, nk_dots_packed_i4_hopper);
    check("dots_pack_i4_hopper", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_hopper,
                                                       nk_dots_packed_shape_i4_hopper, nk_dots_pack_i4_hopper>);
    check("dots_contract_i4_hopper", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_hopper,
                                                               nk_dots_packed_i4_hopper, nk_dots_symmetric_i4_hopper>);
    check("dots_symmetric_i4_hopper", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_hopper);
    check("dots_packed_u4_hopper", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_hopper,
          nk_dots_pack_u4_hopper, nk_dots_packed_u4_hopper);
    check("dots_pack_u4_hopper", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_hopper,
                                                       nk_dots_packed_shape_u4_hopper, nk_dots_pack_u4_hopper>);
    check("dots_contract_u4_hopper", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_hopper,
                                                               nk_dots_packed_u4_hopper, nk_dots_symmetric_u4_hopper>);
    check("dots_symmetric_u4_hopper", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_hopper);
    check("angulars_packed_bf16_hopper", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_hopper,
          nk_dots_pack_bf16_hopper, nk_angulars_packed_bf16_hopper);
    check("angulars_symmetric_bf16_hopper", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_hopper);
    check("euclideans_packed_bf16_hopper", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_hopper, nk_dots_pack_bf16_hopper, nk_euclideans_packed_bf16_hopper);
    check("euclideans_symmetric_bf16_hopper", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_hopper);
    check("angulars_packed_f16_hopper", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_hopper,
          nk_dots_pack_f16_hopper, nk_angulars_packed_f16_hopper);
    check("angulars_symmetric_f16_hopper", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_hopper);
    check("euclideans_packed_f16_hopper", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_hopper,
          nk_dots_pack_f16_hopper, nk_euclideans_packed_f16_hopper);
    check("euclideans_symmetric_f16_hopper", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_hopper);
    check("angulars_packed_e2m3_hopper", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_hopper,
          nk_dots_pack_e2m3_hopper, nk_angulars_packed_e2m3_hopper);
    check("angulars_symmetric_e2m3_hopper", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_hopper);
    check("euclideans_packed_e2m3_hopper", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_hopper, nk_dots_pack_e2m3_hopper, nk_euclideans_packed_e2m3_hopper);
    check("euclideans_symmetric_e2m3_hopper", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_hopper);
    check("angulars_packed_e2m1_hopper", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_hopper,
          nk_dots_pack_e2m1_hopper, nk_angulars_packed_e2m1_hopper);
    check("angulars_symmetric_e2m1_hopper", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_hopper);
    check("euclideans_packed_e2m1_hopper", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_hopper, nk_dots_pack_e2m1_hopper, nk_euclideans_packed_e2m1_hopper);
    check("euclideans_symmetric_e2m1_hopper", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_hopper);
    check("angulars_packed_i8_hopper", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_hopper,
          nk_dots_pack_i8_hopper, nk_angulars_packed_i8_hopper);
    check("angulars_symmetric_i8_hopper", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_hopper);
    check("euclideans_packed_i8_hopper", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_hopper,
          nk_dots_pack_i8_hopper, nk_euclideans_packed_i8_hopper);
    check("euclideans_symmetric_i8_hopper", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_hopper);
    check("angulars_packed_u8_hopper", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_hopper,
          nk_dots_pack_u8_hopper, nk_angulars_packed_u8_hopper);
    check("angulars_symmetric_u8_hopper", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_hopper);
    check("euclideans_packed_u8_hopper", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_hopper,
          nk_dots_pack_u8_hopper, nk_euclideans_packed_u8_hopper);
    check("euclideans_symmetric_u8_hopper", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_hopper);
    check("angulars_packed_i4_hopper", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_hopper,
          nk_dots_pack_i4_hopper, nk_angulars_packed_i4_hopper);
    check("angulars_symmetric_i4_hopper", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_hopper);
    check("euclideans_packed_i4_hopper", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_hopper,
          nk_dots_pack_i4_hopper, nk_euclideans_packed_i4_hopper);
    check("euclideans_symmetric_i4_hopper", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_hopper);
    check("angulars_packed_u4_hopper", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_hopper,
          nk_dots_pack_u4_hopper, nk_angulars_packed_u4_hopper);
    check("angulars_symmetric_u4_hopper", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_hopper);
    check("euclideans_packed_u4_hopper", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_hopper,
          nk_dots_pack_u4_hopper, nk_euclideans_packed_u4_hopper);
    check("euclideans_symmetric_u4_hopper", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_hopper);
    check("attention_packed_bf16_hopper", test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_hopper, nk_attention_pack_bf16_hopper, nk_attention_packed_bf16_hopper);
    check("attention_packed_e4m3_hopper", test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_hopper, nk_attention_pack_e4m3_hopper, nk_attention_packed_e4m3_hopper);
    check("attention_packed_i8_hopper", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_hopper, nk_attention_pack_i8_hopper, nk_attention_packed_i8_hopper);
    check("attention_packed_gradients_bf16_hopper", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_hopper, nk_attention_pack_bf16_hopper, nk_attention_packed_gradients_bf16_hopper);
#endif // NUMKONG_TARGET_HOPPER

#if NUMKONG_TARGET_BLACKWELL
    check.section("Cross Blackwell", nk_cap_blackwell_k);
    check("dots_packed_bf16_blackwell", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_blackwell,
          nk_dots_pack_bf16_blackwell,
          with_workspace(nk_dots_packed_bf16_blackwell, nk_dots_packed_workspace_size_bf16_blackwell));
    check("dots_pack_bf16_blackwell",
          test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_blackwell,
                                nk_dots_packed_shape_bf16_blackwell, nk_dots_pack_bf16_blackwell>);
    check("dots_contract_bf16_blackwell",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_blackwell,
                                    nk_dots_packed_bf16_blackwell, nk_dots_symmetric_bf16_blackwell>);
    check("dots_symmetric_bf16_blackwell", test_dots_symmetric<bf16_t, cuda_backend_t>,
          nk_dots_symmetric_bf16_blackwell);
    check("dots_packed_f16_blackwell", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_blackwell,
          nk_dots_pack_f16_blackwell,
          with_workspace(nk_dots_packed_f16_blackwell, nk_dots_packed_workspace_size_f16_blackwell));
    check("dots_pack_f16_blackwell",
          test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_blackwell,
                                nk_dots_packed_shape_f16_blackwell, nk_dots_pack_f16_blackwell>);
    check("dots_contract_f16_blackwell",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_blackwell,
                                    nk_dots_packed_f16_blackwell, nk_dots_symmetric_f16_blackwell>);
    check("dots_symmetric_f16_blackwell", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_blackwell);
    check("dots_packed_e5m2_blackwell", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_blackwell,
          nk_dots_pack_e5m2_blackwell,
          with_workspace(nk_dots_packed_e5m2_blackwell, nk_dots_packed_workspace_size_e5m2_blackwell));
    check("dots_pack_e5m2_blackwell",
          test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwell,
                                nk_dots_packed_shape_e5m2_blackwell, nk_dots_pack_e5m2_blackwell>);
    check("dots_contract_e5m2_blackwell",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwell,
                                    nk_dots_packed_e5m2_blackwell, nk_dots_symmetric_e5m2_blackwell>);
    check("dots_symmetric_e5m2_blackwell", test_dots_symmetric<e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_e5m2_blackwell);
    check("dots_packed_e4m3_blackwell", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_blackwell,
          nk_dots_pack_e4m3_blackwell,
          with_workspace(nk_dots_packed_e4m3_blackwell, nk_dots_packed_workspace_size_e4m3_blackwell));
    check("dots_pack_e4m3_blackwell",
          test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwell,
                                nk_dots_packed_shape_e4m3_blackwell, nk_dots_pack_e4m3_blackwell>);
    check("dots_contract_e4m3_blackwell",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwell,
                                    nk_dots_packed_e4m3_blackwell, nk_dots_symmetric_e4m3_blackwell>);
    check("dots_symmetric_e4m3_blackwell", test_dots_symmetric<e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_e4m3_blackwell);
    check("dots_packed_e3m2_blackwell", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_blackwell,
          nk_dots_pack_e3m2_blackwell,
          with_workspace(nk_dots_packed_e3m2_blackwell, nk_dots_packed_workspace_size_e3m2_blackwell));
    check("dots_pack_e3m2_blackwell",
          test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwell,
                                nk_dots_packed_shape_e3m2_blackwell, nk_dots_pack_e3m2_blackwell>);
    check("dots_contract_e3m2_blackwell",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwell,
                                    nk_dots_packed_e3m2_blackwell, nk_dots_symmetric_e3m2_blackwell>);
    check("dots_symmetric_e3m2_blackwell", test_dots_symmetric<e3m2_t, cuda_backend_t>,
          nk_dots_symmetric_e3m2_blackwell);
    check("dots_packed_e2m3_blackwell", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_blackwell,
          nk_dots_pack_e2m3_blackwell,
          with_workspace(nk_dots_packed_e2m3_blackwell, nk_dots_packed_workspace_size_e2m3_blackwell));
    check("dots_pack_e2m3_blackwell",
          test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwell,
                                nk_dots_packed_shape_e2m3_blackwell, nk_dots_pack_e2m3_blackwell>);
    check("dots_contract_e2m3_blackwell",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwell,
                                    nk_dots_packed_e2m3_blackwell, nk_dots_symmetric_e2m3_blackwell>);
    check("dots_symmetric_e2m3_blackwell", test_dots_symmetric<e2m3_t, cuda_backend_t>,
          nk_dots_symmetric_e2m3_blackwell);
    check("dots_packed_e2m1_blackwell", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_blackwell,
          nk_dots_pack_e2m1_blackwell,
          with_workspace(nk_dots_packed_e2m1_blackwell, nk_dots_packed_workspace_size_e2m1_blackwell));
    check("dots_pack_e2m1_blackwell",
          test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwell,
                                nk_dots_packed_shape_e2m1_blackwell, nk_dots_pack_e2m1_blackwell>);
    check("dots_contract_e2m1_blackwell",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwell,
                                    nk_dots_packed_e2m1_blackwell, nk_dots_symmetric_e2m1_blackwell>);
    check("dots_symmetric_e2m1_blackwell", test_dots_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_dots_symmetric_e2m1_blackwell);
    check("dots_packed_nvfp4_blackwell", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_blackwell,
          nk_dots_pack_nvfp4_blackwell, nk_dots_packed_nvfp4_blackwell);
    check("dots_pack_nvfp4_blackwell",
          test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_blackwell,
                                nk_dots_packed_shape_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell>);
    check("dots_contract_nvfp4_blackwell",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_blackwell,
                                    nk_dots_packed_nvfp4_blackwell, nk_dots_symmetric_nvfp4_blackwell>);
    check("dots_symmetric_nvfp4_blackwell", test_dots_symmetric<nvfp4_t, cuda_backend_t>,
          nk_dots_symmetric_nvfp4_blackwell);
    check("dots_packed_mxfp4_blackwell", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_blackwell,
          nk_dots_pack_mxfp4_blackwell, nk_dots_packed_mxfp4_blackwell);
    check("dots_pack_mxfp4_blackwell",
          test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_blackwell,
                                nk_dots_packed_shape_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell>);
    check("dots_contract_mxfp4_blackwell",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_blackwell,
                                    nk_dots_packed_mxfp4_blackwell, nk_dots_symmetric_mxfp4_blackwell>);
    check("dots_symmetric_mxfp4_blackwell", test_dots_symmetric<mxfp4_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp4_blackwell);
    check("dots_packed_mxfp8e4m3_blackwell", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell, nk_dots_packed_mxfp8e4m3_blackwell);
    check("dots_pack_mxfp8e4m3_blackwell",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_blackwell,
                                nk_dots_packed_shape_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell>);
    check("dots_contract_mxfp8e4m3_blackwell",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_blackwell,
                                    nk_dots_packed_mxfp8e4m3_blackwell, nk_dots_symmetric_mxfp8e4m3_blackwell>);
    check("dots_symmetric_mxfp8e4m3_blackwell", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_blackwell);
    check("dots_packed_mxfp8e5m2_blackwell", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell, nk_dots_packed_mxfp8e5m2_blackwell);
    check("dots_pack_mxfp8e5m2_blackwell",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_blackwell,
                                nk_dots_packed_shape_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell>);
    check("dots_contract_mxfp8e5m2_blackwell",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_blackwell,
                                    nk_dots_packed_mxfp8e5m2_blackwell, nk_dots_symmetric_mxfp8e5m2_blackwell>);
    check("dots_symmetric_mxfp8e5m2_blackwell", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_blackwell);
    check("angulars_packed_bf16_blackwell", test_angulars_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_blackwell, nk_dots_pack_bf16_blackwell, nk_angulars_packed_bf16_blackwell);
    check("angulars_symmetric_bf16_blackwell", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_blackwell);
    check("euclideans_packed_bf16_blackwell", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_blackwell, nk_dots_pack_bf16_blackwell, nk_euclideans_packed_bf16_blackwell);
    check("euclideans_symmetric_bf16_blackwell", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_blackwell);
    check("angulars_packed_f16_blackwell", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_blackwell,
          nk_dots_pack_f16_blackwell, nk_angulars_packed_f16_blackwell);
    check("angulars_symmetric_f16_blackwell", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_blackwell);
    check("euclideans_packed_f16_blackwell", test_euclideans_packed<f16_t, cuda_backend_t>,
          nk_dots_pack_size_f16_blackwell, nk_dots_pack_f16_blackwell, nk_euclideans_packed_f16_blackwell);
    check("euclideans_symmetric_f16_blackwell", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_blackwell);
    check("angulars_packed_e5m2_blackwell", test_angulars_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwell, nk_dots_pack_e5m2_blackwell, nk_angulars_packed_e5m2_blackwell);
    check("angulars_symmetric_e5m2_blackwell", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_blackwell);
    check("euclideans_packed_e5m2_blackwell", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwell, nk_dots_pack_e5m2_blackwell, nk_euclideans_packed_e5m2_blackwell);
    check("euclideans_symmetric_e5m2_blackwell", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_blackwell);
    check("angulars_packed_e4m3_blackwell", test_angulars_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell, nk_angulars_packed_e4m3_blackwell);
    check("angulars_symmetric_e4m3_blackwell", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_blackwell);
    check("euclideans_packed_e4m3_blackwell", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell, nk_euclideans_packed_e4m3_blackwell);
    check("euclideans_symmetric_e4m3_blackwell", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_blackwell);
    check("angulars_packed_e3m2_blackwell", test_angulars_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwell, nk_dots_pack_e3m2_blackwell, nk_angulars_packed_e3m2_blackwell);
    check("angulars_symmetric_e3m2_blackwell", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_blackwell);
    check("euclideans_packed_e3m2_blackwell", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwell, nk_dots_pack_e3m2_blackwell, nk_euclideans_packed_e3m2_blackwell);
    check("euclideans_symmetric_e3m2_blackwell", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_blackwell);
    check("angulars_packed_e2m3_blackwell", test_angulars_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwell, nk_dots_pack_e2m3_blackwell, nk_angulars_packed_e2m3_blackwell);
    check("angulars_symmetric_e2m3_blackwell", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_blackwell);
    check("euclideans_packed_e2m3_blackwell", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwell, nk_dots_pack_e2m3_blackwell, nk_euclideans_packed_e2m3_blackwell);
    check("euclideans_symmetric_e2m3_blackwell", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_blackwell);
    check("angulars_packed_e2m1_blackwell", test_angulars_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwell, nk_dots_pack_e2m1_blackwell, nk_angulars_packed_e2m1_blackwell);
    check("angulars_symmetric_e2m1_blackwell", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_blackwell);
    check("euclideans_packed_e2m1_blackwell", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwell, nk_dots_pack_e2m1_blackwell, nk_euclideans_packed_e2m1_blackwell);
    check("euclideans_symmetric_e2m1_blackwell", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_blackwell);
    check("angulars_packed_nvfp4_blackwell", test_angulars_packed<nvfp4_t, cuda_backend_t>,
          nk_dots_pack_size_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell, nk_angulars_packed_nvfp4_blackwell);
    check("angulars_symmetric_nvfp4_blackwell", test_angulars_symmetric<nvfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_nvfp4_blackwell);
    check("euclideans_packed_nvfp4_blackwell", test_euclideans_packed<nvfp4_t, cuda_backend_t>,
          nk_dots_pack_size_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell, nk_euclideans_packed_nvfp4_blackwell);
    check("euclideans_symmetric_nvfp4_blackwell", test_euclideans_symmetric<nvfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_nvfp4_blackwell);
    check("angulars_packed_mxfp4_blackwell", test_angulars_packed<mxfp4_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell, nk_angulars_packed_mxfp4_blackwell);
    check("angulars_symmetric_mxfp4_blackwell", test_angulars_symmetric<mxfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp4_blackwell);
    check("euclideans_packed_mxfp4_blackwell", test_euclideans_packed<mxfp4_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell, nk_euclideans_packed_mxfp4_blackwell);
    check("euclideans_symmetric_mxfp4_blackwell", test_euclideans_symmetric<mxfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp4_blackwell);
    check("angulars_packed_mxfp8e4m3_blackwell", test_angulars_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell,
          nk_angulars_packed_mxfp8e4m3_blackwell);
    check("angulars_symmetric_mxfp8e4m3_blackwell", test_angulars_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e4m3_blackwell);
    check("euclideans_packed_mxfp8e4m3_blackwell", test_euclideans_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell,
          nk_euclideans_packed_mxfp8e4m3_blackwell);
    check("euclideans_symmetric_mxfp8e4m3_blackwell", test_euclideans_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e4m3_blackwell);
    check("angulars_packed_mxfp8e5m2_blackwell", test_angulars_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell,
          nk_angulars_packed_mxfp8e5m2_blackwell);
    check("angulars_symmetric_mxfp8e5m2_blackwell", test_angulars_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e5m2_blackwell);
    check("euclideans_packed_mxfp8e5m2_blackwell", test_euclideans_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell,
          nk_euclideans_packed_mxfp8e5m2_blackwell);
    check("euclideans_symmetric_mxfp8e5m2_blackwell", test_euclideans_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e5m2_blackwell);
    check("dots_packed_i8_blackwell", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_blackwell,
          nk_dots_pack_i8_blackwell, nk_dots_packed_i8_blackwell);
    check("dots_pack_i8_blackwell",
          test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_blackwell, nk_dots_packed_shape_i8_blackwell,
                                nk_dots_pack_i8_blackwell>);
    check("dots_contract_i8_blackwell",
          test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_blackwell, nk_dots_packed_i8_blackwell,
                                    nk_dots_symmetric_i8_blackwell>);
    check("dots_symmetric_i8_blackwell", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_blackwell);
    check("dots_packed_u8_blackwell", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_blackwell,
          nk_dots_pack_u8_blackwell, nk_dots_packed_u8_blackwell);
    check("dots_pack_u8_blackwell",
          test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_blackwell, nk_dots_packed_shape_u8_blackwell,
                                nk_dots_pack_u8_blackwell>);
    check("dots_contract_u8_blackwell",
          test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_blackwell, nk_dots_packed_u8_blackwell,
                                    nk_dots_symmetric_u8_blackwell>);
    check("dots_symmetric_u8_blackwell", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_blackwell);
    check("dots_packed_i4_blackwell", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_blackwell,
          nk_dots_pack_i4_blackwell, nk_dots_packed_i4_blackwell);
    check("dots_pack_i4_blackwell",
          test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_blackwell,
                                nk_dots_packed_shape_i4_blackwell, nk_dots_pack_i4_blackwell>);
    check("dots_contract_i4_blackwell",
          test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_blackwell, nk_dots_packed_i4_blackwell,
                                    nk_dots_symmetric_i4_blackwell>);
    check("dots_symmetric_i4_blackwell", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_blackwell);
    check("dots_packed_u4_blackwell", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_blackwell,
          nk_dots_pack_u4_blackwell, nk_dots_packed_u4_blackwell);
    check("dots_pack_u4_blackwell",
          test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_blackwell,
                                nk_dots_packed_shape_u4_blackwell, nk_dots_pack_u4_blackwell>);
    check("dots_contract_u4_blackwell",
          test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_blackwell, nk_dots_packed_u4_blackwell,
                                    nk_dots_symmetric_u4_blackwell>);
    check("dots_symmetric_u4_blackwell", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_blackwell);
    check("angulars_packed_i8_blackwell", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_blackwell,
          nk_dots_pack_i8_blackwell, nk_angulars_packed_i8_blackwell);
    check("angulars_symmetric_i8_blackwell", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_blackwell);
    check("euclideans_packed_i8_blackwell", test_euclideans_packed<i8_t, cuda_backend_t>,
          nk_dots_pack_size_i8_blackwell, nk_dots_pack_i8_blackwell, nk_euclideans_packed_i8_blackwell);
    check("euclideans_symmetric_i8_blackwell", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_blackwell);
    check("angulars_packed_u8_blackwell", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_blackwell,
          nk_dots_pack_u8_blackwell, nk_angulars_packed_u8_blackwell);
    check("angulars_symmetric_u8_blackwell", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_blackwell);
    check("euclideans_packed_u8_blackwell", test_euclideans_packed<u8_t, cuda_backend_t>,
          nk_dots_pack_size_u8_blackwell, nk_dots_pack_u8_blackwell, nk_euclideans_packed_u8_blackwell);
    check("euclideans_symmetric_u8_blackwell", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_blackwell);
    check("angulars_packed_i4_blackwell", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_blackwell,
          nk_dots_pack_i4_blackwell, nk_angulars_packed_i4_blackwell);
    check("angulars_symmetric_i4_blackwell", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_blackwell);
    check("euclideans_packed_i4_blackwell", test_euclideans_packed<i4x2_t, cuda_backend_t>,
          nk_dots_pack_size_i4_blackwell, nk_dots_pack_i4_blackwell, nk_euclideans_packed_i4_blackwell);
    check("euclideans_symmetric_i4_blackwell", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_blackwell);
    check("angulars_packed_u4_blackwell", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_blackwell,
          nk_dots_pack_u4_blackwell, nk_angulars_packed_u4_blackwell);
    check("angulars_symmetric_u4_blackwell", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_blackwell);
    check("euclideans_packed_u4_blackwell", test_euclideans_packed<u4x2_t, cuda_backend_t>,
          nk_dots_pack_size_u4_blackwell, nk_dots_pack_u4_blackwell, nk_euclideans_packed_u4_blackwell);
    check("euclideans_symmetric_u4_blackwell", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_blackwell);
    check("attention_packed_bf16_blackwell",
          test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_blackwell, nk_attention_pack_bf16_blackwell, nk_attention_packed_bf16_blackwell);
    check("attention_packed_f16_blackwell",
          test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_blackwell, nk_attention_pack_f16_blackwell, nk_attention_packed_f16_blackwell);
    check("attention_packed_e4m3_blackwell",
          test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwell, nk_attention_pack_e4m3_blackwell, nk_attention_packed_e4m3_blackwell);
    check("attention_packed_i8_blackwell", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_blackwell, nk_attention_pack_i8_blackwell, nk_attention_packed_i8_blackwell);
    check("attention_packed_gradients_bf16_blackwell", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_blackwell, nk_attention_pack_bf16_blackwell,
          nk_attention_packed_gradients_bf16_blackwell);
    check("rmsds_packed_f16_blackwell", test_rmsds_packed_f16<cuda_backend_t>, nk_rmsds_packed_f16_blackwell);
    check("chamfers_packed_f16_blackwell", test_shapes_flips_f16<cuda_backend_t, nk_kernel_chamfers_packed_k>,
          nk_chamfers_packed_f16_blackwell);
    check("hausdorffs_packed_f16_blackwell", test_shapes_flips_f16<cuda_backend_t, nk_kernel_hausdorffs_packed_k>,
          nk_hausdorffs_packed_f16_blackwell);
#endif // NUMKONG_TARGET_BLACKWELL

#if NUMKONG_TARGET_BLACKWELLRTX
    check.section("Cross Blackwell RTX", nk_cap_blackwellrtx_k);
    check("dots_packed_e5m2_blackwellrtx", test_dots_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_dots_packed_e5m2_blackwellrtx);
    check("dots_pack_e5m2_blackwellrtx",
          test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwellrtx,
                                nk_dots_packed_shape_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx>);
    check("dots_contract_e5m2_blackwellrtx",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwellrtx,
                                    nk_dots_packed_e5m2_blackwellrtx, nk_dots_symmetric_e5m2_blackwellrtx>);
    check("dots_symmetric_e5m2_blackwellrtx", test_dots_symmetric<e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_e5m2_blackwellrtx);
    check("dots_packed_e4m3_blackwellrtx", test_dots_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_dots_packed_e4m3_blackwellrtx);
    check("dots_pack_e4m3_blackwellrtx",
          test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwellrtx,
                                nk_dots_packed_shape_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx>);
    check("dots_contract_e4m3_blackwellrtx",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwellrtx,
                                    nk_dots_packed_e4m3_blackwellrtx, nk_dots_symmetric_e4m3_blackwellrtx>);
    check("dots_symmetric_e4m3_blackwellrtx", test_dots_symmetric<e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_e4m3_blackwellrtx);
    check("dots_packed_e3m2_blackwellrtx", test_dots_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_dots_packed_e3m2_blackwellrtx);
    check("dots_pack_e3m2_blackwellrtx",
          test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwellrtx,
                                nk_dots_packed_shape_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx>);
    check("dots_contract_e3m2_blackwellrtx",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwellrtx,
                                    nk_dots_packed_e3m2_blackwellrtx, nk_dots_symmetric_e3m2_blackwellrtx>);
    check("dots_symmetric_e3m2_blackwellrtx", test_dots_symmetric<e3m2_t, cuda_backend_t>,
          nk_dots_symmetric_e3m2_blackwellrtx);
    check("dots_packed_e2m3_blackwellrtx", test_dots_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_dots_packed_e2m3_blackwellrtx);
    check("dots_pack_e2m3_blackwellrtx",
          test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwellrtx,
                                nk_dots_packed_shape_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx>);
    check("dots_contract_e2m3_blackwellrtx",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwellrtx,
                                    nk_dots_packed_e2m3_blackwellrtx, nk_dots_symmetric_e2m3_blackwellrtx>);
    check("dots_symmetric_e2m3_blackwellrtx", test_dots_symmetric<e2m3_t, cuda_backend_t>,
          nk_dots_symmetric_e2m3_blackwellrtx);
    check("dots_packed_e2m1_blackwellrtx", test_dots_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_dots_packed_e2m1_blackwellrtx);
    check("dots_pack_e2m1_blackwellrtx",
          test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwellrtx,
                                nk_dots_packed_shape_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx>);
    check("dots_contract_e2m1_blackwellrtx",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwellrtx,
                                    nk_dots_packed_e2m1_blackwellrtx, nk_dots_symmetric_e2m1_blackwellrtx>);
    check("dots_symmetric_e2m1_blackwellrtx", test_dots_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_dots_symmetric_e2m1_blackwellrtx);
    check("angulars_packed_e5m2_blackwellrtx", test_angulars_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_angulars_packed_e5m2_blackwellrtx);
    check("angulars_symmetric_e5m2_blackwellrtx", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_blackwellrtx);
    check("euclideans_packed_e5m2_blackwellrtx", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_euclideans_packed_e5m2_blackwellrtx);
    check("euclideans_symmetric_e5m2_blackwellrtx", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_blackwellrtx);
    check("angulars_packed_e4m3_blackwellrtx", test_angulars_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_angulars_packed_e4m3_blackwellrtx);
    check("angulars_symmetric_e4m3_blackwellrtx", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_blackwellrtx);
    check("euclideans_packed_e4m3_blackwellrtx", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_euclideans_packed_e4m3_blackwellrtx);
    check("euclideans_symmetric_e4m3_blackwellrtx", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_blackwellrtx);
    check("angulars_packed_e3m2_blackwellrtx", test_angulars_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_angulars_packed_e3m2_blackwellrtx);
    check("angulars_symmetric_e3m2_blackwellrtx", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_blackwellrtx);
    check("euclideans_packed_e3m2_blackwellrtx", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_euclideans_packed_e3m2_blackwellrtx);
    check("euclideans_symmetric_e3m2_blackwellrtx", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_blackwellrtx);
    check("angulars_packed_e2m3_blackwellrtx", test_angulars_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_angulars_packed_e2m3_blackwellrtx);
    check("angulars_symmetric_e2m3_blackwellrtx", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_blackwellrtx);
    check("euclideans_packed_e2m3_blackwellrtx", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_euclideans_packed_e2m3_blackwellrtx);
    check("euclideans_symmetric_e2m3_blackwellrtx", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_blackwellrtx);
    check("angulars_packed_e2m1_blackwellrtx", test_angulars_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_angulars_packed_e2m1_blackwellrtx);
    check("angulars_symmetric_e2m1_blackwellrtx", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_blackwellrtx);
    check("euclideans_packed_e2m1_blackwellrtx", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_euclideans_packed_e2m1_blackwellrtx);
    check("euclideans_symmetric_e2m1_blackwellrtx", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_blackwellrtx);
    check("attention_packed_e4m3_blackwellrtx",
          test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwellrtx, nk_attention_pack_e4m3_blackwellrtx,
          nk_attention_packed_e4m3_blackwellrtx);
#endif // NUMKONG_TARGET_BLACKWELLRTX

#if NUMKONG_TARGET_BLACKWELLULTRA
    check.section("Cross Blackwell Ultra", nk_cap_blackwellultra_k);
    check("attention_packed_bf16_blackwellultra",
          test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_blackwellultra, nk_attention_pack_bf16_blackwellultra,
          nk_attention_packed_bf16_blackwellultra);
    check("attention_packed_f16_blackwellultra",
          test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_blackwellultra, nk_attention_pack_f16_blackwellultra,
          nk_attention_packed_f16_blackwellultra);
    check("attention_packed_e4m3_blackwellultra",
          test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwellultra, nk_attention_pack_e4m3_blackwellultra,
          nk_attention_packed_e4m3_blackwellultra);
    check("attention_packed_i8_blackwellultra",
          test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_blackwellultra, nk_attention_pack_i8_blackwellultra,
          nk_attention_packed_i8_blackwellultra);
    check("attention_packed_gradients_bf16_blackwellultra", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_blackwellultra, nk_attention_pack_bf16_blackwellultra,
          nk_attention_packed_gradients_bf16_blackwellultra);
#endif // NUMKONG_TARGET_BLACKWELLULTRA

    test_cross_dispatch<cuda_backend_t>(check);
}

#else  // !NUMKONG_ARCH_CUDA_
void test_cross_cuda(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::test
