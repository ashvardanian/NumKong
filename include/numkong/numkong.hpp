/**
 *  @file include/numkong/numkong.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2026
 *  @brief NumKong SDK for C++23 and newer.
 *
 *  C lacks a strong type system and the composable infrastructure that C++ templates and Rust
 *  traits give complex kernels and data structures. Unlike C++, C also lacks function overloading,
 *  namespaces and templates, so it needs verbose signatures and naming conventions like:
 *
 *  @code{.c}
 *  nk_status_t nk_dot_f64_best(nk_f64_t const*, nk_f64_t const*, nk_size_t, nk_f64_t *, nk_capability_t, void *);
 *  nk_status_t nk_dot_f32_best(nk_f32_t const*, nk_f32_t const*, nk_size_t, nk_f64_t *, nk_capability_t, void *);
 *  nk_status_t nk_dot_f16_best(nk_f16_t const*, nk_f16_t const*, nk_size_t, nk_f32_t *, nk_capability_t, void *);
 *  nk_status_t nk_dot_bf16_best(nk_bf16_t const*, nk_bf16_t const*, nk_size_t, nk_f32_t *, nk_capability_t, void *);
 *  nk_status_t nk_dot_e4m3_best(nk_e4m3_t const*, nk_e4m3_t const*, nk_size_t, nk_f32_t *, nk_capability_t, void *);
 *  nk_status_t nk_dot_e5m2_best(nk_e5m2_t const*, nk_e5m2_t const*, nk_size_t, nk_f32_t *, nk_capability_t, void *);
 *  @endcode
 *
 *  As opposed to C++, where the same kernels come in three layers. The raw-pointer layer mirrors
 *  the C ABI one to one, keeping lengths, strides and task windows for sharded launches:
 *
 *  @code{.cpp}
 *  template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t>
 *  status_t dot(in_type_ const *, in_type_ const *, std::size_t, result_type_ *,
 *               nk_capability_t = default_capabilities(), void * = nullptr);
 *  @endcode
 *
 *  The concept layer takes any contiguous run satisfying @c vector_of, or a matrix satisfying
 *  @c const_matrix_of, and returns the result with the @c status_t explaining it:
 *
 *  @code{.cpp}
 *  std::vector<nk::f32_t> a {1, 2, 3}, b {4, 5, 6};
 *  auto [dot, status] = nk::dot<nk::f32_t>(a, b); // nk::expected<nk::f64_t>
 *  @endcode
 *
 *  Runs must fill whole storage values, so odd bit and nibble lengths pad with zero bits up to the
 *  next byte; a view ending mid-value is refused with @c unexpected_dimensions_k.
 *
 *  The owning layer allocates through a caller's allocator for repeated workloads, like
 *  @c tensor, @c packed_matrix, @c packed_maxsim and @c packed_attention:
 *
 *  @code{.cpp}
 *  auto [b_matrix, b_status] = nk::matrix<nk::f32_t>::zeros({256, 512});
 *  auto [packed, packed_status] = nk::packed_matrix<nk::f32_t>::make(b_matrix.view());
 *  @endcode
 *
 *  In HPC implementations, where pretty much every kernel and every datatype uses different
 *  Assembly instructions on different CPU generations/models, those higher-level abstractions
 *  aren't always productive for the primary implementation, but they can still be handy as a
 *  higher-level API for NumKong. Given a mask of no capability, they also verify the algorithms,
 *  upcasting to much larger number types like @c f118_t.
 */

#ifndef NUMKONG_NUMKONG_HPP
#define NUMKONG_NUMKONG_HPP

#include "numkong/random.hpp"
#include "numkong/cast.hpp"
#include "numkong/dot.hpp"
#include "numkong/spatial.hpp"
#include "numkong/spatials.hpp"
#include "numkong/probability.hpp"
#include "numkong/each.hpp"
#include "numkong/reduce.hpp"
#include "numkong/curved.hpp"
#include "numkong/geospatial.hpp"
#include "numkong/sparse.hpp"
#include "numkong/set.hpp"
#include "numkong/mesh.hpp"
#include "numkong/trigonometry.hpp"
#include "numkong/dots.hpp"
#include "numkong/matrix.hpp"
#include "numkong/maxsim.hpp"
#include "numkong/tensor.hpp"

#endif // NUMKONG_NUMKONG_HPP
