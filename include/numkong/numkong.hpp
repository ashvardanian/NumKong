/**
 *  @brief NumKong SDK for C++23 and newer.
 *  @file include/numkong/numkong.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2026
 *
 *  C doesn't have a strong type system or composable infrastructure for complex kernels
 *  and datastructures like the C++ templates and Rust traits. Unlike C++, C also lacks
 *  function overloading, namespaces and templates, thus requiring verbose signatures and
 *  naming conventions, like:
 *
 *  @code{c}
 *  void nk_dot_f64(nk_f64_t const*, nk_f64_t const*, nk_size_t, nk_f64_t *);
 *  void nk_dot_f32(nk_f32_t const*, nk_f32_t const*, nk_size_t, nk_f64_t *);
 *  void nk_dot_f16(nk_f16_t const*, nk_f16_t const*, nk_size_t, nk_f32_t *);
 *  void nk_dot_bf16(nk_bf16_t const*, nk_bf16_t const*, nk_size_t, nk_f32_t *);
 *  void nk_dot_e4m3(nk_e4m3_t const*, nk_e4m3_t const*, nk_size_t, nk_f32_t *);
 *  void nk_dot_e5m2(nk_e5m2_t const*, nk_e5m2_t const*, nk_size_t, nk_f32_t *);
 *  @endcode
 *
 *  As opposed to C++:
 *
 *  @code{cpp}
 *  namespace ashvardanian::numkong {
 *      template <typename input_type_, typename result_type_>
 *      void dot(input_type_ const*, input_type_ const*, size_t, result_type_ *);
 *  }
 *
 *  In HPC implementations, where pretty much every kernel and every datatype uses different
 *  Assembly instructions on different CPU generations/models, those higher-level abstractions
 *  aren't always productive for the primary implementation, but they can still be handy as
 *  a higher-level API for NumKong. They are also used for algorithm verification in no-SIMD
 *  mode, upcasting to much larger number types like `f118_t`.
 */

#ifndef NK_NUMKONG_HPP
#define NK_NUMKONG_HPP

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

#endif // NK_NUMKONG_HPP
