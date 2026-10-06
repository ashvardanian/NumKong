/**
 *  @brief SIMD-accelerated Similarity Measures and Distance Functions.
 *  @file include/numkong.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *
 *  Umbrella header that includes all domain-specific kernel headers
 *  and the runtime capability detection infrastructure.
 */

#ifndef NK_NUMKONG_H
#define NK_NUMKONG_H

#include "numkong/capabilities.h" // Runtime detection, like `nk_capabilities_x86_`
#include "numkong/scalar.h"       // Scalar math: sqrt, rsqrt, fma, saturating, order, like `nk_f32_sqrt`
#include "numkong/cast.h"         // Type conversions, like `nk_cast`
#include "numkong/set.h"          // Hamming, Jaccard, like `nk_hamming_u1`
#include "numkong/curved.h"       // Mahalanobis, Bilinear Forms, like `nk_bilinear_f64`
#include "numkong/dot.h"          // Inner (dot) product and its conjugate, like `nk_dot_f32`
#include "numkong/dots.h"         // GEMM-style MxN batched dot-products, like `nk_dots_packed_size_bf16`
#include "numkong/each.h"         // Weighted Sum, Fused-Multiply-Add, like `nk_each_scale_f64`
#include "numkong/geospatial.h"   // Haversine and Vincenty, like `nk_haversine_f64`
#include "numkong/mesh.h"         // RMSD, Kabsch, Umeyama, like `nk_rmsd_f64`
#include "numkong/probability.h"  // Kullback-Leibler, Jensen-Shannon, like `nk_kld_f16`
#include "numkong/reduce.h"       // Horizontal MinMax & Moments reductions, like `nk_reduce_moments_f64`
#include "numkong/sets.h"         // Hamming & Jaccard for binary sets, like `nk_hammings_packed_u1`
#include "numkong/sparse.h"       // Set Intersections and Sparse Dot Products, like `nk_sparse_intersect_u16`
#include "numkong/spatial.h"      // Euclidean, Angular, like `nk_euclidean_f64`
#include "numkong/spatials.h"     // Batched Angular & Euclidean distances, like `nk_angulars_packed_f32`
#include "numkong/maxsim.h"       // MaxSim: Multi-Vector Maximum Similarity, like `nk_maxsim_packed_f32`
#include "numkong/trigonometry.h" // Sin, Cos, Atan, like `nk_each_sin_f64`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  Returns the output dtype for a given metric kind and input dtype.
 */
NK_PUBLIC nk_dtype_t nk_kernel_output_dtype(nk_kernel_kind_t kind, nk_dtype_t input) {
    switch (kind) {
    case nk_kernel_dot_k:
    case nk_kernel_vdot_k:
    case nk_kernel_dots_packed_k:
    case nk_kernel_dots_symmetric_k: return nk_dot_output_dtype(input);
    case nk_kernel_angular_k:
    case nk_kernel_angulars_packed_k:
    case nk_kernel_angulars_symmetric_k: return nk_angular_output_dtype(input);
    case nk_kernel_euclidean_k:
    case nk_kernel_euclideans_packed_k:
    case nk_kernel_euclideans_symmetric_k: return nk_euclidean_output_dtype(input);
    case nk_kernel_sqeuclidean_k: return nk_sqeuclidean_output_dtype(input);
    case nk_kernel_bilinear_k: return nk_bilinear_output_dtype(input);
    case nk_kernel_mahalanobis_k: return nk_mahalanobis_output_dtype(input);
    case nk_kernel_hamming_k:
    case nk_kernel_hammings_packed_k:
    case nk_kernel_hammings_symmetric_k: return nk_hamming_output_dtype(input);
    case nk_kernel_jaccard_k:
    case nk_kernel_jaccards_packed_k:
    case nk_kernel_jaccards_symmetric_k: return nk_jaccard_output_dtype(input);
    case nk_kernel_haversine_k: return nk_haversine_output_dtype(input);
    case nk_kernel_vincenty_k: return nk_vincenty_output_dtype(input);
    case nk_kernel_kld_k:
    case nk_kernel_jsd_k: return nk_probability_output_dtype(input);
    case nk_kernel_rmsd_k:
    case nk_kernel_kabsch_k:
    case nk_kernel_umeyama_k: return nk_mesh_metric_dtype(input);
    case nk_kernel_sparse_dot_k: return nk_sparse_dot_output_dtype(input);
    case nk_kernel_maxsim_packed_k: return nk_maxsim_output_dtype(input);
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_NUMKONG_H
