/**
 *  @file include/numkong/numkong.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief SIMD-accelerated Similarity Measures and Distance Functions.
 *
 *  Umbrella header that includes all domain-specific kernel headers and the runtime capability
 *  detection infrastructure.
 */

#ifndef NUMKONG_NUMKONG_H
#define NUMKONG_NUMKONG_H

#include "numkong/capabilities.h" // Capabilities, kernel kinds, the pick and detection, like `nk_find_kernel_punned`
#include "numkong/memory.h"       // Host and device allocations
#include "numkong/scalar.h"       // Scalar math: sqrt, rsqrt, fma, saturating, order, like `nk_f32_sqrt_best`
#include "numkong/cast.h"         // Type conversions, like `nk_cast_best`
#include "numkong/set.h"          // Hamming, Jaccard, like `nk_hamming_u1_best`
#include "numkong/curved.h"       // Mahalanobis, Bilinear Forms, like `nk_bilinear_f64_best`
#include "numkong/dot.h"          // Inner (dot) product and its conjugate, like `nk_dot_f32_best`
#include "numkong/dots.h"         // GEMM-style MxN batched dot-products, like `nk_dots_pack_size_bf16_best`
#include "numkong/each.h"         // Weighted Sum, Fused-Multiply-Add, like `nk_each_scale_f64_best`
#include "numkong/geospatial.h"   // Haversine and Vincenty, like `nk_haversines_packed_f64_best`
#include "numkong/probability.h"  // Kullback-Leibler, Jensen-Shannon, like `nk_kld_f16_best`
#include "numkong/reduce.h"       // Horizontal MinMax & Moments reductions, like `nk_reduce_moments_f64_best`
#include "numkong/sets.h"         // Hamming & Jaccard for binary sets, like `nk_hammings_packed_u1_best`
#include "numkong/shapes.h"       // Chamfer, Hausdorff, Wasserstein-Procrustes, Gromov-Wasserstein between point clouds
#include "numkong/sparse.h"       // Set Intersections and Sparse Dot Products, like `nk_sparse_intersect_u16_best`
#include "numkong/spatial.h"      // Euclidean, Angular, like `nk_euclidean_f64_best`
#include "numkong/spatials.h"     // Batched Angular & Euclidean distances, like `nk_angulars_packed_f32_best`
#include "numkong/maxsim.h"       // MaxSim: Multi-Vector Maximum Similarity, like `nk_maxsim_packed_f32_best`
#include "numkong/attention.h"    // Ragged Transformer attention, like `nk_attention_packed_bf16_best`
#include "numkong/trigonometry.h" // Sin, Cos, Atan, like `nk_trig_sin_f64_best`

#if defined(__cplusplus)
extern "C" {
#endif

/** Returns the output dtype for a given metric kind and input dtype. */
NUMKONG_CONSTEXPR nk_dtype_t nk_kernel_output_dtype(nk_kernel_kind_t kind, nk_dtype_t input) {
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
    case nk_kernel_sqeuclideans3d_k: return nk_sqeuclideans3d_output_dtype(input);
    case nk_kernel_haversines_packed_k:
    case nk_kernel_vincentys_packed_k: return nk_geodesics_output_dtype(input);
    case nk_kernel_terrain_heights_packed_k:
    case nk_kernel_clearances_packed_k: return nk_clearances_output_dtype(input);
    case nk_kernel_kld_k:
    case nk_kernel_jsd_k: return nk_probability_output_dtype(input);
    case nk_kernel_sparse_dot_k: return nk_sparse_dot_output_dtype(input);
    case nk_kernel_maxsim_packed_k: return nk_maxsim_output_dtype(input);
    case nk_kernel_chamfers_packed_k:
    case nk_kernel_hausdorffs_packed_k:
    case nk_kernel_wassersteinprocrustes_packed_k:
    case nk_kernel_gromovwassersteins_packed_k: return nk_shapes_output_dtype(input);
    case nk_kernel_attention_packed_k:
    case nk_kernel_attention_packed_gradients_k: return nk_attention_output_dtype(input);
    default: return nk_dtype_unknown_k;
    }
}

/** Whether swapping the two inputs of @p kind leaves its result unchanged. */
NUMKONG_CONSTEXPR int nk_kernel_is_commutative(nk_kernel_kind_t kind) {
    switch (kind) {
    case nk_kernel_kld_k: return 0;
    case nk_kernel_vdot_k: return 0;
    case nk_kernel_bilinear_k: return 0;
    default: return 1;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_NUMKONG_H
