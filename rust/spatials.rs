//! Batched spatial distances — angular (cosine) and Euclidean — over pre-packed matrices.
//!
//! This module provides:
//!
//! - [`Angulars`] / [`Euclideans`]: Low-level per-scalar batched distance traits — FFI-backed
//! - [`AngularsPackedOps`] / [`EuclideansPackedOps`]: `C = dist(A, Bᵀ)` for any [`TensorRef`]
//! - `AngularsPackedParallelOps` / `EuclideansPackedParallelOps`: the same over a thread pool
//! - [`SymmetricAngularsOps`] / [`SymmetricEuclideansOps`]: self-distance upper triangle
//!
//! The right-hand operand is a [`DotsPackedMatrix`] from the [`crate::dots`] module.
#[cfg(feature = "alloc")]
extern crate alloc;

use crate::tensor::{Allocator, Global, Tensor, TensorError, TensorMut, TensorRef, TensorView};
use crate::types::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u4x2, StorageElement};

#[cfg(feature = "parallel")]
use forkunion as fu;

#[cfg(feature = "parallel")]
use crate::dots::compute_thread_rows;
use crate::dots::{validate_matrix_output, validate_packed_input, validate_symmetric_input, Dots, DotsPackedMatrix};

#[link(name = "numkong")]
extern "C" {

    // Batched angular distances
    fn nk_angulars_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_f32(
        vectors: *const f32,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_f64(
        vectors: *const f64,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_f16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_bf16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_i8(
        vectors: *const i8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_u8(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_e4m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_e5m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_e2m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_e3m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_i4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_angulars_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_angulars_symmetric_u4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );

    // Batched euclidean distances
    fn nk_euclideans_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_f32(
        vectors: *const f32,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_f64(
        vectors: *const f64,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_f16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_bf16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_i8(
        vectors: *const i8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_u8(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_e4m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_e5m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_e2m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_e3m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_i4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_euclideans_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_euclideans_symmetric_u4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

// region: Angulars Trait

/// Low-level trait for batched **angular distance** operations.
///
/// Given A ∈ ℝᵐˣᵏ and packed B ∈ ℝⁿˣᵏ, computes C ∈ ℝᵐˣⁿ where:
/// Cᵢⱼ = 1 − cos(θᵢⱼ) = 1 − (aᵢ · bⱼ) / (‖aᵢ‖ × ‖bⱼ‖)
///
/// Packing reuses `Dots::dots_pack` for optimal memory layout.
///
/// # When to use
///
/// Angular (cosine) distance is the standard similarity metric for embeddings
/// from neural language / vision models. The accumulator is widened during
/// the dot-product step — `f32` inputs accumulate in `f64`, `f16` / `bf16` in
/// `f32`, `i8` / `u8` in `i32` / `u32` — then normalised back down to
/// [`Self::SpatialResult`]. Pre-packing the corpus is the standard pattern
/// for repeated-query retrieval.
pub trait Angulars: Dots {
    /// Result type for angular distances.
    type SpatialResult: StorageElement;

    /// Computes angular distances between A rows and packed B columns.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `height * depth` elements with given stride
    /// - `packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `c` must point to valid memory for `height * width` result elements with given stride
    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    /// Computes symmetric angular distance matrix.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for `n_vectors * depth` elements
    /// - `result` must point to valid memory for `n_vectors * n_vectors` result elements
    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

/// Low-level trait for batched **euclidean distance** operations.
///
/// Given A ∈ ℝᵐˣᵏ and packed B ∈ ℝⁿˣᵏ, computes C ∈ ℝᵐˣⁿ where:
/// Cᵢⱼ = √(max(0, ‖aᵢ‖² + ‖bⱼ‖² − 2 · aᵢ · bⱼ))
///
/// Packing reuses `Dots::dots_pack` for optimal memory layout.
///
/// # When to use
///
/// Use for ℓ₂ distance in k-NN / clustering / metric learning. The kernel
/// computes `‖a‖² + ‖b‖² − 2 a·b` in the widened accumulator domain
/// (`f32 × f32 → f64`, `f16 × f16 → f32`, `i8 × i8 → i32`, …) before the
/// square-root, so numerical cancellation is bounded. Pre-pack the reference
/// matrix once when queries are repeated against the same corpus.
pub trait Euclideans: Dots {
    /// Result type for euclidean distances.
    type SpatialResult: StorageElement;

    /// Computes euclidean distances between A rows and packed B columns.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `height * depth` elements with given stride
    /// - `packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `c` must point to valid memory for `height * width` result elements with given stride
    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    /// Computes symmetric euclidean distance matrix.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for `n_vectors * depth` elements
    /// - `result` must point to valid memory for `n_vectors * n_vectors` result elements
    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

impl Angulars for f32 {
    type SpatialResult = f64;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_f32(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_f32(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for f32 {
    type SpatialResult = f64;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_f32(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_f32(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Angulars for f64 {
    type SpatialResult = f64;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_f64(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_f64(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for f64 {
    type SpatialResult = f64;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_f64(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_f64(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Angulars for f16 {
    type SpatialResult = f32;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_f16(a as *const u16, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_f16(
            vectors as *const u16,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for f16 {
    type SpatialResult = f32;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_f16(a as *const u16, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_f16(
            vectors as *const u16,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Angulars for bf16 {
    type SpatialResult = f32;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_bf16(a as *const u16, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_bf16(
            vectors as *const u16,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for bf16 {
    type SpatialResult = f32;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_bf16(a as *const u16, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_bf16(
            vectors as *const u16,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Angulars for i8 {
    type SpatialResult = f32;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_i8(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_i8(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for i8 {
    type SpatialResult = f32;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_i8(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_i8(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Angulars for u8 {
    type SpatialResult = f32;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_u8(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_u8(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for u8 {
    type SpatialResult = f32;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_u8(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_u8(
            vectors,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Angulars for e4m3 {
    type SpatialResult = f32;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_e4m3(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_e4m3(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for e4m3 {
    type SpatialResult = f32;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_e4m3(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_e4m3(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Angulars for e5m2 {
    type SpatialResult = f32;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_e5m2(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_e5m2(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for e5m2 {
    type SpatialResult = f32;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_e5m2(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_e5m2(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Angulars for e2m3 {
    type SpatialResult = f32;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_e2m3(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_e2m3(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for e2m3 {
    type SpatialResult = f32;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_e2m3(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_e2m3(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Angulars for e3m2 {
    type SpatialResult = f32;

    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_e3m2(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_e3m2(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

impl Euclideans for e3m2 {
    type SpatialResult = f32;

    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::SpatialResult,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_e3m2(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::SpatialResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_e3m2(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}
// Manual impls for 4-bit packed types: k must be multiplied by 2 (storage → nibbles).

impl Angulars for u4x2 {
    type SpatialResult = f32;
    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_u4(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }
    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_u4(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}
impl Euclideans for u4x2 {
    type SpatialResult = f32;
    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_u4(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }
    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_u4(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}
impl Angulars for i4x2 {
    type SpatialResult = f32;
    unsafe fn angulars_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_angulars_packed_i4(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }
    unsafe fn angulars_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_angulars_symmetric_i4(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}
impl Euclideans for i4x2 {
    type SpatialResult = f32;
    unsafe fn euclideans_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_euclideans_packed_i4(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }
    unsafe fn euclideans_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_euclideans_symmetric_i4(
            vectors as *const u8,
            n_vectors,
            depth,
            stride,
            result,
            result_stride,
            row_start,
            row_count,
        )
    }
}

// endregion: Angulars Trait

// region: Tensor Spatial Distances

impl<Scalar: Angulars, Alloc: Allocator + Clone, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Computes angular distances between rows of self and packed B matrix.
    pub fn try_angulars_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::SpatialResult, Alloc, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (height, depth) = (self.shape()[0], self.shape()[1]);
        let (width, packed_depth) = packed_b.dims();
        if depth != packed_depth {
            return Err(TensorError::ShapeMismatch {
                axis: 1,
                expected: packed_depth,
                got: depth,
            });
        }
        let mut c = Tensor::try_full_in(&[height, width], Scalar::SpatialResult::default(), self.alloc.clone())?;
        unsafe {
            Scalar::angulars_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                height,
                width,
                depth,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn angulars_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::SpatialResult, Alloc, MAX_RANK> {
        self.try_angulars_packed(packed_b).expect("angulars_packed failed")
    }
}

impl<Scalar: Euclideans, Alloc: Allocator + Clone, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Computes euclidean distances between rows of self and packed B matrix.
    pub fn try_euclideans_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::SpatialResult, Alloc, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (height, depth) = (self.shape()[0], self.shape()[1]);
        let (width, packed_depth) = packed_b.dims();
        if depth != packed_depth {
            return Err(TensorError::ShapeMismatch {
                axis: 1,
                expected: packed_depth,
                got: depth,
            });
        }
        let mut c = Tensor::try_full_in(&[height, width], Scalar::SpatialResult::default(), self.alloc.clone())?;
        unsafe {
            Scalar::euclideans_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                height,
                width,
                depth,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn euclideans_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::SpatialResult, Alloc, MAX_RANK> {
        self.try_euclideans_packed(packed_b).expect("euclideans_packed failed")
    }
}

// Parallel spatial distance implementations
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar: Angulars + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::SpatialResult: Send + Sync,
{
    /// Parallel symmetric angular distance matrix.
    pub fn try_angulars_symmetric_parallel(
        &self,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<Scalar::SpatialResult, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            Scalar::SpatialResult::default(),
        )?;
        self.try_angulars_symmetric_parallel_into(&mut result, pool)?;
        Ok(result)
    }

    /// Parallel symmetric angular distances into pre-allocated output.
    /// Only the upper triangle is written.
    pub fn try_angulars_symmetric_parallel_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        let num_threads = pool.threads_count().max(1);
        let vectors_ptr = fu::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fu::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.broadcast(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let (row_start, row_count) = compute_thread_rows(thread_index, num_threads, n_vectors);
            unsafe {
                Scalar::angulars_symmetric(
                    vectors_ptr.as_ptr(),
                    n_vectors,
                    depth,
                    stride,
                    result_ptr.as_ptr(),
                    result_stride,
                    row_start,
                    row_count,
                );
            }
        });
        Ok(())
    }

    /// Convenience method that panics on error.
    pub fn angulars_symmetric_parallel(
        &self,
        pool: &mut fu::ThreadPool,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_angulars_symmetric_parallel(pool)
            .expect("parallel angulars_symmetric failed")
    }
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar: Euclideans + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::SpatialResult: Send + Sync,
{
    /// Parallel symmetric euclidean distance matrix.
    pub fn try_euclideans_symmetric_parallel(
        &self,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<Scalar::SpatialResult, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            Scalar::SpatialResult::default(),
        )?;
        self.try_euclideans_symmetric_parallel_into(&mut result, pool)?;
        Ok(result)
    }

    /// Parallel symmetric euclidean distances into pre-allocated output.
    /// Only the upper triangle is written.
    pub fn try_euclideans_symmetric_parallel_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        let num_threads = pool.threads_count().max(1);
        let vectors_ptr = fu::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fu::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.broadcast(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let (row_start, row_count) = compute_thread_rows(thread_index, num_threads, n_vectors);
            unsafe {
                Scalar::euclideans_symmetric(
                    vectors_ptr.as_ptr(),
                    n_vectors,
                    depth,
                    stride,
                    result_ptr.as_ptr(),
                    result_stride,
                    row_start,
                    row_count,
                );
            }
        });
        Ok(())
    }

    /// Convenience method that panics on error.
    pub fn euclideans_symmetric_parallel(
        &self,
        pool: &mut fu::ThreadPool,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_euclideans_symmetric_parallel(pool)
            .expect("parallel euclideans_symmetric failed")
    }
}

// endregion: Tensor Spatial Distances

// region: Angulars Packed Ops

/// Extension trait: packed angular distances (`C = angular(A, Bᵀ)`) for any immutable
/// tensor reference — owned [`Tensor`], borrowed [`TensorView`], or [`TensorSpan`](crate::TensorSpan).
///
/// Blanket-implemented for every [`TensorRef`], so an `A` operand backed by an mmap'd
/// view can score against a pre-packed [`DotsPackedMatrix`] without first materializing an
/// owned copy. The allocating entry point returns a globally allocated result, since a
/// bare view carries no allocator of its own.
pub trait AngularsPackedOps<Scalar: Angulars, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    /// Angular distances: C = angular(self, packed_bᵀ)
    ///
    /// self must be 2D (m × k) with contiguous rows.
    /// packed_b contains B (n × k) packed.
    /// Returns C (m × n) using the global allocator.
    ///
    /// Returns `Err` if:
    /// - self is not 2D
    /// - self has non-contiguous rows
    /// - inner dimensions don't match
    /// - output allocation fails
    fn try_angulars_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        let mut c = Tensor::<Scalar::SpatialResult, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::SpatialResult::default(),
        )?;
        unsafe {
            Scalar::angulars_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                height,
                width,
                depth,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    fn angulars_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_angulars_packed(packed_b).expect("angulars_packed failed")
    }

    /// Angular distances into an existing output, avoiding allocation.
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// writable tensor container that implements [`TensorMut`] works. The
    /// kernel overwrites `c` — it need not be pre-initialized.
    fn try_angulars_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, height, width)?;
        unsafe {
            Scalar::angulars_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                height,
                width,
                depth,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(())
    }
}

impl<Scalar: Angulars, const MAX_RANK: usize, A: TensorRef<Scalar, MAX_RANK>> AngularsPackedOps<Scalar, MAX_RANK>
    for A
{
}

/// Extension trait: parallel packed angular distances for any immutable tensor reference.
///
/// The parallel counterpart of [`AngularsPackedOps`], blanket-implemented for every
/// [`TensorRef`] whose scalar can cross thread boundaries.
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub trait AngularsPackedParallelOps<Scalar, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar: Angulars + Clone + Send + Sync,
    Scalar::SpatialResult: Send + Sync,
{
    /// Parallel angular distances into pre-allocated output.
    ///
    /// Distributes rows of A across threads; each computes its portion of C.
    ///
    /// # Arguments
    /// * `packed_b` - Pre-packed B matrix from `DotsPackedMatrix::try_pack[_transposed]`
    /// * `c` - Pre-allocated output tensor (m × n)
    /// * `pool` - Pre-constructed thread pool
    ///
    /// The output may be a `&mut Tensor<...>` or a `&mut TensorSpan<...>` — any
    /// writable tensor container that implements [`TensorMut`]. The kernel
    /// overwrites `c` and need not see initialized memory.
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{DotsPackedMatrix, Tensor};
    /// use forkunion::ThreadPool;
    ///
    /// let topology = forkunion::Topology::new().unwrap();
    /// let mut pool = ThreadPool::try_spawn(&topology, 4).unwrap();
    /// let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
    /// let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();
    /// let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
    /// let mut c = Tensor::<f32>::try_full(&[1024, 256], 0.0).unwrap();
    /// a.try_angulars_packed_parallel_into(&b_packed, &mut c, &mut pool).unwrap();
    /// ```
    fn try_angulars_packed_parallel_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, height, width)?;

        let a_ptr = fu::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fu::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fu::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;
        let num_threads = pool.threads_count().max(1);
        let rows_per_thread = height.div_ceil(num_threads);

        pool.broadcast(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let row_start = thread_index * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(height);
            if row_start < height {
                unsafe {
                    let a_row = (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
                    let c_row = (c_ptr.as_ptr() as *mut u8).add(row_start * c_stride) as *mut Scalar::SpatialResult;
                    Scalar::angulars_packed(
                        a_row,
                        packed_ptr.as_ptr(),
                        c_row,
                        row_end - row_start,
                        width,
                        depth,
                        a_stride,
                        c_stride,
                    );
                }
            }
        });
        Ok(())
    }

    /// Parallel angular distances with allocation.
    fn try_angulars_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        let height = self.shape()[0];
        let (width, _) = packed_b.dims();
        let mut c = Tensor::<Scalar::SpatialResult, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::SpatialResult::default(),
        )?;
        self.try_angulars_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    fn angulars_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_angulars_packed_parallel(packed_b, pool)
            .expect("parallel angulars_packed failed")
    }
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar, const MAX_RANK: usize, A> AngularsPackedParallelOps<Scalar, MAX_RANK> for A
where
    Scalar: Angulars + Clone + Send + Sync,
    Scalar::SpatialResult: Send + Sync,
    A: TensorRef<Scalar, MAX_RANK>,
{
}

// endregion: Angulars Packed Ops

// region: Euclideans Packed Ops

/// Extension trait: packed euclidean distances (`C = euclidean(A, Bᵀ)`) for any immutable
/// tensor reference — owned [`Tensor`], borrowed [`TensorView`], or [`TensorSpan`](crate::TensorSpan).
///
/// Blanket-implemented for every [`TensorRef`], so an `A` operand backed by an mmap'd
/// view can score against a pre-packed [`DotsPackedMatrix`] without first materializing an
/// owned copy. The allocating entry point returns a globally allocated result, since a
/// bare view carries no allocator of its own.
pub trait EuclideansPackedOps<Scalar: Euclideans, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    /// Euclidean distances: C = euclidean(self, packed_bᵀ)
    ///
    /// self must be 2D (m × k) with contiguous rows.
    /// packed_b contains B (n × k) packed.
    /// Returns C (m × n) using the global allocator.
    ///
    /// Returns `Err` if:
    /// - self is not 2D
    /// - self has non-contiguous rows
    /// - inner dimensions don't match
    /// - output allocation fails
    fn try_euclideans_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        let mut c = Tensor::<Scalar::SpatialResult, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::SpatialResult::default(),
        )?;
        unsafe {
            Scalar::euclideans_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                height,
                width,
                depth,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    fn euclideans_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_euclideans_packed(packed_b).expect("euclideans_packed failed")
    }

    /// Euclidean distances into an existing output, avoiding allocation.
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// writable tensor container that implements [`TensorMut`] works. The
    /// kernel overwrites `c` — it need not be pre-initialized.
    fn try_euclideans_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, height, width)?;
        unsafe {
            Scalar::euclideans_packed(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                height,
                width,
                depth,
                self.stride_bytes(0) as usize,
                c.stride_bytes(0) as usize,
            );
        }
        Ok(())
    }
}

impl<Scalar: Euclideans, const MAX_RANK: usize, A: TensorRef<Scalar, MAX_RANK>> EuclideansPackedOps<Scalar, MAX_RANK>
    for A
{
}

/// Extension trait: parallel packed euclidean distances for any immutable tensor reference.
///
/// The parallel counterpart of [`EuclideansPackedOps`], blanket-implemented for every
/// [`TensorRef`] whose scalar can cross thread boundaries.
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub trait EuclideansPackedParallelOps<Scalar, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar: Euclideans + Clone + Send + Sync,
    Scalar::SpatialResult: Send + Sync,
{
    /// Parallel euclidean distances into pre-allocated output.
    ///
    /// Distributes rows of A across threads; each computes its portion of C.
    ///
    /// # Arguments
    /// * `packed_b` - Pre-packed B matrix from `DotsPackedMatrix::try_pack[_transposed]`
    /// * `c` - Pre-allocated output tensor (m × n)
    /// * `pool` - Pre-constructed thread pool
    ///
    /// The output may be a `&mut Tensor<...>` or a `&mut TensorSpan<...>` — any
    /// writable tensor container that implements [`TensorMut`]. The kernel
    /// overwrites `c` and need not see initialized memory.
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{DotsPackedMatrix, Tensor};
    /// use forkunion::ThreadPool;
    ///
    /// let topology = forkunion::Topology::new().unwrap();
    /// let mut pool = ThreadPool::try_spawn(&topology, 4).unwrap();
    /// let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
    /// let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();
    /// let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
    /// let mut c = Tensor::<f32>::try_full(&[1024, 256], 0.0).unwrap();
    /// a.try_euclideans_packed_parallel_into(&b_packed, &mut c, &mut pool).unwrap();
    /// ```
    fn try_euclideans_packed_parallel_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, height, width)?;

        let a_ptr = fu::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fu::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fu::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;
        let num_threads = pool.threads_count().max(1);
        let rows_per_thread = height.div_ceil(num_threads);

        pool.broadcast(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let row_start = thread_index * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(height);
            if row_start < height {
                unsafe {
                    let a_row = (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
                    let c_row = (c_ptr.as_ptr() as *mut u8).add(row_start * c_stride) as *mut Scalar::SpatialResult;
                    Scalar::euclideans_packed(
                        a_row,
                        packed_ptr.as_ptr(),
                        c_row,
                        row_end - row_start,
                        width,
                        depth,
                        a_stride,
                        c_stride,
                    );
                }
            }
        });
        Ok(())
    }

    /// Parallel euclidean distances with allocation.
    fn try_euclideans_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        let height = self.shape()[0];
        let (width, _) = packed_b.dims();
        let mut c = Tensor::<Scalar::SpatialResult, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::SpatialResult::default(),
        )?;
        self.try_euclideans_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    fn euclideans_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_euclideans_packed_parallel(packed_b, pool)
            .expect("parallel euclideans_packed failed")
    }
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar, const MAX_RANK: usize, A> EuclideansPackedParallelOps<Scalar, MAX_RANK> for A
where
    Scalar: Euclideans + Clone + Send + Sync,
    Scalar::SpatialResult: Send + Sync,
    A: TensorRef<Scalar, MAX_RANK>,
{
}

// endregion: Euclideans Packed Ops

// region: TensorView
impl<'a, Scalar: Angulars, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Computes symmetric angular distance matrix for a set of vectors.
    pub fn try_angulars_symmetric(&self) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<Scalar::SpatialResult, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            Scalar::SpatialResult::default(),
        )?;
        self.try_angulars_symmetric_into(&mut result)?;
        Ok(result)
    }

    /// Computes symmetric angular distances into pre-allocated output.
    /// Only the upper triangle is written.
    pub fn try_angulars_symmetric_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        unsafe {
            Scalar::angulars_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride_bytes(0) as usize,
                c.as_mut_ptr(),
                c.stride_bytes(0) as usize,
                0,
                n_vectors,
            );
        }
        Ok(())
    }
}

impl<'a, Scalar: Euclideans, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Computes symmetric euclidean distance matrix for a set of vectors.
    pub fn try_euclideans_symmetric(&self) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<Scalar::SpatialResult, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            Scalar::SpatialResult::default(),
        )?;
        self.try_euclideans_symmetric_into(&mut result)?;
        Ok(result)
    }

    /// Computes symmetric euclidean distances into pre-allocated output.
    /// Only the upper triangle is written.
    pub fn try_euclideans_symmetric_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        unsafe {
            Scalar::euclideans_symmetric(
                self.as_ptr(),
                n_vectors,
                depth,
                self.stride_bytes(0) as usize,
                c.as_mut_ptr(),
                c.stride_bytes(0) as usize,
                0,
                n_vectors,
            );
        }
        Ok(())
    }
}

// endregion: TensorView

// region: Symmetric Extension Traits
pub trait SymmetricAngularsOps<Scalar: Angulars, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn try_angulars_symmetric(&self) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        self.view().try_angulars_symmetric()
    }

    /// Writes the symmetric angular-distance matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_angulars_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(&self, c: &mut Out) -> Result<(), TensorError>
    where
        Out: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        self.view().try_angulars_symmetric_into(c)
    }
}

impl<Scalar: Angulars, const R: usize, OutputTensor: TensorRef<Scalar, R>> SymmetricAngularsOps<Scalar, R>
    for OutputTensor
{
}

/// Extension trait: symmetric euclidean distance matrix for any [`TensorRef`] implementor.
///
/// Blanket-implemented for every `TensorRef<Scalar, R>`, which means
/// `vectors.try_euclideans_symmetric()` compiles whether `vectors` is an
/// owned [`Tensor`] or a borrowed view. The kernel only writes the upper
/// triangle (including the diagonal) — the lower triangle is left alone and
/// callers should mirror it themselves if required.
///
/// Prefer this trait when working through a generic `TensorRef`; reach for
/// the inherent [`TensorView::try_euclideans_symmetric`] method when you
/// already hold a view.
pub trait SymmetricEuclideansOps<Scalar: Euclideans, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn try_euclideans_symmetric(&self) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        self.view().try_euclideans_symmetric()
    }

    /// Writes the symmetric euclidean-distance matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_euclideans_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(&self, c: &mut Out) -> Result<(), TensorError>
    where
        Out: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        self.view().try_euclideans_symmetric_into(c)
    }
}

impl<Scalar: Euclideans, const R: usize, OutputTensor: TensorRef<Scalar, R>> SymmetricEuclideansOps<Scalar, R>
    for OutputTensor
{
}
// endregion: Symmetric Extension Traits

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{align_depth, assert_upper_triangle_eq, init_thread, FloatLike, NumberLike, TestableType, DIMS};

    fn check_angulars_packed<Scalar: TestableType + Angulars>()
    where
        Scalar::SpatialResult: FloatLike + PartialEq + core::fmt::Debug,
    {
        init_thread();
        let tol = Scalar::atol();
        for &(height, width, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
            let c = a.angulars_packed(&b_packed);
            assert_eq!(c.shape(), &[height, width], "shape @ ({height},{width},{depth})");
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    v.to_f64().abs() <= tol,
                    "({height},{width},{depth})[{i}]: {} vs 0.0 (tol={tol})",
                    v.to_f64()
                );
            }
            let mut into_tensor =
                Tensor::<Scalar::SpatialResult>::try_full(&[height, width], Scalar::SpatialResult::default()).unwrap();
            a.try_angulars_packed_into(&b_packed, &mut into_tensor).unwrap();
            assert_eq!(
                c.as_slice(),
                into_tensor.as_slice(),
                "_into(Tensor) @ ({height},{width},{depth})"
            );
            let mut into_span_buf =
                Tensor::<Scalar::SpatialResult>::try_full(&[height, width], Scalar::SpatialResult::default()).unwrap();
            a.try_angulars_packed_into(&b_packed, &mut into_span_buf.span())
                .unwrap();
            assert_eq!(
                c.as_slice(),
                into_span_buf.as_slice(),
                "_into(span) @ ({height},{width},{depth})"
            );
        }
    }

    fn check_euclideans_packed<Scalar: TestableType + Euclideans>()
    where
        Scalar::SpatialResult: FloatLike + PartialEq + core::fmt::Debug,
    {
        init_thread();
        let tol = Scalar::atol();
        for &(height, width, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
            let c = a.euclideans_packed(&b_packed);
            assert_eq!(c.shape(), &[height, width], "shape @ ({height},{width},{depth})");
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    v.to_f64().abs() <= tol,
                    "({height},{width},{depth})[{i}]: {} vs 0.0 (tol={tol})",
                    v.to_f64()
                );
            }
            let mut into_tensor =
                Tensor::<Scalar::SpatialResult>::try_full(&[height, width], Scalar::SpatialResult::default()).unwrap();
            a.try_euclideans_packed_into(&b_packed, &mut into_tensor).unwrap();
            assert_eq!(
                c.as_slice(),
                into_tensor.as_slice(),
                "_into(Tensor) @ ({height},{width},{depth})"
            );
            let mut into_span_buf =
                Tensor::<Scalar::SpatialResult>::try_full(&[height, width], Scalar::SpatialResult::default()).unwrap();
            a.try_euclideans_packed_into(&b_packed, &mut into_span_buf.span())
                .unwrap();
            assert_eq!(
                c.as_slice(),
                into_span_buf.as_slice(),
                "_into(span) @ ({height},{width},{depth})"
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn check_angulars_packed_parallel<Scalar: TestableType + Angulars + Send + Sync>()
    where
        Scalar::SpatialResult: PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let topology = fu::Topology::new().unwrap();
        let mut pool = fu::ThreadPool::try_spawn(&topology, 4).unwrap();
        for &(height, width, depth) in DIMS {
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
            let serial = a.angulars_packed(&b_packed);
            let parallel = a.angulars_packed_parallel(&b_packed, &mut pool);
            assert_eq!(
                serial.as_slice(),
                parallel.as_slice(),
                "serial != parallel @ ({height},{width},{depth})"
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn check_euclideans_packed_parallel<Scalar: TestableType + Euclideans + Send + Sync>()
    where
        Scalar::SpatialResult: PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let topology = fu::Topology::new().unwrap();
        let mut pool = fu::ThreadPool::try_spawn(&topology, 4).unwrap();
        for &(height, width, depth) in DIMS {
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
            let serial = a.euclideans_packed(&b_packed);
            let parallel = a.euclideans_packed_parallel(&b_packed, &mut pool);
            assert_eq!(
                serial.as_slice(),
                parallel.as_slice(),
                "serial != parallel @ ({height},{width},{depth})"
            );
            // Exercise _parallel_into on a span.
            let mut into_span =
                Tensor::<Scalar::SpatialResult>::try_full(&[height, width], Scalar::SpatialResult::default()).unwrap();
            a.try_euclideans_packed_parallel_into(&b_packed, &mut into_span.span(), &mut pool)
                .unwrap();
            assert_eq!(serial.as_slice(), into_span.as_slice(), "_parallel_into(span)");
        }
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn check_spatials_symmetric_parallel<Scalar: TestableType + Angulars + Euclideans + Send + Sync>()
    where
        <Scalar as Angulars>::SpatialResult: Clone + Default + Copy + PartialEq + core::fmt::Debug + Send + Sync,
        <Scalar as Euclideans>::SpatialResult: Clone + Default + Copy + PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let topology = fu::Topology::new().unwrap();
        let mut pool = fu::ThreadPool::try_spawn(&topology, 4).unwrap();
        for &(num_vectors, _, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let vectors = Tensor::<Scalar>::try_full(&[num_vectors, depth], Scalar::one()).unwrap();

            // angulars
            let serial_a = vectors.view().try_angulars_symmetric().unwrap();
            let parallel_a = vectors.angulars_symmetric_parallel(&mut pool);
            assert_upper_triangle_eq(
                serial_a.as_slice(),
                parallel_a.as_slice(),
                num_vectors,
                "angulars_symmetric_parallel",
            );
            let mut into_span_a = Tensor::<<Scalar as Angulars>::SpatialResult>::try_full(
                &[num_vectors, num_vectors],
                <Scalar as Angulars>::SpatialResult::default(),
            )
            .unwrap();
            vectors
                .try_angulars_symmetric_parallel_into(&mut into_span_a.span(), &mut pool)
                .unwrap();
            assert_upper_triangle_eq(
                serial_a.as_slice(),
                into_span_a.as_slice(),
                num_vectors,
                "angulars_symmetric_parallel_into(span)",
            );

            // euclideans
            let serial_e = vectors.view().try_euclideans_symmetric().unwrap();
            let parallel_e = vectors.euclideans_symmetric_parallel(&mut pool);
            assert_upper_triangle_eq(
                serial_e.as_slice(),
                parallel_e.as_slice(),
                num_vectors,
                "euclideans_symmetric_parallel",
            );
            let mut into_span_e = Tensor::<<Scalar as Euclideans>::SpatialResult>::try_full(
                &[num_vectors, num_vectors],
                <Scalar as Euclideans>::SpatialResult::default(),
            )
            .unwrap();
            vectors
                .try_euclideans_symmetric_parallel_into(&mut into_span_e.span(), &mut pool)
                .unwrap();
            assert_upper_triangle_eq(
                serial_e.as_slice(),
                into_span_e.as_slice(),
                num_vectors,
                "euclideans_symmetric_parallel_into(span)",
            );
        }
    }

    fn check_angulars_symmetric<Scalar: TestableType + Angulars>()
    where
        Scalar::SpatialResult: Clone + Default + Copy + FloatLike + PartialEq + core::fmt::Debug + 'static,
    {
        init_thread();
        let tolerance = Scalar::atol();
        for &(num_vectors, _num_targets, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let vectors = Tensor::<Scalar>::try_full(&[num_vectors, depth], Scalar::one()).unwrap();
            let gram_matrix = vectors.view().try_angulars_symmetric().unwrap();
            assert_eq!(gram_matrix.shape(), &[num_vectors, num_vectors]);
            for i in 0..num_vectors {
                for j in i..num_vectors {
                    let value = gram_matrix.as_slice()[i * num_vectors + j];
                    assert!(
                        value.to_f64().abs() <= tolerance,
                        "angular symmetric [{i},{j}]: {}",
                        value.to_f64()
                    );
                }
            }
            let mut into_tensor = Tensor::<Scalar::SpatialResult>::try_full(
                &[num_vectors, num_vectors],
                Scalar::SpatialResult::default(),
            )
            .unwrap();
            vectors.try_angulars_symmetric_into(&mut into_tensor).unwrap();
            assert_upper_triangle_eq(
                gram_matrix.as_slice(),
                into_tensor.as_slice(),
                num_vectors,
                "angulars_symmetric_into(Tensor)",
            );
            let mut into_span_buf = Tensor::<Scalar::SpatialResult>::try_full(
                &[num_vectors, num_vectors],
                Scalar::SpatialResult::default(),
            )
            .unwrap();
            vectors
                .view()
                .try_angulars_symmetric_into(&mut into_span_buf.span())
                .unwrap();
            assert_upper_triangle_eq(
                gram_matrix.as_slice(),
                into_span_buf.as_slice(),
                num_vectors,
                "angulars_symmetric_into(span)",
            );
        }
    }

    fn check_euclideans_symmetric<Scalar: TestableType + Euclideans>()
    where
        Scalar::SpatialResult: Clone + Default + Copy + FloatLike + PartialEq + core::fmt::Debug + 'static,
    {
        init_thread();
        let tolerance = Scalar::atol();
        for &(num_vectors, _num_targets, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let vectors = Tensor::<Scalar>::try_full(&[num_vectors, depth], Scalar::one()).unwrap();
            let gram_matrix = vectors.view().try_euclideans_symmetric().unwrap();
            assert_eq!(gram_matrix.shape(), &[num_vectors, num_vectors]);
            for i in 0..num_vectors {
                for j in i..num_vectors {
                    let value = gram_matrix.as_slice()[i * num_vectors + j];
                    assert!(
                        value.to_f64().abs() <= tolerance,
                        "euclidean symmetric [{i},{j}]: {}",
                        value.to_f64()
                    );
                }
            }
            let mut into_tensor = Tensor::<Scalar::SpatialResult>::try_full(
                &[num_vectors, num_vectors],
                Scalar::SpatialResult::default(),
            )
            .unwrap();
            vectors.try_euclideans_symmetric_into(&mut into_tensor).unwrap();
            assert_upper_triangle_eq(
                gram_matrix.as_slice(),
                into_tensor.as_slice(),
                num_vectors,
                "euclideans_symmetric_into(Tensor)",
            );
            let mut into_span_buf = Tensor::<Scalar::SpatialResult>::try_full(
                &[num_vectors, num_vectors],
                Scalar::SpatialResult::default(),
            )
            .unwrap();
            vectors
                .view()
                .try_euclideans_symmetric_into(&mut into_span_buf.span())
                .unwrap();
            assert_upper_triangle_eq(
                gram_matrix.as_slice(),
                into_span_buf.as_slice(),
                num_vectors,
                "euclideans_symmetric_into(span)",
            );
        }
    }

    #[test]
    fn angulars_packed() {
        check_angulars_packed::<f32>();
        check_angulars_packed::<f64>();
        check_angulars_packed::<f16>();
        check_angulars_packed::<bf16>();
        check_angulars_packed::<e4m3>();
        check_angulars_packed::<e5m2>();
        check_angulars_packed::<e2m3>();
        check_angulars_packed::<e3m2>();
        check_angulars_packed::<i8>();
        check_angulars_packed::<u8>();
        check_angulars_packed::<i4x2>();
        check_angulars_packed::<u4x2>();
    }

    #[test]
    fn euclideans_packed() {
        check_euclideans_packed::<f32>();
        check_euclideans_packed::<f64>();
        check_euclideans_packed::<f16>();
        check_euclideans_packed::<bf16>();
        check_euclideans_packed::<e4m3>();
        check_euclideans_packed::<e5m2>();
        check_euclideans_packed::<e2m3>();
        check_euclideans_packed::<e3m2>();
        check_euclideans_packed::<i8>();
        check_euclideans_packed::<u8>();
        check_euclideans_packed::<i4x2>();
        check_euclideans_packed::<u4x2>();
    }

    /// The `AngularsPackedOps` blanket impl must route a borrowed `TensorView` / `TensorSpan`
    /// `A` operand through the same kernel as an owned `Tensor`. Non-uniform data guards against
    /// a routing bug hiding behind a constant fill.
    #[test]
    fn angulars_packed_accepts_views_and_spans() {
        init_thread();
        let (height, width, depth) = (3usize, 4usize, 5usize);
        let a_data: Vec<f32> = (0..height * depth).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b_data: Vec<f32> = (0..width * depth).map(|i| i as f32 * 0.25 + 0.3).collect();
        let mut a = Tensor::<f32>::from_slice(&a_data, &[height, depth]);
        let b = Tensor::<f32>::from_slice(&b_data, &[width, depth]);
        let packed = DotsPackedMatrix::try_pack(&b).unwrap();

        let expected = a.angulars_packed(&packed);
        assert_eq!(
            a.view().angulars_packed(&packed).as_slice(),
            expected.as_slice(),
            "view A"
        );
        assert_eq!(
            a.span().angulars_packed(&packed).as_slice(),
            expected.as_slice(),
            "span A"
        );
        assert_eq!(
            a.view().try_angulars_packed(&packed).unwrap().as_slice(),
            expected.as_slice(),
            "view A try"
        );
        let mut into = Tensor::<f64>::try_full(&[height, width], 0.0).unwrap();
        a.view().try_angulars_packed_into(&packed, &mut into.span()).unwrap();
        assert_eq!(into.as_slice(), expected.as_slice(), "view A into span");
    }

    /// The `EuclideansPackedOps` blanket impl must route a borrowed `TensorView` / `TensorSpan`
    /// `A` operand through the same kernel as an owned `Tensor`.
    #[test]
    fn euclideans_packed_accepts_views_and_spans() {
        init_thread();
        let (height, width, depth) = (3usize, 4usize, 5usize);
        let a_data: Vec<f32> = (0..height * depth).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b_data: Vec<f32> = (0..width * depth).map(|i| i as f32 * 0.25 + 0.3).collect();
        let mut a = Tensor::<f32>::from_slice(&a_data, &[height, depth]);
        let b = Tensor::<f32>::from_slice(&b_data, &[width, depth]);
        let packed = DotsPackedMatrix::try_pack(&b).unwrap();

        let expected = a.euclideans_packed(&packed);
        assert_eq!(
            a.view().euclideans_packed(&packed).as_slice(),
            expected.as_slice(),
            "view A"
        );
        assert_eq!(
            a.span().euclideans_packed(&packed).as_slice(),
            expected.as_slice(),
            "span A"
        );
        assert_eq!(
            a.view().try_euclideans_packed(&packed).unwrap().as_slice(),
            expected.as_slice(),
            "view A try"
        );
        let mut into = Tensor::<f64>::try_full(&[height, width], 0.0).unwrap();
        a.view().try_euclideans_packed_into(&packed, &mut into.span()).unwrap();
        assert_eq!(into.as_slice(), expected.as_slice(), "view A into span");
    }

    #[test]
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn spatials_packed_parallel() {
        check_angulars_packed_parallel::<f32>();
        check_euclideans_packed_parallel::<f32>();
    }

    #[test]
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn spatials_symmetric_parallel() { check_spatials_symmetric_parallel::<f32>(); }

    #[test]
    fn angulars_symmetric() {
        check_angulars_symmetric::<f32>();
        check_angulars_symmetric::<f64>();
        check_angulars_symmetric::<f16>();
        check_angulars_symmetric::<bf16>();
        check_angulars_symmetric::<e4m3>();
        check_angulars_symmetric::<e5m2>();
        check_angulars_symmetric::<e2m3>();
        check_angulars_symmetric::<e3m2>();
        check_angulars_symmetric::<i8>();
        check_angulars_symmetric::<u8>();
        check_angulars_symmetric::<i4x2>();
        check_angulars_symmetric::<u4x2>();
    }

    #[test]
    fn euclideans_symmetric() {
        check_euclideans_symmetric::<f32>();
        check_euclideans_symmetric::<f64>();
        check_euclideans_symmetric::<f16>();
        check_euclideans_symmetric::<bf16>();
        check_euclideans_symmetric::<e4m3>();
        check_euclideans_symmetric::<e5m2>();
        check_euclideans_symmetric::<e2m3>();
        check_euclideans_symmetric::<e3m2>();
        check_euclideans_symmetric::<i8>();
        check_euclideans_symmetric::<u8>();
        check_euclideans_symmetric::<i4x2>();
        check_euclideans_symmetric::<u4x2>();
    }
}
