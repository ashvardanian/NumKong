//! Batched dot products (GEMM) over pre-packed matrices — the shared `dots` core.
//!
//! This module provides:
//!
//! - [`Dots`]: Low-level per-scalar batched dot-product trait — FFI-backed
//! - [`DotsPackedMatrix`]: Pre-packed right-hand operand reused across many multiplies
//! - [`DotsPackedOps`] / `DotsPackedParallelOps`: `C = A × Bᵀ` for any [`TensorRef`]
//! - [`SymmetricDotsOps`]: `C = A × Aᵀ` (upper triangle) for any [`TensorRef`]
//!
//! The `spatials` (angular/euclidean) and `sets` (hamming/jaccard) modules build on the
//! `DotsPackedMatrix` and validators re-exported here.
#[cfg(feature = "alloc")]
extern crate alloc;

use core::marker::PhantomData;

use crate::tensor::{Allocator, Global, PackedBuffer, Tensor, TensorError, TensorMut, TensorRef, TensorView};
use crate::types::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2, StorageElement};

#[cfg(feature = "parallel")]
use forkunion as fu;

#[link(name = "numkong")]
extern "C" {

    fn nk_dots_pack_size_f32(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_f32(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_f32(b: *const f32, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_f64(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_f64(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_f64(b: *const f64, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_f16(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_f16(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_f16(b: *const u16, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_bf16(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_bf16(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_bf16(b: *const u16, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_i8(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_i8(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_i8(b: *const i8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut i32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_u8(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_u8(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_u8(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_e4m3(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_e4m3(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_e4m3(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_e5m2(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_e5m2(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_e5m2(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_e2m3(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_e2m3(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_e2m3(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_e2m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_e3m2(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_e3m2(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_e3m2(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_e3m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_u4(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_u4(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_u4(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_u4(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    fn nk_dots_pack_size_i4(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_i4(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_i4(b: *const u8, width: usize, depth: usize, b_stride: usize, packed: *mut u8);
    fn nk_dots_packed_i4(
        a: *const u8,
        packed: *const u8,
        c: *mut i32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    // Symmetric Gram matrix (C = A × Aᵀ)
    fn nk_dots_symmetric_f32(
        vectors: *const f32,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_f64(
        vectors: *const f64,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f64,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_f16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_bf16(
        vectors: *const u16,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_i8(
        vectors: *const i8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut i32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_u8(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_e4m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_e5m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_e2m3(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_e3m2(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_u4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
    fn nk_dots_symmetric_i4(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut i32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );

    fn nk_dots_pack_size_u1(width: usize, depth: usize) -> usize;
    fn nk_dots_packed_shape_u1(packed: *const u8, width: *mut usize, depth: *mut usize);
    fn nk_dots_pack_u1(q: *const u8, width: usize, depth: usize, q_stride: usize, q_packed: *mut u8);
    fn nk_dots_packed_u1(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );
    fn nk_dots_symmetric_u1(
        vectors: *const u8,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

// region: Dots Trait

/// Low-level trait for batched **dot product** computation using pre-packed matrices.
///
/// Given A ∈ ℝᵐˣᵏ and packed B ∈ ℝⁿˣᵏ, computes C ∈ ℝᵐˣⁿ where:
/// Cᵢⱼ = aᵢ · bⱼ
///
/// B is pre-packed into a backend-specific layout for optimal memory access.
/// All strides are in bytes.
///
/// # When to use
///
/// Reach for this trait, or the matching `Tensor::try_dots_packed*` wrappers,
/// whenever you multiply many query rows against the **same** matrix of
/// database rows: pre-packing B once amortises layout-conversion cost across
/// every subsequent query. The accumulator type is intentionally widened to
/// avoid precision loss — `f32 × f32 → f64`, `f16 × f16 → f32`,
/// `i8 × i8 → i32`, `u8 × u8 → u32`, and so on; see each impl's
/// [`Dots::Accumulator`].
mod private {
    /// Sealed supertrait for the batch-operation trait family.
    ///
    /// Implementations of [`Dots`] / [`Angulars`] / [`Euclideans`] / [`Hammings`] /
    /// [`Jaccards`] call into NumKong's C kernels via unsafe FFI. External
    /// implementations are not supported — the sealed bound makes that explicit.
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for super::f16 {}
    impl Sealed for super::bf16 {}
    impl Sealed for super::e4m3 {}
    impl Sealed for super::e5m2 {}
    impl Sealed for super::e2m3 {}
    impl Sealed for super::e3m2 {}
    impl Sealed for i8 {}
    impl Sealed for u8 {}
    impl Sealed for super::u4x2 {}
    impl Sealed for super::i4x2 {}
    impl Sealed for super::u1x8 {}
}

pub trait Dots: StorageElement + private::Sealed {
    /// Accumulator type for the multiplication.
    type Accumulator: StorageElement;

    /// Returns the size in bytes needed for the packed B matrix buffer.
    fn dots_pack_size(width: usize, depth: usize) -> usize;

    /// Reads the packed B matrix's width and depth from its header.
    /// # Safety
    /// `packed` must point to a buffer produced by `dots_pack`.
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize);

    /// Packs the B matrix into an optimized backend-specific layout.
    ///
    /// # Safety
    /// - `b` must point to valid memory for `width * depth` elements
    /// - `packed` must point to a buffer of at least `dots_pack_size(width, depth)` bytes
    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8);

    /// Computes C = A × Bᵀ using packed B.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `height * depth` elements with given stride
    /// - `packed` must be a buffer previously filled by `dots_pack`
    /// - `c` must point to valid memory for `height * width` elements with given stride
    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    );

    /// Computes C = A × Aᵀ where C is symmetric.
    ///
    /// Given input matrix A of shape [n, k], computes the symmetric matrix of all pairwise
    /// dot products. Only the upper triangle is computed, then mirrored to the lower triangle.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for `n_vectors × depth` elements with given stride
    /// - `result` must point to valid memory for `n_vectors × n_vectors` elements with given stride
    /// - Strides are in bytes, not elements
    /// - `row_start + row_count` must be <= `n_vectors`
    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

impl Dots for f32 {
    type Accumulator = f64;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_f32(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_f32(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f32(b, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f32(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_f32(
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

impl Dots for f64 {
    type Accumulator = f64;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_f64(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_f64(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f64(b, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f64(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_f64(
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

impl Dots for f16 {
    type Accumulator = f32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_f16(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_f16(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f16(b as *const u16, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f16(a as *const u16, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_f16(
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

impl Dots for bf16 {
    type Accumulator = f32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_bf16(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_bf16(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_bf16(b as *const u16, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_bf16(a as *const u16, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_bf16(
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

impl Dots for i8 {
    type Accumulator = i32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_i8(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_i8(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_i8(b, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_i8(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_i8(
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

impl Dots for u8 {
    type Accumulator = u32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_u8(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_u8(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u8(b, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u8(a, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_u8(
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

impl Dots for e4m3 {
    type Accumulator = f32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_e4m3(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_e4m3(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e4m3(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e4m3(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_e4m3(
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

impl Dots for e5m2 {
    type Accumulator = f32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_e5m2(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_e5m2(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e5m2(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e5m2(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_e5m2(
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

impl Dots for e2m3 {
    type Accumulator = f32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_e2m3(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_e2m3(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e2m3(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e2m3(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_e2m3(
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

impl Dots for e3m2 {
    type Accumulator = f32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_e3m2(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_e3m2(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e3m2(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e3m2(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_e3m2(
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

impl Dots for u4x2 {
    type Accumulator = u32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_u4(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_u4(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u4(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u4(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_u4(
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

impl Dots for i4x2 {
    type Accumulator = i32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_i4(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_i4(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_i4(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_i4(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_i4(
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

impl Dots for u1x8 {
    type Accumulator = u32;

    fn dots_pack_size(width: usize, depth: usize) -> usize { unsafe { nk_dots_pack_size_u1(width, depth) } }
    unsafe fn dots_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut width, mut depth) = (0usize, 0usize);
        nk_dots_packed_shape_u1(packed, &mut width, &mut depth);
        (width, depth)
    }

    unsafe fn dots_pack(b: *const Self, width: usize, depth: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u1(b as *const u8, width, depth, b_stride, packed)
    }

    unsafe fn dots_packed(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        height: usize,
        width: usize,
        depth: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u1(a as *const u8, packed, c, height, width, depth, a_stride, c_stride)
    }

    unsafe fn dots_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::Accumulator,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_dots_symmetric_u1(
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

// endregion: Dots Trait

// region: DotsPackedMatrix

/// Pre-packed B matrix for efficient repeated GEMM operations.
///
/// Uses raw memory allocation, not std::Vec, for maximum control.
///
/// When multiplying A × Bᵀ multiple times with the same B matrix,
/// packing B once and reusing it is much faster than packing each time.
///
/// # Usage
///
/// For C = A × Bᵀ where B is (n × k):
/// ```rust,ignore
/// // Requires linking against libnumkong C library
/// let b_packed = DotsPackedMatrix::try_pack(&b_array).unwrap();
/// let c = a_array.dots_packed(&b_packed);
/// ```
///
/// For C = A × B where B is (k × n) in standard GEMM layout:
/// ```rust,ignore
/// // Requires linking against libnumkong C library
/// let b_packed = DotsPackedMatrix::try_pack_transposed(&b_array).unwrap();
/// let c = a_array.dots_packed(&b_packed);
/// ```
#[derive(Debug)]
pub struct DotsPackedMatrix<Scalar: Dots, Alloc: Allocator = Global> {
    buffer: PackedBuffer<Alloc>,
    /// Output columns (B width).
    width: usize,
    /// Inner dimension (depth).
    depth: usize,
    _marker: PhantomData<Scalar>,
}

// Safety: DotsPackedMatrix owns its data and is just bytes
unsafe impl<Scalar: Dots + Send, Alloc: Allocator + Send> Send for DotsPackedMatrix<Scalar, Alloc> {}
unsafe impl<Scalar: Dots + Sync, Alloc: Allocator + Sync> Sync for DotsPackedMatrix<Scalar, Alloc> {}

impl<Scalar: Dots, Alloc: Allocator + Clone> DotsPackedMatrix<Scalar, Alloc> {
    /// Try to clone this packed matrix, returning an error on allocation failure.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        Ok(Self {
            buffer: self.buffer.try_clone()?,
            width: self.width,
            depth: self.depth,
            _marker: PhantomData,
        })
    }
}

impl<Scalar: Dots, Alloc: Allocator + Clone> Clone for DotsPackedMatrix<Scalar, Alloc> {
    fn clone(&self) -> Self { self.try_clone().expect("DotsPackedMatrix clone allocation failed") }
}

// Generic allocator-aware methods
impl<Scalar: Dots, Alloc: Allocator> DotsPackedMatrix<Scalar, Alloc> {
    /// An empty packed matrix owning no allocation, using a custom allocator.
    ///
    /// Fill it with [`try_pack_into`](Self::try_pack_into).
    pub fn empty_in(alloc: Alloc) -> Self {
        Self {
            buffer: PackedBuffer::empty_in(alloc),
            width: 0,
            depth: 0,
            _marker: PhantomData,
        }
    }

    pub fn try_pack_in<B, const MAX_RANK: usize>(b: &B, alloc: Alloc) -> Result<Self, TensorError>
    where
        B: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let mut packed = Self::empty_in(alloc);
        packed.try_pack_into(b)?;
        Ok(packed)
    }

    /// Repack `b` into this matrix's existing buffer, reusing the allocation when the packed size
    /// fits `capacity` and reallocating through the stored allocator only when it must grow. A
    /// steady-state loop over same-shaped operands then allocates at most once. `b` must be 2D
    /// with contiguous rows.
    pub fn try_pack_into<B, const MAX_RANK: usize>(&mut self, b: &B) -> Result<(), TensorError>
    where
        B: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        if b.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: b.ndim(),
            });
        }
        // The pack kernel reads each row's `depth` elements contiguously — it only takes a row
        // stride — so a view with a non-unit inner stride, e.g. a bare transpose, would be
        // mispacked. Reject it, as the maxsim path does.
        if !b.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (width, depth) = (b.shape()[0], b.shape()[1]);
        let size = Scalar::dots_pack_size(width, depth);
        // The packer zero-fills its header reserved words and panel padding, so it owns every
        // byte of the blob — no pre-zeroing needed here.
        let destination = self.buffer.reset_for_pack(size)?;
        if size > 0 {
            unsafe { Scalar::dots_pack(b.as_ptr(), width, depth, b.stride_bytes(0) as usize, destination) };
        }
        self.width = width;
        self.depth = depth;
        Ok(())
    }

    /// Pack Bᵀ where B is (k × n) row-major, the standard GEMM layout, using a custom allocator.
    ///
    /// Materializes the transpose into a contiguous buffer, then packs normally.
    /// Result computes: C = A × B
    ///
    /// Returns `Err` if:
    /// - b is not 2D
    /// - b is a sub-byte type — transpose unsupported
    /// - allocation fails
    pub fn try_pack_transposed_in<B, const MAX_RANK: usize>(b: &B, alloc: Alloc) -> Result<Self, TensorError>
    where
        B: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        if b.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: b.ndim(),
            });
        }
        // Transpose is a zero-copy strided view, but the pack kernel reads each
        // B row's `depth` elements contiguously (it only takes a row stride — see
        // `nk_dots_pack_*` in `numkong/dots.h`), so the transposed view's
        // non-unit inner stride must be materialized into a contiguous buffer
        // first. For sub-byte types, transpose() returns SubByteUnsupported.
        let transposed = b.view().try_transpose()?.try_to_owned()?;
        Self::try_pack_in(&transposed, alloc)
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { self.buffer.allocator() }

    /// Returns the shape (width, depth) of the original B matrix.
    pub fn shape(&self) -> (usize, usize) { (self.width, self.depth) }

    /// Bytes a packed buffer occupies for a width-by-depth B matrix of this scalar type under the
    /// active backend's layout. Mirrors [`crate::attention::AttentionPackedMatrix::pack_size`] and
    /// lets a caller pre-size an external buffer without packing.
    pub fn pack_size(width: usize, depth: usize) -> usize { Scalar::dots_pack_size(width, depth) }

    /// Adopt an externally-produced packed buffer by copying `bytes` into a container-owned
    /// allocation tagged with the given `width` and `depth`. The dots packed layout is not
    /// self-describing — it carries no dtype tag and its AMX variant drops the exact depth — so the
    /// caller must supply the shape the bytes were packed for.
    ///
    /// # Safety
    /// `bytes` must be a valid packing of a width-by-depth matrix of `Scalar`, produced by this
    /// build's packer for a compatible backend; anything else makes a later `dots_packed` read out
    /// of bounds or return garbage.
    pub unsafe fn from_packed_bytes_in(
        bytes: &[u8],
        width: usize,
        depth: usize,
        alloc: Alloc,
    ) -> Result<Self, TensorError> {
        let mut packed = Self::empty_in(alloc);
        packed.buffer.fill_from_bytes(bytes)?;
        packed.width = width;
        packed.depth = depth;
        Ok(packed)
    }

    /// Bytes currently allocated (>= the live packed size).
    pub fn capacity(&self) -> usize { self.buffer.capacity() }

    /// Reset to logically empty, keeping the allocation so the next `try_pack_into` reuses it.
    pub fn clear(&mut self) { self.buffer.clear(); }

    /// Returns the packed data buffer.
    pub fn as_bytes(&self) -> &[u8] { self.buffer.as_bytes() }

    /// Returns a pointer to the packed data.
    pub fn as_ptr(&self) -> *const u8 { self.buffer.as_ptr() }
}

// Convenience methods using Global allocator
impl<Scalar: Dots> DotsPackedMatrix<Scalar, Global> {
    /// Pack B matrix where B is (n × k) row-major using the global allocator.
    ///
    /// Result computes: C = A × Bᵀ
    pub fn try_pack<B, const MAX_RANK: usize>(b: &B) -> Result<Self, TensorError>
    where
        B: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::try_pack_in(b, Global)
    }

    /// Pack Bᵀ where B is (k × n) row-major, the standard GEMM layout, using the global allocator.
    ///
    /// Result computes: C = A × B
    pub fn try_pack_transposed<B, const MAX_RANK: usize>(b: &B) -> Result<Self, TensorError>
    where
        B: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::try_pack_transposed_in(b, Global)
    }
}

// endregion: DotsPackedMatrix

// region: Shared validators

/// Validate shared preconditions for `*_packed` operations.
///
/// Checks that `a` is a 2D tensor with contiguous rows and that its depth
/// matches that of the packed matrix. Returns `(height, width, depth)` on success.
#[inline]
pub(crate) fn validate_packed_input<Scalar, A, PackedAlloc, const MAX_RANK: usize>(
    a: &A,
    packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
) -> Result<(usize, usize, usize), TensorError>
where
    Scalar: Dots,
    A: TensorRef<Scalar, MAX_RANK> + ?Sized,
    PackedAlloc: Allocator,
{
    if a.ndim() != 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: a.ndim(),
        });
    }
    if !a.has_contiguous_rows() {
        return Err(TensorError::NonContiguousRows);
    }
    let (height, depth) = (a.shape()[0], a.shape()[1]);
    let (width, packed_depth) = packed_b.shape();
    if depth != packed_depth {
        return Err(TensorError::ShapeMismatch {
            axis: 1,
            expected: packed_depth,
            got: depth,
        });
    }
    Ok((height, width, depth))
}

/// Validate that pre-allocated output `c` has shape `[height, width]` and contiguous rows.
#[inline]
pub(crate) fn validate_matrix_output<R, OutputTensor, const OUTPUT_MAX_RANK: usize>(
    c: &OutputTensor,
    height: usize,
    width: usize,
) -> Result<(), TensorError>
where
    R: StorageElement,
    OutputTensor: TensorRef<R, OUTPUT_MAX_RANK> + ?Sized,
{
    if c.shape() != [height, width] {
        return Err(TensorError::ShapeMismatch {
            axis: if c.shape().first().copied() != Some(height) {
                0
            } else {
                1
            },
            expected: if c.shape().first().copied() != Some(height) {
                height
            } else {
                width
            },
            got: if c.shape().first().copied() != Some(height) {
                c.shape().first().copied().unwrap_or(0)
            } else {
                c.shape().get(1).copied().unwrap_or(0)
            },
        });
    }
    if !c.has_contiguous_rows() {
        return Err(TensorError::NonContiguousRows);
    }
    Ok(())
}

/// Validate shared preconditions for `*_symmetric` operations on 2D input.
/// Returns `(n_vectors, depth)` on success.
#[inline]
pub(crate) fn validate_symmetric_input<Scalar, InputTensor, const MAX_RANK: usize>(
    a: &InputTensor,
) -> Result<(usize, usize), TensorError>
where
    Scalar: StorageElement,
    InputTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
{
    if a.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            axis: 0,
            size: a.ndim(),
            reason: "symmetric operations require a 2D tensor",
        });
    }
    // The kernels read each row as `depth` contiguous elements using only the row stride, so a
    // transposed/strided view with non-unit inner stride would read out of bounds — reject it, as the
    // packed and parallel paths already do.
    if !a.has_contiguous_rows() {
        return Err(TensorError::NonContiguousRows);
    }
    Ok((a.shape()[0], a.shape()[1]))
}

// endregion: Shared validators

// region: Tensor GEMM

// Inherent, allocator-preserving entry points on the owning `Tensor`. These
// mirror the [`DotsPackedOps`] methods below but return a result allocated with
// `self`'s own allocator, and remain callable without importing the extension
// trait. For a `Tensor` receiver they shadow the blanket-trait methods of the
// same name; views and spans reach the globally allocating trait versions.
impl<Scalar: Dots, Alloc: Allocator + Clone, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Dot-product multiply: C = self × packed_bᵀ
    ///
    /// self must be 2D (m × k) with contiguous rows.
    /// packed_b contains B (n × k) packed.
    /// Returns C (m × n) using the same allocator as self.
    ///
    /// Returns `Err` if:
    /// - self is not 2D
    /// - self has non-contiguous rows
    /// - inner dimensions don't match
    /// - output allocation fails
    pub fn try_dots_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::Accumulator, Alloc, MAX_RANK>, TensorError> {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        let mut c = Tensor::try_full_in(&[height, width], Scalar::Accumulator::default(), self.alloc.clone())?;
        unsafe {
            Scalar::dots_packed(
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
    pub fn dots_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::Accumulator, Alloc, MAX_RANK> {
        self.try_dots_packed(packed_b).expect("dots_packed failed")
    }
}

/// Extension trait: packed GEMM (`C = A × Bᵀ`) for any immutable tensor
/// reference — owned [`Tensor`], borrowed [`TensorView`], or [`TensorSpan`](crate::TensorSpan).
///
/// Blanket-implemented for every [`TensorRef`], so an `A` operand backed by an
/// mmap'd view can multiply against a pre-packed [`DotsPackedMatrix`] without first
/// materializing an owned copy. The allocating entry point returns a globally
/// allocated result, since a bare view carries no allocator of its own.
pub trait DotsPackedOps<Scalar: Dots, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    /// Dot-product multiply: C = self × packed_bᵀ
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
    fn try_dots_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::Accumulator, Global, MAX_RANK>, TensorError> {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        let mut c = Tensor::<Scalar::Accumulator, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::Accumulator::default(),
        )?;
        unsafe {
            Scalar::dots_packed(
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
    fn dots_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::Accumulator, Global, MAX_RANK> {
        self.try_dots_packed(packed_b).expect("dots_packed failed")
    }

    /// Dot-product multiply into an existing output, avoiding allocation.
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// writable tensor container that implements [`TensorMut`] works. The
    /// kernel overwrites `c` — it need not be pre-initialized.
    fn try_dots_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::Accumulator, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::Accumulator, _, OUTPUT_MAX_RANK>(c, height, width)?;
        unsafe {
            Scalar::dots_packed(
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

impl<Scalar: Dots, const MAX_RANK: usize, A: TensorRef<Scalar, MAX_RANK>> DotsPackedOps<Scalar, MAX_RANK> for A {}

// Parallel dots_packed implementations, if ForkUnion is available.
/// Extension trait: parallel packed GEMM for any immutable tensor reference.
///
/// The parallel counterpart of [`DotsPackedOps`], blanket-implemented for every
/// [`TensorRef`] whose scalar can cross thread boundaries. The `A` operand may
/// therefore be an owned [`Tensor`], a borrowed [`TensorView`], or a
/// [`TensorSpan`](crate::TensorSpan) without materializing an owned copy.
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub trait DotsPackedParallelOps<Scalar, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar: Dots + Clone + Send + Sync,
    Scalar::Accumulator: Send + Sync,
{
    /// Parallel dot-product multiply into pre-allocated output.
    ///
    /// Distributes rows of A across threads; each computes its portion of C.
    /// This is a non-allocating interface - you provide the output tensor.
    ///
    /// # Arguments
    /// * `packed_b` - Pre-packed B matrix from `DotsPackedMatrix::try_pack[_transposed]`
    /// * `c` - Pre-allocated output tensor (m × n)
    /// * `pool` - Pre-constructed thread pool
    ///
    /// The output may be a `&mut Tensor<...>` or a `&mut TensorSpan<...>` —
    /// any writable tensor container that implements [`TensorMut`]. The
    /// kernel overwrites `c` and need not see initialized memory.
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{Tensor, DotsPackedMatrix};
    /// use forkunion::ThreadPool;
    ///
    /// let topology = forkunion::Topology::new().unwrap();
    /// let mut pool = ThreadPool::try_spawn(&topology, 4).unwrap();
    /// let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
    /// let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();
    /// let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
    ///
    /// // Writing into a full tensor:
    /// let mut c = Tensor::<f32>::try_full(&[1024, 256], 0.0).unwrap();
    /// a.try_dots_packed_parallel_into(&b_packed, &mut c, &mut pool).unwrap();
    ///
    /// // Or into a span over a sub-region of a larger buffer:
    /// let mut c_buf = Tensor::<f32>::try_full(&[1024, 256], 0.0).unwrap();
    /// a.try_dots_packed_parallel_into(&b_packed, &mut c_buf.span(), &mut pool).unwrap();
    /// ```
    fn try_dots_packed_parallel_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::Accumulator, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::Accumulator, _, OUTPUT_MAX_RANK>(c, height, width)?;

        let a_ptr = fu::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fu::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fu::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;

        // Get actual thread count from pool
        let num_threads = pool.threads_count().max(1);
        let rows_per_thread = height.div_ceil(num_threads);

        // Distribute rows across threads using the ForkUnion pool
        // Safety: Each thread writes to disjoint rows of C, so no data races.
        pool.broadcast(move |thread_index, _colocation_index| {
            // Configure each worker thread for optimal SIMD, including AMX
            // This is idempotent and safe to call multiple times
            crate::capabilities::configure_thread();

            let row_start = thread_index * rows_per_thread;
            if row_start >= height {
                return;
            }
            let row_end = (row_start + rows_per_thread).min(height);
            unsafe {
                // Byte arithmetic so sub-byte types such as u1x8 stride correctly.
                let a_row = (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
                let c_row = (c_ptr.as_ptr() as *mut u8).add(row_start * c_stride) as *mut Scalar::Accumulator;
                Scalar::dots_packed(
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
        });

        Ok(())
    }

    /// Parallel dot-product multiply with allocation.
    ///
    /// Convenience wrapper that allocates the output tensor.
    /// Prefer `try_dots_packed_parallel_into` for performance-critical code.
    fn try_dots_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<Scalar::Accumulator, Global, MAX_RANK>, TensorError> {
        let height = self.shape()[0];
        let (width, _) = packed_b.shape();
        let mut c = Tensor::<Scalar::Accumulator, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::Accumulator::default(),
        )?;
        self.try_dots_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    fn dots_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Tensor<Scalar::Accumulator, Global, MAX_RANK> {
        self.try_dots_packed_parallel(packed_b, pool)
            .expect("parallel dots_packed failed")
    }
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar, const MAX_RANK: usize, A> DotsPackedParallelOps<Scalar, MAX_RANK> for A
where
    Scalar: Dots + Clone + Send + Sync,
    Scalar::Accumulator: Send + Sync,
    A: TensorRef<Scalar, MAX_RANK>,
{
}

/// Compute row assignment for a thread without allocation
///
/// For a symmetric matrix, cumulative work up to row r is: r*(2n - r + 1)/2
/// Solving r*(2n - r + 1)/2 = work using quadratic formula gives exact row.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
#[inline]
pub(crate) fn compute_thread_rows(thread_index: usize, num_threads: usize, n: usize) -> (usize, usize) {
    let total_work = n * (n + 1) / 2;
    let work_per_thread = total_work.div_ceil(num_threads);

    let work_start = thread_index * work_per_thread;
    let work_end = ((thread_index + 1) * work_per_thread).min(total_work);

    // Solve: r^2 - r(2n + 1) + 2*work = 0
    // Using quadratic formula: r = (2n + 1 - sqrt((2n + 1)^2 - 8*work)) / 2
    let start_row = if work_start == 0 {
        0
    } else {
        let n_f64 = n as f64;
        let work_f64 = work_start as f64;
        let discriminant = (2.0 * n_f64 + 1.0).powi(2) - 8.0 * work_f64;
        let row_f64 = (2.0 * n_f64 + 1.0 - discriminant.sqrt()) / 2.0;
        // Use ceil so thread t's start_row equals thread t-1's end_row,
        // giving threads disjoint row ranges — whole-row scheduling.
        row_f64.ceil() as usize
    };

    let end_row = if work_end >= total_work {
        n
    } else {
        let n_f64 = n as f64;
        let work_f64 = work_end as f64;
        let discriminant = (2.0 * n_f64 + 1.0).powi(2) - 8.0 * work_f64;
        let row_f64 = (2.0 * n_f64 + 1.0 - discriminant.sqrt()) / 2.0;
        row_f64.ceil() as usize
    };

    (start_row, end_row.saturating_sub(start_row))
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar: Dots + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::Accumulator: Send + Sync,
{
    /// Parallel computation of symmetric Gram matrix C = A × Aᵀ.
    ///
    /// Distributes rows across threads with balanced work distribution based on the
    /// triangular structure of symmetric matrix computation.
    ///
    /// # Arguments
    /// * `pool` - Pre-constructed thread pool
    ///
    /// # Example
    /// ```ignore
    /// use numkong::Tensor;
    /// use forkunion::ThreadPool;
    ///
    /// let topology = forkunion::Topology::new().unwrap();
    /// let mut pool = ThreadPool::try_spawn(&topology, 4).unwrap();
    /// let vectors = Tensor::<f32>::try_full(&[100, 768], 1.0).unwrap();
    /// let gram = vectors.try_dots_symmetric_parallel(&mut pool).unwrap();
    /// assert_eq!(gram.shape(), &[100, 100]);
    /// ```
    pub fn try_dots_symmetric_parallel(
        &self,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<Scalar::Accumulator, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<Scalar::Accumulator, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            Scalar::Accumulator::default(),
        )?;
        self.try_dots_symmetric_parallel_into(&mut result, pool)?;
        Ok(result)
    }

    /// Parallel symmetric dot-product matrix into pre-allocated output.
    ///
    /// Only the upper triangle of `c` is written.
    pub fn try_dots_symmetric_parallel_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::Accumulator, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::Accumulator, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;

        let num_threads = pool.threads_count().max(1);
        let vectors_ptr = fu::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fu::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.broadcast(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let (row_start, row_count) = compute_thread_rows(thread_index, num_threads, n_vectors);
            unsafe {
                Scalar::dots_symmetric(
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

    /// Parallel computation of symmetric dot-product matrix (unwrapping version).
    ///
    /// # Panics
    /// Panics if the operation fails, for example on a wrong tensor rank.
    pub fn dots_symmetric_parallel(&self, pool: &mut fu::ThreadPool) -> Tensor<Scalar::Accumulator, Global, MAX_RANK> {
        self.try_dots_symmetric_parallel(pool)
            .expect("parallel dots_symmetric failed")
    }
}

// endregion: Tensor GEMM

// region: TensorView
impl<'a, Scalar: Dots, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK>
where
    Scalar::Accumulator: 'static,
{
    /// Computes the symmetric dot-product matrix C = A × Aᵀ.
    ///
    /// Given a matrix of row vectors, computes the matrix of all pairwise dot products.
    /// The result is a symmetric n×n matrix where result\[i,j\] = dot(row_i, row_j).
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{Tensor, TensorView};
    ///
    /// // 100 vectors of dimension 768
    /// let vectors = Tensor::<f32>::try_full(&[100, 768], 0.0)?;
    ///
    /// // Compute 100×100 symmetric matrix
    /// let gram = vectors.view().try_dots_symmetric()?;
    /// assert_eq!(gram.shape(), &[100, 100]);
    /// ```
    pub fn try_dots_symmetric(&self) -> Result<Tensor<Scalar::Accumulator, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<Scalar::Accumulator, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            Scalar::Accumulator::default(),
        )?;
        self.try_dots_symmetric_into(&mut result)?;
        Ok(result)
    }

    /// Computes the symmetric dot-product matrix into pre-allocated output.
    ///
    /// Only the upper triangle of `c` is written; the lower triangle is left
    /// as-is. The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`.
    pub fn try_dots_symmetric_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::Accumulator, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::Accumulator, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        unsafe {
            Scalar::dots_symmetric(
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

/// Extension trait: symmetric dot-product matrix for any [`TensorRef`] implementor.
///
/// Blanket-implemented for every `TensorRef<Scalar, R>`, so calling
/// `vectors.try_dots_symmetric()` works on both owned [`Tensor`] and borrowed
/// [`TensorView`] / `TensorSpan`. Only the **upper triangle** (including the
/// diagonal) of the output is written by the kernel; the lower triangle is
/// left untouched — callers that need a full dense matrix must mirror it.
///
/// Prefer this extension trait when you have a generic `TensorRef`; use the
/// inherent [`TensorView::try_dots_symmetric`] form when you already hold a
/// view and want to avoid the extra trait import.
pub trait SymmetricDotsOps<Scalar: Dots, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar::Accumulator: 'static,
{
    fn try_dots_symmetric(&self) -> Result<Tensor<Scalar::Accumulator, Global, MAX_RANK>, TensorError> {
        self.view().try_dots_symmetric()
    }

    /// Writes the symmetric dot-product matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_dots_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(&self, c: &mut Out) -> Result<(), TensorError>
    where
        Out: TensorMut<Scalar::Accumulator, OUTPUT_MAX_RANK>,
    {
        self.view().try_dots_symmetric_into(c)
    }
}

impl<Scalar: Dots, const R: usize, OutputTensor: TensorRef<Scalar, R>> SymmetricDotsOps<Scalar, R> for OutputTensor where
    Scalar::Accumulator: 'static
{
}

/// Extension trait: symmetric angular distance matrix for any [`TensorRef`] implementor.
///
/// Blanket-implemented for every `TensorRef<Scalar, R>` so the `try_angulars_symmetric`
/// method is available on both owned [`Tensor`] and borrowed views. Only the
/// upper triangle (including the diagonal) of the output is written by the
/// kernel — mirror it yourself if you need a dense symmetric matrix.
///
/// Prefer this trait when operating on a generic `TensorRef`; use the inherent
/// [`TensorView::try_angulars_symmetric`] form when you already hold a view
/// and want to sidestep the extra trait import.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::SIMD_ALIGNMENT;
    use crate::types::{align_depth, assert_upper_triangle_eq, init_thread, FloatLike, NumberLike, TestableType, DIMS};

    fn check_dots_packed<Scalar: TestableType + Dots>()
    where
        Scalar::Accumulator: FloatLike + PartialEq + core::fmt::Debug,
    {
        init_thread();
        for &(height, width, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
            let c = a.dots_packed(&b_packed);
            assert_eq!(c.shape(), &[height, width], "shape @ ({height},{width},{depth})");
            let expected = depth as f64;
            let tol = Scalar::atol() + Scalar::rtol() * expected.abs();
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    (v.to_f64() - expected).abs() <= tol,
                    "({height},{width},{depth})[{i}]: {} vs {expected} (tol={tol})",
                    v.to_f64()
                );
            }
            // Verify _into(&mut Tensor) and _into(&mut span) produce identical bytes.
            let mut into_tensor =
                Tensor::<Scalar::Accumulator>::try_full(&[height, width], Scalar::Accumulator::default()).unwrap();
            a.try_dots_packed_into(&b_packed, &mut into_tensor).unwrap();
            assert_eq!(
                c.as_slice(),
                into_tensor.as_slice(),
                "_into(Tensor) @ ({height},{width},{depth})"
            );
            let mut into_span_buf =
                Tensor::<Scalar::Accumulator>::try_full(&[height, width], Scalar::Accumulator::default()).unwrap();
            a.try_dots_packed_into(&b_packed, &mut into_span_buf.span()).unwrap();
            assert_eq!(
                c.as_slice(),
                into_span_buf.as_slice(),
                "_into(span) @ ({height},{width},{depth})"
            );
        }
    }

    fn check_dots_packed_transposed<Scalar: TestableType + Dots>()
    where
        Scalar::Accumulator: FloatLike,
    {
        init_thread();
        for &(height, width, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b_t = Tensor::<Scalar>::try_full(&[depth, width], Scalar::from_f32(2.0)).unwrap();
            let b_packed = DotsPackedMatrix::try_pack_transposed(&b_t).unwrap();
            let c = a.dots_packed(&b_packed);
            assert_eq!(c.shape(), &[height, width], "shape @ ({height},{width},{depth})");
            let expected = depth as f64 * 2.0;
            let tol = Scalar::atol() + Scalar::rtol() * expected.abs();
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    (v.to_f64() - expected).abs() <= tol,
                    "({height},{width},{depth})[{i}]: {} vs {expected} (tol={tol})",
                    v.to_f64()
                );
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn check_dots_packed_parallel<Scalar: TestableType + Dots + Send + Sync>()
    where
        Scalar::Accumulator: PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let topology = fu::Topology::new().unwrap();
        let mut pool = fu::ThreadPool::try_spawn(&topology, 4).unwrap();
        for &(height, width, depth) in DIMS {
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
            let serial = a.dots_packed(&b_packed);
            let parallel = a.dots_packed_parallel(&b_packed, &mut pool);
            assert_eq!(
                serial.as_slice(),
                parallel.as_slice(),
                "serial != parallel @ ({height},{width},{depth})"
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn check_dots_symmetric_parallel<Scalar: TestableType + Dots + Send + Sync>()
    where
        Scalar::Accumulator: Clone + Default + Copy + PartialEq + core::fmt::Debug + Send + Sync + 'static,
    {
        init_thread();
        let topology = fu::Topology::new().unwrap();
        let mut pool = fu::ThreadPool::try_spawn(&topology, 4).unwrap();
        for &(num_vectors, _, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let vectors = Tensor::<Scalar>::try_full(&[num_vectors, depth], Scalar::one()).unwrap();

            // dots: compare serial == parallel (upper triangle) and _parallel_into(span)
            let serial = vectors.view().try_dots_symmetric().unwrap();
            let parallel = vectors.dots_symmetric_parallel(&mut pool);
            assert_upper_triangle_eq(
                serial.as_slice(),
                parallel.as_slice(),
                num_vectors,
                "dots_symmetric_parallel",
            );
            let mut into_span =
                Tensor::<Scalar::Accumulator>::try_full(&[num_vectors, num_vectors], Scalar::Accumulator::default())
                    .unwrap();
            vectors
                .try_dots_symmetric_parallel_into(&mut into_span.span(), &mut pool)
                .unwrap();
            assert_upper_triangle_eq(
                serial.as_slice(),
                into_span.as_slice(),
                num_vectors,
                "dots_symmetric_parallel_into(span)",
            );
        }
    }

    fn check_dots_symmetric<Scalar: TestableType + Dots>()
    where
        Scalar::Accumulator: Clone + Default + Copy + FloatLike + PartialEq + core::fmt::Debug + 'static,
    {
        init_thread();
        for &(num_vectors, _num_targets, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let vectors = Tensor::<Scalar>::try_full(&[num_vectors, depth], Scalar::one()).unwrap();
            let gram_matrix = vectors.view().try_dots_symmetric().unwrap();
            assert_eq!(
                gram_matrix.shape(),
                &[num_vectors, num_vectors],
                "shape @ ({num_vectors},{depth})"
            );
            let expected = depth as f64;
            let tolerance = Scalar::atol() + Scalar::rtol() * expected.abs();
            for i in 0..num_vectors {
                for j in i..num_vectors {
                    let value = gram_matrix.as_slice()[i * num_vectors + j];
                    assert!(
                        (value.to_f64() - expected).abs() <= tolerance,
                        "({num_vectors},{depth})[{i},{j}]: {} vs {expected}",
                        value.to_f64()
                    );
                }
            }
            // Verify _into on both &mut Tensor and &mut span via the extension trait.
            let mut into_tensor =
                Tensor::<Scalar::Accumulator>::try_full(&[num_vectors, num_vectors], Scalar::Accumulator::default())
                    .unwrap();
            vectors.try_dots_symmetric_into(&mut into_tensor).unwrap();
            assert_upper_triangle_eq(
                gram_matrix.as_slice(),
                into_tensor.as_slice(),
                num_vectors,
                "dots_symmetric_into(Tensor)",
            );
            let mut into_span_buf =
                Tensor::<Scalar::Accumulator>::try_full(&[num_vectors, num_vectors], Scalar::Accumulator::default())
                    .unwrap();
            vectors
                .view()
                .try_dots_symmetric_into(&mut into_span_buf.span())
                .unwrap();
            assert_upper_triangle_eq(
                gram_matrix.as_slice(),
                into_span_buf.as_slice(),
                num_vectors,
                "dots_symmetric_into(span)",
            );
        }
    }

    #[test]
    fn dots_packed() {
        check_dots_packed::<f32>();
        check_dots_packed::<f64>();
        check_dots_packed::<f16>();
        check_dots_packed::<bf16>();
        check_dots_packed::<e4m3>();
        check_dots_packed::<e5m2>();
        check_dots_packed::<e2m3>();
        check_dots_packed::<e3m2>();
        check_dots_packed::<i8>();
        check_dots_packed::<u8>();
        check_dots_packed::<i4x2>();
        check_dots_packed::<u4x2>();
    }

    #[test]
    fn dots_packed_transposed() {
        check_dots_packed_transposed::<f32>();
        check_dots_packed_transposed::<f64>();
        check_dots_packed_transposed::<f16>();
        check_dots_packed_transposed::<bf16>();
        check_dots_packed_transposed::<e4m3>();
        check_dots_packed_transposed::<e5m2>();
        check_dots_packed_transposed::<e2m3>();
        check_dots_packed_transposed::<e3m2>();
        check_dots_packed_transposed::<i8>();
        check_dots_packed_transposed::<u8>();
    }

    /// Issue #6: the packed GEMM must accept borrowed operands (`TensorView` /
    /// `TensorSpan`) on both the packing input and the `A` operand, matching the
    /// owned-`Tensor` path bit-for-bit. Non-uniform values are used so a stride
    /// bug cannot hide behind a constant fill.
    #[test]
    fn dots_packed_accepts_views_and_spans() {
        init_thread();
        let (height, width, depth) = (3usize, 4usize, 5usize);
        let a_data: Vec<f32> = (0..height * depth).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b_data: Vec<f32> = (0..width * depth).map(|i| i as f32 * 0.25 + 0.3).collect();
        let mut a = Tensor::<f32>::from_slice(&a_data, &[height, depth]);
        let b = Tensor::<f32>::from_slice(&b_data, &[width, depth]);

        // Manual reference: C = A × Bᵀ, so C[i][j] = Σ_l A[i][l] · B[j][l].
        let mut expected = vec![0.0f64; height * width];
        for i in 0..height {
            for j in 0..width {
                let mut acc = 0.0f64;
                for l in 0..depth {
                    acc += a_data[i * depth + l] as f64 * b_data[j * depth + l] as f64;
                }
                expected[i * width + j] = acc;
            }
        }
        let close = |c: &Tensor<f64>, reference: &[f64], label: &str| {
            assert_eq!(c.shape(), &[height, width], "{label} shape");
            for (v, e) in c.as_slice().iter().zip(reference.iter()) {
                assert!((v - e).abs() <= 1e-9 + 1e-6 * e.abs(), "{label}: {v} vs {e}");
            }
        };

        // Packing B from an owned tensor, a borrowed view, and a span must agree.
        let packed_owned = DotsPackedMatrix::try_pack(&b).unwrap();
        let packed_view = DotsPackedMatrix::try_pack(&b.view()).unwrap();
        let packed_span = DotsPackedMatrix::try_pack(&b.clone().span()).unwrap();
        assert_eq!(packed_owned.as_bytes(), packed_view.as_bytes(), "pack(view)");
        assert_eq!(packed_owned.as_bytes(), packed_span.as_bytes(), "pack(span)");

        // The A operand as an owned tensor via the inherent method, then a view and a span
        // via the DotsPackedOps blanket impl.
        close(&a.dots_packed(&packed_view), &expected, "owned A");
        close(&a.view().dots_packed(&packed_view), &expected, "view A");
        close(
            &a.view().try_dots_packed(&packed_view).unwrap(),
            &expected,
            "view A try",
        );
        close(&a.span().dots_packed(&packed_view), &expected, "span A");

        // A view can also write into a caller-provided output.
        let mut into = Tensor::<f64>::try_full(&[height, width], 0.0).unwrap();
        a.view().try_dots_packed_into(&packed_view, &mut into).unwrap();
        close(&into, &expected, "view A into");

        // Transposed packing from a view with B in k×n layout: C = A × B. Non-uniform
        // values here guard the materialization inside `try_pack_transposed_in`.
        let bt_data: Vec<f32> = (0..depth * width).map(|i| i as f32 * 0.2 - 0.7).collect();
        let bt = Tensor::<f32>::from_slice(&bt_data, &[depth, width]);
        let mut expected_t = vec![0.0f64; height * width];
        for i in 0..height {
            for j in 0..width {
                let mut acc = 0.0f64;
                for l in 0..depth {
                    acc += a_data[i * depth + l] as f64 * bt_data[l * width + j] as f64;
                }
                expected_t[i * width + j] = acc;
            }
        }
        let packed_t = DotsPackedMatrix::try_pack_transposed(&bt.view()).unwrap();
        close(&a.view().dots_packed(&packed_t), &expected_t, "transposed view");
    }

    #[test]
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn dots_packed_parallel() {
        check_dots_packed_parallel::<f32>();
        check_dots_packed_parallel::<bf16>();
    }

    #[test]
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn dots_symmetric_parallel() { check_dots_symmetric_parallel::<f32>(); }

    #[test]
    fn dots_symmetric() {
        check_dots_symmetric::<f32>();
        check_dots_symmetric::<f64>();
        check_dots_symmetric::<f16>();
        check_dots_symmetric::<bf16>();
        check_dots_symmetric::<e4m3>();
        check_dots_symmetric::<e5m2>();
        check_dots_symmetric::<e2m3>();
        check_dots_symmetric::<e3m2>();
        check_dots_symmetric::<i8>();
        check_dots_symmetric::<u8>();
        check_dots_symmetric::<i4x2>();
        check_dots_symmetric::<u4x2>();
    }

    #[test]
    fn symmetric_rejects_non_contiguous_rows() {
        // A transposed view has a non-unit inner stride; the symmetric kernels read each row as
        // contiguous, so such a view must be rejected (Err) rather than read out of bounds.
        let m = Tensor::<f32>::try_full(&[4, 6], 1.0f32).unwrap();
        let transposed = m.view().try_transpose().unwrap();
        assert!(!transposed.has_contiguous_rows());
        assert!(matches!(
            transposed.try_dots_symmetric(),
            Err(TensorError::NonContiguousRows)
        ));
    }

    #[test]
    fn packed_size_matches_and_from_bytes_roundtrips() {
        init_thread();
        let (width, depth) = (4usize, 5usize);
        let b_data: Vec<f32> = (0..width * depth).map(|i| i as f32 * 0.25 + 0.3).collect();
        let b = Tensor::<f32>::from_slice(&b_data, &[width, depth]);
        let packed = DotsPackedMatrix::try_pack(&b).unwrap();

        // The static size query matches the live packed byte length and the stored dims.
        assert_eq!(
            DotsPackedMatrix::<f32>::pack_size(width, depth),
            packed.as_bytes().len()
        );
        assert_eq!(packed.shape(), (width, depth));

        // Adopting the packed bytes reproduces byte-identical contents and an identical multiply.
        let adopted =
            unsafe { DotsPackedMatrix::<f32>::from_packed_bytes_in(packed.as_bytes(), width, depth, Global).unwrap() };
        assert_eq!(adopted.as_bytes(), packed.as_bytes());
        assert_eq!(adopted.shape(), (width, depth));

        let a_data: Vec<f32> = (0..3 * depth).map(|i| i as f32 * 0.5 - 1.0).collect();
        let a = Tensor::<f32>::from_slice(&a_data, &[3, depth]);
        assert_eq!(a.dots_packed(&packed).as_slice(), a.dots_packed(&adopted).as_slice());
    }

    #[test]
    fn packed_shape_reads_dims() {
        init_thread();
        fn check<Scalar: TestableType + Dots>(width: usize, depth: usize) {
            let depth = align_depth::<Scalar>(depth);
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let packed = DotsPackedMatrix::try_pack(&b).unwrap();
            let (read_width, read_depth) = unsafe { Scalar::dots_packed_shape(packed.as_ptr()) };
            assert_eq!(
                (read_width, read_depth),
                (width, depth),
                "packed_shape<{}> @ ({width},{depth})",
                core::any::type_name::<Scalar>()
            );
        }
        for &(width, depth) in &[(4usize, 5usize), (17, 33), (1, 8)] {
            check::<f32>(width, depth);
            check::<f16>(width, depth);
            check::<bf16>(width, depth);
            check::<i8>(width, depth);
            check::<u8>(width, depth);
        }
    }

    #[test]
    fn pack_is_hermetic() {
        // Packing is a pure function of its inputs: pre-filling the destination with different garbage
        // must not change a byte of the result. Both windows are 64-aligned so the layout is identical.
        // Non-tile-multiple width/depth exercises the panel and header padding across backends (the
        // cross panel path for f32/f16, the AMX tile path for i8/u8/bf16 on Sapphire Rapids).
        fn check<Scalar: TestableType + Dots>() {
            let (width, depth) = (5usize, align_depth::<Scalar>(20usize));
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let size = <Scalar as Dots>::dots_pack_size(width, depth);
            let b_ptr = b.as_ptr();
            let b_stride = b.stride_bytes(0) as usize;

            let pack_with_fill = |fill: u8| -> Vec<u8> {
                let mut backing = vec![fill; size + SIMD_ALIGNMENT];
                let base = backing.as_mut_ptr();
                let packed = unsafe { base.add(base.align_offset(SIMD_ALIGNMENT)) };
                unsafe {
                    <Scalar as Dots>::dots_pack(b_ptr, width, depth, b_stride, packed);
                    core::slice::from_raw_parts(packed, size).to_vec()
                }
            };
            assert_eq!(
                pack_with_fill(0x00),
                pack_with_fill(0xFF),
                "dots pack must be a pure function of its inputs: {}",
                core::any::type_name::<Scalar>()
            );
        }
        init_thread();
        check::<f32>();
        check::<f16>();
        check::<bf16>();
        check::<i8>();
        check::<u8>();
    }
}
