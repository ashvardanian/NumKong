//! MaxSim (ColBERT-style late-interaction) scoring with pre-packed matrices.
//!
//! MaxSim is the late-interaction similarity introduced by ColBERT: given a
//! query matrix `Q` (one row per query token) and a document matrix `D` (one
//! row per document token), the score is the sum over query rows of the max
//! inner product with any document row. It retains token-level granularity —
//! unlike single-vector retrieval — while remaining cheap enough to run over
//! large candidate sets.
//!
//! [`MaxSimPackedMatrix`] stores those matrices in a quantized format
//! optimized for fast coarse screening followed by full-precision refinement:
//! an i8 pre-pass filters obvious non-matches before the original
//! `f32` / `f16` / `bf16` values resolve the top candidates.
//!
//! # Typical flow
//!
//! 1. Pack both the query set and the document set with
//!    [`MaxSimPackedMatrix::try_pack`] (or [`MaxSimPackOps::try_maxsim_pack`]).
//! 2. Call [`MaxSimPackedMatrix::try_score`] on the pair; the score type is
//!    `f64` for `f32` inputs and `f32` for `f16` / `bf16` inputs.
//!
//! # Example
//!
//! The doctest below is marked `ignore` because doctests compile as separate
//! crates and would need to re-link the `libnumkong` C library that provides
//! the `nk_maxsim_*` FFI symbols used here. The in-crate tests at the bottom
//! of this file exercise the same code path.
//!
//! ```rust,ignore
//! use numkong::{MaxSimPackedMatrix, Tensor};
//!
//! let queries = Tensor::<f32>::try_full(&[32, 128], 1.0).unwrap();
//! let documents = Tensor::<f32>::try_full(&[1024, 128], 1.0).unwrap();
//!
//! let queries_packed = MaxSimPackedMatrix::try_pack(&queries).unwrap();
//! let docs_packed = MaxSimPackedMatrix::try_pack(&documents).unwrap();
//! let score = queries_packed.score(&docs_packed);
//! ```

extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::tensor::{Allocator, Global, TensorError, TensorRef, SIMD_ALIGNMENT};
use crate::types::{bf16, f16, StorageElement};

// region: FFI

#[link(name = "numkong")]
extern "C" {
    fn nk_maxsim_packed_size_f32(vector_count: usize, depth: usize) -> usize;
    fn nk_maxsim_pack_f32(v: *const f32, vector_count: usize, depth: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_f32(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut f64,
    );

    fn nk_maxsim_packed_size_f16(vector_count: usize, depth: usize) -> usize;
    fn nk_maxsim_pack_f16(v: *const f16, vector_count: usize, depth: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_f16(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut f32,
    );

    fn nk_maxsim_packed_size_bf16(vector_count: usize, depth: usize) -> usize;
    fn nk_maxsim_pack_bf16(v: *const bf16, vector_count: usize, depth: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_bf16(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut f32,
    );
}

// endregion: FFI

// region: MaxSim trait

/// Trait abstracting MaxSim pack/score operations per scalar type.
pub trait MaxSim: StorageElement + Clone {
    /// Score type returned by MaxSim scoring.
    type Score: Clone + Default;

    /// Returns the packed buffer size in bytes for `vector_count` vectors of given `depth`.
    fn maxsim_packed_size(vector_count: usize, depth: usize) -> usize;

    /// Pack vectors into backend-specific quantized format.
    ///
    /// # Safety
    /// - `vectors` must point to `vector_count` rows of `depth` elements, byte stride `stride`
    /// - `packed` must have at least `maxsim_packed_size(vector_count, depth)` bytes
    unsafe fn maxsim_pack(vectors: *const Self, vector_count: usize, depth: usize, stride: usize, packed: *mut u8);

    /// Compute MaxSim score on pre-packed buffers.
    ///
    /// # Safety
    /// - Both buffers must have been produced by `maxsim_pack` with matching depth
    /// - `result` must point to valid, writable memory for `Self::Score`
    unsafe fn maxsim_packed(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut Self::Score,
    );
}

// endregion: MaxSim trait

// region: MaxSim impls

impl MaxSim for f32 {
    type Score = f64;

    fn maxsim_packed_size(vector_count: usize, depth: usize) -> usize {
        unsafe { nk_maxsim_packed_size_f32(vector_count, depth) }
    }

    unsafe fn maxsim_pack(vectors: *const Self, vector_count: usize, depth: usize, stride: usize, packed: *mut u8) {
        nk_maxsim_pack_f32(vectors, vector_count, depth, stride, packed)
    }

    unsafe fn maxsim_packed(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut Self::Score,
    ) {
        nk_maxsim_packed_f32(q, d, query_count, document_count, depth, result)
    }
}

impl MaxSim for f16 {
    type Score = f32;

    fn maxsim_packed_size(vector_count: usize, depth: usize) -> usize {
        unsafe { nk_maxsim_packed_size_f16(vector_count, depth) }
    }

    unsafe fn maxsim_pack(vectors: *const Self, vector_count: usize, depth: usize, stride: usize, packed: *mut u8) {
        nk_maxsim_pack_f16(vectors, vector_count, depth, stride, packed)
    }

    unsafe fn maxsim_packed(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut Self::Score,
    ) {
        nk_maxsim_packed_f16(q, d, query_count, document_count, depth, result)
    }
}

impl MaxSim for bf16 {
    type Score = f32;

    fn maxsim_packed_size(vector_count: usize, depth: usize) -> usize {
        unsafe { nk_maxsim_packed_size_bf16(vector_count, depth) }
    }

    unsafe fn maxsim_pack(vectors: *const Self, vector_count: usize, depth: usize, stride: usize, packed: *mut u8) {
        nk_maxsim_pack_bf16(vectors, vector_count, depth, stride, packed)
    }

    unsafe fn maxsim_packed(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut Self::Score,
    ) {
        nk_maxsim_packed_bf16(q, d, query_count, document_count, depth, result)
    }
}

// endregion: MaxSim impls

// region: MaxSimPackedMatrix

/// Pre-packed vector set for MaxSim scoring.
///
/// Both query and document vectors must be packed before scoring.
/// The buffer uses i8 quantization for fast coarse screening,
/// with full-precision originals retained for refinement.
pub struct MaxSimPackedMatrix<Scalar: MaxSim, Alloc: Allocator = Global> {
    data: NonNull<u8>,
    size: usize,
    vector_count: usize,
    depth: usize,
    alloc: Alloc,
    _marker: PhantomData<Scalar>,
}

// Safety: MaxSimPackedMatrix owns its data and is just bytes
unsafe impl<Scalar: MaxSim + Send, Alloc: Allocator + Send> Send for MaxSimPackedMatrix<Scalar, Alloc> {}
unsafe impl<Scalar: MaxSim + Sync, Alloc: Allocator + Sync> Sync for MaxSimPackedMatrix<Scalar, Alloc> {}

impl<Scalar: MaxSim, Alloc: Allocator> Drop for MaxSimPackedMatrix<Scalar, Alloc> {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe {
                let layout = alloc::alloc::Layout::from_size_align_unchecked(self.size, SIMD_ALIGNMENT);
                self.alloc.deallocate(self.data, layout);
            }
        }
    }
}

impl<Scalar: MaxSim, Alloc: Allocator + Clone> MaxSimPackedMatrix<Scalar, Alloc> {
    /// Try to clone this packed matrix, returning an error on allocation failure.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        if self.size == 0 {
            return Ok(Self {
                data: NonNull::dangling(),
                size: 0,
                vector_count: self.vector_count,
                depth: self.depth,
                alloc: self.alloc.clone(),
                _marker: PhantomData,
            });
        }

        let layout = alloc::alloc::Layout::from_size_align(self.size, SIMD_ALIGNMENT)
            .map_err(|_| TensorError::AllocationFailed)?;
        let ptr = self.alloc.allocate(layout).ok_or(TensorError::AllocationFailed)?;
        unsafe {
            core::ptr::copy_nonoverlapping(self.data.as_ptr(), ptr.as_ptr(), self.size);
        }
        Ok(Self {
            data: ptr,
            size: self.size,
            vector_count: self.vector_count,
            depth: self.depth,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        })
    }
}

impl<Scalar: MaxSim, Alloc: Allocator + Clone> Clone for MaxSimPackedMatrix<Scalar, Alloc> {
    fn clone(&self) -> Self { self.try_clone().expect("MaxSimPackedMatrix clone allocation failed") }
}

impl<Scalar: MaxSim, Alloc: Allocator> MaxSimPackedMatrix<Scalar, Alloc> {
    /// Pack vectors from a 2D tensor view using a custom allocator.
    ///
    /// Returns `Err` if the view is not 2D, the depth axis is not contiguous,
    /// the row stride is negative, or allocation fails.
    pub fn try_pack_in<Vectors, const MAX_RANK: usize>(vectors: &Vectors, alloc: Alloc) -> Result<Self, TensorError>
    where
        Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (vector_count, depth, row_stride_bytes) = validate_maxsim_view(vectors)?;
        let size = Scalar::maxsim_packed_size(vector_count, depth);

        let data = if size == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc.allocate(layout).ok_or(TensorError::AllocationFailed)?;
            unsafe {
                core::ptr::write_bytes(ptr.as_ptr(), 0, size);
            }
            ptr
        };

        if size > 0 {
            unsafe {
                Scalar::maxsim_pack(vectors.as_ptr(), vector_count, depth, row_stride_bytes, data.as_ptr());
            }
        }

        Ok(Self {
            data,
            size,
            vector_count,
            depth,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Compute MaxSim score against another packed matrix.
    ///
    /// Returns `Err` if depths don't match.
    pub fn try_score<OtherAlloc: Allocator>(
        &self,
        other: &MaxSimPackedMatrix<Scalar, OtherAlloc>,
    ) -> Result<Scalar::Score, TensorError> {
        if self.depth != other.depth {
            return Err(TensorError::DimensionMismatch {
                expected: self.depth,
                got: other.depth,
            });
        }
        let mut score = Scalar::Score::default();
        unsafe {
            Scalar::maxsim_packed(
                self.as_ptr(),
                other.as_ptr(),
                self.vector_count,
                other.vector_count,
                self.depth,
                &mut score,
            )
        };
        Ok(score)
    }

    /// Convenience that panics on error.
    pub fn score<OtherAlloc: Allocator>(&self, other: &MaxSimPackedMatrix<Scalar, OtherAlloc>) -> Scalar::Score {
        self.try_score(other).expect("MaxSimPackedMatrix::score failed")
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { &self.alloc }

    /// Returns dimensions (vector_count, depth) of the original vector set.
    pub fn dims(&self) -> (usize, usize) { (self.vector_count, self.depth) }

    /// Returns the packed data buffer.
    pub fn as_bytes(&self) -> &[u8] { unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.size) } }

    /// Returns a pointer to the packed data.
    pub fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
}

// Convenience methods using Global allocator
impl<Scalar: MaxSim> MaxSimPackedMatrix<Scalar, Global> {
    /// Pack vectors from a 2D tensor view using the global allocator.
    pub fn try_pack<Vectors, const MAX_RANK: usize>(vectors: &Vectors) -> Result<Self, TensorError>
    where
        Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::try_pack_in(vectors, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn pack<Vectors, const MAX_RANK: usize>(vectors: &Vectors) -> Self
    where
        Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::try_pack(vectors).expect("MaxSimPackedMatrix::pack failed")
    }
}

// endregion: MaxSimPackedMatrix

fn validate_maxsim_view<Scalar, Vectors, const MAX_RANK: usize>(
    vectors: &Vectors,
) -> Result<(usize, usize, usize), TensorError>
where
    Scalar: StorageElement,
    Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
{
    if vectors.ndim() != 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: vectors.ndim(),
        });
    }

    if !vectors.has_contiguous_rows() {
        return Err(TensorError::NonContiguousRows);
    }

    let row_stride_bytes = vectors.stride_bytes(0);
    if row_stride_bytes < 0 {
        return Err(TensorError::InvalidShape {
            axis: 0,
            size: row_stride_bytes as usize,
            reason: "MaxSim requires non-negative row strides",
        });
    }

    Ok((vectors.shape()[0], vectors.shape()[1], row_stride_bytes as usize))
}

// region: MaxSimPackOps convenience

/// Extension trait: MaxSim packing for any [`TensorRef`] implementor.
///
/// Blanket-implemented for `Tensor`, `TensorView`, and `TensorSpan`, so a
/// caller holding any immutable tensor reference can pack for MaxSim scoring
/// without first calling `.view()`.
pub trait MaxSimPackOps<Scalar: MaxSim, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    /// Pack this 2D tensor for MaxSim scoring using the provided allocator.
    fn try_maxsim_pack_in<Alloc: Allocator>(
        &self,
        alloc: Alloc,
    ) -> Result<MaxSimPackedMatrix<Scalar, Alloc>, TensorError> {
        MaxSimPackedMatrix::try_pack_in(self, alloc)
    }

    /// Pack this 2D tensor for MaxSim scoring using the global allocator.
    fn try_maxsim_pack(&self) -> Result<MaxSimPackedMatrix<Scalar, Global>, TensorError> {
        self.try_maxsim_pack_in(Global)
    }
}

impl<Scalar: MaxSim, const MAX_RANK: usize, Vectors: TensorRef<Scalar, MAX_RANK>> MaxSimPackOps<Scalar, MAX_RANK>
    for Vectors
{
}

// endregion: MaxSimPackOps convenience

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{SliceRange, Tensor};

    #[test]
    fn maxsim_packs_from_tensor_view() {
        let queries = Tensor::<f32>::try_full(&[4, 16], 1.0).unwrap();
        let docs = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();

        let queries_packed = queries.try_maxsim_pack().unwrap();
        let docs_packed = docs.try_maxsim_pack().unwrap();

        assert_eq!(queries_packed.dims(), (4, 16));
        assert_eq!(docs_packed.dims(), (8, 16));
        assert!(queries_packed.score(&docs_packed).is_finite());
    }

    #[test]
    fn maxsim_rejects_non_contiguous_depth_axis() {
        let queries = Tensor::<f32>::try_full(&[4, 16], 1.0).unwrap();
        let transposed = queries.transpose().unwrap();
        let result = transposed.try_maxsim_pack();
        assert!(matches!(result, Err(TensorError::NonContiguousRows)));
    }

    #[test]
    fn maxsim_accepts_outer_strided_views() {
        let queries = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();
        let odd_rows = queries
            .slice(&[SliceRange::range_step(1, 7, 2), SliceRange::range_step(0, 16, 1)])
            .unwrap();

        let queries_packed = odd_rows.try_maxsim_pack().unwrap();
        assert_eq!(queries_packed.dims(), (3, 16));
    }

    #[test]
    fn maxsim_rejects_negative_row_stride() {
        let queries = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();
        let reversed_rows = queries
            .slice(&[SliceRange::range_step(7, 0, -1), SliceRange::range_step(0, 16, 1)])
            .unwrap();

        let result = reversed_rows.try_maxsim_pack();
        assert!(matches!(result, Err(TensorError::InvalidShape { .. })));
    }
}
