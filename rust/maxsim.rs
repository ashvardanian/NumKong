//! MaxSim (ColBERT-style late-interaction) scoring with pre-packed matrices.
//!
//! MaxSim is the late-interaction similarity introduced by ColBERT: given a
//! query matrix `Q`, one row per query token, and a document matrix `D`, one
//! row per document token, the score is the sum over query rows of the max
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
//! 1. Pack both the query set and the document set with [`MaxSimPackedMatrix::try_pack`].
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
//! ```rust,no_run
//! use numkong::{MaxSimPackedMatrix, Tensor};
//!
//! // Required once per thread before scoring: enables AMX tile state on x86.
//! numkong::capabilities::configure_thread();
//!
//! let queries = Tensor::<f32>::try_full(&[32, 128], 1.0).unwrap();
//! let documents = Tensor::<f32>::try_full(&[1024, 128], 1.0).unwrap();
//!
//! let queries_packed = MaxSimPackedMatrix::try_pack(&queries).unwrap();
//! let docs_packed = MaxSimPackedMatrix::try_pack(&documents).unwrap();
//! let score = queries_packed.try_score(&docs_packed).unwrap();
//! ```

#[cfg(feature = "alloc")]
extern crate alloc;

use core::marker::PhantomData;

use crate::tensor::{Allocator, Global, PackedBuffer, TensorError, TensorRef};
use crate::types::{bf16, f16, StorageElement};

// region: FFI

#[link(name = "numkong")]
extern "C" {
    fn nk_maxsim_pack_size_f32(vector_count: usize, depth: usize) -> usize;
    fn nk_maxsim_pack_f32(data: *const f32, vector_count: usize, depth: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_f32(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut f64,
    );

    fn nk_maxsim_pack_size_f16(vector_count: usize, depth: usize) -> usize;
    fn nk_maxsim_pack_f16(data: *const f16, vector_count: usize, depth: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_f16(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut f32,
    );

    fn nk_maxsim_pack_size_bf16(vector_count: usize, depth: usize) -> usize;
    fn nk_maxsim_pack_bf16(data: *const bf16, vector_count: usize, depth: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_bf16(
        q: *const u8,
        d: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut f32,
    );

    fn nk_maxsim_packed_shape_f32(packed: *const u8, vector_count: *mut usize, depth: *mut usize);
    fn nk_maxsim_packed_shape_f16(packed: *const u8, vector_count: *mut usize, depth: *mut usize);
    fn nk_maxsim_packed_shape_bf16(packed: *const u8, vector_count: *mut usize, depth: *mut usize);
}

// endregion: FFI

// region: MaxSim trait

/// Trait abstracting MaxSim pack/score operations per scalar type.
pub trait MaxSim: StorageElement + Clone {
    /// Score type returned by MaxSim scoring.
    type Score: Clone + Default;

    /// Returns the packed buffer size in bytes for `vector_count` vector_count of given `depth`.
    fn maxsim_pack_size(vector_count: usize, depth: usize) -> usize;

    /// Reads the packed vector count and depth from the buffer header.
    /// # Safety
    /// `packed` must point to a buffer produced by `maxsim_pack`.
    unsafe fn maxsim_packed_shape(packed: *const u8) -> (usize, usize);

    /// Pack vector_count into backend-specific quantized format.
    ///
    /// # Safety
    /// - `data` must point to `vector_count` rows of `depth` elements, byte stride `stride`
    /// - `packed` must have at least `maxsim_pack_size(vector_count, depth)` bytes
    unsafe fn maxsim_pack(data: *const Self, vector_count: usize, depth: usize, stride: usize, packed: *mut u8);

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

    fn maxsim_pack_size(vector_count: usize, depth: usize) -> usize {
        unsafe { nk_maxsim_pack_size_f32(vector_count, depth) }
    }

    unsafe fn maxsim_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut vector_count, mut depth) = (0usize, 0usize);
        nk_maxsim_packed_shape_f32(packed, &mut vector_count, &mut depth);
        (vector_count, depth)
    }

    unsafe fn maxsim_pack(data: *const Self, vector_count: usize, depth: usize, stride: usize, packed: *mut u8) {
        nk_maxsim_pack_f32(data, vector_count, depth, stride, packed)
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

    fn maxsim_pack_size(vector_count: usize, depth: usize) -> usize {
        unsafe { nk_maxsim_pack_size_f16(vector_count, depth) }
    }

    unsafe fn maxsim_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut vector_count, mut depth) = (0usize, 0usize);
        nk_maxsim_packed_shape_f16(packed, &mut vector_count, &mut depth);
        (vector_count, depth)
    }

    unsafe fn maxsim_pack(data: *const Self, vector_count: usize, depth: usize, stride: usize, packed: *mut u8) {
        nk_maxsim_pack_f16(data, vector_count, depth, stride, packed)
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

    fn maxsim_pack_size(vector_count: usize, depth: usize) -> usize {
        unsafe { nk_maxsim_pack_size_bf16(vector_count, depth) }
    }

    unsafe fn maxsim_packed_shape(packed: *const u8) -> (usize, usize) {
        let (mut vector_count, mut depth) = (0usize, 0usize);
        nk_maxsim_packed_shape_bf16(packed, &mut vector_count, &mut depth);
        (vector_count, depth)
    }

    unsafe fn maxsim_pack(data: *const Self, vector_count: usize, depth: usize, stride: usize, packed: *mut u8) {
        nk_maxsim_pack_bf16(data, vector_count, depth, stride, packed)
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
/// Both query and document vector_count must be packed before scoring.
/// The buffer uses i8 quantization for fast coarse screening,
/// with full-precision originals retained for refinement.
#[derive(Debug)]
pub struct MaxSimPackedMatrix<Scalar: MaxSim, Alloc: Allocator = Global> {
    buffer: PackedBuffer<Alloc>,
    vector_count: usize,
    depth: usize,
    _marker: PhantomData<Scalar>,
}

// Safety: MaxSimPackedMatrix owns its data and is just bytes
unsafe impl<Scalar: MaxSim + Send, Alloc: Allocator + Send> Send for MaxSimPackedMatrix<Scalar, Alloc> {}
unsafe impl<Scalar: MaxSim + Sync, Alloc: Allocator + Sync> Sync for MaxSimPackedMatrix<Scalar, Alloc> {}

impl<Scalar: MaxSim, Alloc: Allocator + Clone> MaxSimPackedMatrix<Scalar, Alloc> {
    /// Try to clone this packed matrix, returning an error on allocation failure.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        Ok(Self {
            buffer: self.buffer.try_clone()?,
            vector_count: self.vector_count,
            depth: self.depth,
            _marker: PhantomData,
        })
    }
}

impl<Scalar: MaxSim, Alloc: Allocator + Clone> Clone for MaxSimPackedMatrix<Scalar, Alloc> {
    fn clone(&self) -> Self { self.try_clone().expect("MaxSimPackedMatrix clone allocation failed") }
}

impl<Scalar: MaxSim, Alloc: Allocator> MaxSimPackedMatrix<Scalar, Alloc> {
    /// An empty packed set owning no allocation; fill it with [`try_pack_into`](Self::try_pack_into).
    pub fn empty_in(alloc: Alloc) -> Self {
        Self {
            buffer: PackedBuffer::empty_in(alloc),
            vector_count: 0,
            depth: 0,
            _marker: PhantomData,
        }
    }

    /// Pack vector_count from a 2D tensor view using a custom allocator.
    ///
    /// Returns `Err` if the view is not 2D, the depth axis is not contiguous,
    /// the row stride is negative, or allocation fails.
    pub fn try_pack_in<Vectors, const MAX_RANK: usize>(data: &Vectors, alloc: Alloc) -> Result<Self, TensorError>
    where
        Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let mut packed = Self::empty_in(alloc);
        packed.try_pack_into(data)?;
        Ok(packed)
    }

    /// Repack `data` into this set's existing buffer, reusing the allocation when the packed size
    /// fits `capacity` and reallocating through the stored allocator only when it must grow.
    pub fn try_pack_into<Vectors, const MAX_RANK: usize>(&mut self, data: &Vectors) -> Result<(), TensorError>
    where
        Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (vector_count, depth, row_stride_bytes) = validate_maxsim_view(data)?;
        let size = Scalar::maxsim_pack_size(vector_count, depth);
        // The packer zeros the whole buffer up front, so it owns every byte — no pre-zeroing here.
        let destination = self.buffer.reset_for_pack(size)?;
        if size > 0 {
            unsafe { Scalar::maxsim_pack(data.as_ptr(), vector_count, depth, row_stride_bytes, destination) };
        }
        self.vector_count = vector_count;
        self.depth = depth;
        Ok(())
    }

    /// Pre-grow the buffer to hold `vector_count` vectors of `depth`, so a later `try_pack_into` that fits
    /// stays allocation-free with a stable pointer — hoist this out of a decode loop.
    pub fn try_reserve(&mut self, vector_count: usize, depth: usize) -> Result<(), TensorError> {
        self.buffer.try_reserve(Scalar::maxsim_pack_size(vector_count, depth))
    }

    /// Compute the MaxSim score — sum over queries of the max cosine to any document vector —
    /// against another packed matrix, treating `self` as the queries and `other` as the documents.
    ///
    /// Returns `Err` if:
    /// - the two matrices were packed at different depths
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

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { self.buffer.allocator() }

    /// Number of vector_count in the packed set.
    pub fn vector_count(&self) -> usize { self.vector_count }

    /// Returns the shape (vector_count, depth) of the original vector set.
    pub fn shape(&self) -> (usize, usize) { (self.vector_count, self.depth) }

    /// Bytes a packed buffer occupies for `vector_count` vector_count of the given `depth` under the
    /// active backend's layout, letting a caller pre-size an external buffer without packing.
    pub fn pack_size(vector_count: usize, depth: usize) -> usize { Scalar::maxsim_pack_size(vector_count, depth) }

    /// Adopt an externally-produced packed buffer by copying `bytes` into a container-owned
    /// allocation tagged with the given `vector_count` and `depth`. The packed layout is not
    /// self-describing, so the caller must supply the shape the bytes were packed for.
    ///
    /// # Safety
    /// `bytes` must be a valid packing of `vector_count` vector_count of `depth` for `Scalar`, produced
    /// by this build's packer; anything else makes a later `try_score` read out of bounds.
    pub unsafe fn from_packed_bytes_in(
        bytes: &[u8],
        vector_count: usize,
        depth: usize,
        alloc: Alloc,
    ) -> Result<Self, TensorError> {
        let mut packed = Self::empty_in(alloc);
        packed.buffer.fill_from_bytes(bytes)?;
        packed.vector_count = vector_count;
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

// endregion: MaxSimPackedMatrix

fn validate_maxsim_view<Scalar, Vectors, const MAX_RANK: usize>(
    data: &Vectors,
) -> Result<(usize, usize, usize), TensorError>
where
    Scalar: StorageElement,
    Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
{
    if data.ndim() != 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: data.ndim(),
        });
    }

    if !data.has_contiguous_rows() {
        return Err(TensorError::NonContiguousRows);
    }

    let row_stride_bytes = data.stride_bytes(0);
    if row_stride_bytes < 0 {
        return Err(TensorError::InvalidShape {
            axis: 0,
            size: row_stride_bytes as usize,
            reason: "MaxSim requires non-negative row strides",
        });
    }

    Ok((data.shape()[0], data.shape()[1], row_stride_bytes as usize))
}

impl<Scalar: MaxSim> MaxSimPackedMatrix<Scalar, Global> {
    /// Pack a 2D tensor of vector_count for MaxSim scoring using the global allocator.
    ///
    /// The `MaxSimPackedMatrix::` qualifier names the packing target — a tensor can be packed
    /// for MaxSim or for dots, and those layouts differ, so construction goes through the typed
    /// constructor rather than a bare `tensor.try_pack()`.
    pub fn try_pack<Vectors, const MAX_RANK: usize>(data: &Vectors) -> Result<Self, TensorError>
    where
        Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::try_pack_in(data, Global)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{SliceRange, Tensor, SIMD_ALIGNMENT};

    #[test]
    fn maxsim_packs_from_tensor_view() {
        crate::capabilities::configure_thread();
        let queries = Tensor::<f32>::try_full(&[4, 16], 1.0).unwrap();
        let docs = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();

        let queries_packed = MaxSimPackedMatrix::try_pack(&queries).unwrap();
        let docs_packed = MaxSimPackedMatrix::try_pack(&docs).unwrap();

        assert_eq!(queries_packed.shape(), (4, 16));
        assert_eq!(queries_packed.vector_count(), 4);
        assert_eq!(docs_packed.shape(), (8, 16));
        assert!(queries_packed.try_score(&docs_packed).unwrap().is_finite());
    }

    #[test]
    fn maxsim_rejects_non_contiguous_depth_axis() {
        let queries = Tensor::<f32>::try_full(&[4, 16], 1.0).unwrap();
        let transposed = queries.try_transpose().unwrap();
        let result = MaxSimPackedMatrix::try_pack(&transposed);
        assert!(matches!(result, Err(TensorError::NonContiguousRows)));
    }

    #[test]
    fn maxsim_accepts_outer_strided_views() {
        let queries = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();
        let odd_rows = queries
            .try_slice(&[SliceRange::range_step(1, 7, 2), SliceRange::range_step(0, 16, 1)])
            .unwrap();

        let queries_packed = MaxSimPackedMatrix::try_pack(&odd_rows).unwrap();
        assert_eq!(queries_packed.shape(), (3, 16));
    }

    #[test]
    fn maxsim_rejects_negative_row_stride() {
        let queries = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();
        let reversed_rows = queries
            .try_slice(&[SliceRange::range_step(7, 0, -1), SliceRange::range_step(0, 16, 1)])
            .unwrap();

        let result = MaxSimPackedMatrix::try_pack(&reversed_rows);
        assert!(matches!(result, Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn packed_shape_reads_dims() {
        fn check<Scalar: MaxSim>(vector_count: usize, depth: usize, fill: Scalar) {
            let data = Tensor::<Scalar>::try_full(&[vector_count, depth], fill).unwrap();
            let packed = MaxSimPackedMatrix::try_pack(&data).unwrap();
            let (read_vectors, read_depth) = unsafe { Scalar::maxsim_packed_shape(packed.as_ptr()) };
            assert_eq!(
                (read_vectors, read_depth),
                (vector_count, depth),
                "maxsim packed_shape<{}>",
                core::any::type_name::<Scalar>()
            );
        }
        for &(vector_count, depth) in &[(4usize, 16usize), (33usize, 65usize)] {
            check::<f32>(vector_count, depth, 1.0);
            check::<f16>(vector_count, depth, f16::from_f32(1.0));
            check::<bf16>(vector_count, depth, bf16::from_f32(1.0));
        }
    }

    #[test]
    fn reserve_then_pack_into_is_allocation_free() {
        // Reserve for the largest geometry once, then repeatedly pack smaller inputs: the pointer
        // must stay stable and capacity must not change — the decode-loop reuse contract.
        crate::capabilities::configure_thread();
        let (max_vector_count, depth) = (64usize, 32usize);
        let mut packed = MaxSimPackedMatrix::<f32>::empty_in(Global);
        packed.try_reserve(max_vector_count, depth).unwrap();
        let reserved_capacity = packed.capacity();
        let reserved_ptr = packed.as_ptr();
        assert!(reserved_capacity >= <f32 as MaxSim>::maxsim_pack_size(max_vector_count, depth));

        for vector_count in [8usize, 33, 64] {
            let data = Tensor::<f32>::try_full(&[vector_count, depth], 1.0).unwrap();
            packed.try_pack_into(&data).unwrap();
            assert_eq!(packed.shape(), (vector_count, depth));
            assert_eq!(
                packed.capacity(),
                reserved_capacity,
                "reserved capacity must not change"
            );
            assert_eq!(packed.as_ptr(), reserved_ptr, "reserved pointer must stay stable");
        }
    }

    #[test]
    fn from_packed_bytes_roundtrips() {
        crate::capabilities::configure_thread();
        let data = Tensor::<f32>::try_full(&[6, 24], 0.7f32).unwrap();
        let packed = MaxSimPackedMatrix::try_pack(&data).unwrap();
        let adopted =
            unsafe { MaxSimPackedMatrix::<f32>::from_packed_bytes_in(packed.as_bytes(), 6, 24, Global) }.unwrap();
        assert_eq!(adopted.shape(), (6, 24));
        assert_eq!(adopted.as_bytes(), packed.as_bytes());
    }

    #[test]
    fn pack_is_hermetic() {
        // Packing is a pure function of its inputs: pre-filling the destination with different garbage
        // must not change a byte of the result. Both windows are 64-aligned so the layout is identical.
        crate::capabilities::configure_thread();
        let (vector_count, depth) = (5usize, 20usize); // non-tile-multiple exercises padding
        let data = Tensor::<f32>::try_full(&[vector_count, depth], 1.5f32).unwrap();
        let size = <f32 as MaxSim>::maxsim_pack_size(vector_count, depth);
        let data_ptr = data.as_ptr();
        let data_stride = data.stride_bytes(0) as usize;

        let pack_with_fill = |fill: u8| -> Vec<u8> {
            let mut backing = vec![fill; size + SIMD_ALIGNMENT];
            let base = backing.as_mut_ptr();
            let packed = unsafe { base.add(base.align_offset(SIMD_ALIGNMENT)) };
            unsafe {
                <f32 as MaxSim>::maxsim_pack(data_ptr, vector_count, depth, data_stride, packed);
                core::slice::from_raw_parts(packed, size).to_vec()
            }
        };
        assert_eq!(
            pack_with_fill(0x00),
            pack_with_fill(0xFF),
            "maxsim pack must be a pure function of its inputs (no allocator garbage in the blob)"
        );
    }
}
