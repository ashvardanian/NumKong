//! MaxSim scoring — ColBERT-style late-interaction — with pre-packed matrices.
//!
//! MaxSim is the late-interaction similarity introduced by ColBERT: given a query matrix `Q`, one
//! row per query token, and a document matrix `D`, one row per document token, the score is the sum
//! over query rows of the max inner product with any document row. It retains token-level
//! granularity — unlike single-vector retrieval — while remaining cheap enough to run over large
//! candidate sets.
//!
//! [`MaxSimPackedMatrix`] stores those matrices in a quantized format optimized for fast coarse
//! screening followed by full-precision refinement: an i8 pre-pass filters obvious non-matches
//! before the original `f32` / `f16` / `bf16` values resolve the top candidates.
//!
//! # Typical flow
//!
//! 1. Pack both the query set and the document set with [`MaxSimPackedMatrix::new`].
//! 2. Call [`MaxSimPackedMatrix::score`] on the pair; the score type is
//!    `f64` for `f32` inputs and `f32` for `f16` / `bf16` inputs.
//!
//! # Example
//!
//! The doctest below is marked `ignore` because doctests compile as separate crates and would need
//! to re-link the `libnumkong` C library that provides the `nk_maxsim_*` FFI symbols used here. The
//! in-crate tests at the bottom of this file exercise the same code path.
//!
//! ```rust,no_run
//! use numkong::{MaxSimPackedMatrix, Tensor};
//!
//! // Required once per thread before scoring: enables AMX tile state on x86.
//! let cpu = numkong::Device::cpu();
//! cpu.configure_thread(cpu.capabilities_enabled().unwrap()).unwrap();
//!
//! let queries = Tensor::<f32>::full(&[32, 128], 1.0).unwrap();
//! let documents = Tensor::<f32>::full(&[1024, 128], 1.0).unwrap();
//!
//! let queries_packed = MaxSimPackedMatrix::new(&queries).unwrap();
//! let docs_packed = MaxSimPackedMatrix::new(&documents).unwrap();
//! let score = queries_packed.score(&docs_packed).unwrap();
//! ```
//!
//! File: rust/maxsim.rs
//! Author: Ash Vardanian

use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr::null_mut;

use crate::capabilities::{enabled_cpu_capabilities_mask, nk_capability_t, nk_size_t, nk_status_t, StatusCode};
use crate::tensor::{Allocator, Global, PackedBuffer, TensorError, TensorRef};
use crate::types::{bf16, f16, StorageElement};

// region: FFI

#[link(name = "numkong")]
extern "C" {
    fn nk_maxsim_pack_size_f32_best(
        vectors: nk_size_t,
        depth: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_maxsim_pack_f32_best(
        data: *const f32,
        vectors: nk_size_t,
        depth: nk_size_t,
        stride: nk_size_t,
        packed: *mut u8,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_maxsim_packed_f32_best(
        queries: *const u8,
        documents: *const u8,
        query_count: nk_size_t,
        document_count: nk_size_t,
        depth: nk_size_t,
        result: *mut f64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_maxsim_pack_size_f16_best(
        vectors: nk_size_t,
        depth: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_maxsim_pack_f16_best(
        data: *const f16,
        vectors: nk_size_t,
        depth: nk_size_t,
        stride: nk_size_t,
        packed: *mut u8,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_maxsim_packed_f16_best(
        queries: *const u8,
        documents: *const u8,
        query_count: nk_size_t,
        document_count: nk_size_t,
        depth: nk_size_t,
        result: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_maxsim_pack_size_bf16_best(
        vectors: nk_size_t,
        depth: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_maxsim_pack_bf16_best(
        data: *const bf16,
        vectors: nk_size_t,
        depth: nk_size_t,
        stride: nk_size_t,
        packed: *mut u8,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_maxsim_packed_bf16_best(
        queries: *const u8,
        documents: *const u8,
        query_count: nk_size_t,
        document_count: nk_size_t,
        depth: nk_size_t,
        result: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_maxsim_packed_shape_f32_best(
        packed: *const u8,
        vectors: *mut nk_size_t,
        depth: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_maxsim_packed_shape_f16_best(
        packed: *const u8,
        vectors: *mut nk_size_t,
        depth: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_maxsim_packed_shape_bf16_best(
        packed: *const u8,
        vectors: *mut nk_size_t,
        depth: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
}

// endregion: FFI

// region: MaxSim trait

/// Trait abstracting MaxSim pack/score operations per scalar type.
///
/// # Errors
///
/// [`TensorError::KernelFailed`] when the kernel refuses to run, like on a buffer packed under
/// other [`Capabilities`](crate::Capabilities) than the current ones.
pub trait MaxSim: StorageElement + Clone {
    /// Score type returned by MaxSim scoring.
    type Score: Clone + Default;

    /// Returns the packed buffer size in bytes for `vectors` vectors of given `depth`.
    fn maxsim_pack_size(vectors: usize, depth: usize) -> Result<usize, TensorError>;

    /// Reads the packed vector count and depth from the buffer header.
    /// # Safety
    /// `packed` must point to a buffer produced by `maxsim_pack`.
    unsafe fn maxsim_packed_shape(packed: *const u8) -> Result<(usize, usize), TensorError>;

    /// Pack vectors into backend-specific quantized format.
    ///
    /// # Safety
    /// - `data` must point to `vectors` rows of `depth` elements, byte stride `stride`
    /// - `packed` must have at least `maxsim_pack_size(vectors, depth)` bytes
    unsafe fn maxsim_pack(
        data: *const Self,
        vectors: usize,
        depth: usize,
        stride: usize,
        packed: *mut u8,
    ) -> Result<(), TensorError>;

    /// Compute MaxSim score on pre-packed buffers.
    ///
    /// # Safety
    /// - Both buffers must have been produced by `maxsim_pack` with matching depth
    /// - `result` must point to valid, writable memory for `Self::Score`
    unsafe fn maxsim_packed(
        queries: *const u8,
        documents: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut Self::Score,
    ) -> Result<(), TensorError>;
}

// endregion: MaxSim trait

// region: MaxSim impls

impl MaxSim for f32 {
    type Score = f64;

    fn maxsim_pack_size(vectors: usize, depth: usize) -> Result<usize, TensorError> {
        let mut bytes = 0;
        unsafe { nk_maxsim_pack_size_f32_best(vectors, depth, enabled_cpu_capabilities_mask(), &mut bytes) }.check()?;
        Ok(bytes)
    }

    unsafe fn maxsim_packed_shape(packed: *const u8) -> Result<(usize, usize), TensorError> {
        let (mut vectors, mut depth) = (0usize, 0usize);
        nk_maxsim_packed_shape_f32_best(
            packed,
            &mut vectors,
            &mut depth,
            enabled_cpu_capabilities_mask(),
            null_mut(),
        )
        .check()?;
        Ok((vectors, depth))
    }

    unsafe fn maxsim_pack(
        data: *const Self,
        vectors: usize,
        depth: usize,
        stride: usize,
        packed: *mut u8,
    ) -> Result<(), TensorError> {
        nk_maxsim_pack_f32_best(
            data,
            vectors,
            depth,
            stride,
            packed,
            enabled_cpu_capabilities_mask(),
            null_mut(),
        )
        .check()
    }

    unsafe fn maxsim_packed(
        queries: *const u8,
        documents: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut Self::Score,
    ) -> Result<(), TensorError> {
        nk_maxsim_packed_f32_best(
            queries,
            documents,
            query_count,
            document_count,
            depth,
            result,
            enabled_cpu_capabilities_mask(),
            null_mut(),
        )
        .check()
    }
}

impl MaxSim for f16 {
    type Score = f32;

    fn maxsim_pack_size(vectors: usize, depth: usize) -> Result<usize, TensorError> {
        let mut bytes = 0;
        unsafe { nk_maxsim_pack_size_f16_best(vectors, depth, enabled_cpu_capabilities_mask(), &mut bytes) }.check()?;
        Ok(bytes)
    }

    unsafe fn maxsim_packed_shape(packed: *const u8) -> Result<(usize, usize), TensorError> {
        let (mut vectors, mut depth) = (0usize, 0usize);
        nk_maxsim_packed_shape_f16_best(
            packed,
            &mut vectors,
            &mut depth,
            enabled_cpu_capabilities_mask(),
            null_mut(),
        )
        .check()?;
        Ok((vectors, depth))
    }

    unsafe fn maxsim_pack(
        data: *const Self,
        vectors: usize,
        depth: usize,
        stride: usize,
        packed: *mut u8,
    ) -> Result<(), TensorError> {
        nk_maxsim_pack_f16_best(
            data,
            vectors,
            depth,
            stride,
            packed,
            enabled_cpu_capabilities_mask(),
            null_mut(),
        )
        .check()
    }

    unsafe fn maxsim_packed(
        queries: *const u8,
        documents: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut Self::Score,
    ) -> Result<(), TensorError> {
        nk_maxsim_packed_f16_best(
            queries,
            documents,
            query_count,
            document_count,
            depth,
            result,
            enabled_cpu_capabilities_mask(),
            null_mut(),
        )
        .check()
    }
}

impl MaxSim for bf16 {
    type Score = f32;

    fn maxsim_pack_size(vectors: usize, depth: usize) -> Result<usize, TensorError> {
        let mut bytes = 0;
        unsafe { nk_maxsim_pack_size_bf16_best(vectors, depth, enabled_cpu_capabilities_mask(), &mut bytes) }
            .check()?;
        Ok(bytes)
    }

    unsafe fn maxsim_packed_shape(packed: *const u8) -> Result<(usize, usize), TensorError> {
        let (mut vectors, mut depth) = (0usize, 0usize);
        nk_maxsim_packed_shape_bf16_best(
            packed,
            &mut vectors,
            &mut depth,
            enabled_cpu_capabilities_mask(),
            null_mut(),
        )
        .check()?;
        Ok((vectors, depth))
    }

    unsafe fn maxsim_pack(
        data: *const Self,
        vectors: usize,
        depth: usize,
        stride: usize,
        packed: *mut u8,
    ) -> Result<(), TensorError> {
        nk_maxsim_pack_bf16_best(
            data,
            vectors,
            depth,
            stride,
            packed,
            enabled_cpu_capabilities_mask(),
            null_mut(),
        )
        .check()
    }

    unsafe fn maxsim_packed(
        queries: *const u8,
        documents: *const u8,
        query_count: usize,
        document_count: usize,
        depth: usize,
        result: *mut Self::Score,
    ) -> Result<(), TensorError> {
        nk_maxsim_packed_bf16_best(
            queries,
            documents,
            query_count,
            document_count,
            depth,
            result,
            enabled_cpu_capabilities_mask(),
            null_mut(),
        )
        .check()
    }
}

// endregion: MaxSim impls

// region: MaxSimPackedMatrix

/// Pre-packed vector set for MaxSim scoring.
///
/// Both query and document vectors must be packed before scoring. The buffer uses i8 quantization
/// for fast coarse screening, with full-precision originals retained for refinement.
#[derive(Debug)]
pub struct MaxSimPackedMatrix<Scalar: MaxSim, Alloc: Allocator = Global> {
    buffer: PackedBuffer<Alloc>,
    vectors: usize,
    depth: usize,
    _marker: PhantomData<Scalar>,
}

// Safety: MaxSimPackedMatrix owns its data and is just bytes
unsafe impl<Scalar: MaxSim + Send, Alloc: Allocator + Send> Send for MaxSimPackedMatrix<Scalar, Alloc> {}
unsafe impl<Scalar: MaxSim + Sync, Alloc: Allocator + Sync> Sync for MaxSimPackedMatrix<Scalar, Alloc> {}

impl<Scalar: MaxSim, Alloc: Allocator + Clone> MaxSimPackedMatrix<Scalar, Alloc> {
    /// Clone this packed matrix, returning an error on allocation failure.
    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Result<Self, TensorError> {
        Ok(Self {
            buffer: self.buffer.clone()?,
            vectors: self.vectors,
            depth: self.depth,
            _marker: PhantomData,
        })
    }
}

impl<Scalar: MaxSim, Alloc: Allocator> MaxSimPackedMatrix<Scalar, Alloc> {
    /// An empty packed set owning no allocation; fill it with
    /// [`pack_into`](Self::pack_into).
    pub fn empty_in(alloc: Alloc) -> Self {
        Self {
            buffer: PackedBuffer::empty_in(alloc),
            vectors: 0,
            depth: 0,
            _marker: PhantomData,
        }
    }

    /// Pack vectors from a 2D tensor view using a custom allocator.
    ///
    /// Returns `Err` if the view is not 2D, the depth axis is not contiguous, the row stride is
    /// negative, or allocation fails.
    pub fn new_in<Vectors, const MAX_RANK: usize>(data: &Vectors, alloc: Alloc) -> Result<Self, TensorError>
    where
        Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let mut packed = Self::empty_in(alloc);
        packed.pack_into(data)?;
        Ok(packed)
    }

    /// Repack `data` into this set's existing buffer, reusing the allocation when the packed size
    /// fits `capacity` and reallocating through the stored allocator only when it must grow.
    pub fn pack_into<Vectors, const MAX_RANK: usize>(&mut self, data: &Vectors) -> Result<(), TensorError>
    where
        Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (vectors, depth, row_stride_bytes) = validate_maxsim_view(data)?;
        let size = Scalar::maxsim_pack_size(vectors, depth)?;
        // The packer zeros the whole buffer up front, so it owns every byte — no pre-zeroing here.
        let destination = self.buffer.reset_for_pack(size)?;
        if size > 0 {
            unsafe { Scalar::maxsim_pack(data.as_ptr(), vectors, depth, row_stride_bytes, destination) }?;
        }
        self.vectors = vectors;
        self.depth = depth;
        Ok(())
    }

    /// Pre-grow the buffer to hold `vectors` vectors of `depth`, so a later `pack_into` that
    /// fits stays allocation-free with a stable pointer — hoist this out of a decode loop.
    pub fn reserve(&mut self, vectors: usize, depth: usize) -> Result<(), TensorError> {
        self.buffer.reserve(Scalar::maxsim_pack_size(vectors, depth)?)
    }

    /// Compute the MaxSim score — sum over queries of the max cosine to any document vector —
    /// against another packed matrix, treating `self` as the queries and `other` as the documents.
    ///
    /// Returns `Err` if:
    /// - the two matrices were packed at different depths
    /// - the kernel refuses a matrix, like one packed under other capabilities
    pub fn score<OtherAlloc: Allocator>(
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
                self.vectors,
                other.vectors,
                self.depth,
                &mut score,
            )
        }?;
        Ok(score)
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { self.buffer.allocator() }

    /// Number of vectors in the packed set.
    pub fn vectors(&self) -> usize { self.vectors }

    /// Returns the shape __[vectors,depth]__ of the original vector set.
    pub fn shape(&self) -> (usize, usize) { (self.vectors, self.depth) }

    /// Bytes a packed buffer occupies for `vectors` vectors of the given `depth` under the active
    /// backend's layout, letting a caller pre-size an external buffer without packing.
    pub fn pack_size(vectors: usize, depth: usize) -> Result<usize, TensorError> {
        Scalar::maxsim_pack_size(vectors, depth)
    }

    /// Adopt an externally-produced packed buffer by copying `bytes` into a container-owned
    /// allocation tagged with the given `vectors` and `depth`. The packed layout is not
    /// self-describing, so the caller must supply the shape the bytes were packed for.
    ///
    /// # Safety
    /// `bytes` must be a valid packing of `vectors` vectors of `depth` for `Scalar`, produced by
    /// this build's packer; anything else makes a later `score` read out of bounds.
    pub unsafe fn from_packed_bytes_in(
        bytes: &[u8],
        vectors: usize,
        depth: usize,
        alloc: Alloc,
    ) -> Result<Self, TensorError> {
        let mut packed = Self::empty_in(alloc);
        packed.buffer.fill_from_bytes(bytes)?;
        packed.vectors = vectors;
        packed.depth = depth;
        Ok(packed)
    }

    /// Bytes currently allocated (>= the live packed size).
    pub fn capacity(&self) -> usize { self.buffer.capacity() }

    /// Reset to logically empty, keeping the allocation so the next `pack_into` reuses it.
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
    /// Pack a 2D tensor of vectors for MaxSim scoring using the global allocator.
    ///
    /// The `MaxSimPackedMatrix::` qualifier names the packing target — a tensor can be packed for
    /// MaxSim or for dots, and those layouts differ, so construction goes through the typed
    /// constructor rather than a bare `tensor.pack()`.
    pub fn new<Vectors, const MAX_RANK: usize>(data: &Vectors) -> Result<Self, TensorError>
    where
        Vectors: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::new_in(data, Global)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{SliceRange, Tensor, SIMD_ALIGNMENT};

    #[test]
    fn maxsim_packs_from_tensor_view() {
        crate::capabilities::configure_cpu_thread().unwrap();
        let queries = Tensor::<f32>::full(&[4, 16], 1.0).unwrap();
        let docs = Tensor::<f32>::full(&[8, 16], 1.0).unwrap();

        let queries_packed = MaxSimPackedMatrix::new(&queries).unwrap();
        let docs_packed = MaxSimPackedMatrix::new(&docs).unwrap();

        assert_eq!(queries_packed.shape(), (4, 16));
        assert_eq!(queries_packed.vectors(), 4);
        assert_eq!(docs_packed.shape(), (8, 16));
        assert!(queries_packed.score(&docs_packed).unwrap().is_finite());
    }

    #[test]
    fn maxsim_rejects_non_contiguous_depth_axis() {
        let queries = Tensor::<f32>::full(&[4, 16], 1.0).unwrap();
        let transposed = queries.transpose().unwrap();
        let result = MaxSimPackedMatrix::new(&transposed);
        assert!(matches!(result, Err(TensorError::NonContiguousRows)));
    }

    #[test]
    fn maxsim_accepts_outer_strided_views() {
        let queries = Tensor::<f32>::full(&[8, 16], 1.0).unwrap();
        let odd_rows = queries
            .slice(&[SliceRange::range_step(1, 7, 2), SliceRange::range_step(0, 16, 1)])
            .unwrap();

        let queries_packed = MaxSimPackedMatrix::new(&odd_rows).unwrap();
        assert_eq!(queries_packed.shape(), (3, 16));
    }

    #[test]
    fn maxsim_rejects_negative_row_stride() {
        let queries = Tensor::<f32>::full(&[8, 16], 1.0).unwrap();
        let reversed_rows = queries
            .slice(&[SliceRange::range_step(7, 0, -1), SliceRange::range_step(0, 16, 1)])
            .unwrap();

        let result = MaxSimPackedMatrix::new(&reversed_rows);
        assert!(matches!(result, Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn packed_shape_reads_dims() {
        fn check<Scalar: MaxSim>(vectors: usize, depth: usize, fill: Scalar) {
            let data = Tensor::<Scalar>::full(&[vectors, depth], fill).unwrap();
            let packed = MaxSimPackedMatrix::new(&data).unwrap();
            let (read_vectors, read_depth) = unsafe { Scalar::maxsim_packed_shape(packed.as_ptr()) }.unwrap();
            assert_eq!(
                (read_vectors, read_depth),
                (vectors, depth),
                "maxsim packed_shape<{}>",
                core::any::type_name::<Scalar>()
            );
        }
        for &(vectors, depth) in &[(4usize, 16usize), (33usize, 65usize)] {
            check::<f32>(vectors, depth, 1.0);
            check::<f16>(vectors, depth, f16::from_f32(1.0));
            check::<bf16>(vectors, depth, bf16::from_f32(1.0));
        }
    }

    #[test]
    fn reserve_then_pack_into_is_allocation_free() {
        // Reserve for the largest geometry once, then repeatedly pack smaller inputs: the pointer
        // must stay stable and capacity must not change — the decode-loop reuse contract.
        crate::capabilities::configure_cpu_thread().unwrap();
        let (max_vectors, depth) = (64usize, 32usize);
        let mut packed = MaxSimPackedMatrix::<f32>::empty_in(Global);
        packed.reserve(max_vectors, depth).unwrap();
        let reserved_capacity = packed.capacity();
        let reserved_ptr = packed.as_ptr();
        assert!(reserved_capacity >= <f32 as MaxSim>::maxsim_pack_size(max_vectors, depth).unwrap());

        for vectors in [8usize, 33, 64] {
            let data = Tensor::<f32>::full(&[vectors, depth], 1.0).unwrap();
            packed.pack_into(&data).unwrap();
            assert_eq!(packed.shape(), (vectors, depth));
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
        crate::capabilities::configure_cpu_thread().unwrap();
        let data = Tensor::<f32>::full(&[6, 24], 0.7f32).unwrap();
        let packed = MaxSimPackedMatrix::new(&data).unwrap();
        let adopted =
            unsafe { MaxSimPackedMatrix::<f32>::from_packed_bytes_in(packed.as_bytes(), 6, 24, Global) }.unwrap();
        assert_eq!(adopted.shape(), (6, 24));
        assert_eq!(adopted.as_bytes(), packed.as_bytes());
    }

    #[test]
    fn pack_is_hermetic() {
        // Packing is a pure function of its inputs: pre-filling the destination with different garbage
        // must not change a byte of the result. Both windows are 64-aligned so the layout is identical.
        crate::capabilities::configure_cpu_thread().unwrap();
        let (vectors, depth) = (5usize, 20usize); // non-tile-multiple exercises padding
        let data = Tensor::<f32>::full(&[vectors, depth], 1.5f32).unwrap();
        let size = <f32 as MaxSim>::maxsim_pack_size(vectors, depth).unwrap();
        let data_ptr = data.as_ptr();
        let data_stride = data.stride_bytes(0) as usize;

        let pack_with_fill = |fill: u8| -> Vec<u8> {
            let mut backing = vec![fill; size + SIMD_ALIGNMENT];
            let base = backing.as_mut_ptr();
            let packed = unsafe { base.add(base.align_offset(SIMD_ALIGNMENT)) };
            unsafe {
                <f32 as MaxSim>::maxsim_pack(data_ptr, vectors, depth, data_stride, packed).unwrap();
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
