//! Batched binary/set metrics — Hamming and Jaccard — over pre-packed matrices.
//!
//! This module provides:
//!
//! - [`Hammings`] / [`Jaccards`]: Low-level per-scalar batched metric traits — FFI-backed
//! - [`HammingsPackedOps`] / [`JaccardsPackedOps`]: `C = metric(A, Bᵀ)` for any [`TensorRef`]
//! - `HammingsPackedParallelOps` / `JaccardsPackedParallelOps`: the same over a thread pool
//! - [`SymmetricHammingsOps`] / [`SymmetricJaccardsOps`]: self-metric upper triangle
//!
//! The right-hand operand is a [`DotsPackedMatrix`] from the [`crate::dots`] module.
use crate::tensor::{Allocator, Global, Tensor, TensorError, TensorMut, TensorRef, TensorView};
use crate::types::{u1x8, StorageElement};

#[cfg(feature = "parallel")]
use forkunion as fu;

#[cfg(feature = "parallel")]
use crate::dots::compute_thread_rows;
use crate::dots::{validate_matrix_output, validate_packed_input, validate_symmetric_input, Dots, DotsPackedMatrix};

#[link(name = "numkong")]
extern "C" {

    fn nk_hammings_packed_u1(
        a: *const u8,
        q_packed: *const u8,
        result: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    );
    fn nk_hammings_symmetric_u1(
        vectors: *const u8,
        n_vectors: usize,
        d: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );

    fn nk_jaccards_packed_u1(
        v: *const u8,
        q_packed: *const u8,
        result: *mut f32,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    );
    fn nk_jaccards_symmetric_u1(
        vectors: *const u8,
        n_vectors: usize,
        d: usize,
        stride: usize,
        result: *mut f32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

// region: Hammings Trait

/// Low-level trait for batched **Hamming distance** operations.
///
/// Given A ∈ {0,1}ᵐˣᵏ and packed B ∈ {0,1}ⁿˣᵏ, computes C ∈ ℕᵐˣⁿ where:
/// Cᵢⱼ = popcount(aᵢ ⊕ bⱼ)
///
/// Packing is inherited from the `Dots` supertrait.
///
/// # When to use
///
/// Binary feature vectors, bit-packed into `u1x8`, are common in approximate
/// nearest-neighbour search and bloom-filter-style retrieval. Packing the
/// query set once and running Hamming distance against many candidate rows is
/// the hot path. Results accumulate in `u32`, wide enough for any practical
/// binary vector length.
pub trait Hammings: Dots {
    /// Computes Hamming distances between values matrix rows and packed query rows.
    ///
    /// # Safety
    /// - `a` must point to valid memory for the values matrix
    /// - `q_packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `result` must point to valid memory for `height * width` u32 elements
    unsafe fn hammings_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    );

    /// Computes symmetric Gram matrix of Hamming distances: C = A × Aᵀ.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for the input matrix
    /// - `result` must point to valid memory for `n_vectors * n_vectors` u32 elements
    unsafe fn hammings_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

impl Hammings for u1x8 {
    unsafe fn hammings_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut u32,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    ) {
        nk_hammings_packed_u1(
            a as *const u8,
            q_packed,
            result,
            height,
            width,
            depth,
            v_stride,
            r_stride,
        )
    }

    unsafe fn hammings_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut u32,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_hammings_symmetric_u1(
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

// endregion: Hammings Trait

// region: Jaccards Trait

/// Low-level trait for batched **Jaccard distance** operations.
///
/// Given A ∈ {0,1}ᵐˣᵏ and packed B ∈ {0,1}ⁿˣᵏ, computes C ∈ ℝᵐˣⁿ where:
/// Cᵢⱼ = 1 − popcount(aᵢ ∧ bⱼ) / popcount(aᵢ ∨ bⱼ)
///
/// Packing is inherited from the `Dots` supertrait.
///
/// # When to use
///
/// Jaccard distance measures set dissimilarity and is the natural metric for
/// binary feature presence/absence. Pack the query set once and query many
/// candidates in a single batched kernel call. The result type is `f32`
/// because the ratio is inherently fractional.
pub trait Jaccards: Dots {
    /// Result type for Jaccard distances.
    type JaccardResult: StorageElement;

    /// Computes Jaccard distances between values matrix rows and packed query rows.
    ///
    /// # Safety
    /// - `a` must point to valid memory for the values matrix
    /// - `q_packed` must be a buffer previously filled by `Dots::dots_pack`
    /// - `result` must point to valid memory for `height * width` elements
    unsafe fn jaccards_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut Self::JaccardResult,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    );

    /// Computes symmetric Gram matrix of Jaccard distances.
    ///
    /// # Safety
    /// - `vectors` must point to valid memory for the input matrix
    /// - `result` must point to valid memory for `n_vectors * n_vectors` elements
    unsafe fn jaccards_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::JaccardResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    );
}

impl Jaccards for u1x8 {
    type JaccardResult = f32;

    unsafe fn jaccards_packed(
        a: *const Self,
        q_packed: *const u8,
        result: *mut Self::JaccardResult,
        height: usize,
        width: usize,
        depth: usize,
        v_stride: usize,
        r_stride: usize,
    ) {
        nk_jaccards_packed_u1(
            a as *const u8,
            q_packed,
            result,
            height,
            width,
            depth,
            v_stride,
            r_stride,
        )
    }

    unsafe fn jaccards_symmetric(
        vectors: *const Self,
        n_vectors: usize,
        depth: usize,
        stride: usize,
        result: *mut Self::JaccardResult,
        result_stride: usize,
        row_start: usize,
        row_count: usize,
    ) {
        nk_jaccards_symmetric_u1(
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

// endregion: Jaccards Trait

// region: Tensor Hammings/Jaccards

impl<Scalar: Hammings, Alloc: Allocator + Clone, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Computes Hamming distances between rows of self and packed B matrix.
    pub fn try_hammings_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<u32, Alloc, MAX_RANK>, TensorError> {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        let mut c = Tensor::try_full_in(&[height, width], u32::default(), self.alloc.clone())?;
        unsafe {
            Scalar::hammings_packed(
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
    pub fn hammings_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<u32, Alloc, MAX_RANK> {
        self.try_hammings_packed(packed_b).expect("hammings_packed failed")
    }
}

/// Extension trait: packed Hamming distances (`C = hamming(A, Bᵀ)`) for any immutable tensor
/// reference — owned [`Tensor`], borrowed [`TensorView`], or [`TensorSpan`](crate::TensorSpan).
///
/// Blanket-implemented for every [`TensorRef`], so an `A` operand backed by an mmap'd view can
/// score against a pre-packed [`DotsPackedMatrix`] without first materializing an owned copy. The
/// allocating entry point returns a globally allocated result, since a bare view carries no
/// allocator of its own.
pub trait HammingsPackedOps<Scalar: Hammings, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    /// Hamming distances: C = self × packed_bᵀ
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
    fn try_hammings_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        let mut c = Tensor::<u32, Global, MAX_RANK>::try_full(&[height, width], u32::default())?;
        unsafe {
            Scalar::hammings_packed(
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
    fn hammings_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<u32, Global, MAX_RANK> {
        self.try_hammings_packed(packed_b).expect("hammings_packed failed")
    }

    /// Hamming distances into an existing output, avoiding allocation.
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// writable tensor container that implements [`TensorMut`] works. The
    /// kernel overwrites `c` — it need not be pre-initialized.
    fn try_hammings_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<u32, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<u32, _, OUTPUT_MAX_RANK>(c, height, width)?;
        unsafe {
            Scalar::hammings_packed(
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

impl<Scalar: Hammings, const MAX_RANK: usize, A: TensorRef<Scalar, MAX_RANK>> HammingsPackedOps<Scalar, MAX_RANK>
    for A
{
}

impl<Scalar: Jaccards, Alloc: Allocator + Clone, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Computes Jaccard distances between rows of self and packed B matrix.
    pub fn try_jaccards_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::JaccardResult, Alloc, MAX_RANK>, TensorError> {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        let mut c = Tensor::try_full_in(&[height, width], Scalar::JaccardResult::default(), self.alloc.clone())?;
        unsafe {
            Scalar::jaccards_packed(
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
    pub fn jaccards_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::JaccardResult, Alloc, MAX_RANK> {
        self.try_jaccards_packed(packed_b).expect("jaccards_packed failed")
    }
}

/// Extension trait: packed Jaccard distances (`C = jaccard(A, Bᵀ)`) for any immutable tensor
/// reference — owned [`Tensor`], borrowed [`TensorView`], or [`TensorSpan`](crate::TensorSpan).
///
/// Blanket-implemented for every [`TensorRef`], so an `A` operand backed by an mmap'd view can
/// score against a pre-packed [`DotsPackedMatrix`] without first materializing an owned copy. The
/// allocating entry point returns a globally allocated result, since a bare view carries no
/// allocator of its own.
pub trait JaccardsPackedOps<Scalar: Jaccards, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    /// Jaccard distances: C = self × packed_bᵀ
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
    fn try_jaccards_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::JaccardResult, Global, MAX_RANK>, TensorError> {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        let mut c = Tensor::<Scalar::JaccardResult, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::JaccardResult::default(),
        )?;
        unsafe {
            Scalar::jaccards_packed(
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
    fn jaccards_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::JaccardResult, Global, MAX_RANK> {
        self.try_jaccards_packed(packed_b).expect("jaccards_packed failed")
    }

    /// Jaccard distances into an existing output, avoiding allocation.
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// writable tensor container that implements [`TensorMut`] works. The
    /// kernel overwrites `c` — it need not be pre-initialized.
    fn try_jaccards_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::JaccardResult, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::JaccardResult, _, OUTPUT_MAX_RANK>(c, height, width)?;
        unsafe {
            Scalar::jaccards_packed(
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

impl<Scalar: Jaccards, const MAX_RANK: usize, A: TensorRef<Scalar, MAX_RANK>> JaccardsPackedOps<Scalar, MAX_RANK>
    for A
{
}

// endregion: Tensor Hammings/Jaccards

// region: Parallel Hammings/Jaccards

/// Extension trait: parallel packed Hamming distances over a [`fu::ThreadPool`], for any immutable
/// tensor reference. Blanket-implemented for every [`TensorRef`] whose scalar is thread-shareable,
/// so views and spans schedule across the pool without an owned copy.
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub trait HammingsPackedParallelOps<Scalar, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar: Hammings + Clone + Send + Sync,
{
    /// Parallel Hamming distances into pre-allocated output.
    ///
    /// The kernel overwrites `c`; callers need not pre-initialize.
    fn try_hammings_packed_parallel_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<u32, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<u32, _, OUTPUT_MAX_RANK>(c, height, width)?;

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
            if row_start >= height {
                return;
            }
            let row_end = (row_start + rows_per_thread).min(height);
            unsafe {
                let a_row = (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
                let c_row = (c_ptr.as_ptr() as *mut u8).add(row_start * c_stride) as *mut u32;
                Scalar::hammings_packed(
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

    /// Parallel Hamming distances with allocation.
    fn try_hammings_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        let height = self.shape()[0];
        let (width, _) = packed_b.shape();
        let mut c = Tensor::<u32, Global, MAX_RANK>::try_full(&[height, width], 0u32)?;
        self.try_hammings_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    fn hammings_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Tensor<u32, Global, MAX_RANK> {
        self.try_hammings_packed_parallel(packed_b, pool)
            .expect("parallel hammings_packed failed")
    }
}

#[cfg(feature = "parallel")]
impl<Scalar, const MAX_RANK: usize, A> HammingsPackedParallelOps<Scalar, MAX_RANK> for A
where
    Scalar: Hammings + Clone + Send + Sync,
    A: TensorRef<Scalar, MAX_RANK>,
{
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar: Hammings + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Parallel symmetric Hamming-distance matrix.
    ///
    /// Only the upper triangle of the result is guaranteed to be initialized.
    pub fn try_hammings_symmetric_parallel(
        &self,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<u32, Global, MAX_RANK>::try_full(&[n_vectors, n_vectors], 0u32)?;
        self.try_hammings_symmetric_parallel_into(&mut result, pool)?;
        Ok(result)
    }

    /// Parallel symmetric Hamming distances into pre-allocated output.
    /// Only the upper triangle is written.
    pub fn try_hammings_symmetric_parallel_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<u32, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<u32, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        let num_threads = pool.threads_count().max(1);
        let vectors_ptr = fu::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fu::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.broadcast(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let (row_start, row_count) = compute_thread_rows(thread_index, num_threads, n_vectors);
            unsafe {
                Scalar::hammings_symmetric(
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
    pub fn hammings_symmetric_parallel(&self, pool: &mut fu::ThreadPool) -> Tensor<u32, Global, MAX_RANK> {
        self.try_hammings_symmetric_parallel(pool)
            .expect("parallel hammings_symmetric failed")
    }
}

/// Extension trait: parallel packed Jaccard distances over a [`fu::ThreadPool`], for any immutable
/// tensor reference. Blanket-implemented for every [`TensorRef`] whose scalar is thread-shareable,
/// so views and spans schedule across the pool without an owned copy.
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub trait JaccardsPackedParallelOps<Scalar, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar: Jaccards + Clone + Send + Sync,
    Scalar::JaccardResult: Send + Sync,
{
    /// Parallel Jaccard distances into pre-allocated output.
    ///
    /// The kernel overwrites `c`; callers need not pre-initialize.
    fn try_jaccards_packed_parallel_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::JaccardResult, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::JaccardResult, _, OUTPUT_MAX_RANK>(c, height, width)?;

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
            if row_start >= height {
                return;
            }
            let row_end = (row_start + rows_per_thread).min(height);
            unsafe {
                let a_row = (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
                let c_row = (c_ptr.as_ptr() as *mut u8).add(row_start * c_stride) as *mut Scalar::JaccardResult;
                Scalar::jaccards_packed(
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

    /// Parallel Jaccard distances with allocation.
    fn try_jaccards_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<Scalar::JaccardResult, Global, MAX_RANK>, TensorError> {
        let height = self.shape()[0];
        let (width, _) = packed_b.shape();
        let mut c = Tensor::<Scalar::JaccardResult, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::JaccardResult::default(),
        )?;
        self.try_jaccards_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    fn jaccards_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &DotsPackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fu::ThreadPool,
    ) -> Tensor<Scalar::JaccardResult, Global, MAX_RANK> {
        self.try_jaccards_packed_parallel(packed_b, pool)
            .expect("parallel jaccards_packed failed")
    }
}

#[cfg(feature = "parallel")]
impl<Scalar, const MAX_RANK: usize, A> JaccardsPackedParallelOps<Scalar, MAX_RANK> for A
where
    Scalar: Jaccards + Clone + Send + Sync,
    Scalar::JaccardResult: Send + Sync,
    A: TensorRef<Scalar, MAX_RANK>,
{
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar: Jaccards + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::JaccardResult: Send + Sync,
{
    /// Parallel symmetric Jaccard-distance matrix.
    ///
    /// Only the upper triangle of the result is guaranteed to be initialized.
    pub fn try_jaccards_symmetric_parallel(
        &self,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<Scalar::JaccardResult, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<Scalar::JaccardResult, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            Scalar::JaccardResult::default(),
        )?;
        self.try_jaccards_symmetric_parallel_into(&mut result, pool)?;
        Ok(result)
    }

    /// Parallel symmetric Jaccard distances into pre-allocated output.
    /// Only the upper triangle is written.
    pub fn try_jaccards_symmetric_parallel_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::JaccardResult, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::JaccardResult, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        let num_threads = pool.threads_count().max(1);
        let vectors_ptr = fu::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fu::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.broadcast(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let (row_start, row_count) = compute_thread_rows(thread_index, num_threads, n_vectors);
            unsafe {
                Scalar::jaccards_symmetric(
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
    pub fn jaccards_symmetric_parallel(
        &self,
        pool: &mut fu::ThreadPool,
    ) -> Tensor<Scalar::JaccardResult, Global, MAX_RANK> {
        self.try_jaccards_symmetric_parallel(pool)
            .expect("parallel jaccards_symmetric failed")
    }
}

// endregion: Parallel Hammings/Jaccards

// region: TensorView
impl<'a, Scalar: Hammings, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Computes symmetric Hamming distance matrix for a set of binary vectors.
    pub fn try_hammings_symmetric(&self) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<u32, Global, MAX_RANK>::try_full(&[n_vectors, n_vectors], u32::default())?;
        self.try_hammings_symmetric_into(&mut result)?;
        Ok(result)
    }

    /// Computes symmetric Hamming distances into pre-allocated output.
    /// Only the upper triangle is written.
    pub fn try_hammings_symmetric_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<u32, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<u32, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        unsafe {
            Scalar::hammings_symmetric(
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

impl<'a, Scalar: Jaccards, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Computes symmetric Jaccard distance matrix for a set of binary vectors.
    pub fn try_jaccards_symmetric(&self) -> Result<Tensor<Scalar::JaccardResult, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result = Tensor::<Scalar::JaccardResult, Global, MAX_RANK>::try_full(
            &[n_vectors, n_vectors],
            Scalar::JaccardResult::default(),
        )?;
        self.try_jaccards_symmetric_into(&mut result)?;
        Ok(result)
    }

    /// Computes symmetric Jaccard distances into pre-allocated output.
    /// Only the upper triangle is written.
    pub fn try_jaccards_symmetric_into<OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::JaccardResult, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::JaccardResult, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        unsafe {
            Scalar::jaccards_symmetric(
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
/// Extension trait: symmetric Hamming distance matrix for any [`TensorRef`] implementor.
///
/// Blanket-implemented for every `TensorRef<Scalar, R>`, exposing
/// `try_hammings_symmetric` on owned [`Tensor`] as well as borrowed views.
/// The kernel writes only the upper triangle (including the diagonal) — the
/// lower triangle is not touched, so mirror it if you need a fully-populated
/// matrix.
///
/// Prefer this trait when writing generic code over `TensorRef`; use the
/// inherent [`TensorView::try_hammings_symmetric`] when you already hold a
/// view.
pub trait SymmetricHammingsOps<Scalar: Hammings, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn try_hammings_symmetric(&self) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        self.view().try_hammings_symmetric()
    }

    /// Writes the symmetric Hamming-distance matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_hammings_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(&self, c: &mut Out) -> Result<(), TensorError>
    where
        Out: TensorMut<u32, OUTPUT_MAX_RANK>,
    {
        self.view().try_hammings_symmetric_into(c)
    }
}

impl<Scalar: Hammings, const R: usize, OutputTensor: TensorRef<Scalar, R>> SymmetricHammingsOps<Scalar, R>
    for OutputTensor
{
}

/// Extension trait: symmetric Jaccard distance matrix for any [`TensorRef`] implementor.
///
/// Blanket-implemented for every `TensorRef<Scalar, R>`, so
/// `vectors.try_jaccards_symmetric()` is available on both owned [`Tensor`]
/// and borrowed views. The kernel writes only the upper triangle (including
/// the diagonal); mirror to the lower triangle yourself if a dense symmetric
/// result is required.
///
/// Prefer this trait when writing generic code over `TensorRef`; use the
/// inherent [`TensorView::try_jaccards_symmetric`] when you already hold a
/// view.
pub trait SymmetricJaccardsOps<Scalar: Jaccards, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn try_jaccards_symmetric(&self) -> Result<Tensor<Scalar::JaccardResult, Global, MAX_RANK>, TensorError> {
        self.view().try_jaccards_symmetric()
    }

    /// Writes the symmetric Jaccard-distance matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_jaccards_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(&self, c: &mut Out) -> Result<(), TensorError>
    where
        Out: TensorMut<Scalar::JaccardResult, OUTPUT_MAX_RANK>,
    {
        self.view().try_jaccards_symmetric_into(c)
    }
}

impl<Scalar: Jaccards, const R: usize, OutputTensor: TensorRef<Scalar, R>> SymmetricJaccardsOps<Scalar, R>
    for OutputTensor
{
}

// endregion: Symmetric Extension Traits

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "parallel")]
    use crate::types::{align_depth, DIMS};
    use crate::types::{assert_upper_triangle_eq, init_thread};

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn check_hammings_packed_parallel_u1() {
        init_thread();
        let topology = fu::Topology::new().unwrap();
        let mut pool = fu::ThreadPool::try_spawn(&topology, 4).unwrap();
        for &(height, width, depth) in DIMS {
            let depth = align_depth::<u1x8>(depth); // logical bit-count, multiple of 8
            let a = Tensor::<u1x8>::try_full(&[height, depth], u1x8(0xFF)).unwrap();
            let b = Tensor::<u1x8>::try_full(&[width, depth], u1x8(0xFF)).unwrap();
            let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();
            let serial = a.hammings_packed(&b_packed);
            let parallel = a.hammings_packed_parallel(&b_packed, &mut pool);
            assert_eq!(
                serial.as_slice(),
                parallel.as_slice(),
                "hammings @ ({height},{width},{depth})"
            );
            let mut into_span = Tensor::<u32>::try_full(&[height, width], 0u32).unwrap();
            a.try_hammings_packed_parallel_into(&b_packed, &mut into_span.span(), &mut pool)
                .unwrap();
            assert_eq!(serial.as_slice(), into_span.as_slice(), "hammings _parallel_into(span)");

            let serial_j = a.jaccards_packed(&b_packed);
            let parallel_j = a.jaccards_packed_parallel(&b_packed, &mut pool);
            assert_eq!(
                serial_j.as_slice(),
                parallel_j.as_slice(),
                "jaccards @ ({height},{width},{depth})"
            );
            let mut into_span_j = Tensor::<f32>::try_full(&[height, width], 0.0f32).unwrap();
            a.try_jaccards_packed_parallel_into(&b_packed, &mut into_span_j.span(), &mut pool)
                .unwrap();
            assert_eq!(
                serial_j.as_slice(),
                into_span_j.as_slice(),
                "jaccards _parallel_into(span)"
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn check_sets_symmetric_parallel() {
        init_thread();
        let topology = fu::Topology::new().unwrap();
        let mut pool = fu::ThreadPool::try_spawn(&topology, 4).unwrap();
        for &(num_vectors, _, depth) in DIMS {
            let depth = align_depth::<u1x8>(depth); // logical bit-count, multiple of 8
            let vectors = Tensor::<u1x8>::try_full(&[num_vectors, depth], u1x8(0xFF)).unwrap();

            let serial_h = vectors.view().try_hammings_symmetric().unwrap();
            let parallel_h = vectors.hammings_symmetric_parallel(&mut pool);
            assert_upper_triangle_eq(
                serial_h.as_slice(),
                parallel_h.as_slice(),
                num_vectors,
                "hammings_symmetric_parallel",
            );
            let mut into_span_h = Tensor::<u32>::try_full(&[num_vectors, num_vectors], 0u32).unwrap();
            vectors
                .try_hammings_symmetric_parallel_into(&mut into_span_h.span(), &mut pool)
                .unwrap();
            assert_upper_triangle_eq(
                serial_h.as_slice(),
                into_span_h.as_slice(),
                num_vectors,
                "hammings_symmetric_parallel_into(span)",
            );

            let serial_j = vectors.view().try_jaccards_symmetric().unwrap();
            let parallel_j = vectors.jaccards_symmetric_parallel(&mut pool);
            assert_upper_triangle_eq(
                serial_j.as_slice(),
                parallel_j.as_slice(),
                num_vectors,
                "jaccards_symmetric_parallel",
            );
            let mut into_span_j = Tensor::<f32>::try_full(&[num_vectors, num_vectors], 0.0f32).unwrap();
            vectors
                .try_jaccards_symmetric_parallel_into(&mut into_span_j.span(), &mut pool)
                .unwrap();
            assert_upper_triangle_eq(
                serial_j.as_slice(),
                into_span_j.as_slice(),
                num_vectors,
                "jaccards_symmetric_parallel_into(span)",
            );
        }
    }

    /// The `HammingsPackedOps` blanket impl must route a borrowed `TensorView` / `TensorSpan`
    /// `A` operand through the same kernel as an owned `Tensor`. Distinct bit patterns give a
    /// non-zero distance so a routing bug cannot hide behind an all-equal result.
    #[test]
    fn hammings_packed_accepts_views_and_spans() {
        init_thread();
        let (height, width, depth) = (3usize, 4usize, 16usize);
        let mut a = Tensor::<u1x8>::try_full(&[height, depth], u1x8(0b1011_0100)).unwrap();
        let b = Tensor::<u1x8>::try_full(&[width, depth], u1x8(0b1100_1010)).unwrap();
        let packed = DotsPackedMatrix::try_pack(&b).unwrap();

        let expected = a.hammings_packed(&packed);
        assert_eq!(
            a.view().hammings_packed(&packed).as_slice(),
            expected.as_slice(),
            "view A"
        );
        assert_eq!(
            a.span().hammings_packed(&packed).as_slice(),
            expected.as_slice(),
            "span A"
        );
        assert_eq!(
            a.view().try_hammings_packed(&packed).unwrap().as_slice(),
            expected.as_slice(),
            "view A try"
        );
        let mut into = Tensor::<u32>::try_full(&[height, width], 0u32).unwrap();
        a.view().try_hammings_packed_into(&packed, &mut into.span()).unwrap();
        assert_eq!(into.as_slice(), expected.as_slice(), "view A into span");
    }

    /// The `JaccardsPackedOps` blanket impl must route a borrowed `TensorView` / `TensorSpan`
    /// `A` operand through the same kernel as an owned `Tensor`.
    #[test]
    fn jaccards_packed_accepts_views_and_spans() {
        init_thread();
        let (height, width, depth) = (3usize, 4usize, 16usize);
        let mut a = Tensor::<u1x8>::try_full(&[height, depth], u1x8(0b1011_0100)).unwrap();
        let b = Tensor::<u1x8>::try_full(&[width, depth], u1x8(0b1100_1010)).unwrap();
        let packed = DotsPackedMatrix::try_pack(&b).unwrap();

        let expected = a.jaccards_packed(&packed);
        assert_eq!(
            a.view().jaccards_packed(&packed).as_slice(),
            expected.as_slice(),
            "view A"
        );
        assert_eq!(
            a.span().jaccards_packed(&packed).as_slice(),
            expected.as_slice(),
            "span A"
        );
        assert_eq!(
            a.view().try_jaccards_packed(&packed).unwrap().as_slice(),
            expected.as_slice(),
            "view A try"
        );
        let mut into = Tensor::<f32>::try_full(&[height, width], 0.0f32).unwrap();
        a.view().try_jaccards_packed_into(&packed, &mut into.span()).unwrap();
        assert_eq!(into.as_slice(), expected.as_slice(), "view A into span");
    }

    #[test]
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn sets_packed_parallel() { check_hammings_packed_parallel_u1(); }

    #[test]
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    fn sets_symmetric_parallel() { check_sets_symmetric_parallel(); }

    #[test]
    fn binary_packed_u1() {
        init_thread();
        let a = Tensor::<u1x8>::try_full(&[4, 64], u1x8(0xFF)).unwrap();
        let b = Tensor::<u1x8>::try_full(&[16, 64], u1x8(0xFF)).unwrap();
        let b_packed = DotsPackedMatrix::try_pack(&b).unwrap();

        let c = a.dots_packed(&b_packed);
        assert_eq!(c.shape(), &[4, 16]);
        assert_eq!(c.as_slice()[0], 64);

        let c_h = a.hammings_packed(&b_packed);
        assert_eq!(c_h.shape(), &[4, 16]);
        assert_eq!(c_h.as_slice()[0], 0);
        let mut c_h_into = Tensor::<u32>::try_full(&[4, 16], 0u32).unwrap();
        a.try_hammings_packed_into(&b_packed, &mut c_h_into.span()).unwrap();
        assert_eq!(c_h.as_slice(), c_h_into.as_slice());

        let c_j = a.jaccards_packed(&b_packed);
        assert_eq!(c_j.shape(), &[4, 16]);
        assert!(c_j.as_slice()[0].abs() < 1e-5);
        let mut c_j_into = Tensor::<f32>::try_full(&[4, 16], 0.0f32).unwrap();
        a.try_jaccards_packed_into(&b_packed, &mut c_j_into.span()).unwrap();
        assert_eq!(c_j.as_slice(), c_j_into.as_slice());
    }

    #[test]
    fn binary_symmetric_u1() {
        init_thread();
        let a = Tensor::<u1x8>::try_full(&[4, 64], u1x8(0xFF)).unwrap();

        let gram = a.view().try_dots_symmetric().unwrap();
        assert_eq!(gram.shape(), &[4, 4]);
        assert_eq!(gram.as_slice()[0], 64);

        let gram_h = a.try_hammings_symmetric().unwrap();
        assert_eq!(gram_h.shape(), &[4, 4]);
        assert_eq!(gram_h.as_slice()[0], 0);
        let mut gram_h_into = Tensor::<u32>::try_full(&[4, 4], 0u32).unwrap();
        a.view().try_hammings_symmetric_into(&mut gram_h_into.span()).unwrap();
        assert_upper_triangle_eq(gram_h.as_slice(), gram_h_into.as_slice(), 4, "hammings");

        let gram_j = a.try_jaccards_symmetric().unwrap();
        assert_eq!(gram_j.shape(), &[4, 4]);
        assert!(gram_j.as_slice()[0].abs() < 1e-5);
        let mut gram_j_into = Tensor::<f32>::try_full(&[4, 4], 0.0f32).unwrap();
        a.view().try_jaccards_symmetric_into(&mut gram_j_into.span()).unwrap();
        assert_upper_triangle_eq(gram_j.as_slice(), gram_j_into.as_slice(), 4, "jaccards");
    }
}
