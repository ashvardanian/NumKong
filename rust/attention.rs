//! Ragged scaled-dot-product attention with a pre-packed KV-cache.
//!
//! The attention family operates on ragged batches: a directory of variable-length
//! segments shares one packed KV-cache blob, and every `(segment, head)` pair is an
//! independent task. Packing rearranges K and V into a backend-opaque layout — AMX
//! tiles on Sapphire Rapids, dtype-preserving planes on the AVX tiers — so the hot
//! kernel streams data in its native format.
//!
//! # Typical flow
//!
//! 1. Pack the K/V token matrices once per layer with [`AttentionPackedMatrix::try_pack`].
//! 2. Call [`AttentionPackedMatrix::try_attention`] with the query tokens and the
//!    cumulative `query_offsets`; `arange` offsets turn the call into a batched
//!    single-query pool over the same packed cache.
//!
//! # Example
//!
//! The doctest below is marked `ignore` because doctests compile as separate crates
//! and would need to re-link the `libnumkong` C library providing the
//! `nk_attention_*` FFI symbols. The in-crate tests at the bottom of this file
//! exercise the same code path.
//!
//! ```rust,no_run
//! use numkong::{bf16, AttentionPackedMatrix, Tensor};
//!
//! let tokens = 128;
//! let (heads, depth) = (8, 128);
//! let keys = Tensor::<bf16>::try_full(&[tokens, heads * depth], bf16::from_f32(0.1)).unwrap();
//! let values = keys.try_clone().unwrap();
//! let offsets = [0u32, tokens as u32];
//!
//! let kv = AttentionPackedMatrix::try_pack(&keys.view(), &values.view(), depth, &offsets, None).unwrap();
//! let outputs = kv.try_attention(&keys.view(), &offsets, None).unwrap();
//! ```

#[cfg(feature = "alloc")]
extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::scalar::Roots;
use crate::tensor::{Allocator, Global, Tensor, TensorError, TensorMut, TensorRef, SIMD_ALIGNMENT};
use crate::types::{bf16, e4m3, StorageElement};

#[cfg(feature = "parallel")]
use forkunion as fu;

// region: FFI

#[link(name = "numkong")]
extern "C" {
    fn nk_attention_pack_size_bf16(
        heads: usize,
        depth: usize,
        segment_lengths: *const u32,
        segment_count: usize,
    ) -> usize;
    fn nk_attention_pack_bf16(
        keys: *const bf16,
        values: *const bf16,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride_bytes: usize,
        value_stride_bytes: usize,
        key_value_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    );
    fn nk_attention_packed_bf16(
        queries: *const bf16,
        key_value_packed: *const u8,
        output: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );

    fn nk_attention_pack_size_e4m3(
        heads: usize,
        depth: usize,
        segment_lengths: *const u32,
        segment_count: usize,
    ) -> usize;
    fn nk_attention_pack_e4m3(
        keys: *const e4m3,
        values: *const e4m3,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride_bytes: usize,
        value_stride_bytes: usize,
        key_value_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    );
    fn nk_attention_packed_e4m3(
        queries: *const e4m3,
        key_value_packed: *const u8,
        output: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );

    fn nk_attention_pack_size_i8(
        heads: usize,
        depth: usize,
        segment_lengths: *const u32,
        segment_count: usize,
    ) -> usize;
    fn nk_attention_pack_i8(
        keys: *const i8,
        values: *const i8,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride_bytes: usize,
        value_stride_bytes: usize,
        key_value_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    );
    fn nk_attention_packed_i8(
        queries: *const i8,
        key_value_packed: *const u8,
        output: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );

    fn nk_attention_packed_shape_bf16(packed: *const u8, heads: *mut usize, depth: *mut usize, segments: *mut usize);
    fn nk_attention_packed_shape_e4m3(packed: *const u8, heads: *mut usize, depth: *mut usize, segments: *mut usize);
    fn nk_attention_packed_shape_i8(packed: *const u8, heads: *mut usize, depth: *mut usize, segments: *mut usize);
}

// endregion: FFI

// region: Attention trait

/// Trait abstracting ragged-attention pack/compute operations per scalar type.
pub trait Attention: StorageElement + Clone {
    /// Returns the packed KV-cache size in bytes for the given segment geometry.
    fn attention_pack_size(heads: usize, depth: usize, segment_lengths: &[u32], segment_count: usize) -> usize;

    /// Reads the KV-cache geometry — heads, depth, segments — from the packed header.
    /// # Safety
    /// `packed` must point to a buffer produced by `attention_pack`.
    unsafe fn attention_packed_shape(packed: *const u8) -> (usize, usize, usize);

    /// Pack a window of the `(segment, kv_head)` task grid into the KV-cache blob.
    /// # Safety
    /// - `k` / `v` must point to token matrices with `key_stride_bytes` / `value_stride_bytes` byte rows
    ///   covering every token addressed by `segment_offsets` + `segment_lengths`
    /// - `key_value_packed` must have at least `attention_pack_size(..)` bytes
    /// - a window with `first_task > 0` requires the header already initialized by a
    ///   prior or concurrent window covering task 0
    #[allow(clippy::too_many_arguments)]
    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride_bytes: usize,
        value_stride_bytes: usize,
        key_value_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    );

    /// Compute a window of the `(segment, head)` attention task grid.
    /// # Safety
    /// - `key_value_packed` must have been produced by `attention_pack` with matching geometry
    /// - `queries` rows addressed by `query_offsets` must be valid, `output` writable
    ///   with `output_stride_bytes` byte rows
    #[allow(clippy::too_many_arguments)]
    unsafe fn attention_packed(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );
}

impl Attention for bf16 {
    fn attention_pack_size(heads: usize, depth: usize, segment_lengths: &[u32], segment_count: usize) -> usize {
        unsafe { nk_attention_pack_size_bf16(heads, depth, segment_lengths.as_ptr(), segment_count) }
    }

    unsafe fn attention_packed_shape(packed: *const u8) -> (usize, usize, usize) {
        let (mut heads, mut depth, mut segments) = (0usize, 0usize, 0usize);
        nk_attention_packed_shape_bf16(packed, &mut heads, &mut depth, &mut segments);
        (heads, depth, segments)
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride_bytes: usize,
        value_stride_bytes: usize,
        key_value_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    ) {
        unsafe {
            nk_attention_pack_bf16(
                keys,
                values,
                heads,
                depth,
                segment_offsets,
                segment_lengths,
                segment_count,
                key_stride_bytes,
                value_stride_bytes,
                key_value_packed,
                first_task,
                task_count,
            )
        }
    }

    unsafe fn attention_packed(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    ) {
        unsafe {
            nk_attention_packed_bf16(
                queries,
                key_value_packed,
                output,
                head_count,
                heads,
                depth,
                query_offsets,
                query_stride_bytes,
                output_stride_bytes,
                scale,
                first_task,
                task_count,
            )
        }
    }
}

impl Attention for e4m3 {
    fn attention_pack_size(heads: usize, depth: usize, segment_lengths: &[u32], segment_count: usize) -> usize {
        unsafe { nk_attention_pack_size_e4m3(heads, depth, segment_lengths.as_ptr(), segment_count) }
    }

    unsafe fn attention_packed_shape(packed: *const u8) -> (usize, usize, usize) {
        let (mut heads, mut depth, mut segments) = (0usize, 0usize, 0usize);
        nk_attention_packed_shape_e4m3(packed, &mut heads, &mut depth, &mut segments);
        (heads, depth, segments)
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride_bytes: usize,
        value_stride_bytes: usize,
        key_value_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    ) {
        unsafe {
            nk_attention_pack_e4m3(
                keys,
                values,
                heads,
                depth,
                segment_offsets,
                segment_lengths,
                segment_count,
                key_stride_bytes,
                value_stride_bytes,
                key_value_packed,
                first_task,
                task_count,
            )
        }
    }

    unsafe fn attention_packed(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    ) {
        unsafe {
            nk_attention_packed_e4m3(
                queries,
                key_value_packed,
                output,
                head_count,
                heads,
                depth,
                query_offsets,
                query_stride_bytes,
                output_stride_bytes,
                scale,
                first_task,
                task_count,
            )
        }
    }
}

impl Attention for i8 {
    fn attention_pack_size(heads: usize, depth: usize, segment_lengths: &[u32], segment_count: usize) -> usize {
        unsafe { nk_attention_pack_size_i8(heads, depth, segment_lengths.as_ptr(), segment_count) }
    }

    unsafe fn attention_packed_shape(packed: *const u8) -> (usize, usize, usize) {
        let (mut heads, mut depth, mut segments) = (0usize, 0usize, 0usize);
        nk_attention_packed_shape_i8(packed, &mut heads, &mut depth, &mut segments);
        (heads, depth, segments)
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride_bytes: usize,
        value_stride_bytes: usize,
        key_value_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    ) {
        unsafe {
            nk_attention_pack_i8(
                keys,
                values,
                heads,
                depth,
                segment_offsets,
                segment_lengths,
                segment_count,
                key_stride_bytes,
                value_stride_bytes,
                key_value_packed,
                first_task,
                task_count,
            )
        }
    }

    unsafe fn attention_packed(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    ) {
        unsafe {
            nk_attention_packed_i8(
                queries,
                key_value_packed,
                output,
                head_count,
                heads,
                depth,
                query_offsets,
                query_stride_bytes,
                output_stride_bytes,
                scale,
                first_task,
                task_count,
            )
        }
    }
}

// endregion: Attention trait

// region: AttentionPackedMatrix

/// Pre-packed ragged KV-cache for scaled-dot-product attention.
/// Owns the backend-opaque blob plus the geometry needed to validate query batches:
/// segment count, KV head count, head width, and total token count.
#[derive(Debug)]
pub struct AttentionPackedMatrix<Scalar: Attention, Alloc: Allocator = Global> {
    data: NonNull<u8>,
    /// Bytes of live packed content.
    size: usize,
    /// Bytes actually allocated (>= size); lets `try_pack_into` reuse the buffer across steps.
    capacity: usize,
    heads: usize,
    depth: usize,
    segment_count: usize,
    total_tokens: usize,
    alloc: Alloc,
    _marker: PhantomData<Scalar>,
}

// Safety: AttentionPackedMatrix owns its data and is just bytes
unsafe impl<Scalar: Attention + Send, Alloc: Allocator + Send> Send for AttentionPackedMatrix<Scalar, Alloc> {}
unsafe impl<Scalar: Attention + Sync, Alloc: Allocator + Sync> Sync for AttentionPackedMatrix<Scalar, Alloc> {}

impl<Scalar: Attention, Alloc: Allocator> Drop for AttentionPackedMatrix<Scalar, Alloc> {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }
        unsafe { crate::tensor::dealloc_aligned(&self.alloc, self.data, self.capacity) };
    }
}

impl<Scalar: Attention, Alloc: Allocator + Clone> AttentionPackedMatrix<Scalar, Alloc> {
    /// Try to clone this packed KV-cache, returning an error on allocation failure.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        let data = if self.size == 0 {
            NonNull::dangling()
        } else {
            let layout = core::alloc::Layout::from_size_align(self.size, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = self.alloc.allocate(layout).ok_or(TensorError::AllocationFailed)?;
            unsafe { core::ptr::copy_nonoverlapping(self.data.as_ptr(), ptr.as_ptr(), self.size) };
            ptr
        };
        Ok(Self {
            data,
            size: self.size,
            capacity: self.size,
            heads: self.heads,
            depth: self.depth,
            segment_count: self.segment_count,
            total_tokens: self.total_tokens,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        })
    }
}

impl<Scalar: Attention, Alloc: Allocator + Clone> Clone for AttentionPackedMatrix<Scalar, Alloc> {
    fn clone(&self) -> Self { self.try_clone().expect("AttentionPackedMatrix clone allocation failed") }
}

/// Validates a `[tokens, heads * depth]` token-matrix view against a head width.
fn validate_token_view<Scalar, View, const MAX_RANK: usize>(
    view: &View,
    depth: usize,
) -> Result<(usize, usize, usize), TensorError>
where
    Scalar: StorageElement,
    View: TensorRef<Scalar, MAX_RANK> + ?Sized,
{
    if view.ndim() != 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: view.ndim(),
        });
    }
    if !view.has_contiguous_rows() {
        return Err(TensorError::NonContiguousRows);
    }
    let row_stride_bytes = view.stride_bytes(0);
    if row_stride_bytes < 0 {
        return Err(TensorError::InvalidShape {
            axis: 0,
            size: row_stride_bytes as usize,
            reason: "attention requires non-negative row strides",
        });
    }
    if depth == 0 || view.shape()[1] % depth != 0 {
        return Err(TensorError::DimensionMismatch {
            expected: depth,
            got: view.shape()[1],
        });
    }
    Ok((view.shape()[0], view.shape()[1] / depth, row_stride_bytes as usize))
}

/// Validates the `keys` and `values` token views together and returns their shared geometry as
/// `(tokens, heads, keys_stride_bytes, values_stride_bytes)`. Both views must be 2D
/// `[tokens, heads * depth]` with contiguous rows and matching token and head counts.
fn validate_attention_views<Scalar, Keys, Values, const MAX_RANK: usize>(
    keys: &Keys,
    values: &Values,
    depth: usize,
) -> Result<(usize, usize, usize, usize), TensorError>
where
    Scalar: StorageElement,
    Keys: TensorRef<Scalar, MAX_RANK> + ?Sized,
    Values: TensorRef<Scalar, MAX_RANK> + ?Sized,
{
    let (keys_tokens, keys_heads, keys_stride_bytes) = validate_token_view(keys, depth)?;
    let (values_tokens, values_heads, values_stride_bytes) = validate_token_view(values, depth)?;
    if keys_tokens != values_tokens || keys_heads != values_heads {
        return Err(TensorError::DimensionMismatch {
            expected: keys_tokens,
            got: values_tokens,
        });
    }
    Ok((keys_tokens, keys_heads, keys_stride_bytes, values_stride_bytes))
}

/// Validates that `offsets` is cumulative and covers at most `tokens` rows.
fn validate_offsets(offsets: &[u32], tokens: usize) -> Result<usize, TensorError> {
    if offsets.len() < 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: offsets.len(),
        });
    }
    for pair in offsets.windows(2) {
        if pair[1] < pair[0] {
            return Err(TensorError::InvalidShape {
                axis: 0,
                size: pair[1] as usize,
                reason: "offsets must be non-decreasing",
            });
        }
    }
    let last = *offsets.last().unwrap() as usize;
    if last > tokens {
        return Err(TensorError::DimensionMismatch {
            expected: tokens,
            got: last,
        });
    }
    Ok(offsets.len() - 1)
}

impl<Scalar: Attention, Alloc: Allocator> AttentionPackedMatrix<Scalar, Alloc> {
    /// An empty cache that owns no allocation, holding only the given allocator.
    ///
    /// Nothing is packed yet: every geometry field reads zero and [`as_bytes`](Self::as_bytes)
    /// is empty until the first [`try_pack_into`](Self::try_pack_into), which allocates on demand
    /// and can then be re-run each decode step to reuse the buffer.
    pub fn empty_in(alloc: Alloc) -> Self {
        Self {
            data: NonNull::dangling(),
            size: 0,
            capacity: 0,
            heads: 0,
            depth: 0,
            segment_count: 0,
            total_tokens: 0,
            alloc,
            _marker: PhantomData,
        }
    }

    /// Pack `keys`/`values` into a freshly allocated cache using the given allocator.
    pub fn try_pack_in<KIn, VIn, const MAX_RANK: usize>(
        keys: &KIn,
        values: &VIn,
        depth: usize,
        segment_offsets: &[u32],
        segment_lengths: Option<&[u32]>,
        alloc: Alloc,
    ) -> Result<Self, TensorError>
    where
        KIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        VIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let mut cache = Self::empty_in(alloc);
        cache.try_pack_into(keys, values, depth, segment_offsets, segment_lengths)?;
        Ok(cache)
    }

    /// Repack `keys`/`values` into this cache's existing blob, reusing the allocation when the
    /// packed size fits `capacity` and reallocating through the stored allocator only when it must
    /// grow. The decode-loop path: allocate the cache once at layer-init, then refresh it every
    /// forward step with no further allocation. Packing overwrites, so a grow discards the old
    /// contents rather than copying them.
    pub fn try_pack_into<KIn, VIn, const MAX_RANK: usize>(
        &mut self,
        keys: &KIn,
        values: &VIn,
        depth: usize,
        segment_offsets: &[u32],
        segment_lengths: Option<&[u32]>,
    ) -> Result<(), TensorError>
    where
        KIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        VIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (tokens, heads, keys_stride_bytes, values_stride_bytes) = validate_attention_views(keys, values, depth)?;
        let segment_count = validate_offsets(segment_offsets, tokens)?;

        #[cfg(feature = "alloc")]
        let mut lengths_storage;
        let segment_lengths = match segment_lengths {
            Some(lengths) => {
                if lengths.len() != segment_count {
                    return Err(TensorError::DimensionMismatch {
                        expected: segment_count,
                        got: lengths.len(),
                    });
                }
                lengths
            }
            #[cfg(feature = "alloc")]
            None => {
                lengths_storage = alloc::vec::Vec::with_capacity(segment_count);
                for pair in segment_offsets.windows(2) {
                    lengths_storage.push(pair[1] - pair[0]);
                }
                &lengths_storage[..]
            }
            #[cfg(not(feature = "alloc"))]
            None => {
                return Err(TensorError::InvalidShape {
                    axis: 0,
                    size: 0,
                    reason: "segment_lengths must be supplied without the `alloc` feature",
                })
            }
        };

        let size = Scalar::attention_pack_size(heads, depth, segment_lengths, segment_count);
        if size > self.capacity {
            self.grow_to(size)?;
        }
        self.size = size;
        self.heads = heads;
        self.depth = depth;
        self.segment_count = segment_count;
        self.total_tokens = *segment_offsets.last().unwrap() as usize;
        if size == 0 {
            return Ok(());
        }
        unsafe {
            // Zero first so any alignment padding the packer leaves untouched is deterministic —
            // packed blobs stay byte-reproducible across allocations and match the parallel path.
            core::ptr::write_bytes(self.data.as_ptr(), 0, size);
            Scalar::attention_pack(
                keys.as_ptr(),
                values.as_ptr(),
                heads,
                depth,
                segment_offsets.as_ptr(),
                segment_lengths.as_ptr(),
                segment_count,
                keys_stride_bytes,
                values_stride_bytes,
                self.data.as_ptr(),
                0,
                0,
            );
        }
        Ok(())
    }

    /// Grow the backing allocation to at least `needed` bytes. Frees the old buffer without copying
    /// — the caller repacks, overwriting the contents.
    fn grow_to(&mut self, needed: usize) -> Result<(), TensorError> {
        let new_data = if needed == 0 {
            NonNull::dangling()
        } else {
            let layout = core::alloc::Layout::from_size_align(needed, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            self.alloc.allocate(layout).ok_or(TensorError::AllocationFailed)?
        };
        if self.capacity > 0 {
            unsafe {
                let old = core::alloc::Layout::from_size_align_unchecked(self.capacity, SIMD_ALIGNMENT);
                self.alloc.deallocate(self.data, old);
            }
        }
        self.data = new_data;
        self.capacity = needed;
        Ok(())
    }

    /// Ragged attention into a caller-provided `f32` output tensor of the same
    /// logical shape as `q` — `[tokens, heads * depth]`, contiguous rows.
    pub fn try_attention_into<QIn, OutTensor, const MAX_RANK: usize, const OUT_MAX_RANK: usize>(
        &self,
        queries: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
        output: &mut OutTensor,
    ) -> Result<(), TensorError>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutTensor: TensorMut<f32, OUT_MAX_RANK> + ?Sized,
    {
        let (query_tokens, query_head_count, query_stride_bytes) = validate_token_view(queries, self.depth)?;
        if query_head_count % self.heads != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: self.heads,
                got: query_head_count,
            });
        }
        let segment_count = validate_offsets(query_offsets, query_tokens)?;
        if segment_count != self.segment_count {
            return Err(TensorError::DimensionMismatch {
                expected: self.segment_count,
                got: segment_count,
            });
        }
        let row_values = query_head_count * self.depth;
        if output.shape() != [query_tokens, row_values] {
            return Err(TensorError::DimensionMismatch {
                expected: query_tokens * row_values,
                got: output.shape().iter().product(),
            });
        }
        let scale = scale.unwrap_or_else(|| (self.depth as f32).rsqrt());
        unsafe {
            Scalar::attention_packed(
                queries.as_ptr(),
                self.data.as_ptr(),
                output.as_mut_ptr(),
                query_head_count,
                self.heads,
                self.depth,
                query_offsets.as_ptr(),
                query_stride_bytes,
                output.stride_bytes(0) as usize,
                scale,
                0,
                0,
            );
        }
        Ok(())
    }

    /// Bytes a packed cache needs for the given ragged geometry — the infallible size query
    /// mirroring [`Dots::dots_pack_size`](crate::Dots::dots_pack_size). `segment_lengths`
    /// carries one token count per segment, so its length is the segment count. Useful for
    /// pre-sizing an external buffer before packing, without constructing a cache.
    pub fn pack_size(heads: usize, depth: usize, segment_lengths: &[u32]) -> usize {
        Scalar::attention_pack_size(heads, depth, segment_lengths, segment_lengths.len())
    }

    /// Read the geometry of an externally-produced packed KV blob straight from its self-describing
    /// header, returning `(heads, depth, segments)`, or `None` if the slice is too short to hold
    /// one. Unlike the dots packed matrix, the attention blob records its own shape, so no
    /// caller-supplied dimensions are needed.
    ///
    /// Reads through the C `nk_attention_packed_shape_<dtype>` accessor for this cache's scalar type.
    pub fn shape(packed: &[u8]) -> Option<(usize, usize, usize)> {
        if packed.len() < 12 {
            return None;
        }
        // Safety: the length check guarantees the header prefix the accessor reads is present.
        Some(unsafe { Scalar::attention_packed_shape(packed.as_ptr()) })
    }

    /// Returns the number of ragged segments in the packed cache.
    pub fn segments(&self) -> usize { self.segment_count }

    /// Returns the number of KV heads per token.
    pub fn heads(&self) -> usize { self.heads }

    /// Returns the number of channels per head.
    pub fn depth(&self) -> usize { self.depth }

    /// Returns the total KV token count across segments.
    pub fn tokens(&self) -> usize { self.total_tokens }

    /// Bytes currently allocated for the packed blob (>= the live packed size).
    pub fn capacity(&self) -> usize { self.capacity }

    /// Reset to logically empty, keeping the allocation so the next `try_pack_into` reuses it.
    pub fn clear(&mut self) {
        self.size = 0;
        self.segment_count = 0;
        self.total_tokens = 0;
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { &self.alloc }

    /// Returns the packed data buffer.
    pub fn as_bytes(&self) -> &[u8] { unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.size) } }

    /// Returns a pointer to the packed data.
    pub fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
}

// Convenience methods using the Global allocator
impl<Scalar: Attention> AttentionPackedMatrix<Scalar, Global> {
    /// Pack ragged K/V token matrices using the global allocator.
    pub fn try_pack<KIn, VIn, const MAX_RANK: usize>(
        keys: &KIn,
        values: &VIn,
        depth: usize,
        segment_offsets: &[u32],
        segment_lengths: Option<&[u32]>,
    ) -> Result<Self, TensorError>
    where
        KIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        VIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::try_pack_in(keys, values, depth, segment_offsets, segment_lengths, Global)
    }

    /// Convenience method that panics on error.
    pub fn pack<KIn, VIn, const MAX_RANK: usize>(
        keys: &KIn,
        values: &VIn,
        depth: usize,
        segment_offsets: &[u32],
        segment_lengths: Option<&[u32]>,
    ) -> Self
    where
        KIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        VIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::try_pack(keys, values, depth, segment_offsets, segment_lengths).expect("attention pack failed")
    }

    /// Ragged attention allocating a fresh `f32` output tensor.
    pub fn try_attention<QIn, const MAX_RANK: usize>(
        &self,
        queries: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
    ) -> Result<Tensor<f32>, TensorError>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (query_tokens, query_head_count, _) = validate_token_view(queries, self.depth)?;
        let mut output = Tensor::<f32>::try_full(&[query_tokens, query_head_count * self.depth], 0.0)?;
        self.try_attention_into(queries, query_offsets, scale, &mut output)?;
        Ok(output)
    }

    /// Convenience method that panics on error.
    pub fn attention<QIn, const MAX_RANK: usize>(
        &self,
        queries: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
    ) -> Tensor<f32>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        self.try_attention(queries, query_offsets, scale)
            .expect("attention failed")
    }
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar: Attention, Alloc: Allocator> AttentionPackedMatrix<Scalar, Alloc> {
    /// Ragged attention parallelized over the `(segment, head)` task grid with a
    /// ForkUnion thread pool; each worker computes a contiguous task window.
    pub fn try_attention_parallel_into<QIn, OutTensor, const MAX_RANK: usize, const OUT_MAX_RANK: usize>(
        &self,
        queries: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
        output: &mut OutTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutTensor: TensorMut<f32, OUT_MAX_RANK> + ?Sized,
    {
        let (query_tokens, query_head_count, query_stride_bytes) = validate_token_view(queries, self.depth)?;
        if query_head_count % self.heads != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: self.heads,
                got: query_head_count,
            });
        }
        let segment_count = validate_offsets(query_offsets, query_tokens)?;
        if segment_count != self.segment_count {
            return Err(TensorError::DimensionMismatch {
                expected: self.segment_count,
                got: segment_count,
            });
        }
        let row_values = query_head_count * self.depth;
        if output.shape() != [query_tokens, row_values] {
            return Err(TensorError::DimensionMismatch {
                expected: query_tokens * row_values,
                got: output.shape().iter().product(),
            });
        }
        let scale = scale.unwrap_or_else(|| (self.depth as f32).rsqrt());
        let output_stride_bytes = output.stride_bytes(0) as usize;

        let q_ptr = fu::SyncConstPtr::new(queries.as_ptr());
        let kv_ptr = fu::SyncConstPtr::new(self.data.as_ptr());
        let out_ptr = fu::SyncMutPtr::new(output.as_mut_ptr());
        let offsets_ptr = fu::SyncConstPtr::new(query_offsets.as_ptr());
        let (heads, depth) = (self.heads, self.depth);

        // Self-attention task cost is `q_len × kv_len × depth` — quadratic in segment
        // length, so equal-count windows can be ~64× unbalanced on ragged batches.
        // Dynamic per-task scheduling mirrors the Python layer's `schedule(dynamic, 1)`;
        // GQA-sibling heads stay adjacent in task order, preserving packed-KV reuse.
        let total_tasks = segment_count * query_head_count;
        pool.for_n_dynamic(total_tasks, move |prong| {
            // Configure the worker for AMX and other thread-local SIMD state — idempotent.
            crate::capabilities::configure_thread();
            unsafe {
                Scalar::attention_packed(
                    q_ptr.as_ptr(),
                    kv_ptr.as_ptr(),
                    out_ptr.as_ptr(),
                    query_head_count,
                    heads,
                    depth,
                    offsets_ptr.as_ptr(),
                    query_stride_bytes,
                    output_stride_bytes,
                    scale,
                    prong.task_index,
                    1,
                );
            }
        }); // executes and synchronizes on drop

        Ok(())
    }

    /// Ragged attention parallelized over the task grid, allocating a fresh `f32` output tensor
    /// of shape `[tokens, heads * depth]`. The allocating twin of
    /// [`try_attention_parallel_into`](Self::try_attention_parallel_into).
    pub fn try_attention_parallel<QIn, const MAX_RANK: usize>(
        &self,
        queries: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<f32>, TensorError>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (query_tokens, query_head_count, _) = validate_token_view(queries, self.depth)?;
        let mut output = Tensor::<f32>::try_full(&[query_tokens, query_head_count * self.depth], 0.0)?;
        self.try_attention_parallel_into(queries, query_offsets, scale, &mut output, pool)?;
        Ok(output)
    }

    /// Convenience method that panics on error.
    pub fn attention_parallel<QIn, const MAX_RANK: usize>(
        &self,
        queries: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
        pool: &mut fu::ThreadPool,
    ) -> Tensor<f32>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        self.try_attention_parallel(queries, query_offsets, scale, pool)
            .expect("parallel attention failed")
    }

    /// Pack ragged K/V token matrices into this cache in parallel over the `(segment, kv_head)`
    /// task grid with a ForkUnion thread pool. Sizing and any (re)allocation run serially up
    /// front; only the per-task packing fans out. Like [`try_pack_into`](Self::try_pack_into),
    /// packing overwrites, so a grow discards the old contents rather than copying them.
    pub fn try_pack_parallel_into<KIn, VIn, const MAX_RANK: usize>(
        &mut self,
        keys: &KIn,
        values: &VIn,
        depth: usize,
        segment_offsets: &[u32],
        segment_lengths: Option<&[u32]>,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), TensorError>
    where
        KIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        VIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (tokens, heads, keys_stride_bytes, values_stride_bytes) = validate_attention_views(keys, values, depth)?;
        let segment_count = validate_offsets(segment_offsets, tokens)?;

        #[cfg(feature = "alloc")]
        let mut lengths_storage;
        let segment_lengths = match segment_lengths {
            Some(lengths) => {
                if lengths.len() != segment_count {
                    return Err(TensorError::DimensionMismatch {
                        expected: segment_count,
                        got: lengths.len(),
                    });
                }
                lengths
            }
            #[cfg(feature = "alloc")]
            None => {
                lengths_storage = alloc::vec::Vec::with_capacity(segment_count);
                for pair in segment_offsets.windows(2) {
                    lengths_storage.push(pair[1] - pair[0]);
                }
                &lengths_storage[..]
            }
            #[cfg(not(feature = "alloc"))]
            None => {
                return Err(TensorError::InvalidShape {
                    axis: 0,
                    size: 0,
                    reason: "segment_lengths must be supplied without the `alloc` feature",
                })
            }
        };

        let size = Scalar::attention_pack_size(heads, depth, segment_lengths, segment_count);
        if size > self.capacity {
            self.grow_to(size)?;
        }
        self.size = size;
        self.heads = heads;
        self.depth = depth;
        self.segment_count = segment_count;
        self.total_tokens = *segment_offsets.last().unwrap() as usize;
        if size == 0 {
            return Ok(());
        }

        // Zero the blob so alignment padding is deterministic and the parallel result is
        // byte-identical to the serial path regardless of allocator-provided contents.
        unsafe { core::ptr::write_bytes(self.data.as_ptr(), 0, size) };

        // Task 0's window writes the shared header + payload-offsets directory that every other
        // window reads to locate its plane (see the `attention_pack` safety note). Run it serially
        // first: fanning task 0 out with the rest races — a worker could read the directory before
        // it is populated and scatter its plane to a garbage offset.
        unsafe {
            Scalar::attention_pack(
                keys.as_ptr(),
                values.as_ptr(),
                heads,
                depth,
                segment_offsets.as_ptr(),
                segment_lengths.as_ptr(),
                segment_count,
                keys_stride_bytes,
                values_stride_bytes,
                self.data.as_ptr(),
                0,
                1,
            );
        }

        // Remaining `(segment, kv_head)` windows read the now-populated directory. Dynamic
        // scheduling keeps GQA-sibling heads adjacent to preserve locality.
        let total_tasks = segment_count * heads;
        if total_tasks <= 1 {
            return Ok(());
        }
        let keys_ptr = fu::SyncConstPtr::new(keys.as_ptr());
        let values_ptr = fu::SyncConstPtr::new(values.as_ptr());
        let offsets_ptr = fu::SyncConstPtr::new(segment_offsets.as_ptr());
        let lengths_ptr = fu::SyncConstPtr::new(segment_lengths.as_ptr());
        let packed_ptr = fu::SyncMutPtr::new(self.data.as_ptr());
        pool.for_n_dynamic(total_tasks - 1, move |prong| {
            crate::capabilities::configure_thread();
            unsafe {
                Scalar::attention_pack(
                    keys_ptr.as_ptr(),
                    values_ptr.as_ptr(),
                    heads,
                    depth,
                    offsets_ptr.as_ptr(),
                    lengths_ptr.as_ptr(),
                    segment_count,
                    keys_stride_bytes,
                    values_stride_bytes,
                    packed_ptr.as_ptr(),
                    prong.task_index + 1,
                    1,
                );
            }
        }); // executes and synchronizes on drop
        Ok(())
    }
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar: Attention> AttentionPackedMatrix<Scalar, Global> {
    /// Pack ragged K/V token matrices in parallel using the global allocator. The allocating
    /// twin of [`try_pack_parallel_into`](Self::try_pack_parallel_into).
    pub fn try_pack_parallel<KIn, VIn, const MAX_RANK: usize>(
        keys: &KIn,
        values: &VIn,
        depth: usize,
        segment_offsets: &[u32],
        segment_lengths: Option<&[u32]>,
        pool: &mut fu::ThreadPool,
    ) -> Result<Self, TensorError>
    where
        KIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        VIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let mut cache = Self::empty_in(Global);
        cache.try_pack_parallel_into(keys, values, depth, segment_offsets, segment_lengths, pool)?;
        Ok(cache)
    }
}

// endregion: AttentionPackedMatrix

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_matches_packed_cache() {
        crate::capabilities::configure_thread();
        let (tokens, heads, depth) = (6usize, 2usize, 8usize);
        let keys = Tensor::<bf16>::try_full(&[tokens, heads * depth], bf16::from_f32(0.1)).unwrap();
        let values = keys.try_clone().unwrap();
        let offsets = [0u32, tokens as u32]; // one ragged segment
        let cache = AttentionPackedMatrix::try_pack(&keys.view(), &values.view(), depth, &offsets, None).unwrap();

        // shape reads the C-written header; it must agree with the cache's own getters.
        let read = AttentionPackedMatrix::<bf16>::shape(cache.as_bytes()).unwrap();
        assert_eq!(read, (cache.heads(), cache.depth(), cache.segments()));
        assert_eq!(AttentionPackedMatrix::<bf16>::shape(&[]), None);
    }
}
