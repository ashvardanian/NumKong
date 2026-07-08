//! Ragged scaled-dot-product attention with a pre-packed KV-cache.
//!
//! The attention family operates on ragged batches: a directory of variable-length
//! segments shares one packed KV-cache blob, and every `(segment, head)` pair is an
//! independent task. Packing rearranges K and V into a backend-opaque layout (AMX
//! tiles on Sapphire Rapids, dtype-preserving planes on the AVX tiers), so the hot
//! kernel streams data in its native format.
//!
//! # Typical flow
//!
//! 1. Pack the K/V token matrices once per layer with [`AttentionKeyValueCache::try_pack`].
//! 2. Call [`AttentionKeyValueCache::try_attention`] with the query tokens and the
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
//! ```rust,ignore
//! use numkong::{bf16, AttentionKeyValueCache, Tensor};
//!
//! let tokens = 128;
//! let (kv_heads, depth) = (8, 128);
//! let keys = Tensor::<bf16>::try_full(&[tokens, kv_heads * depth], bf16::from_f32(0.1)).unwrap();
//! let values = keys.try_clone().unwrap();
//! let offsets = [0u32, tokens as u32];
//!
//! let kv = AttentionKeyValueCache::try_pack(&keys.view(), &values.view(), depth, &offsets, None).unwrap();
//! let outputs = kv.try_attention(&keys.view(), &offsets, None).unwrap();
//! ```

#[cfg(feature = "alloc")]
extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::tensor::{Allocator, Global, Tensor, TensorError, TensorMut, TensorRef, SIMD_ALIGNMENT};
use crate::types::Roots;
use crate::types::{bf16, e4m3, StorageElement};

// region: FFI

#[link(name = "numkong")]
extern "C" {
    fn nk_attention_packed_size_bf16(
        key_value_head_count: usize,
        depth: usize,
        segment_lengths: *const u32,
        segment_count: usize,
    ) -> usize;
    fn nk_attention_pack_bf16(
        keys: *const bf16,
        values: *const bf16,
        key_value_head_count: usize,
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
        key_value_head_count: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );

    fn nk_attention_packed_size_e4m3(
        key_value_head_count: usize,
        depth: usize,
        segment_lengths: *const u32,
        segment_count: usize,
    ) -> usize;
    fn nk_attention_pack_e4m3(
        keys: *const e4m3,
        values: *const e4m3,
        key_value_head_count: usize,
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
        key_value_head_count: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );

    fn nk_attention_packed_size_i8(
        key_value_head_count: usize,
        depth: usize,
        segment_lengths: *const u32,
        segment_count: usize,
    ) -> usize;
    fn nk_attention_pack_i8(
        keys: *const i8,
        values: *const i8,
        key_value_head_count: usize,
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
        key_value_head_count: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride_bytes: usize,
        output_stride_bytes: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );
}

// endregion: FFI

// region: Attention trait

/// Trait abstracting ragged-attention pack/compute operations per scalar type.
pub trait Attention: StorageElement + Clone {
    /// Returns the packed KV-cache size in bytes for the given segment geometry.
    fn attention_packed_size(
        key_value_head_count: usize,
        depth: usize,
        segment_lengths: &[u32],
        segment_count: usize,
    ) -> usize;

    /// Pack a window of the `(segment, kv_head)` task grid into the KV-cache blob.
    /// # Safety
    /// - `k` / `v` must point to token matrices with `key_stride_bytes` / `value_stride_bytes` byte rows
    ///   covering every token addressed by `segment_offsets` + `segment_lengths`
    /// - `key_value_packed` must have at least `attention_packed_size(..)` bytes
    /// - a window with `first_task > 0` requires the header already initialized by a
    ///   prior (or concurrent) window covering task 0
    #[allow(clippy::too_many_arguments)]
    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        key_value_head_count: usize,
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
        key_value_head_count: usize,
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
    fn attention_packed_size(
        key_value_head_count: usize,
        depth: usize,
        segment_lengths: &[u32],
        segment_count: usize,
    ) -> usize {
        unsafe { nk_attention_packed_size_bf16(key_value_head_count, depth, segment_lengths.as_ptr(), segment_count) }
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        key_value_head_count: usize,
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
                key_value_head_count,
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
        key_value_head_count: usize,
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
                key_value_head_count,
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
    fn attention_packed_size(
        key_value_head_count: usize,
        depth: usize,
        segment_lengths: &[u32],
        segment_count: usize,
    ) -> usize {
        unsafe { nk_attention_packed_size_e4m3(key_value_head_count, depth, segment_lengths.as_ptr(), segment_count) }
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        key_value_head_count: usize,
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
                key_value_head_count,
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
        key_value_head_count: usize,
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
                key_value_head_count,
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
    fn attention_packed_size(
        key_value_head_count: usize,
        depth: usize,
        segment_lengths: &[u32],
        segment_count: usize,
    ) -> usize {
        unsafe { nk_attention_packed_size_i8(key_value_head_count, depth, segment_lengths.as_ptr(), segment_count) }
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        key_value_head_count: usize,
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
                key_value_head_count,
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
        key_value_head_count: usize,
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
                key_value_head_count,
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

// region: AttentionKeyValueCache

/// Pre-packed ragged KV-cache for scaled-dot-product attention.
/// Owns the backend-opaque blob plus the geometry needed to validate query batches:
/// segment count, KV head count, head width, and total token count.
pub struct AttentionKeyValueCache<Scalar: Attention, Alloc: Allocator = Global> {
    data: NonNull<u8>,
    /// Bytes of live packed content.
    size: usize,
    /// Bytes actually allocated (>= size); lets `try_pack_into` reuse the buffer across steps.
    capacity: usize,
    key_value_head_count: usize,
    depth: usize,
    segment_count: usize,
    total_tokens: usize,
    alloc: Alloc,
    _marker: PhantomData<Scalar>,
}

// Safety: AttentionKeyValueCache owns its data and is just bytes
unsafe impl<Scalar: Attention + Send, Alloc: Allocator + Send> Send for AttentionKeyValueCache<Scalar, Alloc> {}
unsafe impl<Scalar: Attention + Sync, Alloc: Allocator + Sync> Sync for AttentionKeyValueCache<Scalar, Alloc> {}

impl<Scalar: Attention, Alloc: Allocator> Drop for AttentionKeyValueCache<Scalar, Alloc> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                let layout = core::alloc::Layout::from_size_align_unchecked(self.capacity, SIMD_ALIGNMENT);
                self.alloc.deallocate(self.data, layout);
            }
        }
    }
}

impl<Scalar: Attention, Alloc: Allocator + Clone> AttentionKeyValueCache<Scalar, Alloc> {
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
            key_value_head_count: self.key_value_head_count,
            depth: self.depth,
            segment_count: self.segment_count,
            total_tokens: self.total_tokens,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        })
    }
}

impl<Scalar: Attention, Alloc: Allocator + Clone> Clone for AttentionKeyValueCache<Scalar, Alloc> {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("AttentionKeyValueCache clone allocation failed")
    }
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

impl<Scalar: Attention, Alloc: Allocator> AttentionKeyValueCache<Scalar, Alloc> {
    /// Pack ragged K/V token matrices into a KV-cache blob using a custom allocator.
    /// `k` and `v` are `[tokens, kv_heads * depth]` views (rows may be strided
    /// interior slices of a fused QKV buffer). `segment_offsets` holds cumulative token
    /// offsets (`segments + 1` entries); `segment_lengths` defaults to the adjacent
    /// offset differences — the self-attention geometry.
    /// An empty cache owning no allocation; fill it with [`try_pack_into`](Self::try_pack_into).
    pub fn empty_in(alloc: Alloc) -> Self {
        Self {
            data: NonNull::dangling(),
            size: 0,
            capacity: 0,
            key_value_head_count: 0,
            depth: 0,
            segment_count: 0,
            total_tokens: 0,
            alloc,
            _marker: PhantomData,
        }
    }

    /// Pack `keys`/`values` into a freshly allocated cache (via the given allocator).
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
    /// packed size fits `capacity` and reallocating (via the stored allocator) only when it must
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
        let (k_tokens, key_value_head_count, key_stride_bytes) = validate_token_view(keys, depth)?;
        let (v_tokens, v_heads, value_stride_bytes) = validate_token_view(values, depth)?;
        if k_tokens != v_tokens || key_value_head_count != v_heads {
            return Err(TensorError::DimensionMismatch {
                expected: k_tokens,
                got: v_tokens,
            });
        }
        let segment_count = validate_offsets(segment_offsets, k_tokens)?;

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

        let size = Scalar::attention_packed_size(key_value_head_count, depth, segment_lengths, segment_count);
        if size > self.capacity {
            self.grow_to(size)?;
        }
        if size > 0 {
            unsafe {
                Scalar::attention_pack(
                    keys.as_ptr(),
                    values.as_ptr(),
                    key_value_head_count,
                    depth,
                    segment_offsets.as_ptr(),
                    segment_lengths.as_ptr(),
                    segment_count,
                    key_stride_bytes,
                    value_stride_bytes,
                    self.data.as_ptr(),
                    0,
                    0,
                );
            }
        }

        self.size = size;
        self.key_value_head_count = key_value_head_count;
        self.depth = depth;
        self.segment_count = segment_count;
        self.total_tokens = *segment_offsets.last().unwrap() as usize;
        Ok(())
    }

    /// Grow the backing allocation to at least `needed` bytes. Frees the old buffer without copying
    /// (the caller repacks, overwriting the contents).
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
    /// logical shape as `q` (`[tokens, heads * depth]`, contiguous rows).
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
        let (q_tokens, head_count, query_stride_bytes) = validate_token_view(queries, self.depth)?;
        if head_count % self.key_value_head_count != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: self.key_value_head_count,
                got: head_count,
            });
        }
        let segment_count = validate_offsets(query_offsets, q_tokens)?;
        if segment_count != self.segment_count {
            return Err(TensorError::DimensionMismatch {
                expected: self.segment_count,
                got: segment_count,
            });
        }
        let row_values = head_count * self.depth;
        if output.shape() != [q_tokens, row_values] {
            return Err(TensorError::DimensionMismatch {
                expected: q_tokens * row_values,
                got: output.shape().iter().product(),
            });
        }
        let scale = scale.unwrap_or_else(|| (self.depth as f32).rsqrt());
        unsafe {
            Scalar::attention_packed(
                queries.as_ptr(),
                self.data.as_ptr(),
                output.as_mut_ptr(),
                head_count,
                self.key_value_head_count,
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

    /// Returns the number of ragged segments in the packed cache.
    pub fn segments(&self) -> usize { self.segment_count }

    /// Returns the number of KV heads per token.
    pub fn kv_heads(&self) -> usize { self.key_value_head_count }

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
impl<Scalar: Attention> AttentionKeyValueCache<Scalar, Global> {
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
        let (q_tokens, head_count, _) = validate_token_view(queries, self.depth)?;
        let mut output = Tensor::<f32>::try_full(&[q_tokens, head_count * self.depth], 0.0)?;
        self.try_attention_into(queries, query_offsets, scale, &mut output)?;
        Ok(output)
    }
}

#[cfg(feature = "parallel")]
impl<Scalar: Attention, Alloc: Allocator> AttentionKeyValueCache<Scalar, Alloc> {
    /// Ragged attention parallelized over the `(segment, head)` task grid with a
    /// `fork_union` thread pool; each worker computes a contiguous task window.
    pub fn try_attention_parallel_into<QIn, OutTensor, const MAX_RANK: usize, const OUT_MAX_RANK: usize>(
        &self,
        queries: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
        output: &mut OutTensor,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutTensor: TensorMut<f32, OUT_MAX_RANK> + ?Sized,
    {
        let (q_tokens, head_count, query_stride_bytes) = validate_token_view(queries, self.depth)?;
        if head_count % self.key_value_head_count != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: self.key_value_head_count,
                got: head_count,
            });
        }
        let segment_count = validate_offsets(query_offsets, q_tokens)?;
        if segment_count != self.segment_count {
            return Err(TensorError::DimensionMismatch {
                expected: self.segment_count,
                got: segment_count,
            });
        }
        let row_values = head_count * self.depth;
        if output.shape() != [q_tokens, row_values] {
            return Err(TensorError::DimensionMismatch {
                expected: q_tokens * row_values,
                got: output.shape().iter().product(),
            });
        }
        let scale = scale.unwrap_or_else(|| (self.depth as f32).rsqrt());
        let output_stride_bytes = output.stride_bytes(0) as usize;

        let q_ptr = fork_union::SyncConstPtr::new(queries.as_ptr());
        let kv_ptr = fork_union::SyncConstPtr::new(self.data.as_ptr());
        let out_ptr = fork_union::SyncMutPtr::new(output.as_mut_ptr());
        let offsets_ptr = fork_union::SyncConstPtr::new(query_offsets.as_ptr());
        let (key_value_head_count, depth) = (self.key_value_head_count, self.depth);

        // Self-attention task cost is `q_len × kv_len × depth` — quadratic in segment
        // length, so equal-count windows can be ~64× unbalanced on ragged batches.
        // Dynamic per-task scheduling mirrors the Python layer's `schedule(dynamic, 1)`;
        // GQA-sibling heads stay adjacent in task order, preserving packed-KV reuse.
        let total_tasks = segment_count * head_count;
        pool.for_n_dynamic(total_tasks, move |prong| {
            // Configure the worker for AMX and other thread-local SIMD state (idempotent).
            crate::capabilities::configure_thread();
            unsafe {
                Scalar::attention_packed(
                    q_ptr.as_ptr(),
                    kv_ptr.as_ptr(),
                    out_ptr.as_ptr(),
                    head_count,
                    key_value_head_count,
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
}

// endregion: AttentionKeyValueCache
