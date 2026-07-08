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
//! 1. Pack the K/V token matrices once per layer with [`AttentionPackedKV::try_pack`].
//! 2. Call [`AttentionPackedKV::try_attention`] with the query tokens and the
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
//! use numkong::{bf16, AttentionPackedKV, Tensor};
//!
//! let tokens = 128;
//! let (kv_heads, head_dim) = (8, 128);
//! let keys = Tensor::<bf16>::try_full(&[tokens, kv_heads * head_dim], bf16::from_f32(0.1)).unwrap();
//! let values = keys.try_clone().unwrap();
//! let offsets = [0u32, tokens as u32];
//!
//! let kv = AttentionPackedKV::try_pack(&keys.view(), &values.view(), head_dim, &offsets, None).unwrap();
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
        num_kv_heads: usize,
        head_dim: usize,
        segment_lengths: *const u32,
        segment_count: usize,
    ) -> usize;
    fn nk_attention_pack_bf16(
        k: *const bf16,
        v: *const bf16,
        num_kv_heads: usize,
        head_dim: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        k_stride: usize,
        v_stride: usize,
        kv_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    );
    fn nk_attention_packed_bf16(
        queries: *const bf16,
        kv_packed: *const u8,
        output: *mut f32,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        query_offsets: *const u32,
        q_stride: usize,
        o_stride: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );

    fn nk_attention_packed_size_e4m3(
        num_kv_heads: usize,
        head_dim: usize,
        segment_lengths: *const u32,
        segment_count: usize,
    ) -> usize;
    fn nk_attention_pack_e4m3(
        k: *const e4m3,
        v: *const e4m3,
        num_kv_heads: usize,
        head_dim: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        k_stride: usize,
        v_stride: usize,
        kv_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    );
    fn nk_attention_packed_e4m3(
        queries: *const e4m3,
        kv_packed: *const u8,
        output: *mut f32,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        query_offsets: *const u32,
        q_stride: usize,
        o_stride: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );

    fn nk_attention_packed_size_i8(
        num_kv_heads: usize,
        head_dim: usize,
        segment_lengths: *const u32,
        segment_count: usize,
    ) -> usize;
    fn nk_attention_pack_i8(
        k: *const i8,
        v: *const i8,
        num_kv_heads: usize,
        head_dim: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        k_stride: usize,
        v_stride: usize,
        kv_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    );
    fn nk_attention_packed_i8(
        queries: *const i8,
        kv_packed: *const u8,
        output: *mut f32,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        query_offsets: *const u32,
        q_stride: usize,
        o_stride: usize,
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
        num_kv_heads: usize,
        head_dim: usize,
        segment_lengths: &[u32],
        segment_count: usize,
    ) -> usize;

    /// Pack a window of the `(segment, kv_head)` task grid into the KV-cache blob.
    /// # Safety
    /// - `k` / `v` must point to token matrices with `k_stride` / `v_stride` byte rows
    ///   covering every token addressed by `segment_offsets` + `segment_lengths`
    /// - `kv_packed` must have at least `attention_packed_size(..)` bytes
    /// - a window with `first_task > 0` requires the header already initialized by a
    ///   prior (or concurrent) window covering task 0
    #[allow(clippy::too_many_arguments)]
    unsafe fn attention_pack(
        k: *const Self,
        v: *const Self,
        num_kv_heads: usize,
        head_dim: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        k_stride: usize,
        v_stride: usize,
        kv_packed: *mut u8,
        first_task: usize,
        task_count: usize,
    );

    /// Compute a window of the `(segment, head)` attention task grid.
    /// # Safety
    /// - `kv_packed` must have been produced by `attention_pack` with matching geometry
    /// - `queries` rows addressed by `query_offsets` must be valid, `output` writable
    ///   with `o_stride` byte rows
    #[allow(clippy::too_many_arguments)]
    unsafe fn attention_packed(
        queries: *const Self,
        kv_packed: *const u8,
        output: *mut f32,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        query_offsets: *const u32,
        q_stride: usize,
        o_stride: usize,
        scale: f32,
        first_task: usize,
        task_count: usize,
    );
}

macro_rules! impl_attention {
    ($scalar:ty, $size_fn:ident, $pack_fn:ident, $packed_fn:ident) => {
        impl Attention for $scalar {
            fn attention_packed_size(
                num_kv_heads: usize,
                head_dim: usize,
                segment_lengths: &[u32],
                segment_count: usize,
            ) -> usize {
                unsafe { $size_fn(num_kv_heads, head_dim, segment_lengths.as_ptr(), segment_count) }
            }

            unsafe fn attention_pack(
                k: *const Self,
                v: *const Self,
                num_kv_heads: usize,
                head_dim: usize,
                segment_offsets: *const u32,
                segment_lengths: *const u32,
                segment_count: usize,
                k_stride: usize,
                v_stride: usize,
                kv_packed: *mut u8,
                first_task: usize,
                task_count: usize,
            ) {
                $pack_fn(
                    k,
                    v,
                    num_kv_heads,
                    head_dim,
                    segment_offsets,
                    segment_lengths,
                    segment_count,
                    k_stride,
                    v_stride,
                    kv_packed,
                    first_task,
                    task_count,
                )
            }

            unsafe fn attention_packed(
                queries: *const Self,
                kv_packed: *const u8,
                output: *mut f32,
                num_heads: usize,
                num_kv_heads: usize,
                head_dim: usize,
                query_offsets: *const u32,
                q_stride: usize,
                o_stride: usize,
                scale: f32,
                first_task: usize,
                task_count: usize,
            ) {
                $packed_fn(
                    queries,
                    kv_packed,
                    output,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    query_offsets,
                    q_stride,
                    o_stride,
                    scale,
                    first_task,
                    task_count,
                )
            }
        }
    };
}

impl_attention!(
    bf16,
    nk_attention_packed_size_bf16,
    nk_attention_pack_bf16,
    nk_attention_packed_bf16
);
impl_attention!(
    e4m3,
    nk_attention_packed_size_e4m3,
    nk_attention_pack_e4m3,
    nk_attention_packed_e4m3
);
impl_attention!(
    i8,
    nk_attention_packed_size_i8,
    nk_attention_pack_i8,
    nk_attention_packed_i8
);

// endregion: Attention trait

// region: AttentionPackedKV

/// Pre-packed ragged KV-cache for scaled-dot-product attention.
/// Owns the backend-opaque blob plus the geometry needed to validate query batches:
/// segment count, KV head count, head width, and total token count.
pub struct AttentionPackedKV<Scalar: Attention, Alloc: Allocator = Global> {
    data: NonNull<u8>,
    size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    segment_count: usize,
    total_tokens: usize,
    alloc: Alloc,
    _marker: PhantomData<Scalar>,
}

// Safety: AttentionPackedKV owns its data and is just bytes
unsafe impl<Scalar: Attention + Send, Alloc: Allocator + Send> Send for AttentionPackedKV<Scalar, Alloc> {}
unsafe impl<Scalar: Attention + Sync, Alloc: Allocator + Sync> Sync for AttentionPackedKV<Scalar, Alloc> {}

impl<Scalar: Attention, Alloc: Allocator> Drop for AttentionPackedKV<Scalar, Alloc> {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe {
                let layout = core::alloc::Layout::from_size_align_unchecked(self.size, SIMD_ALIGNMENT);
                self.alloc.deallocate(self.data, layout);
            }
        }
    }
}

impl<Scalar: Attention, Alloc: Allocator + Clone> AttentionPackedKV<Scalar, Alloc> {
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
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            segment_count: self.segment_count,
            total_tokens: self.total_tokens,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        })
    }
}

impl<Scalar: Attention, Alloc: Allocator + Clone> Clone for AttentionPackedKV<Scalar, Alloc> {
    fn clone(&self) -> Self { self.try_clone().expect("AttentionPackedKV clone allocation failed") }
}

/// Validates a `[tokens, heads * head_dim]` token-matrix view against a head width.
fn validate_token_view<Scalar, View, const MAX_RANK: usize>(
    view: &View,
    head_dim: usize,
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
    if head_dim == 0 || view.shape()[1] % head_dim != 0 {
        return Err(TensorError::DimensionMismatch {
            expected: head_dim,
            got: view.shape()[1],
        });
    }
    Ok((view.shape()[0], view.shape()[1] / head_dim, row_stride_bytes as usize))
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

impl<Scalar: Attention, Alloc: Allocator> AttentionPackedKV<Scalar, Alloc> {
    /// Pack ragged K/V token matrices into a KV-cache blob using a custom allocator.
    /// `k` and `v` are `[tokens, kv_heads * head_dim]` views (rows may be strided
    /// interior slices of a fused QKV buffer). `segment_offsets` holds cumulative token
    /// offsets (`segments + 1` entries); `segment_lengths` defaults to the adjacent
    /// offset differences — the self-attention geometry.
    pub fn try_pack_in<KIn, VIn, const MAX_RANK: usize>(
        k: &KIn,
        v: &VIn,
        head_dim: usize,
        segment_offsets: &[u32],
        segment_lengths: Option<&[u32]>,
        alloc: Alloc,
    ) -> Result<Self, TensorError>
    where
        KIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        VIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (k_tokens, num_kv_heads, k_stride) = validate_token_view(k, head_dim)?;
        let (v_tokens, v_heads, v_stride) = validate_token_view(v, head_dim)?;
        if k_tokens != v_tokens || num_kv_heads != v_heads {
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

        let size = Scalar::attention_packed_size(num_kv_heads, head_dim, segment_lengths, segment_count);
        let data = if size == 0 {
            NonNull::dangling()
        } else {
            let layout = core::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            alloc.allocate(layout).ok_or(TensorError::AllocationFailed)?
        };

        if size > 0 {
            unsafe {
                Scalar::attention_pack(
                    k.as_ptr(),
                    v.as_ptr(),
                    num_kv_heads,
                    head_dim,
                    segment_offsets.as_ptr(),
                    segment_lengths.as_ptr(),
                    segment_count,
                    k_stride,
                    v_stride,
                    data.as_ptr(),
                    0,
                    0,
                );
            }
        }

        Ok(Self {
            data,
            size,
            num_kv_heads,
            head_dim,
            segment_count,
            total_tokens: *segment_offsets.last().unwrap() as usize,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Ragged attention into a caller-provided `f32` output tensor of the same
    /// logical shape as `q` (`[tokens, heads * head_dim]`, contiguous rows).
    pub fn try_attention_into<QIn, OutTensor, const MAX_RANK: usize, const OUT_MAX_RANK: usize>(
        &self,
        q: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
        output: &mut OutTensor,
    ) -> Result<(), TensorError>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutTensor: TensorMut<f32, OUT_MAX_RANK> + ?Sized,
    {
        let (q_tokens, num_heads, q_stride) = validate_token_view(q, self.head_dim)?;
        if num_heads % self.num_kv_heads != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: self.num_kv_heads,
                got: num_heads,
            });
        }
        let segment_count = validate_offsets(query_offsets, q_tokens)?;
        if segment_count != self.segment_count {
            return Err(TensorError::DimensionMismatch {
                expected: self.segment_count,
                got: segment_count,
            });
        }
        let row_values = num_heads * self.head_dim;
        if output.shape() != [q_tokens, row_values] {
            return Err(TensorError::DimensionMismatch {
                expected: q_tokens * row_values,
                got: output.shape().iter().product(),
            });
        }
        let scale = scale.unwrap_or_else(|| (self.head_dim as f32).rsqrt());
        unsafe {
            Scalar::attention_packed(
                q.as_ptr(),
                self.data.as_ptr(),
                output.as_mut_ptr(),
                num_heads,
                self.num_kv_heads,
                self.head_dim,
                query_offsets.as_ptr(),
                q_stride,
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
    pub fn kv_heads(&self) -> usize { self.num_kv_heads }

    /// Returns the number of channels per head.
    pub fn head_dim(&self) -> usize { self.head_dim }

    /// Returns the total KV token count across segments.
    pub fn tokens(&self) -> usize { self.total_tokens }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { &self.alloc }

    /// Returns the packed data buffer.
    pub fn as_bytes(&self) -> &[u8] { unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.size) } }

    /// Returns a pointer to the packed data.
    pub fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
}

// Convenience methods using the Global allocator
impl<Scalar: Attention> AttentionPackedKV<Scalar, Global> {
    /// Pack ragged K/V token matrices using the global allocator.
    pub fn try_pack<KIn, VIn, const MAX_RANK: usize>(
        k: &KIn,
        v: &VIn,
        head_dim: usize,
        segment_offsets: &[u32],
        segment_lengths: Option<&[u32]>,
    ) -> Result<Self, TensorError>
    where
        KIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        VIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::try_pack_in(k, v, head_dim, segment_offsets, segment_lengths, Global)
    }

    /// Ragged attention allocating a fresh `f32` output tensor.
    pub fn try_attention<QIn, const MAX_RANK: usize>(
        &self,
        q: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
    ) -> Result<Tensor<f32>, TensorError>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (q_tokens, num_heads, _) = validate_token_view(q, self.head_dim)?;
        let mut output = Tensor::<f32>::try_full(&[q_tokens, num_heads * self.head_dim], 0.0)?;
        self.try_attention_into(q, query_offsets, scale, &mut output)?;
        Ok(output)
    }
}

#[cfg(feature = "parallel")]
impl<Scalar: Attention, Alloc: Allocator> AttentionPackedKV<Scalar, Alloc> {
    /// Ragged attention parallelized over the `(segment, head)` task grid with a
    /// `fork_union` thread pool; each worker computes a contiguous task window.
    pub fn try_attention_parallel_into<QIn, OutTensor, const MAX_RANK: usize, const OUT_MAX_RANK: usize>(
        &self,
        q: &QIn,
        query_offsets: &[u32],
        scale: Option<f32>,
        output: &mut OutTensor,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        QIn: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutTensor: TensorMut<f32, OUT_MAX_RANK> + ?Sized,
    {
        let (q_tokens, num_heads, q_stride) = validate_token_view(q, self.head_dim)?;
        if num_heads % self.num_kv_heads != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: self.num_kv_heads,
                got: num_heads,
            });
        }
        let segment_count = validate_offsets(query_offsets, q_tokens)?;
        if segment_count != self.segment_count {
            return Err(TensorError::DimensionMismatch {
                expected: self.segment_count,
                got: segment_count,
            });
        }
        let row_values = num_heads * self.head_dim;
        if output.shape() != [q_tokens, row_values] {
            return Err(TensorError::DimensionMismatch {
                expected: q_tokens * row_values,
                got: output.shape().iter().product(),
            });
        }
        let scale = scale.unwrap_or_else(|| (self.head_dim as f32).rsqrt());
        let o_stride = output.stride_bytes(0) as usize;

        let q_ptr = fork_union::SyncConstPtr::new(q.as_ptr());
        let kv_ptr = fork_union::SyncConstPtr::new(self.data.as_ptr());
        let out_ptr = fork_union::SyncMutPtr::new(output.as_mut_ptr());
        let offsets_ptr = fork_union::SyncConstPtr::new(query_offsets.as_ptr());
        let (num_kv_heads, head_dim) = (self.num_kv_heads, self.head_dim);

        // Self-attention task cost is `q_len × kv_len × head_dim` — quadratic in segment
        // length, so equal-count windows can be ~64× unbalanced on ragged batches.
        // Dynamic per-task scheduling mirrors the Python layer's `schedule(dynamic, 1)`;
        // GQA-sibling heads stay adjacent in task order, preserving packed-KV reuse.
        let total_tasks = segment_count * num_heads;
        pool.for_n_dynamic(total_tasks, move |prong| {
            // Configure the worker for AMX and other thread-local SIMD state (idempotent).
            crate::capabilities::configure_thread();
            unsafe {
                Scalar::attention_packed(
                    q_ptr.as_ptr(),
                    kv_ptr.as_ptr(),
                    out_ptr.as_ptr(),
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    offsets_ptr.as_ptr(),
                    q_stride,
                    o_stride,
                    scale,
                    prong.task_index,
                    1,
                );
            }
        }); // executes and synchronizes on drop

        Ok(())
    }
}

// endregion: AttentionPackedKV
