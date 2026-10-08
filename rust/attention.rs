//! Ragged scaled-dot-product attention with a pre-packed KV-cache.
//!
//! The attention family operates on ragged batches: a directory of variable-length segments shares
//! one packed KV-cache blob, and every `(query token, head)` pair is an independent task. Packing
//! rearranges K and V into a backend-opaque layout — AMX tiles on Sapphire Rapids, dtype-preserving
//! planes on the AVX capabilities — so the hot kernel streams data in its native format.
//!
//! # Typical flow
//!
//! 1. Pack the K/V token matrices once per layer with [`AttentionPackedMatrix::new`].
//! 2. Call [`AttentionPackedMatrix::attention`] with the query tokens, the
//!    cumulative `query_offsets` and the `keys_before, keys_after` band of visible keys;
//!    `arange` offsets turn the call into a batched single-query pool over the same packed cache.
//! 3. Pass `(usize::MAX, usize::MAX)` for bidirectional attention, `(usize::MAX, 0)` for causal
//!    and `(window - 1, 0)` for a sliding window of `window` keys.
//!
//! # Examples
//!
//! The doctest below is marked `ignore` because doctests compile as separate crates and would need
//! to re-link the `libnumkong` C library providing the `nk_attention_*` FFI symbols. The in-crate
//! tests at the bottom of this file exercise the same code path.
//!
//! ```rust,no_run
//! use numkong::{bf16, AttentionPackedMatrix, Tensor};
//!
//! let tokens = 128;
//! let (heads, depth) = (8, 128);
//! let keys = Tensor::<bf16>::full(&[tokens, heads * depth], bf16::from_f32(0.1)).unwrap();
//! let values = keys.clone().unwrap();
//! let offsets = [0u32, tokens as u32];
//!
//! let kv = AttentionPackedMatrix::new(&keys.view(), &values.view(), depth, &offsets).unwrap();
//! let outputs = kv.attention(&keys.view(), &offsets, None, usize::MAX, usize::MAX).unwrap();
//! ```
//!
//! File: rust/attention.rs
//! Author: Ash Vardanian

use core::{ffi::c_void, marker::PhantomData, ptr::null_mut};

#[cfg(feature = "parallel")]
use forkunion as fu;

use crate::{
    capabilities::{nk_capability_t, nk_size_t, nk_status_t, Capabilities, StatusCode},
    scalar::Roots,
    tensor::{Allocator, Error, Global, PackedBuffer, Tensor, TensorMut, TensorRef},
    types::{bf16, e4m3, f16, StorageElement},
    vector::Vector,
};

#[cfg(feature = "parallel")]
use crate::capabilities::WorkerStatus;

// region: FFI

#[link(name = "numkong")]
extern "C" {
    fn nk_attention_pack_size_bf16_best(
        heads: nk_size_t,
        depth: nk_size_t,
        token_count: nk_size_t,
        segment_count: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_attention_pack_bf16_best(
        keys: *const bf16,
        values: *const bf16,
        heads: nk_size_t,
        depth: nk_size_t,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: nk_size_t,
        key_stride: nk_size_t,
        value_stride: nk_size_t,
        key_value_packed: *mut u8,
        tasks_begin: nk_size_t,
        tasks_end: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_packed_bf16_best(
        queries: *const bf16,
        key_value_packed: *const u8,
        output: *mut f32,
        log_sum_exp: *mut f32,
        head_count: nk_size_t,
        key_value_head_count: nk_size_t,
        depth: nk_size_t,
        query_offsets: *const u32,
        query_stride: nk_size_t,
        output_stride: nk_size_t,
        scale: f32,
        keys_before: nk_size_t,
        keys_after: nk_size_t,
        tasks_begin: nk_size_t,
        tasks_end: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_packed_gradients_bf16_best(
        queries: *const bf16,
        key_value_packed: *const u8,
        output: *const f32,
        output_gradient: *const f32,
        log_sum_exp: *const f32,
        query_gradient: *mut f32,
        key_gradient: *mut f32,
        value_gradient: *mut f32,
        head_count: nk_size_t,
        key_value_head_count: nk_size_t,
        depth: nk_size_t,
        query_offsets: *const u32,
        key_offsets: *const u32,
        query_stride: nk_size_t,
        output_stride: nk_size_t,
        query_gradient_stride: nk_size_t,
        key_value_gradient_stride: nk_size_t,
        scale: f32,
        keys_before: nk_size_t,
        keys_after: nk_size_t,
        tasks_begin: nk_size_t,
        tasks_end: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_attention_pack_size_f16_best(
        heads: nk_size_t,
        depth: nk_size_t,
        token_count: nk_size_t,
        segment_count: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_attention_pack_f16_best(
        keys: *const f16,
        values: *const f16,
        heads: nk_size_t,
        depth: nk_size_t,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: nk_size_t,
        key_stride: nk_size_t,
        value_stride: nk_size_t,
        key_value_packed: *mut u8,
        tasks_begin: nk_size_t,
        tasks_end: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_packed_f16_best(
        queries: *const f16,
        key_value_packed: *const u8,
        output: *mut f32,
        log_sum_exp: *mut f32,
        head_count: nk_size_t,
        key_value_head_count: nk_size_t,
        depth: nk_size_t,
        query_offsets: *const u32,
        query_stride: nk_size_t,
        output_stride: nk_size_t,
        scale: f32,
        keys_before: nk_size_t,
        keys_after: nk_size_t,
        tasks_begin: nk_size_t,
        tasks_end: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_attention_pack_size_e4m3_best(
        heads: nk_size_t,
        depth: nk_size_t,
        token_count: nk_size_t,
        segment_count: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_attention_pack_e4m3_best(
        keys: *const e4m3,
        values: *const e4m3,
        heads: nk_size_t,
        depth: nk_size_t,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: nk_size_t,
        key_stride: nk_size_t,
        value_stride: nk_size_t,
        key_value_packed: *mut u8,
        tasks_begin: nk_size_t,
        tasks_end: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_packed_e4m3_best(
        queries: *const e4m3,
        key_value_packed: *const u8,
        output: *mut f32,
        log_sum_exp: *mut f32,
        head_count: nk_size_t,
        key_value_head_count: nk_size_t,
        depth: nk_size_t,
        query_offsets: *const u32,
        query_stride: nk_size_t,
        output_stride: nk_size_t,
        scale: f32,
        keys_before: nk_size_t,
        keys_after: nk_size_t,
        tasks_begin: nk_size_t,
        tasks_end: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_attention_pack_size_i8_best(
        heads: nk_size_t,
        depth: nk_size_t,
        token_count: nk_size_t,
        segment_count: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_attention_pack_i8_best(
        keys: *const i8,
        values: *const i8,
        heads: nk_size_t,
        depth: nk_size_t,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: nk_size_t,
        key_stride: nk_size_t,
        value_stride: nk_size_t,
        key_value_packed: *mut u8,
        tasks_begin: nk_size_t,
        tasks_end: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_packed_i8_best(
        queries: *const i8,
        key_value_packed: *const u8,
        output: *mut f32,
        log_sum_exp: *mut f32,
        head_count: nk_size_t,
        key_value_head_count: nk_size_t,
        depth: nk_size_t,
        query_offsets: *const u32,
        query_stride: nk_size_t,
        output_stride: nk_size_t,
        scale: f32,
        keys_before: nk_size_t,
        keys_after: nk_size_t,
        tasks_begin: nk_size_t,
        tasks_end: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_attention_packed_shape_bf16_best(
        packed: *const u8,
        heads: *mut nk_size_t,
        depth: *mut nk_size_t,
        segments: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_packed_shape_f16_best(
        packed: *const u8,
        heads: *mut nk_size_t,
        depth: *mut nk_size_t,
        segments: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_packed_shape_e4m3_best(
        packed: *const u8,
        heads: *mut nk_size_t,
        depth: *mut nk_size_t,
        segments: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_packed_shape_i8_best(
        packed: *const u8,
        heads: *mut nk_size_t,
        depth: *mut nk_size_t,
        segments: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_rope_f32_best(
        x: *const f32,
        cos: *const f32,
        sin: *const f32,
        y: *mut f32,
        rows: nk_size_t,
        head_count: nk_size_t,
        depth: nk_size_t,
        x_stride: nk_size_t,
        y_stride: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_rope_bf16_best(
        x: *const u16,
        cos: *const f32,
        sin: *const f32,
        y: *mut u16,
        rows: nk_size_t,
        head_count: nk_size_t,
        depth: nk_size_t,
        x_stride: nk_size_t,
        y_stride: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_attention_rope_e4m3_best(
        x: *const u8,
        cos: *const f32,
        sin: *const f32,
        y: *mut u8,
        rows: nk_size_t,
        head_count: nk_size_t,
        depth: nk_size_t,
        x_stride: nk_size_t,
        y_stride: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
}

// endregion: FFI

// region: Attention trait

/// Trait abstracting ragged-attention pack/compute operations per scalar type.
///
/// # Errors
///
/// [`Error::KernelFailed`] when the kernel refuses to run, like on a buffer packed under
/// other [`Capabilities`](crate::Capabilities) than the current ones.
pub trait Attention: StorageElement + Clone {
    /// Returns bytes enough to pack `token_count` keys in `segment_count` segments, however split.
    fn attention_pack_size(
        heads: usize,
        depth: usize,
        token_count: usize,
        segment_count: usize,
    ) -> Result<usize, Error>;

    /// Reads the KV-cache geometry — heads, depth, segments — from the packed header.
    /// # Safety
    /// `packed` must point to a buffer produced by `attention_pack`.
    unsafe fn attention_packed_shape(packed: *const u8) -> Result<(usize, usize, usize), Error>;

    /// Pack a window of the `(segment, kv_head)` task grid into the KV-cache blob.
    /// # Safety
    /// - `k` / `v` must point to token matrices with `key_stride` / `value_stride` byte
    ///   rows covering every token addressed by `segment_offsets` + `segment_lengths`
    /// - `key_value_packed` must have at least `attention_pack_size(..)` bytes
    ///
    /// Windows run in any order: the one starting at task 0 also writes the header and directory,
    /// which no window reads.
    #[allow(clippy::too_many_arguments)]
    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride: usize,
        value_stride: usize,
        key_value_packed: *mut u8,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error>;

    /// Compute attention over the half-open task window `tasks_begin..tasks_end` of the
    /// `(query token, head)` grid, task `token * head_count + head`, the end clipped to the grid.
    ///
    /// Each segment's queries align to the end of its keys: row `r` of a segment with `q` queries
    /// and `k` keys sits at position `p = r + k - q` and sees key `j` when
    /// `p - keys_before <= j <= p + keys_after`. `usize::MAX` leaves a side unbounded, so causal is
    /// `(usize::MAX, 0)`, a sliding window of `w` keys is `(w - 1, 0)` and bidirectional is
    /// `(usize::MAX, usize::MAX)`. Rows that see no key are written as zeros.
    /// # Safety
    /// - `key_value_packed` must have been produced by `attention_pack` with matching geometry
    /// - `queries` rows addressed by `query_offsets` must be valid, `output` writable
    ///   with `output_stride` byte rows
    /// - `log_sum_exp` is null or writable for one value per query token and head
    #[allow(clippy::too_many_arguments)]
    unsafe fn attention_packed(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *mut f32,
        log_sum_exp: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride: usize,
        output_stride: usize,
        scale: f32,
        keys_before: usize,
        keys_after: usize,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error>;
}

impl Attention for bf16 {
    fn attention_pack_size(
        heads: usize,
        depth: usize,
        token_count: usize,
        segment_count: usize,
    ) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe {
            nk_attention_pack_size_bf16_best(
                heads,
                depth,
                token_count,
                segment_count,
                Capabilities::CPUS.bits(),
                &mut bytes,
            )
        }
        .check()?;
        Ok(bytes)
    }

    unsafe fn attention_packed_shape(packed: *const u8) -> Result<(usize, usize, usize), Error> {
        let (mut heads, mut depth, mut segments) = (0usize, 0usize, 0usize);
        nk_attention_packed_shape_bf16_best(
            packed,
            &mut heads,
            &mut depth,
            &mut segments,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()?;
        Ok((heads, depth, segments))
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride: usize,
        value_stride: usize,
        key_value_packed: *mut u8,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error> {
        unsafe {
            nk_attention_pack_bf16_best(
                keys,
                values,
                heads,
                depth,
                segment_offsets,
                segment_lengths,
                segment_count,
                key_stride,
                value_stride,
                key_value_packed,
                tasks_begin,
                tasks_end,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()
        }
    }

    unsafe fn attention_packed(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *mut f32,
        log_sum_exp: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride: usize,
        output_stride: usize,
        scale: f32,
        keys_before: usize,
        keys_after: usize,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error> {
        unsafe {
            nk_attention_packed_bf16_best(
                queries,
                key_value_packed,
                output,
                log_sum_exp,
                head_count,
                heads,
                depth,
                query_offsets,
                query_stride,
                output_stride,
                scale,
                keys_before,
                keys_after,
                tasks_begin,
                tasks_end,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()
        }
    }
}

impl bf16 {
    /// Compute the gradients of [`attention_packed`](Attention::attention_packed) with respect to
    /// the queries, keys and values over the task window `tasks_begin..tasks_end` of the
    /// `(segment, kv_head)` grid, the end clipped to the grid.
    ///
    /// `output` and `log_sum_exp` are the forward's results. The query gradient holds one row of
    /// `head_count * depth` values per query token, spaced `query_gradient_stride` bytes apart. Key
    /// and value gradients hold one row of `heads * depth` values per key token at
    /// `key_offsets[segment] + token`, spaced `key_value_gradient_stride` bytes apart; rows past a
    /// segment's live keys are never written, and all others are overwritten.
    /// # Safety
    /// - the forward's safety contract holds for `queries`, `key_value_packed` and `query_offsets`
    /// - `key_offsets` must be the segment offsets the cache was packed with
    /// - every gradient buffer must be writable for its extent, and `output` and `output_gradient`
    ///   share `output_stride`
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_packed_gradients(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *const f32,
        output_gradient: *const f32,
        log_sum_exp: *const f32,
        query_gradient: *mut f32,
        key_gradient: *mut f32,
        value_gradient: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        key_offsets: *const u32,
        query_stride: usize,
        output_stride: usize,
        query_gradient_stride: usize,
        key_value_gradient_stride: usize,
        scale: f32,
        keys_before: usize,
        keys_after: usize,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error> {
        unsafe {
            nk_attention_packed_gradients_bf16_best(
                queries,
                key_value_packed,
                output,
                output_gradient,
                log_sum_exp,
                query_gradient,
                key_gradient,
                value_gradient,
                head_count,
                heads,
                depth,
                query_offsets,
                key_offsets,
                query_stride,
                output_stride,
                query_gradient_stride,
                key_value_gradient_stride,
                scale,
                keys_before,
                keys_after,
                tasks_begin,
                tasks_end,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()
        }
    }
}

impl Attention for f16 {
    fn attention_pack_size(
        heads: usize,
        depth: usize,
        token_count: usize,
        segment_count: usize,
    ) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe {
            nk_attention_pack_size_f16_best(
                heads,
                depth,
                token_count,
                segment_count,
                Capabilities::CPUS.bits(),
                &mut bytes,
            )
        }
        .check()?;
        Ok(bytes)
    }

    unsafe fn attention_packed_shape(packed: *const u8) -> Result<(usize, usize, usize), Error> {
        let (mut heads, mut depth, mut segments) = (0usize, 0usize, 0usize);
        nk_attention_packed_shape_f16_best(
            packed,
            &mut heads,
            &mut depth,
            &mut segments,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()?;
        Ok((heads, depth, segments))
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride: usize,
        value_stride: usize,
        key_value_packed: *mut u8,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error> {
        unsafe {
            nk_attention_pack_f16_best(
                keys,
                values,
                heads,
                depth,
                segment_offsets,
                segment_lengths,
                segment_count,
                key_stride,
                value_stride,
                key_value_packed,
                tasks_begin,
                tasks_end,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()
        }
    }

    unsafe fn attention_packed(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *mut f32,
        log_sum_exp: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride: usize,
        output_stride: usize,
        scale: f32,
        keys_before: usize,
        keys_after: usize,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error> {
        unsafe {
            nk_attention_packed_f16_best(
                queries,
                key_value_packed,
                output,
                log_sum_exp,
                head_count,
                heads,
                depth,
                query_offsets,
                query_stride,
                output_stride,
                scale,
                keys_before,
                keys_after,
                tasks_begin,
                tasks_end,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()
        }
    }
}

impl Attention for e4m3 {
    fn attention_pack_size(
        heads: usize,
        depth: usize,
        token_count: usize,
        segment_count: usize,
    ) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe {
            nk_attention_pack_size_e4m3_best(
                heads,
                depth,
                token_count,
                segment_count,
                Capabilities::CPUS.bits(),
                &mut bytes,
            )
        }
        .check()?;
        Ok(bytes)
    }

    unsafe fn attention_packed_shape(packed: *const u8) -> Result<(usize, usize, usize), Error> {
        let (mut heads, mut depth, mut segments) = (0usize, 0usize, 0usize);
        nk_attention_packed_shape_e4m3_best(
            packed,
            &mut heads,
            &mut depth,
            &mut segments,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()?;
        Ok((heads, depth, segments))
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride: usize,
        value_stride: usize,
        key_value_packed: *mut u8,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error> {
        unsafe {
            nk_attention_pack_e4m3_best(
                keys,
                values,
                heads,
                depth,
                segment_offsets,
                segment_lengths,
                segment_count,
                key_stride,
                value_stride,
                key_value_packed,
                tasks_begin,
                tasks_end,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()
        }
    }

    unsafe fn attention_packed(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *mut f32,
        log_sum_exp: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride: usize,
        output_stride: usize,
        scale: f32,
        keys_before: usize,
        keys_after: usize,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error> {
        unsafe {
            nk_attention_packed_e4m3_best(
                queries,
                key_value_packed,
                output,
                log_sum_exp,
                head_count,
                heads,
                depth,
                query_offsets,
                query_stride,
                output_stride,
                scale,
                keys_before,
                keys_after,
                tasks_begin,
                tasks_end,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()
        }
    }
}

impl Attention for i8 {
    fn attention_pack_size(
        heads: usize,
        depth: usize,
        token_count: usize,
        segment_count: usize,
    ) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe {
            nk_attention_pack_size_i8_best(
                heads,
                depth,
                token_count,
                segment_count,
                Capabilities::CPUS.bits(),
                &mut bytes,
            )
        }
        .check()?;
        Ok(bytes)
    }

    unsafe fn attention_packed_shape(packed: *const u8) -> Result<(usize, usize, usize), Error> {
        let (mut heads, mut depth, mut segments) = (0usize, 0usize, 0usize);
        nk_attention_packed_shape_i8_best(
            packed,
            &mut heads,
            &mut depth,
            &mut segments,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()?;
        Ok((heads, depth, segments))
    }

    unsafe fn attention_pack(
        keys: *const Self,
        values: *const Self,
        heads: usize,
        depth: usize,
        segment_offsets: *const u32,
        segment_lengths: *const u32,
        segment_count: usize,
        key_stride: usize,
        value_stride: usize,
        key_value_packed: *mut u8,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error> {
        unsafe {
            nk_attention_pack_i8_best(
                keys,
                values,
                heads,
                depth,
                segment_offsets,
                segment_lengths,
                segment_count,
                key_stride,
                value_stride,
                key_value_packed,
                tasks_begin,
                tasks_end,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()
        }
    }

    unsafe fn attention_packed(
        queries: *const Self,
        key_value_packed: *const u8,
        output: *mut f32,
        log_sum_exp: *mut f32,
        head_count: usize,
        heads: usize,
        depth: usize,
        query_offsets: *const u32,
        query_stride: usize,
        output_stride: usize,
        scale: f32,
        keys_before: usize,
        keys_after: usize,
        tasks_begin: usize,
        tasks_end: usize,
    ) -> Result<(), Error> {
        unsafe {
            nk_attention_packed_i8_best(
                queries,
                key_value_packed,
                output,
                log_sum_exp,
                head_count,
                heads,
                depth,
                query_offsets,
                query_stride,
                output_stride,
                scale,
                keys_before,
                keys_after,
                tasks_begin,
                tasks_end,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()
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
    buffer: PackedBuffer<Alloc>,
    /// Per-segment token counts, when the caller left them to be derived from the offsets.
    ///
    /// Held on this container's own allocator, and reused by every later pack, so a decode loop
    /// that repacks each step does not allocate again — the same contract [`PackedBuffer`] offers
    /// for the blob itself. Empty whenever the caller supplies the lengths directly.
    derived_lengths: Vector<u32, Alloc>,
    heads: usize,
    depth: usize,
    segment_count: usize,
    total_tokens: usize,
    _marker: PhantomData<Scalar>,
}

// Safety: AttentionPackedMatrix owns its data and is just bytes
unsafe impl<Scalar: Attention + Send, Alloc: Allocator + Send> Send for AttentionPackedMatrix<Scalar, Alloc> {}
unsafe impl<Scalar: Attention + Sync, Alloc: Allocator + Sync> Sync for AttentionPackedMatrix<Scalar, Alloc> {}

impl<Scalar: Attention, Alloc: Allocator + Clone> AttentionPackedMatrix<Scalar, Alloc> {
    /// Clone this packed KV-cache, returning an error on allocation failure.
    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Result<Self, Error> {
        Ok(Self {
            buffer: self.buffer.clone()?,
            derived_lengths: self.derived_lengths.clone()?,
            heads: self.heads,
            depth: self.depth,
            segment_count: self.segment_count,
            total_tokens: self.total_tokens,
            _marker: PhantomData,
        })
    }
}

/// Validates a __[tokens,heads×depth]__ token-matrix view against a head width.
fn validate_token_view<Scalar, View, const MAX_RANK: usize>(
    view: &View,
    depth: usize,
) -> Result<(usize, usize, usize), Error>
where
    Scalar: StorageElement,
    View: TensorRef<Scalar, MAX_RANK> + ?Sized,
{
    let &[rows, width] = view.shape() else {
        return Err(Error::DimensionMismatch {
            expected: 2,
            got: view.ndim(),
        });
    };
    if !view.has_contiguous_rows() {
        return Err(Error::NonContiguousRows);
    }
    let row_stride = view.stride_bytes(0);
    if row_stride < 0 {
        return Err(Error::InvalidShape {
            axis: 0,
            size: row_stride as usize,
            reason: "attention requires non-negative row strides",
        });
    }
    // A zero head count would survive to the query path and divide by zero there, so reject the
    // zero-width view that produces it here, where the geometry is established.
    if depth == 0 || width == 0 || width % depth != 0 {
        return Err(Error::DimensionMismatch {
            expected: depth,
            got: width,
        });
    }
    Ok((rows, width / depth, row_stride as usize))
}

/// Validates the `keys` and `values` token views together and returns their shared geometry as
/// `(tokens, heads, keys_stride, values_stride)`. Both views must be 2D
/// __[tokens,heads*depth]__ with contiguous rows and matching token and head counts.
fn validate_attention_views<Scalar, Keys, Values, const MAX_RANK: usize>(
    keys: &Keys,
    values: &Values,
    depth: usize,
) -> Result<(usize, usize, usize, usize), Error>
where
    Scalar: StorageElement,
    Keys: TensorRef<Scalar, MAX_RANK> + ?Sized,
    Values: TensorRef<Scalar, MAX_RANK> + ?Sized,
{
    let (keys_tokens, keys_heads, keys_stride) = validate_token_view(keys, depth)?;
    let (values_tokens, values_heads, values_stride) = validate_token_view(values, depth)?;
    if keys_tokens != values_tokens || keys_heads != values_heads {
        return Err(Error::DimensionMismatch {
            expected: keys_tokens,
            got: values_tokens,
        });
    }
    Ok((keys_tokens, keys_heads, keys_stride, values_stride))
}

/// Validates that `offsets` is cumulative and covers at most `tokens` rows.
fn validate_offsets(offsets: &[u32], tokens: usize) -> Result<usize, Error> {
    let [_, .., last] = offsets else {
        return Err(Error::DimensionMismatch {
            expected: 2,
            got: offsets.len(),
        });
    };
    for pair in offsets.windows(2) {
        if pair[1] < pair[0] {
            return Err(Error::InvalidShape {
                axis: 0,
                size: pair[1] as usize,
                reason: "offsets must be non-decreasing",
            });
        }
    }
    let last = *last as usize;
    if last > tokens {
        return Err(Error::DimensionMismatch {
            expected: tokens,
            got: last,
        });
    }
    Ok(offsets.len() - 1)
}

/// Everything a pack window needs once the geometry is validated and the buffer is sized.
///
/// `segment_lengths` is raw because it borrows the cache's own scratch while `destination` borrows
/// its buffer; handing both back as pointers is what lets one `&mut self` call produce both, and it
/// is the form the FFI consumes anyway.
struct PackPlan {
    heads: usize,
    keys_stride: usize,
    values_stride: usize,
    segment_lengths: *const u32,
    segment_count: usize,
    destination: *mut u8,
}

/// Everything a query window needs once the batch is validated against the packed geometry.
struct QueryPlan {
    query_head_count: usize,
    query_stride: usize,
    segment_count: usize,
    scale: f32,
}

/// Fill `storage` with the per-segment token counts implied by `segment_offsets`.
///
/// The offsets are cumulative, so each length is one first difference. `storage` grows through the
/// cache's own allocator and keeps its capacity, so repeated packs of the same geometry reuse it
/// without allocating.
fn derive_segment_lengths<Alloc: Allocator>(
    storage: &mut Vector<u32, Alloc>,
    segment_offsets: &[u32],
) -> Result<(), Error> {
    // `validate_offsets` has already established at least two offsets, so this cannot underflow.
    let segment_count = segment_offsets.len() - 1;
    storage.reserve(segment_count)?;
    storage.resize(segment_count)?;
    for (slot, pair) in storage.as_mut_slice().iter_mut().zip(segment_offsets.windows(2)) {
        *slot = pair[1] - pair[0];
    }
    Ok(())
}

impl<Scalar: Attention, Alloc: Allocator + Clone> AttentionPackedMatrix<Scalar, Alloc> {
    /// An empty cache that owns no allocation, holding only the given allocator.
    ///
    /// Nothing is packed yet: every geometry field reads zero and [`as_bytes`](Self::as_bytes) is
    /// empty until the first [`pack_into`](Self::pack_into), which allocates on demand and can then
    /// be re-run each decode step to reuse the buffer.
    pub fn empty_in(alloc: Alloc) -> Self {
        Self {
            buffer: PackedBuffer::empty_in(alloc.clone()),
            derived_lengths: Vector::empty_in(alloc),
            heads: 0,
            depth: 0,
            segment_count: 0,
            total_tokens: 0,
            _marker: PhantomData,
        }
    }

    /// Pack `keys`/`values` into a freshly allocated cache using the given allocator.
    pub fn new_in<KeysTensor, ValuesTensor, const MAX_RANK: usize>(
        keys: &KeysTensor,
        values: &ValuesTensor,
        depth: usize,
        segment_offsets: &[u32],
        alloc: Alloc,
    ) -> Result<Self, Error>
    where
        KeysTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        ValuesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let mut cache = Self::empty_in(alloc);
        cache.pack_into(keys, values, depth, segment_offsets)?;
        Ok(cache)
    }

    /// Repack `keys`/`values` into this cache's existing blob, reusing the allocation when the
    /// packed size fits `capacity` and reallocating through the stored allocator only when it must
    /// grow. The decode-loop path: allocate the cache once at layer-init, then refresh it every
    /// forward step with no further allocation. Packing overwrites, so a grow discards the old
    /// contents rather than copying them.
    ///
    /// Per-segment token counts are derived from `segment_offsets` into storage this cache owns on
    /// its own allocator, reused by every later pack, so the derivation costs no allocation after
    /// the first call.
    pub fn pack_into<KeysTensor, ValuesTensor, const MAX_RANK: usize>(
        &mut self,
        keys: &KeysTensor,
        values: &ValuesTensor,
        depth: usize,
        segment_offsets: &[u32],
    ) -> Result<(), Error>
    where
        KeysTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        ValuesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let Some(PackPlan {
            heads,
            keys_stride,
            values_stride,
            segment_lengths,
            segment_count,
            destination,
        }) = self.prepare_pack(keys, values, depth, segment_offsets)?
        else {
            return Ok(());
        };
        // The packer writes every byte it owns — the directory (header + offsets table incl. its
        // aligned tail) and every payload plane's padding are zero-filled by the kernel — so the
        // blob is a pure function of the inputs and needs no pre-zeroing here.
        unsafe {
            Scalar::attention_pack(
                keys.as_ptr(),
                values.as_ptr(),
                heads,
                depth,
                segment_offsets.as_ptr(),
                segment_lengths,
                segment_count,
                keys_stride,
                values_stride,
                destination,
                0,
                segment_count * heads,
            )
        }
    }

    /// Pre-grow the cache to hold the given ragged geometry, so a later `pack_into` that fits
    /// stays allocation-free with a stable pointer — allocate once at layer-init for the maximum
    /// sequence length, then refresh every decode step with no further allocation.
    /// `segment_lengths` carries one token count per segment.
    pub fn reserve(&mut self, heads: usize, depth: usize, segment_lengths: &[u32]) -> Result<(), Error> {
        let token_count = segment_lengths.iter().map(|&length| length as usize).sum();
        self.buffer.reserve(Scalar::attention_pack_size(
            heads,
            depth,
            token_count,
            segment_lengths.len(),
        )?)
    }

    /// Ragged attention into a caller-provided `f32` output tensor of the same logical shape as `q`
    /// — __[tokens,heads * depth]__, contiguous rows.
    ///
    /// Each segment's queries align to the end of its keys, and query row `r` of a segment with
    /// `q` queries and `k` keys sees key `j` when `r + k - q - keys_before <= j <= r + k - q +
    /// keys_after`. `usize::MAX` leaves a side unbounded: causal is `(usize::MAX, 0)`, a sliding
    /// window of `w` keys is `(w - 1, 0)` and bidirectional is `(usize::MAX, usize::MAX)`. Rows
    /// that see no key are written as zeros.
    pub fn attention_into<QueriesTensor, OutTensor, const MAX_RANK: usize, const OUT_MAX_RANK: usize>(
        &self,
        queries: &QueriesTensor,
        query_offsets: &[u32],
        scale: Option<f32>,
        keys_before: usize,
        keys_after: usize,
        output: &mut OutTensor,
    ) -> Result<(), Error>
    where
        QueriesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutTensor: TensorMut<f32, OUT_MAX_RANK> + ?Sized,
    {
        let QueryPlan {
            query_head_count,
            query_stride,
            segment_count,
            scale,
        } = self.prepare_query(queries, output, query_offsets, scale)?;
        unsafe {
            Scalar::attention_packed(
                queries.as_ptr(),
                self.buffer.as_ptr(),
                output.as_mut_ptr(),
                null_mut(),
                query_head_count,
                self.heads,
                self.depth,
                query_offsets.as_ptr(),
                query_stride,
                output.stride_bytes(0) as usize,
                scale,
                keys_before,
                keys_after,
                0,
                query_offsets[segment_count] as usize * query_head_count,
            )
        }
    }

    /// Bytes a packed cache needs for `token_count` keys in `segment_count` segments however they
    /// split, the size query mirroring [`Dots::dots_pack_size`](crate::Dots::dots_pack_size).
    /// Useful for pre-sizing an external buffer before packing, without constructing a cache.
    pub fn pack_size(heads: usize, depth: usize, token_count: usize, segment_count: usize) -> Result<usize, Error> {
        Scalar::attention_pack_size(heads, depth, token_count, segment_count)
    }

    /// Read the geometry of an externally-produced packed KV blob straight from its self-describing
    /// header, returning `(heads, depth, segments)`, or [`Error::InvalidShape`] if the slice is too
    /// short to hold one. Unlike the dots packed matrix, the attention blob records its own shape,
    /// so no caller-supplied dimensions are needed.
    ///
    /// Reads via the C `nk_attention_packed_shape_<dtype>` accessor for this cache's scalar type.
    pub fn peek_shape(packed: &[u8]) -> Result<(usize, usize, usize), Error> {
        if packed.len() < 12 {
            return Err(Error::InvalidShape {
                axis: 0,
                size: packed.len(),
                reason: "packed KV blob too short for a header",
            });
        }
        // Safety: the length check guarantees the header prefix the accessor reads is present.
        unsafe { Scalar::attention_packed_shape(packed.as_ptr()) }
    }

    /// Adopt an externally-produced packed KV blob by copying `bytes` into a container-owned
    /// allocation. The attention blob is self-describing, so `(heads, depth, segments)` are read
    /// back from its header; [`tokens`](Self::tokens) reports `0` for an adopted blob because the
    /// total token count is not part of the queryable header.
    ///
    /// # Safety
    /// `bytes` must be a valid attention packing for `Scalar`, produced by this build's packer;
    /// anything else makes a later `attention` read out of bounds.
    pub unsafe fn from_packed_bytes_in(bytes: &[u8], alloc: Alloc) -> Result<Self, Error> {
        let (heads, depth, segment_count) = Self::peek_shape(bytes)?;
        let mut cache = Self::empty_in(alloc);
        cache.buffer.fill_from_bytes(bytes)?;
        cache.heads = heads;
        cache.depth = depth;
        cache.segment_count = segment_count;
        Ok(cache)
    }

    /// Returns the shape `(heads, depth, segments)` of the packed cache.
    pub fn shape(&self) -> (usize, usize, usize) { (self.heads, self.depth, self.segment_count) }

    /// Returns the number of ragged segments in the packed cache.
    pub fn segments(&self) -> usize { self.segment_count }

    /// Returns the number of KV heads per token.
    pub fn heads(&self) -> usize { self.heads }

    /// Returns the number of channels per head.
    pub fn depth(&self) -> usize { self.depth }

    /// Returns the total KV token count across segments.
    pub fn tokens(&self) -> usize { self.total_tokens }

    /// Bytes currently allocated for the packed blob (>= the live packed size).
    pub fn capacity(&self) -> usize { self.buffer.capacity() }

    /// Reset to logically empty, keeping the allocation so the next `pack_into` reuses it.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.segment_count = 0;
        self.total_tokens = 0;
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { self.buffer.allocator() }

    /// Returns the packed data buffer.
    pub fn as_bytes(&self) -> &[u8] { self.buffer.as_bytes() }

    /// Returns a pointer to the packed data.
    pub fn as_ptr(&self) -> *const u8 { self.buffer.as_ptr() }
}

// Convenience methods using the Global allocator
impl<Scalar: Attention> AttentionPackedMatrix<Scalar, Global> {
    /// Pack ragged K/V token matrices using the global allocator.
    pub fn new<KeysTensor, ValuesTensor, const MAX_RANK: usize>(
        keys: &KeysTensor,
        values: &ValuesTensor,
        depth: usize,
        segment_offsets: &[u32],
    ) -> Result<Self, Error>
    where
        KeysTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        ValuesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        Self::new_in(keys, values, depth, segment_offsets, Global)
    }
}

impl<Scalar: Attention, Alloc: Allocator> AttentionPackedMatrix<Scalar, Alloc> {
    /// Ragged attention allocating a fresh `f32` output tensor through a clone of this cache's
    /// allocator. The allocating twin of [`attention_into`](Self::attention_into).
    pub fn attention<QueriesTensor, const MAX_RANK: usize>(
        &self,
        queries: &QueriesTensor,
        query_offsets: &[u32],
        scale: Option<f32>,
        keys_before: usize,
        keys_after: usize,
    ) -> Result<Tensor<f32, Alloc>, Error>
    where
        QueriesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        Alloc: Clone,
    {
        let (query_tokens, query_head_count, _) = validate_token_view(queries, self.depth)?;
        let shape = [query_tokens, query_head_count * self.depth];
        let mut output = Tensor::full_in(&shape, 0.0, self.buffer.allocator().clone())?;
        self.attention_into(queries, query_offsets, scale, keys_before, keys_after, &mut output)?;
        Ok(output)
    }

    /// Validate the token views and offsets, size the buffer, and record the packed geometry.
    ///
    /// Shared by the serial and parallel packs, which differ only in how they fan the task grid out
    /// afterwards. Returns `None` when the geometry packs to nothing, so the caller returns early
    /// rather than handing a zero-length blob to the kernel.
    fn prepare_pack<KeysTensor, ValuesTensor, const MAX_RANK: usize>(
        &mut self,
        keys: &KeysTensor,
        values: &ValuesTensor,
        depth: usize,
        segment_offsets: &[u32],
    ) -> Result<Option<PackPlan>, Error>
    where
        KeysTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        ValuesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let (tokens, heads, keys_stride, values_stride) = validate_attention_views(keys, values, depth)?;
        let segment_count = validate_offsets(segment_offsets, tokens)?;
        derive_segment_lengths(&mut self.derived_lengths, segment_offsets)?;

        let token_count = (segment_offsets[segment_count] - segment_offsets[0]) as usize;
        let size = Scalar::attention_pack_size(heads, depth, token_count, segment_count)?;
        let destination = self.buffer.reset_for_pack(size)?;
        self.heads = heads;
        self.depth = depth;
        self.segment_count = segment_count;
        self.total_tokens = segment_offsets[segment_count] as usize;
        if size == 0 {
            return Ok(None);
        }
        Ok(Some(PackPlan {
            heads,
            keys_stride,
            values_stride,
            segment_lengths: self.derived_lengths.as_ptr(),
            segment_count,
            destination,
        }))
    }

    /// Validate a query batch against the packed geometry and resolve the softmax scale.
    ///
    /// Shared by the serial and parallel query paths, which differ only in how they fan the
    /// `(query token, head)` grid out afterwards.
    fn prepare_query<QueriesTensor, OutTensor, const MAX_RANK: usize, const OUT_MAX_RANK: usize>(
        &self,
        queries: &QueriesTensor,
        output: &OutTensor,
        query_offsets: &[u32],
        scale: Option<f32>,
    ) -> Result<QueryPlan, Error>
    where
        QueriesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutTensor: TensorMut<f32, OUT_MAX_RANK> + ?Sized,
    {
        let (query_tokens, query_head_count, query_stride) = validate_token_view(queries, self.depth)?;
        if query_head_count % self.heads != 0 {
            return Err(Error::DimensionMismatch {
                expected: self.heads,
                got: query_head_count,
            });
        }
        let segment_count = validate_offsets(query_offsets, query_tokens)?;
        if segment_count != self.segment_count {
            return Err(Error::DimensionMismatch {
                expected: self.segment_count,
                got: segment_count,
            });
        }
        let row_values = query_head_count * self.depth;
        if output.shape() != [query_tokens, row_values] {
            return Err(Error::DimensionMismatch {
                expected: query_tokens * row_values,
                got: output.shape().iter().product(),
            });
        }
        Ok(QueryPlan {
            query_head_count,
            query_stride,
            segment_count,
            scale: scale.unwrap_or_else(|| (self.depth as f32).rsqrt()),
        })
    }
}

/// Most windows one parallel attention call cuts its `(query token, head)` grid into.
#[cfg(feature = "parallel")]
const ATTENTION_WINDOWS_LIMIT: usize = 1024;

/// The cost of one head of query `row` of a segment of `queries` queries over `keys` keys: the keys
/// the band shows it, plus one for writing the row.
#[cfg(feature = "parallel")]
fn attention_row_cost(row: usize, queries: usize, keys: usize, keys_before: usize, keys_after: usize) -> u64 {
    let position = keys as i64 - queries as i64 + row as i64;
    let (begin, end) = if position < 0 {
        let lag = position.unsigned_abs() as usize;
        let end = keys_after
            .checked_sub(lag)
            .map_or(0, |reach| reach.saturating_add(1).min(keys));
        (0, end)
    } else {
        let lead = position as usize;
        let end = lead.saturating_add(keys_after).saturating_add(1).min(keys);
        (lead.saturating_sub(keys_before).min(keys), end)
    };
    (end.saturating_sub(begin) + 1) as u64
}

/// Cuts the `(query token, head)` grid into `bounds.len() - 1` windows of about equal cost, window
/// `w` spanning tasks `bounds[w]..bounds[w + 1]`, cutting inside a token where a share ends, so a
/// single decode token still spreads over its heads.
#[cfg(feature = "parallel")]
fn attention_windows(
    query_offsets: &[u32],
    segment_lengths: &[u32],
    head_count: usize,
    keys_before: usize,
    keys_after: usize,
    bounds: &mut [usize],
) {
    let window_count = bounds.len() - 1;
    let heads = head_count as u64;
    let segments = || query_offsets.windows(2).zip(segment_lengths);
    let mut total = 0u64;
    for (pair, &keys) in segments() {
        let queries = (pair[1] - pair[0]) as usize;
        for row in 0..queries {
            total += attention_row_cost(row, queries, keys as usize, keys_before, keys_after) * heads;
        }
    }
    let (mut done, mut window) = (0u64, 1usize);
    bounds[0] = query_offsets[0] as usize * head_count;
    for (pair, &keys) in segments() {
        let queries = (pair[1] - pair[0]) as usize;
        for row in 0..queries {
            let cost = attention_row_cost(row, queries, keys as usize, keys_before, keys_after);
            while window < window_count {
                let target = total * window as u64 / window_count as u64;
                if target > done + cost * heads {
                    break;
                }
                let heads_before = if target > done {
                    (target - done).div_ceil(cost)
                } else {
                    0
                };
                bounds[window] = (pair[0] as usize + row) * head_count + heads_before as usize;
                window += 1;
            }
            done += cost * heads;
        }
    }
    let grid_end = query_offsets[query_offsets.len() - 1] as usize * head_count;
    bounds[window..].fill(grid_end);
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar: Attention, Alloc: Allocator> AttentionPackedMatrix<Scalar, Alloc> {
    /// Key counts of the packed segments, which every pack's directory keeps past its 64-byte
    /// header and `segments + 1` payload offsets.
    fn packed_lengths(&self) -> &[u32] {
        let offset = 64 + (self.segment_count + 1) * core::mem::size_of::<u64>();
        unsafe { core::slice::from_raw_parts(self.buffer.as_ptr().add(offset) as *const u32, self.segment_count) }
    }

    /// Ragged attention parallelized over the `(query token, head)` task grid with a ForkUnion
    /// thread pool, cut into a few windows of equal cost per thread, a row costing the keys it
    /// sees. The band of visible keys is the one that [`attention_into`](Self::attention_into)
    /// takes.
    pub fn attention_parallel_into<QueriesTensor, OutTensor, const MAX_RANK: usize, const OUT_MAX_RANK: usize>(
        &self,
        queries: &QueriesTensor,
        query_offsets: &[u32],
        scale: Option<f32>,
        keys_before: usize,
        keys_after: usize,
        output: &mut OutTensor,
        pool: &mut fu::ThreadPool,
    ) -> Result<(), Error>
    where
        QueriesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutTensor: TensorMut<f32, OUT_MAX_RANK> + ?Sized,
    {
        let QueryPlan {
            query_head_count,
            query_stride,
            segment_count,
            scale,
        } = self.prepare_query(queries, output, query_offsets, scale)?;
        let output_stride = output.stride_bytes(0) as usize;

        let q_ptr = fu::SyncConstPtr::new(queries.as_ptr());
        let kv_ptr = fu::SyncConstPtr::new(self.buffer.as_ptr());
        let out_ptr = fu::SyncMutPtr::new(output.as_mut_ptr());
        let offsets_ptr = fu::SyncConstPtr::new(query_offsets.as_ptr());
        let (heads, depth) = (self.heads, self.depth);

        let task_count = (query_offsets[segment_count] - query_offsets[0]) as usize * query_head_count;
        let threads = pool.threads_count();
        let window_count = if threads > 1 { 4 * threads } else { 1 }
            .min(ATTENTION_WINDOWS_LIMIT)
            .min(task_count);
        let mut window_bounds = [0usize; ATTENTION_WINDOWS_LIMIT + 1];
        attention_windows(
            query_offsets,
            self.packed_lengths(),
            query_head_count,
            keys_before,
            keys_after,
            &mut window_bounds[..=window_count],
        );
        let window_bounds = &window_bounds;
        let failure = WorkerStatus::default();
        let failure = &failure;
        pool.for_n_dynamic(window_count, move |prong| {
            // Configure the worker for AMX and other thread-local SIMD state — idempotent.
            if let Err(error) = crate::capabilities::configure_cpu_thread() {
                failure.record(Err(error));
                return;
            }
            unsafe {
                failure.record(Scalar::attention_packed(
                    q_ptr.as_ptr(),
                    kv_ptr.as_ptr(),
                    out_ptr.as_ptr(),
                    null_mut(),
                    query_head_count,
                    heads,
                    depth,
                    offsets_ptr.as_ptr(),
                    query_stride,
                    output_stride,
                    scale,
                    keys_before,
                    keys_after,
                    window_bounds[prong.task_index],
                    window_bounds[prong.task_index + 1],
                ));
            }
        }); // executes and synchronizes on drop
        failure.check()
    }

    /// Ragged attention parallelized over the task grid, allocating a fresh `f32` output tensor
    /// of shape [tokens, heads × depth]. The allocating twin of
    /// [`attention_parallel_into`](Self::attention_parallel_into).
    pub fn attention_parallel<QueriesTensor, const MAX_RANK: usize>(
        &self,
        queries: &QueriesTensor,
        query_offsets: &[u32],
        scale: Option<f32>,
        keys_before: usize,
        keys_after: usize,
        pool: &mut fu::ThreadPool,
    ) -> Result<Tensor<f32, Alloc>, Error>
    where
        QueriesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        Alloc: Clone,
    {
        let (query_tokens, query_head_count, _) = validate_token_view(queries, self.depth)?;
        let shape = [query_tokens, query_head_count * self.depth];
        let mut output = Tensor::full_in(&shape, 0.0, self.buffer.allocator().clone())?;
        self.attention_parallel_into(
            queries,
            query_offsets,
            scale,
            keys_before,
            keys_after,
            &mut output,
            pool,
        )?;
        Ok(output)
    }

    /// Pack ragged K/V token matrices into this cache in parallel over the `(segment, kv_head)`
    /// task grid with a ForkUnion thread pool. Sizing and any allocation or reallocation run
    /// serially up front; only the per-task packing fans out. Like [`pack_into`](Self::pack_into),
    /// packing overwrites, so a grow discards the old contents rather than copying them.
    pub fn pack_parallel_into<KeysTensor, ValuesTensor, const MAX_RANK: usize>(
        &mut self,
        keys: &KeysTensor,
        values: &ValuesTensor,
        depth: usize,
        segment_offsets: &[u32],
        pool: &mut fu::ThreadPool,
    ) -> Result<(), Error>
    where
        KeysTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        ValuesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let Some(PackPlan {
            heads,
            keys_stride,
            values_stride,
            segment_lengths,
            segment_count,
            destination,
        }) = self.prepare_pack(keys, values, depth, segment_offsets)?
        else {
            return Ok(());
        };

        let keys_ptr = fu::SyncConstPtr::new(keys.as_ptr());
        let values_ptr = fu::SyncConstPtr::new(values.as_ptr());
        let offsets_ptr = fu::SyncConstPtr::new(segment_offsets.as_ptr());
        let lengths_ptr = fu::SyncConstPtr::new(segment_lengths);
        let packed_ptr = fu::SyncMutPtr::new(destination);
        let failure = WorkerStatus::default();
        let failure = &failure;
        pool.for_n_dynamic(segment_count * heads, move |prong| {
            if let Err(error) = crate::capabilities::configure_cpu_thread() {
                failure.record(Err(error));
                return;
            }
            unsafe {
                failure.record(Scalar::attention_pack(
                    keys_ptr.as_ptr(),
                    values_ptr.as_ptr(),
                    heads,
                    depth,
                    offsets_ptr.as_ptr(),
                    lengths_ptr.as_ptr(),
                    segment_count,
                    keys_stride,
                    values_stride,
                    packed_ptr.as_ptr(),
                    prong.task_index,
                    prong.task_index + 1,
                ));
            }
        }); // executes and synchronizes on drop
        failure.check()
    }
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
impl<Scalar: Attention> AttentionPackedMatrix<Scalar, Global> {
    /// Pack ragged K/V token matrices in parallel using the global allocator. The allocating
    /// twin of [`pack_parallel_into`](Self::pack_parallel_into).
    pub fn new_parallel<KeysTensor, ValuesTensor, const MAX_RANK: usize>(
        keys: &KeysTensor,
        values: &ValuesTensor,
        depth: usize,
        segment_offsets: &[u32],
        pool: &mut fu::ThreadPool,
    ) -> Result<Self, Error>
    where
        KeysTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        ValuesTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
    {
        let mut cache = Self::empty_in(Global);
        cache.pack_parallel_into(keys, values, depth, segment_offsets, pool)?;
        Ok(cache)
    }
}

// endregion: AttentionPackedMatrix

// region: RoPE

/// In-place NeoX split-half RoPE over a row-major __[rows,head_count × depth]__ tensor.
pub trait AttentionRope: Sized + StorageElement {
    /// Rotates a 2D __[rows,head_count × depth]__ tensor in place using the __[rows,depth/2]__
    /// `cos`/`sin` angle grids, shared across heads, for an even `depth` of channels per head.
    ///
    /// The row stride is read from the tensor, so `x` may be a non-contiguous sub-span, for example
    /// the Q or K column-section of a fused QKV buffer. Returns `Err` on a shape mismatch.
    fn attention_rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[f32],
        sin: &[f32],
        head_count: usize,
        depth: usize,
    ) -> Result<(), Error>
    where
        XMut: TensorMut<Self, RX> + ?Sized;
}

impl AttentionRope for f32 {
    fn attention_rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[f32],
        sin: &[f32],
        head_count: usize,
        depth: usize,
    ) -> Result<(), Error>
    where
        XMut: TensorMut<Self, RX> + ?Sized,
    {
        if x.ndim() != 2 {
            return Err(Error::DimensionMismatch {
                expected: 2,
                got: x.ndim(),
            });
        }
        if depth % 2 != 0 {
            return Err(Error::InvalidShape {
                axis: 1,
                size: depth,
                reason: "RoPE depth must be even",
            });
        }
        let &[rows, width] = x.shape() else {
            return Err(Error::DimensionMismatch {
                expected: 2,
                got: x.ndim(),
            });
        };
        if width < head_count * depth {
            return Err(Error::ShapeMismatch {
                axis: 1,
                expected: head_count * depth,
                got: width,
            });
        }
        if cos.len() < rows * depth / 2 || sin.len() < rows * depth / 2 {
            return Err(Error::ShapeMismatch {
                axis: 0,
                expected: rows * depth / 2,
                got: cos.len().min(sin.len()),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let stride = x.stride_bytes(0) as usize;
        let yp = x.as_mut_ptr();
        unsafe {
            nk_attention_rope_f32_best(
                yp,
                cos.as_ptr(),
                sin.as_ptr(),
                yp,
                rows,
                head_count,
                depth,
                stride,
                stride,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()?;
        }
        Ok(())
    }
}

impl AttentionRope for bf16 {
    fn attention_rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[f32],
        sin: &[f32],
        head_count: usize,
        depth: usize,
    ) -> Result<(), Error>
    where
        XMut: TensorMut<Self, RX> + ?Sized,
    {
        if x.ndim() != 2 {
            return Err(Error::DimensionMismatch {
                expected: 2,
                got: x.ndim(),
            });
        }
        if depth % 2 != 0 {
            return Err(Error::InvalidShape {
                axis: 1,
                size: depth,
                reason: "RoPE depth must be even",
            });
        }
        let &[rows, width] = x.shape() else {
            return Err(Error::DimensionMismatch {
                expected: 2,
                got: x.ndim(),
            });
        };
        if width < head_count * depth {
            return Err(Error::ShapeMismatch {
                axis: 1,
                expected: head_count * depth,
                got: width,
            });
        }
        if cos.len() < rows * depth / 2 || sin.len() < rows * depth / 2 {
            return Err(Error::ShapeMismatch {
                axis: 0,
                expected: rows * depth / 2,
                got: cos.len().min(sin.len()),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let stride = x.stride_bytes(0) as usize;
        let yp = x.as_mut_ptr() as *mut u16;
        unsafe {
            nk_attention_rope_bf16_best(
                yp,
                cos.as_ptr(),
                sin.as_ptr(),
                yp,
                rows,
                head_count,
                depth,
                stride,
                stride,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()?;
        }
        Ok(())
    }
}

impl AttentionRope for e4m3 {
    fn attention_rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[f32],
        sin: &[f32],
        head_count: usize,
        depth: usize,
    ) -> Result<(), Error>
    where
        XMut: TensorMut<Self, RX> + ?Sized,
    {
        if x.ndim() != 2 {
            return Err(Error::DimensionMismatch {
                expected: 2,
                got: x.ndim(),
            });
        }
        if depth % 2 != 0 {
            return Err(Error::InvalidShape {
                axis: 1,
                size: depth,
                reason: "RoPE depth must be even",
            });
        }
        let &[rows, width] = x.shape() else {
            return Err(Error::DimensionMismatch {
                expected: 2,
                got: x.ndim(),
            });
        };
        if width < head_count * depth {
            return Err(Error::ShapeMismatch {
                axis: 1,
                expected: head_count * depth,
                got: width,
            });
        }
        if cos.len() < rows * depth / 2 || sin.len() < rows * depth / 2 {
            return Err(Error::ShapeMismatch {
                axis: 0,
                expected: rows * depth / 2,
                got: cos.len().min(sin.len()),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let stride = x.stride_bytes(0) as usize;
        let yp = x.as_mut_ptr() as *mut u8;
        unsafe {
            nk_attention_rope_e4m3_best(
                yp,
                cos.as_ptr(),
                sin.as_ptr(),
                yp,
                rows,
                head_count,
                depth,
                stride,
                stride,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
            .check()?;
        }
        Ok(())
    }
}

// endregion: RoPE

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        tensor::SIMD_ALIGNMENT,
        types::{assert_close, FloatLike, TestableType},
    };

    #[test]
    fn shape_matches_packed_cache() {
        crate::capabilities::configure_cpu_thread().unwrap();
        let (tokens, heads, depth) = (6usize, 2usize, 8usize);
        let keys = Tensor::<bf16>::full(&[tokens, heads * depth], bf16::from_f32(0.1)).unwrap();
        let values = keys.clone().unwrap();
        let offsets = [0u32, tokens as u32]; // one ragged segment
        let cache = AttentionPackedMatrix::new(&keys.view(), &values.view(), depth, &offsets).unwrap();

        // peek_shape reads the C-written header; it must agree with the cache's own getters and
        // its instance `shape()`.
        let read = AttentionPackedMatrix::<bf16>::peek_shape(cache.as_bytes()).unwrap();
        assert_eq!(read, (cache.heads(), cache.depth(), cache.segments()));
        assert_eq!(cache.shape(), read);
        assert!(matches!(
            AttentionPackedMatrix::<bf16>::peek_shape(&[]),
            Err(Error::InvalidShape { size: 0, .. })
        ));
    }

    #[test]
    fn reserve_then_pack_into_is_allocation_free() {
        crate::capabilities::configure_cpu_thread().unwrap();
        let (heads, depth) = (2usize, 8usize);
        let max_lengths = [64u32];
        let mut cache = AttentionPackedMatrix::<bf16>::empty_in(Global);
        cache.reserve(heads, depth, &max_lengths).unwrap();
        let reserved_capacity = cache.capacity();
        let reserved_ptr = cache.as_ptr();
        assert!(reserved_capacity >= AttentionPackedMatrix::<bf16>::pack_size(heads, depth, &max_lengths).unwrap());

        for tokens in [8usize, 33, 64] {
            let keys = Tensor::<bf16>::full(&[tokens, heads * depth], bf16::from_f32(0.1)).unwrap();
            let values = keys.clone().unwrap();
            let offsets = [0u32, tokens as u32];
            cache.pack_into(&keys.view(), &values.view(), depth, &offsets).unwrap();
            assert_eq!(cache.shape(), (heads, depth, 1));
            assert_eq!(cache.capacity(), reserved_capacity, "reserved capacity must not change");
            assert_eq!(cache.as_ptr(), reserved_ptr, "reserved pointer must stay stable");
        }
    }

    #[test]
    fn from_packed_bytes_roundtrips() {
        crate::capabilities::configure_cpu_thread().unwrap();
        let (tokens, heads, depth) = (6usize, 2usize, 8usize);
        let keys = Tensor::<bf16>::full(&[tokens, heads * depth], bf16::from_f32(0.1)).unwrap();
        let values = keys.clone().unwrap();
        let offsets = [0u32, tokens as u32];
        let cache = AttentionPackedMatrix::new(&keys.view(), &values.view(), depth, &offsets).unwrap();

        let adopted = unsafe { AttentionPackedMatrix::<bf16>::from_packed_bytes_in(cache.as_bytes(), Global) }.unwrap();
        assert_eq!(adopted.shape(), cache.shape());
        assert_eq!(adopted.as_bytes(), cache.as_bytes());
    }

    #[test]
    fn bands_select_visible_keys() {
        crate::capabilities::configure_cpu_thread().unwrap();
        let (tokens, depth) = (8usize, 8usize);
        let keys = Tensor::<bf16>::full(&[tokens, depth], bf16::from_f32(0.1)).unwrap();
        let rows: Vec<bf16> = (0..tokens)
            .flat_map(|token| core::iter::repeat(bf16::from_f32(token as f32)).take(depth))
            .collect();
        let values = Tensor::<bf16>::from_slice(&rows, &[tokens, depth]).unwrap();
        let cache = AttentionPackedMatrix::new(&keys.view(), &values.view(), depth, &[0u32, tokens as u32]).unwrap();

        // Equal scores weight every visible key alike, so a row is the mean index of its band.
        let bands = [(usize::MAX, usize::MAX), (usize::MAX, 0), (0, 0), (1, 0), (1, 2)];
        for (keys_before, keys_after) in bands {
            let outputs = cache
                .attention(&keys.view(), &[0u32, tokens as u32], None, keys_before, keys_after)
                .unwrap();
            for row in 0..tokens {
                let first = row.saturating_sub(keys_before);
                let last = row.saturating_add(keys_after).min(tokens - 1);
                let expected = (first + last) as f32 / 2.0;
                let got = outputs.as_slice()[row * depth];
                assert!(
                    (got - expected).abs() < 5e-2,
                    "band ({keys_before}, {keys_after}) row {row}: expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn single_query_aligns_to_segment_end() {
        crate::capabilities::configure_cpu_thread().unwrap();
        let (tokens, depth) = (8usize, 8usize);
        let keys = Tensor::<bf16>::full(&[tokens, depth], bf16::from_f32(0.1)).unwrap();
        let rows: Vec<bf16> = (0..tokens)
            .flat_map(|token| core::iter::repeat(bf16::from_f32(token as f32)).take(depth))
            .collect();
        let values = Tensor::<bf16>::from_slice(&rows, &[tokens, depth]).unwrap();
        let cache = AttentionPackedMatrix::new(&keys.view(), &values.view(), depth, &[0u32, tokens as u32]).unwrap();
        let query = Tensor::<bf16>::full(&[1, depth], bf16::from_f32(0.1)).unwrap();

        let causal = cache.attention(&query.view(), &[0u32, 1], None, usize::MAX, 0).unwrap();
        assert!((causal.as_slice()[0] - 3.5).abs() < 5e-2, "decode sees the whole cache");
        let latest = cache.attention(&query.view(), &[0u32, 1], None, 0, 0).unwrap();
        assert!(
            (latest.as_slice()[0] - 7.0).abs() < 5e-2,
            "a one-key window sees the last key"
        );
    }

    #[test]
    fn pack_is_hermetic() {
        // Packing must be a pure function of its inputs: pre-filling the destination with different
        // garbage must not change a single byte of the result. This is the invariant that lets the
        // container skip pre-zeroing — the packer owns every byte, including the directory's aligned
        // tail and each plane's padding. Both windows are 64-aligned so the layout is identical.
        crate::capabilities::configure_cpu_thread().unwrap();
        let (heads, depth) = (2usize, 64usize);
        let offsets = [0u32, 7, 7, 40]; // three segments, including a 0-length pad (7..7)
        let segment_lengths: Vec<u32> = offsets.windows(2).map(|pair| pair[1] - pair[0]).collect();
        let segment_count = segment_lengths.len();
        let tokens = *offsets.last().unwrap() as usize;
        let keys = Tensor::<bf16>::full(&[tokens, heads * depth], bf16::from_f32(0.1)).unwrap();
        let values = Tensor::<bf16>::full(&[tokens, heads * depth], bf16::from_f32(0.2)).unwrap();
        let (keys_view, values_view) = (keys.view(), values.view());
        let size = <bf16 as Attention>::attention_pack_size(heads, depth, tokens, segment_count).unwrap();
        let keys_stride = keys_view.stride_bytes(0) as usize;
        let values_stride = values_view.stride_bytes(0) as usize;

        let pack_with_fill = |fill: u8| -> Vec<u8> {
            let mut backing = vec![fill; size + SIMD_ALIGNMENT];
            let base = backing.as_mut_ptr();
            let packed = unsafe { base.add(base.align_offset(SIMD_ALIGNMENT)) };
            unsafe {
                <bf16 as Attention>::attention_pack(
                    keys_view.as_ptr(),
                    values_view.as_ptr(),
                    heads,
                    depth,
                    offsets.as_ptr(),
                    segment_lengths.as_ptr(),
                    segment_count,
                    keys_stride,
                    values_stride,
                    packed,
                    0,
                    segment_count * heads,
                )
                .unwrap();
                core::slice::from_raw_parts(packed, size).to_vec()
            }
        };
        assert_eq!(
            pack_with_fill(0x00),
            pack_with_fill(0xFF),
            "attention pack must be a pure function of its inputs (no allocator garbage in the blob)"
        );
    }

    fn check_attention_rope<Scalar>(values: &[f32], head_count: usize, depth: usize)
    where
        Scalar: FloatLike + TestableType + AttentionRope,
    {
        let (rows, half_depth) = (2, depth / 2);
        let width = head_count * depth;
        assert_eq!(values.len(), rows * width);
        let x: Vec<Scalar> = values.iter().map(|&v| Scalar::from_f32(v)).collect();
        // Per-token angle grids [rows, half_depth].
        let cos: Vec<f32> = (0..rows * half_depth).map(|k| (0.1 * k as f32).cos()).collect();
        let sin: Vec<f32> = (0..rows * half_depth).map(|k| (0.1 * k as f32).sin()).collect();
        let reference = x.clone();
        let mut x_t = crate::tensor::Tensor::<Scalar>::from_slice(&x, &[rows, width]).unwrap();
        Scalar::attention_rope_into(&mut x_t, &cos, &sin, head_count, depth).unwrap();
        let x = x_t.as_slice().to_vec();
        for r in 0..rows {
            for h in 0..head_count {
                let base = r * width + h * depth;
                for i in 0..half_depth {
                    let low = reference[base + i].to_f64();
                    let high = reference[base + i + half_depth].to_f64();
                    let cosine = cos[r * half_depth + i] as f64;
                    let sine = sin[r * half_depth + i] as f64;
                    let expected_low = Scalar::from_f32((low * cosine - high * sine) as f32).to_f64();
                    let expected_high = Scalar::from_f32((low * sine + high * cosine) as f32).to_f64();
                    assert_close(
                        x[base + i].to_f64(),
                        expected_low,
                        Scalar::atol() * 4.0,
                        Scalar::rtol() * 4.0,
                        "rope low",
                    );
                    assert_close(
                        x[base + i + half_depth].to_f64(),
                        expected_high,
                        Scalar::atol() * 4.0,
                        Scalar::rtol() * 4.0,
                        "rope high",
                    );
                }
            }
        }
    }

    #[test]
    fn rope_split_half() {
        let values: Vec<f32> = (0..2 * 2 * 2 * 8).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        check_attention_rope::<f32>(&values, 2, 16);
        check_attention_rope::<bf16>(&values, 2, 16);
        check_attention_rope::<e4m3>(&values, 2, 16);
    }

    #[test]
    fn rope_strided_section() {
        use crate::tensor::{SliceRange, Tensor};

        // Rotate the left __[rows,width]__ column-section of a __[rows,2×width]__ buffer in place
        // with a row stride of twice the width, the Q or K section of a fused QKV buffer.
        let (rows, head_count, depth) = (3, 2, 8);
        let (width, half_depth) = (head_count * depth, depth / 2);
        let full = 2 * width;
        let section: Vec<f32> = (0..rows * width).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let mut wide_vec = vec![999.0f32; rows * full]; // right half is a sentinel
        for r in 0..rows {
            for c in 0..width {
                wide_vec[r * full + c] = section[r * width + c];
            }
        }
        let cos: Vec<f32> = (0..rows * half_depth).map(|k| (0.1 * k as f32).cos()).collect();
        let sin: Vec<f32> = (0..rows * half_depth).map(|k| (0.1 * k as f32).sin()).collect();

        let mut wide = Tensor::<f32>::from_slice(&wide_vec, &[rows, full]).unwrap();
        {
            let mut span = wide.span();
            let mut sec = span
                .slice_mut(&[SliceRange::Full, SliceRange::range(0, width)][..])
                .unwrap();
            f32::attention_rope_into(&mut sec, &cos, &sin, head_count, depth).unwrap();
        }

        let mut contig = Tensor::<f32>::from_slice(&section, &[rows, width]).unwrap();
        f32::attention_rope_into(&mut contig, &cos, &sin, head_count, depth).unwrap();

        let wide_after = wide.as_slice();
        let contig_after = contig.as_slice();
        for r in 0..rows {
            for c in 0..width {
                assert!(
                    (wide_after[r * full + c] - contig_after[r * width + c]).abs() < 1e-5,
                    "strided RoPE section mismatch at [{r},{c}]"
                );
            }
            for c in width..full {
                assert_eq!(
                    wide_after[r * full + c],
                    999.0,
                    "RoPE wrote outside its strided section"
                );
            }
        }
    }
}
