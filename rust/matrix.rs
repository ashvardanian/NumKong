//! Batch matrix operations: GEMM, packed spatial distances.
//!
//! This module provides:
//!
//! - [`Dots`]: Batch dot-product (GEMM-like) between two matrices
//! - [`Angulars`]: Batch angular (cosine) distances
//! - [`Euclideans`]: Batch squared-L2 distances
//! - [`Hammings`]: Batch Hamming distances
//! - [`Jaccards`]: Batch Jaccard distances
//! - [`PackedMatrix`]: Pre-packed matrix for accelerated batch operations
//! - [`SymmetricDots`], [`SymmetricAngulars`], [`SymmetricEuclideans`],
//!   [`SymmetricHammings`], [`SymmetricJaccards`]: extension traits that add
//!   `try_*_symmetric` methods to any [`TensorRef`] implementor, computing the
//!   upper triangle of the pairwise Gram / distance matrix.
//!
//! # Pack once, query many
//!
//! When the right-hand side of a batch operation is reused across many queries
//! (e.g. a fixed corpus of document embeddings), pack it once with
//! [`PackedMatrix::try_pack`] and reuse the packed buffer:
//!
//! ```rust,ignore
//! use numkong::{Tensor, PackedMatrix};
//!
//! // Corpus: 1 million vectors of dimension 768 (built once).
//! let corpus = Tensor::<f32>::try_full(&[1_000_000, 768], 0.0)?;
//! let corpus_packed = PackedMatrix::try_pack(&corpus)?;
//!
//! // Each query is a batch of 32 vectors — kernel reads the packed layout
//! // directly, avoiding re-packing overhead per query.
//! let queries = Tensor::<f32>::try_full(&[32, 768], 0.0)?;
//! let scores = queries.dots_packed(&corpus_packed);       // f32 × f32 → f64
//! let angles = queries.angulars_packed(&corpus_packed);   // f32 × f32 → f32
//! ```

extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::tensor::{
    Allocator, Global, Tensor, TensorError, TensorMut, TensorRef, TensorView, SIMD_ALIGNMENT,
};
use crate::types::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2, StorageElement};

#[link(name = "numkong")]
extern "C" {

    fn nk_dots_packed_size_f32(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_f32(
        b: *const f32,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
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

    fn nk_dots_packed_size_f64(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_f64(
        b: *const f64,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
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

    fn nk_dots_packed_size_f16(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_f16(
        b: *const u16,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
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

    fn nk_dots_packed_size_bf16(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_bf16(
        b: *const u16,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
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

    fn nk_dots_packed_size_i8(width: usize, depth: usize) -> usize;
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

    fn nk_dots_packed_size_u8(width: usize, depth: usize) -> usize;
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

    fn nk_dots_packed_size_e4m3(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_e4m3(
        b: *const u8,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
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

    fn nk_dots_packed_size_e5m2(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_e5m2(
        b: *const u8,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
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

    fn nk_dots_packed_size_e2m3(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_e2m3(
        b: *const u8,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
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

    fn nk_dots_packed_size_e3m2(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_e3m2(
        b: *const u8,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );
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

    fn nk_dots_packed_size_u4(width: usize, depth: usize) -> usize;
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

    fn nk_dots_packed_size_i4(width: usize, depth: usize) -> usize;
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

    fn nk_dots_packed_size_u1(width: usize, depth: usize) -> usize;
    fn nk_dots_pack_u1(
        q: *const u8,
        width: usize,
        depth: usize,
        q_stride: usize,
        q_packed: *mut u8,
    );
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
/// Reach for this trait (or the matching `Tensor::try_dots_packed*` wrappers)
/// whenever you multiply many query rows against the **same** matrix of
/// database rows: pre-packing B once amortises layout-conversion cost across
/// every subsequent query. The accumulator type is intentionally widened to
/// avoid precision loss — `f32 × f32 → f64`, `f16 × f16 → f32`,
/// `i8 × i8 → i32`, `u8 × u8 → u32`, and so on (see each impl's
/// [`Dots::Accumulator`]).
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
    fn dots_packed_size(width: usize, depth: usize) -> usize;

    /// Packs the B matrix into an optimized backend-specific layout.
    ///
    /// # Safety
    /// - `b` must point to valid memory for `width * depth` elements
    /// - `packed` must point to a buffer of at least `dots_packed_size(width, depth)` bytes
    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    );

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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_f32(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_f64(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_f16(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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
        nk_dots_packed_f16(
            a as *const u16,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_bf16(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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
        nk_dots_packed_bf16(
            a as *const u16,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_i8(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_u8(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_e4m3(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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
        nk_dots_packed_e4m3(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_e5m2(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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
        nk_dots_packed_e5m2(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_e2m3(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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
        nk_dots_packed_e2m3(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_e3m2(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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
        nk_dots_packed_e3m2(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_u4(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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
        nk_dots_packed_u4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_i4(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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
        nk_dots_packed_i4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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

    fn dots_packed_size(width: usize, depth: usize) -> usize {
        unsafe { nk_dots_packed_size_u1(width, depth) }
    }

    unsafe fn dots_pack(
        b: *const Self,
        width: usize,
        depth: usize,
        b_stride: usize,
        packed: *mut u8,
    ) {
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
        nk_dots_packed_u1(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
/// Binary feature vectors (bit-packed into `u1x8`) are common in approximate
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
        nk_angulars_packed_f16(
            a as *const u16,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_euclideans_packed_f16(
            a as *const u16,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_angulars_packed_bf16(
            a as *const u16,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_euclideans_packed_bf16(
            a as *const u16,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_angulars_packed_e4m3(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_euclideans_packed_e4m3(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_angulars_packed_e5m2(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_euclideans_packed_e5m2(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_angulars_packed_e2m3(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_euclideans_packed_e2m3(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_angulars_packed_e3m2(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_euclideans_packed_e3m2(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_angulars_packed_u4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_euclideans_packed_u4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_angulars_packed_i4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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
        nk_euclideans_packed_i4(
            a as *const u8,
            packed,
            c,
            height,
            width,
            depth,
            a_stride,
            c_stride,
        )
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

// region: PackedMatrix

/// Pre-packed B matrix for efficient repeated GEMM operations.
///
/// Uses raw memory allocation (no std::Vec) for maximum control.
///
/// When multiplying A × Bᵀ multiple times with the same B matrix,
/// packing B once and reusing it is much faster than packing each time.
///
/// # Usage
///
/// For C = A × Bᵀ where B is (n × k):
/// ```rust,ignore
/// // Requires linking against libnumkong C library
/// let b_packed = PackedMatrix::try_pack(&b_array).unwrap();
/// let c = a_array.dots_packed(&b_packed);
/// ```
///
/// For C = A × B where B is (k × n) (standard GEMM layout):
/// ```rust,ignore
/// // Requires linking against libnumkong C library
/// let b_packed = PackedMatrix::try_pack_transposed(&b_array).unwrap();
/// let c = a_array.dots_packed(&b_packed);
/// ```
pub struct PackedMatrix<Scalar: Dots, Alloc: Allocator = Global> {
    /// Raw pointer to packed data buffer.
    data: NonNull<u8>,
    /// Size of the packed buffer in bytes.
    size: usize,
    /// Output columns (B width).
    width: usize,
    /// Inner dimension (depth).
    depth: usize,
    /// Allocator instance.
    alloc: Alloc,
    _marker: PhantomData<Scalar>,
}

// Safety: PackedMatrix owns its data and is just bytes
unsafe impl<Scalar: Dots + Send, Alloc: Allocator + Send> Send for PackedMatrix<Scalar, Alloc> {}
unsafe impl<Scalar: Dots + Sync, Alloc: Allocator + Sync> Sync for PackedMatrix<Scalar, Alloc> {}

impl<Scalar: Dots, Alloc: Allocator> Drop for PackedMatrix<Scalar, Alloc> {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe {
                let layout =
                    alloc::alloc::Layout::from_size_align_unchecked(self.size, SIMD_ALIGNMENT);
                self.alloc.deallocate(self.data, layout);
            }
        }
    }
}

impl<Scalar: Dots, Alloc: Allocator + Clone> PackedMatrix<Scalar, Alloc> {
    /// Try to clone this packed matrix, returning an error on allocation failure.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        if self.size == 0 {
            return Ok(Self {
                data: NonNull::dangling(),
                size: 0,
                width: self.width,
                depth: self.depth,
                alloc: self.alloc.clone(),
                _marker: PhantomData,
            });
        }

        let layout = alloc::alloc::Layout::from_size_align(self.size, SIMD_ALIGNMENT)
            .map_err(|_| TensorError::AllocationFailed)?;
        let ptr = self
            .alloc
            .allocate(layout)
            .ok_or(TensorError::AllocationFailed)?;
        unsafe {
            core::ptr::copy_nonoverlapping(self.data.as_ptr(), ptr.as_ptr(), self.size);
        }
        Ok(Self {
            data: ptr,
            size: self.size,
            width: self.width,
            depth: self.depth,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        })
    }
}

impl<Scalar: Dots, Alloc: Allocator + Clone> Clone for PackedMatrix<Scalar, Alloc> {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("PackedMatrix clone allocation failed")
    }
}

// Generic allocator-aware methods
impl<Scalar: Dots, Alloc: Allocator> PackedMatrix<Scalar, Alloc> {
    /// Pack B matrix where B is (n × k) row-major using a custom allocator.
    ///
    /// Result computes: C = A × Bᵀ
    ///
    /// Returns `Err` if:
    /// - b is not 2D
    /// - allocation fails
    pub fn try_pack_in<PackedAlloc: Allocator, const MAX_RANK: usize>(
        b: &Tensor<Scalar, PackedAlloc, MAX_RANK>,
        alloc: Alloc,
    ) -> Result<Self, TensorError> {
        if b.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: b.ndim(),
            });
        }
        let (width, depth) = (b.shape()[0], b.shape()[1]);
        let size = Scalar::dots_packed_size(width, depth);

        let data = if size == 0 {
            NonNull::dangling()
        } else {
            // Allocate with SIMD alignment
            let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Zero the memory
            unsafe {
                core::ptr::write_bytes(ptr.as_ptr(), 0, size);
            }
            ptr
        };

        if size > 0 {
            unsafe {
                Scalar::dots_pack(
                    b.as_ptr(),
                    width,
                    depth,
                    b.stride_bytes(0) as usize,
                    data.as_ptr(),
                );
            }
        }

        Ok(Self {
            data,
            size,
            width,
            depth,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Pack Bᵀ where B is (k × n) row-major (standard GEMM layout) using a custom allocator.
    ///
    /// Materializes the transpose into a contiguous buffer, then packs normally.
    /// Result computes: C = A × B
    ///
    /// Returns `Err` if:
    /// - b is not 2D
    /// - b is a sub-byte type (transpose unsupported)
    /// - allocation fails
    pub fn try_pack_transposed_in<PackedAlloc: Allocator, const MAX_RANK: usize>(
        b: &Tensor<Scalar, PackedAlloc, MAX_RANK>,
        alloc: Alloc,
    ) -> Result<Self, TensorError> {
        if b.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: b.ndim(),
            });
        }
        // Transpose returns a strided view (no copy), then to_owned materializes it.
        // For sub-byte types, transpose() returns SubByteUnsupported.
        let transposed = b.transpose()?.to_owned()?;
        Self::try_pack_in(&transposed, alloc)
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc {
        &self.alloc
    }

    /// Returns dimensions (width, depth) of the original B matrix.
    pub fn dims(&self) -> (usize, usize) {
        (self.width, self.depth)
    }

    /// Returns the packed data buffer.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.size) }
    }

    /// Returns a pointer to the packed data.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

// Convenience methods using Global allocator
impl<Scalar: Dots> PackedMatrix<Scalar, Global> {
    /// Pack B matrix where B is (n × k) row-major using the global allocator.
    ///
    /// Result computes: C = A × Bᵀ
    pub fn try_pack<PackedAlloc: Allocator, const MAX_RANK: usize>(
        b: &Tensor<Scalar, PackedAlloc, MAX_RANK>,
    ) -> Result<Self, TensorError> {
        Self::try_pack_in(b, Global)
    }

    /// Pack Bᵀ where B is (k × n) row-major (standard GEMM layout) using the global allocator.
    ///
    /// Result computes: C = A × B
    pub fn try_pack_transposed<PackedAlloc: Allocator, const MAX_RANK: usize>(
        b: &Tensor<Scalar, PackedAlloc, MAX_RANK>,
    ) -> Result<Self, TensorError> {
        Self::try_pack_transposed_in(b, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn pack<PackedAlloc: Allocator, const MAX_RANK: usize>(
        b: &Tensor<Scalar, PackedAlloc, MAX_RANK>,
    ) -> Self {
        Self::try_pack(b).expect("PackedMatrix::pack failed")
    }

    /// Convenience constructor that panics on error.
    pub fn pack_transposed<PackedAlloc: Allocator, const MAX_RANK: usize>(
        b: &Tensor<Scalar, PackedAlloc, MAX_RANK>,
    ) -> Self {
        Self::try_pack_transposed(b).expect("PackedMatrix::pack_transposed failed")
    }
}

// endregion: PackedMatrix

// region: Shared validators

/// Validate shared preconditions for `*_packed` operations.
///
/// Checks that `a` is a 2D tensor with contiguous rows and that its depth
/// matches that of the packed matrix. Returns `(height, width, depth)` on success.
#[inline]
fn validate_packed_input<Scalar, Alloc, PackedAlloc, const MAX_RANK: usize>(
    a: &Tensor<Scalar, Alloc, MAX_RANK>,
    packed_b: &PackedMatrix<Scalar, PackedAlloc>,
) -> Result<(usize, usize, usize), TensorError>
where
    Scalar: Dots,
    Alloc: Allocator,
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
    let (width, packed_depth) = packed_b.dims();
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
fn validate_matrix_output<R, OutputTensor, const OUTPUT_MAX_RANK: usize>(
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

/// Validate shared preconditions for `*_symmetric` operations (2D input).
/// Returns `(n_vectors, depth)` on success.
#[inline]
fn validate_symmetric_input<Scalar, InputTensor, const MAX_RANK: usize>(
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
    Ok((a.shape()[0], a.shape()[1]))
}

// endregion: Shared validators

// region: Tensor GEMM

impl<Scalar: Dots, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
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
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::Accumulator, Alloc, MAX_RANK>, TensorError> {
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

        let mut c = Tensor::try_full_in(
            &[height, width],
            Scalar::Accumulator::default(),
            self.alloc.clone(),
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
    pub fn dots_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::Accumulator, Alloc, MAX_RANK> {
        self.try_dots_packed(packed_b).expect("dots_packed failed")
    }
}

impl<Scalar: Dots, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Dot-product multiply into existing output (avoids allocation).
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// writable tensor container that implements [`TensorMut`] works. The
    /// kernel overwrites `c` — it need not be pre-initialized.
    pub fn try_dots_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
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

// Parallel dots_packed implementations, if ForkUnion is available
#[cfg(feature = "parallel")]
impl<Scalar: Dots + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::Accumulator: Send + Sync,
{
    /// Parallel dot-product multiply into pre-allocated output.
    ///
    /// Distributes rows of A across threads; each computes its portion of C.
    /// This is a non-allocating interface - you provide the output tensor.
    ///
    /// # Arguments
    /// * `packed_b` - Pre-packed B matrix from `PackedMatrix::try_pack[_transposed]`
    /// * `c` - Pre-allocated output tensor (m × n)
    /// * `pool` - Pre-constructed thread pool
    ///
    /// The output may be a `&mut Tensor<...>` or a `&mut TensorSpan<...>` —
    /// any writable tensor container that implements [`TensorMut`]. The
    /// kernel overwrites `c` and need not see initialized memory.
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{Tensor, PackedMatrix};
    /// use fork_union::ThreadPool;
    ///
    /// let mut pool = ThreadPool::try_spawn(4).unwrap();
    /// let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
    /// let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();
    /// let b_packed = PackedMatrix::try_pack(&b).unwrap();
    ///
    /// // Writing into a full tensor:
    /// let mut c = Tensor::<f32>::try_full(&[1024, 256], 0.0).unwrap();
    /// a.try_dots_packed_parallel_into(&b_packed, &mut c, &mut pool).unwrap();
    ///
    /// // Or into a span over a sub-region of a larger buffer:
    /// let mut c_buf = Tensor::<f32>::try_full(&[1024, 256], 0.0).unwrap();
    /// a.try_dots_packed_parallel_into(&b_packed, &mut c_buf.span(), &mut pool).unwrap();
    /// ```
    pub fn try_dots_packed_parallel_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::Accumulator, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::Accumulator, _, OUTPUT_MAX_RANK>(c, height, width)?;

        let a_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fork_union::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;

        // Get actual thread count from pool
        let num_threads = pool.threads().max(1);
        let rows_per_thread = height.div_ceil(num_threads);

        // Distribute rows across threads using fork_union
        // Safety: Each thread writes to disjoint rows of C, so no data races.
        pool.for_threads(move |thread_index, _colocation_index| {
            // Configure each worker thread for optimal SIMD (including AMX)
            // This is idempotent and safe to call multiple times
            crate::capabilities::configure_thread();

            let row_start = thread_index * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(height);

            if row_start < height {
                unsafe {
                    // Byte arithmetic so sub-byte types (u1x8 etc.) stride correctly.
                    let a_row =
                        (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
                    let c_row = (c_ptr.as_ptr() as *mut u8).add(row_start * c_stride)
                        as *mut Scalar::Accumulator;
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
            }
        })
        .join();

        Ok(())
    }

    /// Parallel dot-product multiply with allocation.
    ///
    /// Convenience wrapper that allocates the output tensor.
    /// Prefer `try_dots_packed_parallel_into` for performance-critical code.
    pub fn try_dots_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<Scalar::Accumulator, Global, MAX_RANK>, TensorError> {
        let height = self.shape()[0];
        let (width, _) = packed_b.dims();
        let mut c = Tensor::<Scalar::Accumulator, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::Accumulator::default(),
        )?;
        self.try_dots_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn dots_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<Scalar::Accumulator, Global, MAX_RANK> {
        self.try_dots_packed_parallel(packed_b, pool)
            .expect("parallel dots_packed failed")
    }
}

/// Compute row assignment for a thread without allocation
///
/// For a symmetric matrix, cumulative work up to row r is: r*(2n - r + 1)/2
/// Solving r*(2n - r + 1)/2 = work using quadratic formula gives exact row.
#[cfg(feature = "std")]
#[inline]
fn compute_thread_rows(thread_index: usize, num_threads: usize, n: usize) -> (usize, usize) {
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
        // giving threads disjoint row ranges (whole-row scheduling).
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
    /// use fork_union::ThreadPool;
    ///
    /// let mut pool = ThreadPool::try_spawn(4).unwrap();
    /// let vectors = Tensor::<f32>::try_full(&[100, 768], 1.0).unwrap();
    /// let gram = vectors.try_dots_symmetric_parallel(&mut pool).unwrap();
    /// assert_eq!(gram.shape(), &[100, 100]);
    /// ```
    pub fn try_dots_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
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
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::Accumulator, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::Accumulator, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;

        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.for_threads(move |thread_index, _colocation_index| {
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
        })
        .join();
        Ok(())
    }

    /// Parallel computation of symmetric dot-product matrix (unwrapping version).
    ///
    /// # Panics
    /// Panics if the operation fails (e.g., wrong tensor rank).
    pub fn dots_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<Scalar::Accumulator, Global, MAX_RANK> {
        self.try_dots_symmetric_parallel(pool)
            .expect("parallel dots_symmetric failed")
    }
}

// endregion: Tensor GEMM

// region: Tensor Spatial Distances

impl<Scalar: Angulars, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Computes angular distances between rows of self and packed B matrix.
    pub fn try_angulars_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
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
        let mut c = Tensor::try_full_in(
            &[height, width],
            Scalar::SpatialResult::default(),
            self.alloc.clone(),
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
    pub fn angulars_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::SpatialResult, Alloc, MAX_RANK> {
        self.try_angulars_packed(packed_b)
            .expect("angulars_packed failed")
    }
}

impl<Scalar: Angulars, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Computes angular distances into pre-allocated output.
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// [`TensorMut`] implementor works. The kernel overwrites `c`.
    pub fn try_angulars_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
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

impl<Scalar: Euclideans, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Computes euclidean distances between rows of self and packed B matrix.
    pub fn try_euclideans_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
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
        let mut c = Tensor::try_full_in(
            &[height, width],
            Scalar::SpatialResult::default(),
            self.alloc.clone(),
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
    pub fn euclideans_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::SpatialResult, Alloc, MAX_RANK> {
        self.try_euclideans_packed(packed_b)
            .expect("euclideans_packed failed")
    }
}

impl<Scalar: Euclideans, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Computes euclidean distances into pre-allocated output.
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// [`TensorMut`] implementor works. The kernel overwrites `c`.
    pub fn try_euclideans_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
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

// Parallel spatial distance implementations
#[cfg(feature = "parallel")]
impl<Scalar: Angulars + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::SpatialResult: Send + Sync,
{
    /// Parallel angular distances into pre-allocated output.
    ///
    /// The kernel overwrites `c`; callers need not pre-initialize.
    pub fn try_angulars_packed_parallel_into<
        PackedAlloc,
        OutputTensor,
        const OUTPUT_MAX_RANK: usize,
    >(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, height, width)?;

        let a_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fork_union::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;
        let num_threads = pool.threads().max(1);
        let rows_per_thread = height.div_ceil(num_threads);

        pool.for_threads(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let row_start = thread_index * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(height);
            if row_start < height {
                unsafe {
                    let a_row =
                        (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
                    let c_row = (c_ptr.as_ptr() as *mut u8).add(row_start * c_stride)
                        as *mut Scalar::SpatialResult;
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
        })
        .join();
        Ok(())
    }

    /// Parallel angular distances with allocation.
    pub fn try_angulars_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
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
    pub fn angulars_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_angulars_packed_parallel(packed_b, pool)
            .expect("parallel angulars_packed failed")
    }

    /// Parallel symmetric angular distance matrix.
    pub fn try_angulars_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
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
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(
            c, n_vectors, n_vectors,
        )?;
        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.for_threads(move |thread_index, _colocation_index| {
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
        })
        .join();
        Ok(())
    }

    /// Convenience method that panics on error.
    pub fn angulars_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_angulars_symmetric_parallel(pool)
            .expect("parallel angulars_symmetric failed")
    }
}

#[cfg(feature = "parallel")]
impl<Scalar: Euclideans + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::SpatialResult: Send + Sync,
{
    /// Parallel euclidean distances into pre-allocated output.
    ///
    /// The kernel overwrites `c`; callers need not pre-initialize.
    pub fn try_euclideans_packed_parallel_into<
        PackedAlloc,
        OutputTensor,
        const OUTPUT_MAX_RANK: usize,
    >(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(c, height, width)?;

        let a_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fork_union::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;
        let num_threads = pool.threads().max(1);
        let rows_per_thread = height.div_ceil(num_threads);

        pool.for_threads(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let row_start = thread_index * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(height);
            if row_start < height {
                unsafe {
                    let a_row =
                        (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
                    let c_row = (c_ptr.as_ptr() as *mut u8).add(row_start * c_stride)
                        as *mut Scalar::SpatialResult;
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
        })
        .join();
        Ok(())
    }

    /// Parallel euclidean distances with allocation.
    pub fn try_euclideans_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
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
    pub fn euclideans_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_euclideans_packed_parallel(packed_b, pool)
            .expect("parallel euclideans_packed failed")
    }

    /// Parallel symmetric euclidean distance matrix.
    pub fn try_euclideans_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
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
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(
            c, n_vectors, n_vectors,
        )?;
        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.for_threads(move |thread_index, _colocation_index| {
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
        })
        .join();
        Ok(())
    }

    /// Convenience method that panics on error.
    pub fn euclideans_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<Scalar::SpatialResult, Global, MAX_RANK> {
        self.try_euclideans_symmetric_parallel(pool)
            .expect("parallel euclideans_symmetric failed")
    }
}

// endregion: Tensor Spatial Distances

// region: Tensor Hammings/Jaccards

impl<Scalar: Hammings, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Computes Hamming distances between rows of self and packed B matrix.
    pub fn try_hammings_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<u32, Alloc, MAX_RANK>, TensorError> {
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
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<u32, Alloc, MAX_RANK> {
        self.try_hammings_packed(packed_b)
            .expect("hammings_packed failed")
    }
}

impl<Scalar: Hammings, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Computes Hamming distances into pre-allocated output.
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// [`TensorMut`] implementor works. The kernel overwrites `c`.
    pub fn try_hammings_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
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

impl<Scalar: Jaccards, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Computes Jaccard distances between rows of self and packed B matrix.
    pub fn try_jaccards_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
    ) -> Result<Tensor<Scalar::JaccardResult, Alloc, MAX_RANK>, TensorError> {
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
        let mut c = Tensor::try_full_in(
            &[height, width],
            Scalar::JaccardResult::default(),
            self.alloc.clone(),
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
    pub fn jaccards_packed<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
    ) -> Tensor<Scalar::JaccardResult, Alloc, MAX_RANK> {
        self.try_jaccards_packed(packed_b)
            .expect("jaccards_packed failed")
    }
}

impl<Scalar: Jaccards, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Computes Jaccard distances into pre-allocated output.
    ///
    /// The output may be a `&mut Tensor<...>` or `&mut TensorSpan<...>`; any
    /// [`TensorMut`] implementor works. The kernel overwrites `c`.
    pub fn try_jaccards_packed_into<PackedAlloc, OutputTensor, const OUTPUT_MAX_RANK: usize>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
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

// endregion: Tensor Hammings/Jaccards

// region: Parallel Hammings/Jaccards

#[cfg(feature = "parallel")]
impl<Scalar: Hammings + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Parallel Hamming distances into pre-allocated output.
    ///
    /// The kernel overwrites `c`; callers need not pre-initialize.
    pub fn try_hammings_packed_parallel_into<
        PackedAlloc,
        OutputTensor,
        const OUTPUT_MAX_RANK: usize,
    >(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<u32, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<u32, _, OUTPUT_MAX_RANK>(c, height, width)?;

        let a_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fork_union::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;
        let num_threads = pool.threads().max(1);
        let rows_per_thread = height.div_ceil(num_threads);

        pool.for_threads(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let row_start = thread_index * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(height);
            if row_start < height {
                unsafe {
                    let a_row =
                        (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
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
            }
        })
        .join();
        Ok(())
    }

    /// Parallel Hamming distances with allocation.
    pub fn try_hammings_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        let height = self.shape()[0];
        let (width, _) = packed_b.dims();
        let mut c = Tensor::<u32, Global, MAX_RANK>::try_full(&[height, width], 0u32)?;
        self.try_hammings_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn hammings_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<u32, Global, MAX_RANK> {
        self.try_hammings_packed_parallel(packed_b, pool)
            .expect("parallel hammings_packed failed")
    }

    /// Parallel symmetric Hamming-distance matrix.
    ///
    /// Only the upper triangle of the result is guaranteed to be initialized.
    pub fn try_hammings_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
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
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<u32, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<u32, _, OUTPUT_MAX_RANK>(c, n_vectors, n_vectors)?;
        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.for_threads(move |thread_index, _colocation_index| {
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
        })
        .join();
        Ok(())
    }

    /// Convenience method that panics on error.
    pub fn hammings_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<u32, Global, MAX_RANK> {
        self.try_hammings_symmetric_parallel(pool)
            .expect("parallel hammings_symmetric failed")
    }
}

#[cfg(feature = "parallel")]
impl<Scalar: Jaccards + Clone + Send + Sync, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::JaccardResult: Send + Sync,
{
    /// Parallel Jaccard distances into pre-allocated output.
    ///
    /// The kernel overwrites `c`; callers need not pre-initialize.
    pub fn try_jaccards_packed_parallel_into<
        PackedAlloc,
        OutputTensor,
        const OUTPUT_MAX_RANK: usize,
    >(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        c: &mut OutputTensor,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        PackedAlloc: Allocator,
        OutputTensor: TensorMut<Scalar::JaccardResult, OUTPUT_MAX_RANK>,
    {
        let (height, width, depth) = validate_packed_input(self, packed_b)?;
        validate_matrix_output::<Scalar::JaccardResult, _, OUTPUT_MAX_RANK>(c, height, width)?;

        let a_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fork_union::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride_bytes(0) as usize;
        let c_stride = c.stride_bytes(0) as usize;
        let num_threads = pool.threads().max(1);
        let rows_per_thread = height.div_ceil(num_threads);

        pool.for_threads(move |thread_index, _colocation_index| {
            crate::capabilities::configure_thread();
            let row_start = thread_index * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(height);
            if row_start < height {
                unsafe {
                    let a_row =
                        (a_ptr.as_ptr() as *const u8).add(row_start * a_stride) as *const Scalar;
                    let c_row = (c_ptr.as_ptr() as *mut u8).add(row_start * c_stride)
                        as *mut Scalar::JaccardResult;
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
            }
        })
        .join();
        Ok(())
    }

    /// Parallel Jaccard distances with allocation.
    pub fn try_jaccards_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<Scalar::JaccardResult, Global, MAX_RANK>, TensorError> {
        let height = self.shape()[0];
        let (width, _) = packed_b.dims();
        let mut c = Tensor::<Scalar::JaccardResult, Global, MAX_RANK>::try_full(
            &[height, width],
            Scalar::JaccardResult::default(),
        )?;
        self.try_jaccards_packed_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn jaccards_packed_parallel<PackedAlloc: Allocator>(
        &self,
        packed_b: &PackedMatrix<Scalar, PackedAlloc>,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<Scalar::JaccardResult, Global, MAX_RANK> {
        self.try_jaccards_packed_parallel(packed_b, pool)
            .expect("parallel jaccards_packed failed")
    }

    /// Parallel symmetric Jaccard-distance matrix.
    ///
    /// Only the upper triangle of the result is guaranteed to be initialized.
    pub fn try_jaccards_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
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
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>
    where
        OutputTensor: TensorMut<Scalar::JaccardResult, OUTPUT_MAX_RANK>,
    {
        let (n_vectors, depth) = validate_symmetric_input(self)?;
        validate_matrix_output::<Scalar::JaccardResult, _, OUTPUT_MAX_RANK>(
            c, n_vectors, n_vectors,
        )?;
        let num_threads = pool.threads().max(1);
        let vectors_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let result_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let stride = self.stride_bytes(0) as usize;
        let result_stride = c.stride_bytes(0) as usize;

        pool.for_threads(move |thread_index, _colocation_index| {
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
        })
        .join();
        Ok(())
    }

    /// Convenience method that panics on error.
    pub fn jaccards_symmetric_parallel(
        &self,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<Scalar::JaccardResult, Global, MAX_RANK> {
        self.try_jaccards_symmetric_parallel(pool)
            .expect("parallel jaccards_symmetric failed")
    }
}

// endregion: Parallel Hammings/Jaccards

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
    pub fn try_dots_symmetric(
        &self,
    ) -> Result<Tensor<Scalar::Accumulator, Global, MAX_RANK>, TensorError> {
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

impl<'a, Scalar: Angulars, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Computes symmetric angular distance matrix for a set of vectors.
    pub fn try_angulars_symmetric(
        &self,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
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
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(
            c, n_vectors, n_vectors,
        )?;
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
    pub fn try_euclideans_symmetric(
        &self,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
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
        validate_matrix_output::<Scalar::SpatialResult, _, OUTPUT_MAX_RANK>(
            c, n_vectors, n_vectors,
        )?;
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

impl<'a, Scalar: Hammings, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Computes symmetric Hamming distance matrix for a set of binary vectors.
    pub fn try_hammings_symmetric(&self) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        let (n_vectors, _) = validate_symmetric_input(self)?;
        let mut result =
            Tensor::<u32, Global, MAX_RANK>::try_full(&[n_vectors, n_vectors], u32::default())?;
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
    pub fn try_jaccards_symmetric(
        &self,
    ) -> Result<Tensor<Scalar::JaccardResult, Global, MAX_RANK>, TensorError> {
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
        validate_matrix_output::<Scalar::JaccardResult, _, OUTPUT_MAX_RANK>(
            c, n_vectors, n_vectors,
        )?;
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
pub trait SymmetricDots<Scalar: Dots, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar::Accumulator: 'static,
{
    fn try_dots_symmetric(
        &self,
    ) -> Result<Tensor<Scalar::Accumulator, Global, MAX_RANK>, TensorError> {
        self.view().try_dots_symmetric()
    }

    /// Writes the symmetric dot-product matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_dots_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut Out,
    ) -> Result<(), TensorError>
    where
        Out: TensorMut<Scalar::Accumulator, OUTPUT_MAX_RANK>,
    {
        self.view().try_dots_symmetric_into(c)
    }
}

impl<Scalar: Dots, const R: usize, OutputTensor: TensorRef<Scalar, R>> SymmetricDots<Scalar, R>
    for OutputTensor
where
    Scalar::Accumulator: 'static,
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
pub trait SymmetricAngulars<Scalar: Angulars, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
{
    fn try_angulars_symmetric(
        &self,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        self.view().try_angulars_symmetric()
    }

    /// Writes the symmetric angular-distance matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_angulars_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut Out,
    ) -> Result<(), TensorError>
    where
        Out: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        self.view().try_angulars_symmetric_into(c)
    }
}

impl<Scalar: Angulars, const R: usize, OutputTensor: TensorRef<Scalar, R>>
    SymmetricAngulars<Scalar, R> for OutputTensor
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
pub trait SymmetricEuclideans<Scalar: Euclideans, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
{
    fn try_euclideans_symmetric(
        &self,
    ) -> Result<Tensor<Scalar::SpatialResult, Global, MAX_RANK>, TensorError> {
        self.view().try_euclideans_symmetric()
    }

    /// Writes the symmetric euclidean-distance matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_euclideans_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut Out,
    ) -> Result<(), TensorError>
    where
        Out: TensorMut<Scalar::SpatialResult, OUTPUT_MAX_RANK>,
    {
        self.view().try_euclideans_symmetric_into(c)
    }
}

impl<Scalar: Euclideans, const R: usize, OutputTensor: TensorRef<Scalar, R>>
    SymmetricEuclideans<Scalar, R> for OutputTensor
{
}

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
pub trait SymmetricHammings<Scalar: Hammings, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
{
    fn try_hammings_symmetric(&self) -> Result<Tensor<u32, Global, MAX_RANK>, TensorError> {
        self.view().try_hammings_symmetric()
    }

    /// Writes the symmetric Hamming-distance matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_hammings_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut Out,
    ) -> Result<(), TensorError>
    where
        Out: TensorMut<u32, OUTPUT_MAX_RANK>,
    {
        self.view().try_hammings_symmetric_into(c)
    }
}

impl<Scalar: Hammings, const R: usize, OutputTensor: TensorRef<Scalar, R>>
    SymmetricHammings<Scalar, R> for OutputTensor
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
pub trait SymmetricJaccards<Scalar: Jaccards, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
{
    fn try_jaccards_symmetric(
        &self,
    ) -> Result<Tensor<Scalar::JaccardResult, Global, MAX_RANK>, TensorError> {
        self.view().try_jaccards_symmetric()
    }

    /// Writes the symmetric Jaccard-distance matrix into pre-allocated output.
    /// Only the upper triangle is written.
    fn try_jaccards_symmetric_into<Out, const OUTPUT_MAX_RANK: usize>(
        &self,
        c: &mut Out,
    ) -> Result<(), TensorError>
    where
        Out: TensorMut<Scalar::JaccardResult, OUTPUT_MAX_RANK>,
    {
        self.view().try_jaccards_symmetric_into(c)
    }
}

impl<Scalar: Jaccards, const R: usize, OutputTensor: TensorRef<Scalar, R>>
    SymmetricJaccards<Scalar, R> for OutputTensor
{
}

// endregion: Symmetric Extension Traits

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FloatLike, NumberLike, TestableType};
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init_thread() {
        INIT.call_once(|| {
            crate::capabilities::configure_thread();
        });
    }

    const DIMS: &[(usize, usize, usize)] =
        &[(1, 1, 1), (1, 8, 3), (3, 1, 7), (7, 5, 3), (33, 17, 65)];

    /// Round `depth` up to the nearest multiple of `Scalar::dimensions_per_value()`.
    fn align_depth<Scalar: StorageElement>(depth: usize) -> usize {
        let dims_per_value = Scalar::dimensions_per_value();
        depth.div_ceil(dims_per_value) * dims_per_value
    }

    fn check_dots_packed<Scalar: TestableType + Dots>()
    where
        Scalar::Accumulator: FloatLike + PartialEq + core::fmt::Debug,
    {
        init_thread();
        for &(height, width, depth) in DIMS {
            let depth = align_depth::<Scalar>(depth);
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let c = a.dots_packed(&b_packed);
            assert_eq!(
                c.shape(),
                &[height, width],
                "shape @ ({height},{width},{depth})"
            );
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
            let mut into_tensor = Tensor::<Scalar::Accumulator>::try_full(
                &[height, width],
                Scalar::Accumulator::default(),
            )
            .unwrap();
            a.try_dots_packed_into(&b_packed, &mut into_tensor).unwrap();
            assert_eq!(
                c.as_slice(),
                into_tensor.as_slice(),
                "_into(Tensor) @ ({height},{width},{depth})"
            );
            let mut into_span_buf = Tensor::<Scalar::Accumulator>::try_full(
                &[height, width],
                Scalar::Accumulator::default(),
            )
            .unwrap();
            a.try_dots_packed_into(&b_packed, &mut into_span_buf.span())
                .unwrap();
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
            let b_packed = PackedMatrix::try_pack_transposed(&b_t).unwrap();
            let c = a.dots_packed(&b_packed);
            assert_eq!(
                c.shape(),
                &[height, width],
                "shape @ ({height},{width},{depth})"
            );
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
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let c = a.angulars_packed(&b_packed);
            assert_eq!(
                c.shape(),
                &[height, width],
                "shape @ ({height},{width},{depth})"
            );
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    v.to_f64().abs() <= tol,
                    "({height},{width},{depth})[{i}]: {} vs 0.0 (tol={tol})",
                    v.to_f64()
                );
            }
            let mut into_tensor = Tensor::<Scalar::SpatialResult>::try_full(
                &[height, width],
                Scalar::SpatialResult::default(),
            )
            .unwrap();
            a.try_angulars_packed_into(&b_packed, &mut into_tensor)
                .unwrap();
            assert_eq!(
                c.as_slice(),
                into_tensor.as_slice(),
                "_into(Tensor) @ ({height},{width},{depth})"
            );
            let mut into_span_buf = Tensor::<Scalar::SpatialResult>::try_full(
                &[height, width],
                Scalar::SpatialResult::default(),
            )
            .unwrap();
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
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let c = a.euclideans_packed(&b_packed);
            assert_eq!(
                c.shape(),
                &[height, width],
                "shape @ ({height},{width},{depth})"
            );
            for (i, &v) in c.as_slice().iter().enumerate() {
                assert!(
                    v.to_f64().abs() <= tol,
                    "({height},{width},{depth})[{i}]: {} vs 0.0 (tol={tol})",
                    v.to_f64()
                );
            }
            let mut into_tensor = Tensor::<Scalar::SpatialResult>::try_full(
                &[height, width],
                Scalar::SpatialResult::default(),
            )
            .unwrap();
            a.try_euclideans_packed_into(&b_packed, &mut into_tensor)
                .unwrap();
            assert_eq!(
                c.as_slice(),
                into_tensor.as_slice(),
                "_into(Tensor) @ ({height},{width},{depth})"
            );
            let mut into_span_buf = Tensor::<Scalar::SpatialResult>::try_full(
                &[height, width],
                Scalar::SpatialResult::default(),
            )
            .unwrap();
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
    fn check_dots_packed_parallel<Scalar: TestableType + Dots + Send + Sync>()
    where
        Scalar::Accumulator: PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        for &(height, width, depth) in DIMS {
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
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
    fn check_angulars_packed_parallel<Scalar: TestableType + Angulars + Send + Sync>()
    where
        Scalar::SpatialResult: PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        for &(height, width, depth) in DIMS {
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
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
    fn check_euclideans_packed_parallel<Scalar: TestableType + Euclideans + Send + Sync>()
    where
        Scalar::SpatialResult: PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        for &(height, width, depth) in DIMS {
            let a = Tensor::<Scalar>::try_full(&[height, depth], Scalar::one()).unwrap();
            let b = Tensor::<Scalar>::try_full(&[width, depth], Scalar::one()).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
            let serial = a.euclideans_packed(&b_packed);
            let parallel = a.euclideans_packed_parallel(&b_packed, &mut pool);
            assert_eq!(
                serial.as_slice(),
                parallel.as_slice(),
                "serial != parallel @ ({height},{width},{depth})"
            );
            // Exercise _parallel_into on a span.
            let mut into_span = Tensor::<Scalar::SpatialResult>::try_full(
                &[height, width],
                Scalar::SpatialResult::default(),
            )
            .unwrap();
            a.try_euclideans_packed_parallel_into(&b_packed, &mut into_span.span(), &mut pool)
                .unwrap();
            assert_eq!(
                serial.as_slice(),
                into_span.as_slice(),
                "_parallel_into(span)"
            );
        }
    }

    #[cfg(feature = "parallel")]
    fn check_hammings_packed_parallel_u1() {
        init_thread();
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        for &(height, width, depth) in DIMS {
            let depth = align_depth::<u1x8>(depth); // logical bit-count, multiple of 8
            let a = Tensor::<u1x8>::try_full(&[height, depth], u1x8(0xFF)).unwrap();
            let b = Tensor::<u1x8>::try_full(&[width, depth], u1x8(0xFF)).unwrap();
            let b_packed = PackedMatrix::try_pack(&b).unwrap();
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
            assert_eq!(
                serial.as_slice(),
                into_span.as_slice(),
                "hammings _parallel_into(span)"
            );

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
    fn check_symmetric_parallel<Scalar: TestableType + Dots + Angulars + Euclideans + Send + Sync>()
    where
        Scalar::Accumulator:
            Clone + Default + Copy + PartialEq + core::fmt::Debug + Send + Sync + 'static,
        <Scalar as Angulars>::SpatialResult:
            Clone + Default + Copy + PartialEq + core::fmt::Debug + Send + Sync,
        <Scalar as Euclideans>::SpatialResult:
            Clone + Default + Copy + PartialEq + core::fmt::Debug + Send + Sync,
    {
        init_thread();
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
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
            let mut into_span = Tensor::<Scalar::Accumulator>::try_full(
                &[num_vectors, num_vectors],
                Scalar::Accumulator::default(),
            )
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

    #[cfg(feature = "parallel")]
    fn check_symmetric_parallel_u1() {
        init_thread();
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
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
            let mut into_span_h =
                Tensor::<u32>::try_full(&[num_vectors, num_vectors], 0u32).unwrap();
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
            let mut into_span_j =
                Tensor::<f32>::try_full(&[num_vectors, num_vectors], 0.0f32).unwrap();
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

    #[test]
    #[cfg(feature = "parallel")]
    fn packed_parallel() {
        check_dots_packed_parallel::<f32>();
        check_dots_packed_parallel::<bf16>();
        check_angulars_packed_parallel::<f32>();
        check_euclideans_packed_parallel::<f32>();
        check_hammings_packed_parallel_u1();
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn symmetric_parallel() {
        check_symmetric_parallel::<f32>();
        check_symmetric_parallel_u1();
    }

    /// Compare only the upper-triangle elements of two NxN buffers; symmetric kernels
    /// do not write the lower triangle.
    fn assert_upper_triangle_eq<X: Copy + PartialEq + core::fmt::Debug>(
        left: &[X],
        right: &[X],
        n: usize,
        tag: &str,
    ) {
        for i in 0..n {
            for j in i..n {
                let index = i * n + j;
                assert_eq!(left[index], right[index], "{tag}[{i},{j}]");
            }
        }
    }

    fn check_dots_symmetric<Scalar: TestableType + Dots>()
    where
        Scalar::Accumulator:
            Clone + Default + Copy + FloatLike + PartialEq + core::fmt::Debug + 'static,
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
            let mut into_tensor = Tensor::<Scalar::Accumulator>::try_full(
                &[num_vectors, num_vectors],
                Scalar::Accumulator::default(),
            )
            .unwrap();
            vectors.try_dots_symmetric_into(&mut into_tensor).unwrap();
            assert_upper_triangle_eq(
                gram_matrix.as_slice(),
                into_tensor.as_slice(),
                num_vectors,
                "dots_symmetric_into(Tensor)",
            );
            let mut into_span_buf = Tensor::<Scalar::Accumulator>::try_full(
                &[num_vectors, num_vectors],
                Scalar::Accumulator::default(),
            )
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

    fn check_angulars_symmetric<Scalar: TestableType + Angulars>()
    where
        Scalar::SpatialResult:
            Clone + Default + Copy + FloatLike + PartialEq + core::fmt::Debug + 'static,
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
            vectors
                .try_angulars_symmetric_into(&mut into_tensor)
                .unwrap();
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
        Scalar::SpatialResult:
            Clone + Default + Copy + FloatLike + PartialEq + core::fmt::Debug + 'static,
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
            vectors
                .try_euclideans_symmetric_into(&mut into_tensor)
                .unwrap();
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

    #[test]
    fn binary_packed_u1() {
        init_thread();
        let a = Tensor::<u1x8>::try_full(&[4, 64], u1x8(0xFF)).unwrap();
        let b = Tensor::<u1x8>::try_full(&[16, 64], u1x8(0xFF)).unwrap();
        let b_packed = PackedMatrix::try_pack(&b).unwrap();

        let c = a.dots_packed(&b_packed);
        assert_eq!(c.shape(), &[4, 16]);
        assert_eq!(c.as_slice()[0], 64);

        let c_h = a.hammings_packed(&b_packed);
        assert_eq!(c_h.shape(), &[4, 16]);
        assert_eq!(c_h.as_slice()[0], 0);
        let mut c_h_into = Tensor::<u32>::try_full(&[4, 16], 0u32).unwrap();
        a.try_hammings_packed_into(&b_packed, &mut c_h_into.span())
            .unwrap();
        assert_eq!(c_h.as_slice(), c_h_into.as_slice());

        let c_j = a.jaccards_packed(&b_packed);
        assert_eq!(c_j.shape(), &[4, 16]);
        assert!(c_j.as_slice()[0].abs() < 1e-5);
        let mut c_j_into = Tensor::<f32>::try_full(&[4, 16], 0.0f32).unwrap();
        a.try_jaccards_packed_into(&b_packed, &mut c_j_into.span())
            .unwrap();
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
        a.view()
            .try_hammings_symmetric_into(&mut gram_h_into.span())
            .unwrap();
        assert_upper_triangle_eq(gram_h.as_slice(), gram_h_into.as_slice(), 4, "hammings");

        let gram_j = a.try_jaccards_symmetric().unwrap();
        assert_eq!(gram_j.shape(), &[4, 4]);
        assert!(gram_j.as_slice()[0].abs() < 1e-5);
        let mut gram_j_into = Tensor::<f32>::try_full(&[4, 4], 0.0f32).unwrap();
        a.view()
            .try_jaccards_symmetric_into(&mut gram_j_into.span())
            .unwrap();
        assert_upper_triangle_eq(gram_j.as_slice(), gram_j_into.as_slice(), 4, "jaccards");
    }
}
