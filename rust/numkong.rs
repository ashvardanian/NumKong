//! # NumKong - Hardware-Accelerated Numerics
//!
//! Provides SIMD-accelerated distance metrics, elementwise operations, and tensor algebra targeting
//! ARM NEON/SVE/SME and x86 AVX2/AVX-512 backends.
//!
//! ## Modules
//!
//! - [`types`]: Mixed-precision scalar types (`f16`, `bf16`, FP8, packed integers) and
//!   [`FloatLike`] trait
//! - [`dot`]: Real and complex dot products
//! - [`spatial`]: Angular, also called cosine, and Euclidean distances
//! - [`each`]: Elementwise operations — scale, sum, blend, FMA
//! - [`trigonometry`]: Elementwise trigonometry — sin, cos, atan
//! - [`reduce`]: Statistical reductions — moments, min/max
//! - [`scalar`]: Scalar math primitives — square root, reciprocal square root
//! - [`set`]: Binary set similarity — Hamming, Jaccard
//! - [`probability`]: Probability divergences — KL, JS
//! - [`curved`]: Curved metric spaces — Bilinear, Mahalanobis
//! - [`mesh`]: Mesh alignment — Kabsch, Umeyama, RMSD
//! - [`geospatial`]: Geospatial distances — Haversine, Vincenty
//! - [`sparse`]: Sparse set operations
//! - [`mod@cast`]: Type casting between scalar formats
//! - [`capabilities`]: Devices and the capabilities they detect, compile and enable
//! - [`dots`]: Batched GEMM over pre-packed matrices
//! - [`spatials`]: Batched spatial distances — angular, Euclidean — over pre-packed matrices
//! - [`sets`]: Batched binary/set metrics — Hamming, Jaccard — over pre-packed matrices
//! - [`maxsim`]: Late-interaction MaxSim scoring over pre-packed matrices
//! - [`attention`]: Ragged scaled-dot-product attention with a packed KV-cache
//! - [`tensor`]: N-dimensional tensors with elementwise/reduction operations
//! - [`vector`]: Strided 1-D vector views
//!
//! ## Implemented operations include:
//!
//! - Euclidean L2, inner-product, and angular cosine spatial distances.
//! - Hamming and Jaccard binary distances.
//! - Kullback-Leibler divergence and Jensen-Shannon distance.
//! - Elementwise scale, sum, blend, and FMA operations.
//! - Trigonometric functions — sin, cos, atan.
//! - Type casting between all scalar formats.
//! - Matrix multiplication with pre-packing, i.e. GEMM.
//!
//! ## Example
//!
//! ```rust
//! use numkong::{Dot, Angular, Euclidean};
//!
//! let a = &[1.0_f32, 2.0, 3.0];
//! let b = &[4.0_f32, 5.0, 6.0];
//!
//! let dot_product = f32::dot(a, b);
//! let angular_dist = f32::angular(a, b);
//! let l2sq_dist = f32::sqeuclidean(a, b);
//!
//! // Enable AMX and other platform-specific SIMD features
//! let cpu = numkong::Device::cpu();
//! cpu.configure_thread(cpu.capabilities_enabled().unwrap()).unwrap();
//! ```
//!
//! ## Mixed Precision Support
//!
//! ```rust
//! use numkong::{Angular, f16, bf16};
//!
//! // Work with half-precision floats
//! let half_a: Vec<f16> = vec![1.0, 2.0, 3.0].iter().map(|&x| f16::from_f32(x)).collect();
//! let half_b: Vec<f16> = vec![4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x)).collect();
//! let half_angular_dist = f16::angular(&half_a, &half_b);
//!
//! // Work with brain floats
//! let brain_a: Vec<bf16> = vec![1.0, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
//! let brain_b: Vec<bf16> = vec![4.0, 5.0, 6.0].iter().map(|&x| bf16::from_f32(x)).collect();
//! let brain_angular_dist = bf16::angular(&brain_a, &brain_b);
//!
//! // Direct bit manipulation
//! let half = f16::from_f32(3.14);
//! let bits = half.0; // Access raw u16 representation
//! let reconstructed = f16(bits);
//! ```
//!
//! ## Traits
//!
//! The `SpatialSimilarity` trait, combining `Dot`, `Angular`, `Euclidean`, covers:
//!
//! - `dot(a, b)`: Computes dot product between two slices.
//! - `angular(a, b)` / `cosine(a, b)`: Computes angular distance (1 − cosine similarity).
//! - `sqeuclidean(a, b)`: Computes squared Euclidean distance.
//! - `euclidean(a, b)`: Computes Euclidean distance.
//!
//! The `BinarySimilarity` trait, combining `Hamming`, `Jaccard`, covers:
//!
//! - `hamming(a, b)`: Computes Hamming distance between two slices.
//! - `jaccard(a, b)`: Computes Jaccard distance between two slices.
//!
//! The `ProbabilitySimilarity` trait, combining `KullbackLeibler`, `JensenShannon`, covers:
//!
//! - `jensenshannon(a, b)`: Computes Jensen-Shannon distance.
//! - `kullbackleibler(a, b)`: Computes Kullback-Leibler divergence.
//!
//! The elementwise traits, including `EachScale`, `EachSum`, `EachBlend`, `EachFma`, cover:
//!
//! - `scale(a, alpha, beta, result)`: Element-wise `result[i] = α × a[i] + β`.
//! - `sum(a, b, result)`: Element-wise `result[i] = a[i] + b[i]`.
//! - `blend(a, b, alpha, beta, result)`: Blend `result[i] = α × a[i] + β × b[i]`.
//! - `fma(a, b, c, alpha, beta, result)`: FMA `result[i] = α × a[i] × b[i] + β × c[i]`.
//!
//! The `Trigonometry` trait, combining `TrigSin`, `TrigCos`, `TrigAtan`, covers:
//!
//! - `sin(input, result)`: Element-wise sine.
//! - `cos(input, result)`: Element-wise cosine.
//! - `atan(input, result)`: Element-wise arctangent.
//!
//! Additional traits: `VDot` for complex dot products, `Roots` for scalar square root and
//! reciprocal square root, and `SparseDot` / `SparseIntersect` for sparse set operations.
//!
//! File: rust/numkong.rs
//! Author: Ash Vardanian
#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]
// docs.rs builds with `--cfg docsrs`, per Cargo.toml; this enables the "Available on feature …"
// badges on feature-gated items. It's a nightly-only feature, gated behind `docsrs` so stable
// builds ignore it.
#![cfg_attr(docsrs, feature(doc_cfg))]

// Domain modules
pub mod attention;
pub mod capabilities;
pub mod cast;
pub mod curved;
pub mod dot;
pub mod each;
pub mod geospatial;
pub mod maxsim;
pub mod mesh;
pub mod probability;
pub mod reduce;
pub mod scalar;
pub mod set;
pub mod sparse;
pub mod spatial;
pub mod trigonometry;

// Containers
pub mod dots;
pub mod sets;
pub mod spatials;
pub mod tensor;
pub mod types;
pub mod vector;

// Re-export scalar types at crate root
pub use types::{
    bf16, bf16c, e2m1x2, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, i4x2, is_close, u1x8, u4x2, DimMut, DimRef,
    FloatConvertible, FloatLike, NumberLike, StorageElement, Ue4m3, Ue8m0,
};

// Re-export scalar-math traits
pub use scalar::Roots;

// Re-export dot / spatial traits
pub use dot::{Dot, VDot};
pub use spatial::{Angular, Euclidean, SpatialSimilarity};

// Re-export set traits
pub use set::{BinarySimilarity, Hamming, Jaccard};

// Re-export probability traits
pub use probability::{JensenShannon, KullbackLeibler, ProbabilitySimilarity};

// Re-export elementwise and trig traits
pub use each::{
    AllCloseOps, BlendOps, EachBlend, EachFma, EachRmsNorm, EachScale, EachSum, EachSwiGlu, FmaOps, ScaleOps, SumOps,
};

pub use reduce::{BitwiseReductionsOps, MinMaxOps, MomentsOps, ReduceMinMax, ReduceMoments, Reductions};
pub use trigonometry::{TrigAtan, TrigAtanOps, TrigCos, TrigCosOps, TrigSin, TrigSinOps, Trigonometry};

// Re-export curved metric traits
pub use curved::{Bilinear, Mahalanobis};

// Re-export mesh alignment
pub use mesh::{MeshAlignment, MeshAlignmentResult};

// Re-export geospatial
pub use geospatial::{Geospatial, Haversine, Vincenty};

// Re-export sparse
pub use sparse::{SparseDot, SparseIntersect};

// Re-export cast operations
pub use cast::{cast, CastDType, CastOps, DenseToScaledOps};

// Re-export block-scaled formats and casts
pub use cast::{
    BlockScaledDescriptor, BlockScaledFormat, Mxfp4, Mxfp6E2m3, Mxfp6E3m2, Mxfp8E4m3, Mxfp8E5m2, Mxint8, Nvfp4,
};

// Re-export capabilities
pub use capabilities::{Capabilities, Capability, Device, DeviceKind, Status};

// Re-export tensor types
pub use tensor::{
    AllocError, Allocator, AxisIterator, AxisIteratorMut, CopyFrom, Fill, Global, Matrix, MatrixSpan, MatrixView,
    MinMaxResult, RangeStep, ScaledTensor, ScaledTensorSpan, ScaledTensorView, SliceArg, SliceRange, SliceSpec, Tensor,
    TensorDims, TensorError, TensorIterator, TensorMut, TensorRef, TensorSpan, TensorSpanDims, TensorSpanIterator,
    TensorView, TensorViewDims, TensorViewIterator, DEFAULT_MAX_RANK, SIMD_ALIGNMENT,
};

// Re-export batched GEMM types
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub use dots::DotsPackedParallelOps;
pub use dots::{Dots, DotsPackedMatrix, DotsPackedOps, SymmetricDotsOps};

// Re-export batched spatial-distance types
pub use spatials::{
    Angulars, AngularsPackedOps, Euclideans, EuclideansPackedOps, SymmetricAngularsOps, SymmetricEuclideansOps,
};
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub use spatials::{AngularsPackedParallelOps, EuclideansPackedParallelOps};

// Re-export batched binary/set-metric types
pub use sets::{Hammings, HammingsPackedOps, Jaccards, JaccardsPackedOps, SymmetricHammingsOps, SymmetricJaccardsOps};
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub use sets::{HammingsPackedParallelOps, JaccardsPackedParallelOps};

// Re-export vector types
pub use vector::{Vector, VectorIndex, VectorIterator, VectorSpan, VectorSpanIterator, VectorView, VectorViewIterator};

// Re-export maxsim and attention types
pub use attention::{Attention, AttentionPackedMatrix, AttentionRope};

pub use maxsim::{MaxSim, MaxSimPackedMatrix};

/// Prelude: `use numkong::prelude::*;` brings the core containers and every extension trait into
/// scope, so the `.sum_all` / `.dots_packed` / `.sin` methods light up without importing each
/// trait by name.
pub mod prelude {
    pub use crate::{
        AllCloseOps, AngularsPackedOps, AttentionRope, BitwiseReductionsOps, BlendOps, CastOps, DenseToScaledOps,
        DotsPackedMatrix, DotsPackedOps, EachRmsNorm, EachSwiGlu, EuclideansPackedOps, FmaOps, HammingsPackedOps,
        JaccardsPackedOps, Matrix, MinMaxOps, MomentsOps, Reductions, ScaleOps, ScaledTensor, SumOps, Tensor,
        TensorMut, TensorRef, TensorSpan, TensorView, TrigAtanOps, TrigCosOps, TrigSinOps, Vector, VectorSpan,
        VectorView,
    };

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    pub use crate::{
        AngularsPackedParallelOps, DotsPackedParallelOps, EuclideansPackedParallelOps, HammingsPackedParallelOps,
        JaccardsPackedParallelOps,
    };
}

// region: Tests

#[cfg(test)]
mod tests {
    #[cfg(feature = "parallel")]
    use forkunion as fu;

    use super::*;

    #[test]
    fn dot_smoke() {
        let first = [1.0_f32, 2.0, 3.0];
        let second = [4.0_f32, 5.0, 6.0];
        assert!((<f32 as Dot>::dot(&first, &second).unwrap() - 32.0).abs() < 0.01);
    }

    #[test]
    fn angular_smoke() {
        let first = [1.0_f32, 0.0];
        let second = [0.0_f32, 1.0];
        // Orthogonal vectors → angular distance = 1.0
        assert!((f32::angular(&first, &second).unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn euclidean_smoke() {
        let first = [0.0_f32, 0.0, 0.0];
        let second = [3.0_f32, 4.0, 0.0];
        assert!((f32::euclidean(&first, &second).unwrap() - 5.0).abs() < 0.1);
    }

    #[test]
    fn maxsim_smoke() {
        capabilities::configure_cpu_thread().unwrap();
        let queries = Tensor::<f32>::full(&[4, 16], 1.0).unwrap();
        let documents = Tensor::<f32>::full(&[8, 16], 1.0).unwrap();
        let queries_view = queries.view();
        let docs_view = documents.view();
        let queries_packed = MaxSimPackedMatrix::new(&queries_view).unwrap();
        let docs_packed = MaxSimPackedMatrix::new(&docs_view).unwrap();
        assert_eq!(queries_packed.shape(), (4, 16));
        assert_eq!(docs_packed.shape(), (8, 16));
        let score = queries_packed.score(&docs_packed).unwrap();
        assert!(score.is_finite(), "MaxSim score must be finite, got {score}");
    }

    #[test]
    fn attention_smoke() {
        capabilities::configure_cpu_thread().unwrap();
        let (tokens, heads, head_dim) = (24usize, 2usize, 32usize);
        let keys = Tensor::<bf16>::full(&[tokens, heads * head_dim], bf16::from_f32(0.25)).unwrap();
        let values = Tensor::<bf16>::full(&[tokens, heads * head_dim], bf16::from_f32(0.5)).unwrap();
        let offsets = [0u32, 10, 24];

        let kv = AttentionPackedMatrix::new(&keys.view(), &values.view(), head_dim, &offsets).unwrap();
        assert_eq!(kv.segments(), 2);
        assert_eq!(kv.heads(), heads);
        assert_eq!(kv.depth(), head_dim);
        assert_eq!(kv.tokens(), tokens);

        // With constant V, softmax weights sum to 1 → every output equals V's value.
        let outputs = kv.attention(&keys.view(), &offsets, None).unwrap();
        assert_eq!(outputs.shape(), [tokens, heads * head_dim]);
        for &x in outputs.as_slice() {
            assert!((x - 0.5).abs() < 1e-2, "expected 0.5, got {x}");
        }
    }

    #[test]
    fn attention_kv_cache_reuse() {
        capabilities::configure_cpu_thread().unwrap();
        let (heads, head_dim) = (2usize, 32usize);
        let small = Tensor::<bf16>::full(&[10, heads * head_dim], bf16::from_f32(0.25)).unwrap();
        let big = Tensor::<bf16>::full(&[24, heads * head_dim], bf16::from_f32(0.25)).unwrap();
        let small_off = [0u32, 4, 10];
        let big_off = [0u32, 10, 24];

        let mut kv = AttentionPackedMatrix::new(&small.view(), &small.view(), head_dim, &small_off).unwrap();
        let cap0 = kv.capacity();
        assert!(cap0 > 0);

        // Same geometry repacked in place → allocation reused, capacity unchanged.
        kv.pack_into(&small.view(), &small.view(), head_dim, &small_off)
            .unwrap();
        assert_eq!(kv.capacity(), cap0, "same-size repack must reuse the buffer");
        assert_eq!(kv.tokens(), 10);

        // Larger geometry → capacity grows, never shrinks.
        kv.pack_into(&big.view(), &big.view(), head_dim, &big_off).unwrap();
        assert!(kv.capacity() >= cap0, "grow must not shrink capacity");
        assert_eq!(kv.tokens(), 24);
        assert_eq!(kv.segments(), 2);

        // clear() keeps the allocation for the next reuse.
        let cap_big = kv.capacity();
        kv.clear();
        assert_eq!(kv.tokens(), 0);
        assert_eq!(kv.capacity(), cap_big, "clear keeps the allocation");

        kv.pack_into(&small.view(), &small.view(), head_dim, &small_off)
            .unwrap();
        let outputs = kv.attention(&small.view(), &small_off, None).unwrap();
        assert_eq!(outputs.shape(), [10, heads * head_dim]);
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    #[test]
    fn attention_parallel_matches_serial() {
        capabilities::configure_cpu_thread().unwrap();
        let (heads, head_dim) = (4usize, 64usize);
        let lengths = [7u32, 250, 0, 33, 129]; // ragged mix: tiny, sub-panel, pad, odd
        let mut offsets = vec![0u32];
        for length in lengths {
            offsets.push(offsets.last().unwrap() + length);
        }
        let tokens = *offsets.last().unwrap() as usize;

        let keys = Tensor::<bf16>::full(&[tokens, heads * head_dim], bf16::from_f32(0.125)).unwrap();
        let values = Tensor::<bf16>::full(&[tokens, heads * head_dim], bf16::from_f32(0.75)).unwrap();
        let kv = AttentionPackedMatrix::new(&keys.view(), &values.view(), head_dim, &offsets).unwrap();

        let sequential = kv.attention(&keys.view(), &offsets, None).unwrap();
        let topology = fu::Topology::new().unwrap();
        let mut pool = fu::ThreadPool::try_spawn(&topology, 4).unwrap();
        let mut parallel = Tensor::<f32>::full(&[tokens, heads * head_dim], 0.0).unwrap();
        kv.attention_parallel_into(&keys.view(), &offsets, None, &mut parallel, &mut pool)
            .unwrap();

        // Per-task dynamic scheduling must be bit-identical to the single-window run:
        // tasks are independent and write disjoint rows.
        for (index, (a, b)) in sequential.as_slice().iter().zip(parallel.as_slice()).enumerate() {
            assert!(a.to_bits() == b.to_bits(), "mismatch at {index}: {a} vs {b}");
        }
    }

    /// Exercises the Wave 2 attention surface: the `pack_size` query, the `new` / `attention` /
    /// `attention_parallel` entry points, and the allocating `new_parallel` / `attention_parallel`.
    /// Parallel packing must reproduce the serial blob byte-for-byte, and every compute path must
    /// agree bit-for-bit.
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    #[test]
    fn attention_capabilities_symmetry() {
        capabilities::configure_cpu_thread().unwrap();
        let (heads, head_dim) = (4usize, 64usize);
        let lengths = [7u32, 250, 0, 33, 129]; // ragged mix incl. a pad segment
        let mut offsets = vec![0u32];
        for length in lengths {
            offsets.push(offsets.last().unwrap() + length);
        }
        let tokens = *offsets.last().unwrap() as usize;

        let keys = Tensor::<bf16>::full(&[tokens, heads * head_dim], bf16::from_f32(0.125)).unwrap();
        let values = Tensor::<bf16>::full(&[tokens, heads * head_dim], bf16::from_f32(0.75)).unwrap();

        // The `pack_size` query must predict the produced blob size exactly.
        let seg_lengths: Vec<u32> = offsets.windows(2).map(|p| p[1] - p[0]).collect();
        let predicted = AttentionPackedMatrix::<bf16>::pack_size(heads, head_dim, &seg_lengths).unwrap();

        // Serial pack via the typed constructor.
        let kv_serial = AttentionPackedMatrix::new(&keys.view(), &values.view(), head_dim, &offsets).unwrap();
        assert_eq!(
            kv_serial.as_bytes().len(),
            predicted,
            "pack_size must predict the blob size"
        );

        // Parallel pack must yield a byte-identical blob.
        let topology = fu::Topology::new().unwrap();
        let mut pool = fu::ThreadPool::try_spawn(&topology, 4).unwrap();
        let kv_parallel =
            AttentionPackedMatrix::new_parallel(&keys.view(), &values.view(), head_dim, &offsets, &mut pool).unwrap();
        assert_eq!(
            kv_serial.as_bytes(),
            kv_parallel.as_bytes(),
            "parallel pack must match serial pack"
        );

        // Serial and parallel attention must agree bit-for-bit.
        let serial = kv_serial.attention(&keys.view(), &offsets, None).unwrap();
        let par_alloc = kv_parallel
            .attention_parallel(&keys.view(), &offsets, None, &mut pool)
            .unwrap();
        for (index, (a, b)) in serial.as_slice().iter().zip(par_alloc.as_slice()).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "attention_parallel mismatch at {index}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn tensor_dots_smoke() {
        capabilities::configure_cpu_thread().unwrap();
        let queries = Tensor::<f32>::full(&[2, 4], 1.0).unwrap();
        let targets = Tensor::<f32>::full(&[3, 4], 1.0).unwrap();
        let packed_targets = DotsPackedMatrix::new(&targets).unwrap();
        let products = queries.dots_packed(&packed_targets).unwrap();
        assert_eq!(products.shape(), &[2, 3]);
        assert!((products.as_slice()[0] - 4.0).abs() < 0.01);
    }
}

// endregion: Tests
