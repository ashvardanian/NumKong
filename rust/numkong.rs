//! # NumKong - Hardware-Accelerated Numerics
//!
//! Provides SIMD-accelerated distance metrics, elementwise operations, and tensor algebra
//! targeting ARM NEON/SVE/SME and x86 AVX2/AVX-512 backends.
//!
//! ## Modules
//!
//! - [`types`]: Mixed-precision scalar types (`f16`, `bf16`, FP8, packed integers) and [`FloatLike`] trait
//! - [`spatial`]: Dot products, angular (cosine), and Euclidean distances
//! - [`each`]: Elementwise operations and trigonometry
//! - [`reduce`]: Statistical reductions (moments, min/max)
//! - [`set`]: Binary set similarity (Hamming, Jaccard)
//! - [`probability`]: Probability divergences (KL, JS)
//! - [`curved`]: Curved metric spaces (Bilinear, Mahalanobis)
//! - [`mesh`]: Mesh alignment (Kabsch, Umeyama, RMSD)
//! - [`geospatial`]: Geospatial distances (Haversine, Vincenty)
//! - [`sparse`]: Sparse set operations
//! - [`cast`]: Type casting between scalar formats
//! - [`capabilities`]: Runtime SIMD feature detection
//! - [`matrix`]: Batch matrix operations (GEMM, packed spatial distances)
//! - [`tensor`]: N-dimensional tensors with elementwise/reduction operations
//!
//! ## Implemented operations include:
//!
//! * Euclidean (L2), inner product, and angular (cosine) spatial distances.
//! * Hamming and Jaccard binary distances.
//! * Kullback-Leibler divergence and Jensen-Shannon distance.
//! * Elementwise scale, sum, blend, and FMA operations.
//! * Trigonometric functions (sin, cos, atan).
//! * Type casting between all scalar formats.
//! * Matrix multiplication with pre-packing (GEMM).
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
//! numkong::capabilities::configure_thread();
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
//! The `SpatialSimilarity` trait (combining `Dot`, `Angular`, `Euclidean`) covers:
//!
//! - `dot(a, b)`: Computes dot product between two slices.
//! - `angular(a, b)` / `cosine(a, b)`: Computes angular distance (1 − cosine similarity).
//! - `sqeuclidean(a, b)`: Computes squared Euclidean distance.
//! - `euclidean(a, b)`: Computes Euclidean distance.
//!
//! The `BinarySimilarity` trait (combining `Hamming`, `Jaccard`) covers:
//!
//! - `hamming(a, b)`: Computes Hamming distance between two slices.
//! - `jaccard(a, b)`: Computes Jaccard distance between two slices.
//!
//! The `ProbabilitySimilarity` trait (combining `KullbackLeibler`, `JensenShannon`) covers:
//!
//! - `jensenshannon(a, b)`: Computes Jensen-Shannon distance.
//! - `kullbackleibler(a, b)`: Computes Kullback-Leibler divergence.
//!
//! The elementwise traits (including `EachScale`, `EachSum`, `EachBlend`, `EachFMA`) covers:
//!
//! - `scale(a, alpha, beta, result)`: Element-wise `result[i] = α × a[i] + β`.
//! - `sum(a, b, result)`: Element-wise `result[i] = a[i] + b[i]`.
//! - `blend(a, b, alpha, beta, result)`: Blend `result[i] = α × a[i] + β × b[i]`.
//! - `fma(a, b, c, alpha, beta, result)`: Fused multiply-add `result[i] = α × a[i] × b[i] + β × c[i]`.
//!
//! The `Trigonometry` trait (combining `EachSin`, `EachCos`, `EachATan`) covers:
//!
//! - `sin(input, result)`: Element-wise sine.
//! - `cos(input, result)`: Element-wise cosine.
//! - `atan(input, result)`: Element-wise arctangent.
//!
//! Additional traits: `VDot`, `Roots`, `SparseIntersect`, `SparseDot`.
//!
#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

// Domain modules
pub mod capabilities;
pub mod cast;
pub mod curved;
pub mod each;
pub mod geospatial;
pub mod maxsim;
pub mod mesh;
pub mod probability;
pub mod reduce;
pub mod set;
pub mod sparse;
pub mod spatial;

// Containers
pub mod matrix;
pub mod tensor;
pub mod types;
pub mod vector;

// Re-export scalar types at crate root
pub use types::{
    bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, i4x2, is_close, u1x8, u4x2, DimMut,
    DimRef, FloatConvertible, FloatLike, NumberLike, StorageElement,
};

// Re-export spatial traits
pub use spatial::{Angular, Dot, Euclidean, Roots, SpatialSimilarity, VDot};

// Re-export set traits
pub use set::{BinarySimilarity, Hamming, Jaccard};

// Re-export probability traits
pub use probability::{JensenShannon, KullbackLeibler, ProbabilitySimilarity};

// Re-export elementwise and trig traits
pub use each::{
    AllCloseOps, BlendOps, EachATan, EachBlend, EachCos, EachFMA, EachScale, EachSin, EachSum,
    FmaOps, ScaleOps, SumOps, TrigAtanOps, TrigCosOps, TrigSinOps, Trigonometry,
};

// Re-export reduction traits
pub use reduce::{
    BitwiseReductions, MinMaxOps, MomentsOps, ReduceMinMax, ReduceMoments, Reductions, SumSqToF64,
};

// Re-export curved metric traits
pub use curved::{Bilinear, Mahalanobis};

// Re-export mesh alignment
pub use mesh::{MeshAlignment, MeshAlignmentResult};

// Re-export geospatial
pub use geospatial::{Geospatial, Haversine, Vincenty};

// Re-export sparse
pub use sparse::{SparseDot, SparseIntersect};

// Re-export cast operations
pub use cast::{cast, CastDtype, CastOps};

// Re-export capabilities
pub use capabilities::cap;
pub use capabilities::{available, configure_thread, uses_dynamic_dispatch};

// Re-export tensor types
pub use tensor::{
    Allocator, AxisIterator, AxisIteratorMut, CopyFrom, Fill, Global, Matrix, MatrixSpan,
    MatrixView, MinMaxResult, RangeStep, SliceArg, SliceRange, SliceSpec, Tensor, TensorDims,
    TensorError, TensorIterator, TensorMut, TensorRef, TensorSpan, TensorSpanDims,
    TensorSpanIterator, TensorView, TensorViewDims, TensorViewIterator, DEFAULT_MAX_RANK,
    SIMD_ALIGNMENT,
};

// Re-export matrix types
pub use matrix::{
    Angulars, Dots, Euclideans, Hammings, Jaccards, PackedMatrix, SymmetricAngulars, SymmetricDots,
    SymmetricEuclideans, SymmetricHammings, SymmetricJaccards,
};

// Re-export vector types
pub use vector::{
    Vector, VectorIndex, VectorIterator, VectorSpan, VectorSpanIterator, VectorView,
    VectorViewIterator,
};

// Re-export maxsim types
pub use maxsim::{MaxSim, MaxSimPackedMatrix};

// region: Tests

#[cfg(test)]
mod tests {
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
        capabilities::configure_thread();
        let queries = Tensor::<f32>::try_full(&[4, 16], 1.0).unwrap();
        let documents = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();
        let queries_view = queries.view();
        let docs_view = documents.view();
        let queries_packed = MaxSimPackedMatrix::try_pack(&queries_view).unwrap();
        let docs_packed = MaxSimPackedMatrix::try_pack(&docs_view).unwrap();
        assert_eq!(queries_packed.dims(), (4, 16));
        assert_eq!(docs_packed.dims(), (8, 16));
        let score = queries_packed.score(&docs_packed);
        assert!(
            score.is_finite(),
            "MaxSim score must be finite, got {score}"
        );
    }

    #[test]
    fn tensor_dots_smoke() {
        capabilities::configure_thread();
        let queries = Tensor::<f32>::try_full(&[2, 4], 1.0).unwrap();
        let targets = Tensor::<f32>::try_full(&[3, 4], 1.0).unwrap();
        let packed_targets = PackedMatrix::try_pack(&targets).unwrap();
        let products = queries.dots_packed(&packed_targets);
        assert_eq!(products.shape(), &[2, 3]);
        assert!((products.as_slice()[0] - 4.0).abs() < 0.01);
    }
}

// endregion: Tests

// region: WASM Runtime Tests

/// WASM runtime integration tests using Wasmtime
/// These tests validate that WASI builds work correctly with standalone runtimes
#[cfg(all(test, feature = "wasm-runtime"))]
mod wasm_runtime_tests {
    use std::fs;
    use std::path::Path;
    use wasmtime::{
        Config, Engine, Extern, ExternType, Linker, Memory, MemoryType, Module, SharedMemory, Store,
    };
    use wasmtime_wasi::WasiCtx;

    fn resolve_wasi_module() -> Option<String> {
        if let Ok(path) = std::env::var("NK_WASI_MODULE") {
            if Path::new(&path).exists() {
                return Some(path);
            }
        }
        if Path::new("build-wasi/nk_test.wasm").exists() {
            Some("build-wasi/nk_test.wasm".to_string())
        } else if Path::new("build-wasi/test.wasm").exists() {
            Some("build-wasi/test.wasm".to_string())
        } else {
            None
        }
    }

    /// Test that WASI WASM module can be loaded and executed with Wasmtime
    /// This validates the dual-path capability detection (EM_ASM vs WASI imports)
    #[test]
    fn wasi_with_wasmtime() -> wasmtime::Result<()> {
        // Check if WASI build exists
        let Some(wasm_path) = resolve_wasi_module() else {
            eprintln!("WASI build not found. Run:");
            eprintln!("  export WASI_SDK_PATH=~/wasi-sdk");
            eprintln!("  cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake -DNK_BUILD_TEST=ON");
            eprintln!("  cmake --build build-wasi --target nk_test");
            return Ok(()); // Skip test if build doesn't exist
        };

        println!("Loading WASI module from {}", wasm_path);

        let mut config = Config::new();
        config.wasm_simd(true);
        config.wasm_relaxed_simd(true);
        config.wasm_threads(true);
        config.shared_memory(true);

        // Create Wasmtime engine and linker
        let engine = Engine::new(&config)?;
        let mut linker = Linker::new(&engine);

        // Create WASI context (Wasmtime 41+ API)
        // Don't inherit_args() — cargo's test filter args would confuse the WASM test binary.
        let wasi = WasiCtx::builder().inherit_stdio().inherit_env().build_p1();
        let mut store = Store::new(&engine, wasi);

        // Add WASI support (Wasmtime 41+ requires p1 module)
        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s| s)?;

        // Provide capability detection imports (required for WASI build)
        // These functions are called from nk_capabilities_v128relaxed_() in C code
        linker.func_wrap("env", "nk_has_v128", || -> i32 {
            // Return 1 (true) - assume SIMD128 is available in Wasmtime
            println!("  nk_has_v128() called from WASM -> returning 1");
            1
        })?;

        linker.func_wrap("env", "nk_has_relaxed", || -> i32 {
            // Return 1 (true) - assume Relaxed SIMD is available in Wasmtime
            println!("  nk_has_relaxed() called from WASM -> returning 1");
            1
        })?;

        linker.func_wrap("wasi", "thread-spawn", |_start_arg: i32| -> i32 { -1 })?;

        // Load WASM module
        let wasm_bytes = fs::read(&wasm_path)?;
        let module = Module::new(&engine, wasm_bytes)?;

        for import in module.imports() {
            if import.module() != "env" || import.name() != "memory" {
                continue;
            }

            let ExternType::Memory(memory_ty) = import.ty() else {
                continue;
            };

            let minimum = u32::try_from(memory_ty.minimum()).map_err(|_| {
                wasmtime::Error::msg(format!(
                    "memory minimum {} does not fit in u32",
                    memory_ty.minimum()
                ))
            })?;
            let maximum = memory_ty
                .maximum()
                .ok_or_else(|| wasmtime::Error::msg("shared memory import is missing a maximum"))?;
            let maximum = u32::try_from(maximum).map_err(|_| {
                wasmtime::Error::msg(format!("memory maximum {maximum} does not fit in u32"))
            })?;

            if memory_ty.is_shared() {
                let memory = SharedMemory::new(&engine, MemoryType::shared(minimum, maximum))?;
                linker.define(&store, "env", "memory", Extern::from(memory))?;
            } else {
                let memory = Memory::new(&mut store, MemoryType::new(minimum, Some(maximum)))?;
                linker.define(&store, "env", "memory", Extern::from(memory))?;
            }
        }

        // Instantiate module
        println!("Instantiating WASM module...");
        let instance = linker.instantiate(&mut store, &module)?;

        // Get _start entry point (WASI convention; exit code comes via proc_exit trap)
        let start = instance.get_typed_func::<(), ()>(&mut store, "_start")?;

        // Run tests — _start calls exit(0) which triggers proc_exit(0).
        // Wasmtime reports proc_exit as an I32Exit error; exit code 0 means success.
        println!("Running WASM tests...");
        match start.call(&mut store, ()) {
            Ok(()) => {}
            Err(e) => {
                if let Some(exit) = e.downcast_ref::<wasmtime_wasi::I32Exit>() {
                    assert_eq!(exit.0, 0, "WASI tests failed with exit code {}", exit.0);
                } else {
                    return Err(e);
                }
            }
        }

        println!("WASM tests completed successfully");
        Ok(())
    }

    /// Test capability detection mechanism works in WASI environment
    #[test]
    fn capability_imports() -> wasmtime::Result<()> {
        println!("Testing capability import mechanism...");

        // Create minimal engine for import testing
        let engine = Engine::default();
        let mut linker = Linker::<()>::new(&engine);

        // Test that we can define the required imports
        linker.func_wrap("env", "nk_has_v128", || -> i32 { 1 })?;
        linker.func_wrap("env", "nk_has_relaxed", || -> i32 { 0 })?;

        println!("  ✓ Capability imports defined successfully");

        Ok(())
    }
}

// endregion: WASM Runtime Tests
