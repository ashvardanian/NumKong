//! Spatial similarity: angular (cosine) and Euclidean distances.
//!
//! This module provides:
//!
//! - [`Angular`]: Cosine distance (1 - cosine similarity)
//! - [`Euclidean`]: Squared L2 distance
//! - [`SpatialSimilarity`]: Blanket trait combining [`Dot`] + `Angular` + `Euclidean`
//!
//! # Accumulator Widening — The Core Value Proposition
//!
//! Every spatial kernel promotes its accumulator to a wider type than the inputs.
//! This is not cosmetic — on long vectors of low-precision data, naive same-width
//! accumulation overflows silently for integers or loses huge amounts of precision
//! for half-floats. NumKong widens systematically so every kernel in this module
//! returns a result that matches a textbook-accurate reference in the wide type:
//!
//! - **`f32` → `f64`**: single-precision inputs accumulate in double precision.
//! - **`f16` → `f32`** and **`bf16` → `f32`**: half-precision norms and distances
//!   accumulate in `f32` rather than clamping at `f16::MAX = 65 504`.
//! - **`i8` → `i32`** (unsigned `u8` → `u32`): byte-level quantised inputs widen
//!   into 32-bit integer accumulators.
//! - **FP8 variants** — `e4m3` / `e5m2` / `e2m3` / `e3m2` — all accumulate in `f32`.
//! - **4-bit packed** `i4x2` / `u4x2` behave like `i8` / `u8` but compute over
//!   double the element count because each byte holds two logical values.
//!
//! # Example — Euclidean Distance
//!
//! ```
//! use numkong::Euclidean;
//!
//! let a = vec![1.0_f32, 2.0, 3.0];
//! let b = vec![4.0_f32, 5.0, 6.0];
//! let dist = f32::euclidean(&a, &b).unwrap();
//! assert!((dist - 27.0_f64.sqrt()).abs() < 1e-6);
//! ```

// Supplies the `Dot` supertrait bound for the `SpatialSimilarity` bundle below.
use crate::dot::Dot;
use crate::types::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u4x2, StorageElement};

#[link(name = "numkong")]
extern "C" {

    // Spatial similarity/distance functions
    fn nk_angular_i8(a: *const i8, b: *const i8, c: usize, d: *mut f32);
    fn nk_angular_u8(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_angular_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_angular_e4m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_e5m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_e2m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_e3m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_angular_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_angular_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    fn nk_sqeuclidean_i8(a: *const i8, b: *const i8, c: usize, d: *mut u32);
    fn nk_sqeuclidean_u8(a: *const u8, b: *const u8, c: usize, d: *mut u32);
    fn nk_sqeuclidean_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_sqeuclidean_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_sqeuclidean_e4m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_sqeuclidean_e5m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_sqeuclidean_e2m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_sqeuclidean_e3m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_sqeuclidean_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_sqeuclidean_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    fn nk_euclidean_i8(a: *const i8, b: *const i8, c: usize, d: *mut f32);
    fn nk_euclidean_u8(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_euclidean_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_euclidean_e4m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_e5m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_e2m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_e3m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_euclidean_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_euclidean_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    // 4-bit integer kernels
    fn nk_sqeuclidean_i4(a: *const u8, b: *const u8, n: usize, result: *mut u32);
    fn nk_sqeuclidean_u4(a: *const u8, b: *const u8, n: usize, result: *mut u32);
    fn nk_euclidean_i4(a: *const u8, b: *const u8, n: usize, result: *mut f32);
    fn nk_euclidean_u4(a: *const u8, b: *const u8, n: usize, result: *mut f32);
    fn nk_angular_i4(a: *const u8, b: *const u8, n: usize, result: *mut f32);
    fn nk_angular_u4(a: *const u8, b: *const u8, n: usize, result: *mut f32);
}

// region: Angular

/// Computes the **angular distance** (cosine distance) between two vectors.
///
/// d = 1 − (a · b) / (‖a‖ × ‖b‖)
///
/// Range: \[0, 2\]. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`, `i4x2`, `u4x2`.
pub trait Angular: StorageElement {
    type Output;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `angular`.
    fn cosine(a: &[Self], b: &[Self]) -> Option<Self::Output> { Self::angular(a, b) }
}

impl Angular for f64 {
    type Output = f64;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for f32 {
    type Output = f64;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for f16 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_f16(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for bf16 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_bf16(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for i8 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_i8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for u8 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_u8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for e4m3 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_e4m3(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for e5m2 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_e5m2(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for e2m3 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_e2m3(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for e3m2 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_angular_e3m2(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Angular for i4x2 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        let element_count = a.len() * 2; // Each i4x2 contains 2 elements
        unsafe {
            nk_angular_i4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                element_count,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Angular for u4x2 {
    type Output = f32;
    fn angular(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        let element_count = a.len() * 2; // Each u4x2 contains 2 elements
        unsafe {
            nk_angular_u4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                element_count,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: Angular

// region: Euclidean

/// Computes the **Euclidean distance** (L2) between two vectors.
///
/// d = √(∑ᵢ (aᵢ − bᵢ)²)
///
/// Range: \[0, ∞). Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`, `i4x2`, `u4x2`.
pub trait Euclidean: StorageElement {
    type SqEuclideanOutput;
    type EuclideanOutput;

    /// Squared Euclidean distance (L2²). Faster than `euclidean` for comparisons.
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput>;

    /// Euclidean distance (L2). True metric distance.
    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput>;
}

impl Euclidean for f64 {
    type SqEuclideanOutput = f64;
    type EuclideanOutput = f64;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for f32 {
    type SqEuclideanOutput = f64;
    type EuclideanOutput = f64;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for f16 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_f16(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_f16(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for bf16 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_bf16(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_bf16(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for i8 {
    type SqEuclideanOutput = u32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0;
        unsafe { nk_sqeuclidean_i8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_i8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for u8 {
    type SqEuclideanOutput = u32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0;
        unsafe { nk_sqeuclidean_u8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_u8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for e4m3 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_e4m3(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_e4m3(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for e5m2 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_e5m2(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_e5m2(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for e2m3 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_e2m3(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_e2m3(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for e3m2 {
    type SqEuclideanOutput = f32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0.0;
        unsafe { nk_sqeuclidean_e3m2(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        unsafe { nk_euclidean_e3m2(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Euclidean for i4x2 {
    type SqEuclideanOutput = u32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0;
        let element_count = a.len() * 2; // Each i4x2 contains 2 elements
        unsafe {
            nk_sqeuclidean_i4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                element_count,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        let element_count = a.len() * 2; // Each i4x2 contains 2 elements
        unsafe {
            nk_euclidean_i4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                element_count,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Euclidean for u4x2 {
    type SqEuclideanOutput = u32;
    type EuclideanOutput = f32;

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<Self::SqEuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::SqEuclideanOutput = 0;
        let element_count = a.len() * 2; // Each u4x2 contains 2 elements
        unsafe {
            nk_sqeuclidean_u4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                element_count,
                &mut result,
            )
        };
        Some(result)
    }

    fn euclidean(a: &[Self], b: &[Self]) -> Option<Self::EuclideanOutput> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::EuclideanOutput = 0.0;
        let element_count = a.len() * 2; // Each u4x2 contains 2 elements
        unsafe {
            nk_euclidean_u4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                element_count,
                &mut result,
            )
        };
        Some(result)
    }
}

// endregion: Euclidean

/// `SpatialSimilarity` bundles spatial distance metrics: Dot, Angular, and Euclidean.
pub trait SpatialSimilarity: Dot + Angular + Euclidean {}
impl<Scalar: Dot + Angular + Euclidean> SpatialSimilarity for Scalar {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u4x2, FloatLike, TestableType};

    /// Test a two-input metric: convert f32 inputs to Scalar, call `op`, compare to `expected`.
    pub(crate) fn check_binary<Scalar, R, F>(a_vals: &[f32], b_vals: &[f32], op: F, expected: f64, label: &str)
    where
        Scalar: FloatLike + TestableType,
        R: FloatLike,
        F: FnOnce(&[Scalar], &[Scalar]) -> Option<R>,
    {
        let a: Vec<Scalar> = a_vals.iter().map(|&v| Scalar::from_f32(v)).collect();
        let b: Vec<Scalar> = b_vals.iter().map(|&v| Scalar::from_f32(v)).collect();
        let result = op(&a, &b).unwrap().to_f64();
        assert_close(
            result,
            expected,
            Scalar::atol(),
            Scalar::rtol(),
            &format!("{}<{}>", label, core::any::type_name::<Scalar>()),
        );
    }

    // region: Angular Distances

    fn check_angular<Scalar>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        Scalar: FloatLike + TestableType + Angular,
        Scalar::Output: FloatLike,
    {
        check_binary::<Scalar, Scalar::Output, _>(a_vals, b_vals, Scalar::angular, expected, "angular");
    }

    #[test]
    fn angular() {
        // angular([1,2,3],[4,5,6]) = 1 - 32/sqrt(14*77) ≈ 0.025368
        let expected = 1.0 - 32.0 / (14.0_f64.sqrt() * 77.0_f64.sqrt());
        check_angular::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        // e2m3 max is 7.5, use values in range
        let expected_e2m3 = 1.0 - 6.0 / (14.0_f64.sqrt() * 3.0_f64.sqrt());
        check_angular::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], expected_e2m3);
        check_angular::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_angular::<i4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], expected_e2m3);
        check_angular::<u4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], expected_e2m3);
    }

    // endregion

    // region: Euclidean Distances

    fn check_sqeuclidean<Scalar>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        Scalar: FloatLike + TestableType + Euclidean,
        Scalar::SqEuclideanOutput: FloatLike,
    {
        check_binary::<Scalar, Scalar::SqEuclideanOutput, _>(
            a_vals,
            b_vals,
            Scalar::sqeuclidean,
            expected,
            "sqeuclidean",
        );
    }

    fn check_euclidean<Scalar>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        Scalar: FloatLike + TestableType + Euclidean,
        Scalar::EuclideanOutput: FloatLike,
    {
        check_binary::<Scalar, Scalar::EuclideanOutput, _>(a_vals, b_vals, Scalar::euclidean, expected, "euclidean");
    }

    #[test]
    fn sqeuclidean() {
        check_sqeuclidean::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<e2m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
        check_sqeuclidean::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 27.0);
    }

    #[test]
    fn euclidean() {
        let expected = 27.0_f64.sqrt();
        check_euclidean::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e2m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        check_euclidean::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected);
        let expected_packed = 54.0_f64.sqrt(); // i4x2 duplicates each value into both nibbles
        check_euclidean::<i4x2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected_packed);
        check_euclidean::<u4x2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], expected_packed);
    }

    // endregion

    // region: Denormal (subnormal) inputs
    //
    // Verify that distance kernels produce correct results when fed IEEE-754
    // denormal (subnormal) values. This guards against FTZ/DAZ silently
    // flushing tiny values to zero.

    #[test]
    fn sqeuclidean_f32_denormals() {
        // Two vectors of denormals: differences are also denormal.
        let a_val = f32::from_bits(0x007F_FFFF); // largest f32 denormal
        let b_val = f32::from_bits(0x003F_FFFF); // half-way f32 denormal
        let diff = (a_val as f64) - (b_val as f64);
        let expected = 3.0 * diff * diff;
        let a = [a_val; 3];
        let b = [b_val; 3];
        let result = f32::sqeuclidean(&a, &b).unwrap();
        assert!(
            result.is_finite(),
            "sqeuclidean<f32> denormal produced non-finite: {result}"
        );
        assert_close(result as f64, expected, 1e-50, 1e-6, "sqeuclidean<f32> denormal");
    }

    #[test]
    fn sqeuclidean_f64_denormals() {
        let a_val = f64::from_bits(0x000F_FFFF_FFFF_FFFF);
        let b_val = f64::from_bits(0x0007_FFFF_FFFF_FFFF);
        let diff = a_val - b_val;
        let expected = 3.0 * diff * diff;
        let a = [a_val; 3];
        let b = [b_val; 3];
        let result = f64::sqeuclidean(&a, &b).unwrap();
        assert!(
            result.is_finite(),
            "sqeuclidean<f64> denormal produced non-finite: {result}"
        );
        assert_close(result, expected, 1e-300, 1e-6, "sqeuclidean<f64> denormal");
    }

    #[test]
    fn angular_f32_denormals() {
        // Two identical denormal vectors: angular distance should be 0 (cosine = 1).
        let d = f32::from_bits(0x007F_FFFF);
        let a = [d, d, d];
        let result = f32::angular(&a, &a).unwrap();
        assert!(
            result.is_finite(),
            "angular<f32> denormal produced non-finite: {result}"
        );
        assert_close(result as f64, 0.0, 1e-4, 0.0, "angular<f32> identical denormals");
    }

    #[test]
    fn angular_f64_denormals() {
        let d = f64::from_bits(0x000F_FFFF_FFFF_FFFF);
        let a = [d, d, d];
        let result = f64::angular(&a, &a).unwrap();
        assert!(
            result.is_finite(),
            "angular<f64> denormal produced non-finite: {result}"
        );
        assert_close(result, 0.0, 1e-9, 0.0, "angular<f64> identical denormals");
    }

    // endregion
}
