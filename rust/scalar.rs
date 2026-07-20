//! Scalar math primitives — square root and reciprocal square root.
//!
//! Small per-scalar kernels that route into NumKong's runtime-dispatched implementations
//! rather than the standard library.
//!
//! This module provides:
//!
//! - [`Roots`]: Scalar square root and reciprocal square root

use crate::types::f16;

#[link(name = "numkong")]
extern "C" {
    // Scalar square-root / reciprocal-square-root, backing the `Roots` trait.
    fn nk_f32_sqrt(x: f32) -> f32;
    fn nk_f32_rsqrt(x: f32) -> f32;
    fn nk_f64_sqrt(x: f64) -> f64;
    fn nk_f64_rsqrt(x: f64) -> f64;
    fn nk_f16_sqrt(x: u16) -> u16;
    fn nk_f16_rsqrt(x: u16) -> u16;
}

/// Scalar square-root and reciprocal-square-root operations backed by NumKong's exported kernels.
///
/// Unlike `f32::sqrt` / `f64::sqrt`, this routes into the hand-tuned NumKong C kernels; on ISAs with
/// a dedicated reciprocal-sqrt, such as `vrsqrte` on NEON or `vrsqrt14` on AVX-512, `rsqrt` uses a single
/// Newton refinement to ~1 ULP — roughly 2-4× faster than computing `1.0 / sqrt(x)` explicitly.
pub trait Roots: Sized {
    /// Non-negative square root of `self`, routed through NumKong's runtime-dispatched kernel.
    fn sqrt(self) -> Self;
    /// Reciprocal square root `1 / sqrt(self)` as a single primitive op where the hardware allows.
    fn rsqrt(self) -> Self;
}

impl Roots for f32 {
    fn sqrt(self) -> Self { unsafe { nk_f32_sqrt(self) } }
    fn rsqrt(self) -> Self { unsafe { nk_f32_rsqrt(self) } }
}

impl Roots for f64 {
    fn sqrt(self) -> Self { unsafe { nk_f64_sqrt(self) } }
    fn rsqrt(self) -> Self { unsafe { nk_f64_rsqrt(self) } }
}

impl Roots for f16 {
    fn sqrt(self) -> Self { f16(unsafe { nk_f16_sqrt(self.0) }) }
    fn rsqrt(self) -> Self { f16(unsafe { nk_f16_rsqrt(self.0) }) }
}
