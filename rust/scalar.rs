//! Scalar math primitives — square root and reciprocal square root.
//!
//! Small per-scalar kernels that route into NumKong's runtime-dispatched implementations rather
//! than the standard library.
//!
//! This module provides:
//!
//! - [`Roots`]: Scalar square root and reciprocal square root
//!
//! File: rust/scalar.rs
//! Author: Ash Vardanian

use crate::{
    capabilities::{enabled_cpu_capabilities_mask, nk_capability_t},
    types::f16,
};

#[link(name = "numkong")]
extern "C" {
    // Scalar square-root / reciprocal-square-root, backing the `Roots` trait.
    fn nk_f32_sqrt_best(x: f32, capabilities: nk_capability_t) -> f32;
    fn nk_f32_rsqrt_best(x: f32, capabilities: nk_capability_t) -> f32;
    fn nk_f64_sqrt_best(x: f64, capabilities: nk_capability_t) -> f64;
    fn nk_f64_rsqrt_best(x: f64, capabilities: nk_capability_t) -> f64;
    fn nk_f16_sqrt_best(x: u16, capabilities: nk_capability_t) -> u16;
    fn nk_f16_rsqrt_best(x: u16, capabilities: nk_capability_t) -> u16;
}

/// Scalar square-root and reciprocal-square-root operations backed by NumKong's exported kernels.
///
/// Unlike `f32::sqrt` / `f64::sqrt`, this routes into the hand-tuned NumKong C kernels; on ISAs
/// with a dedicated reciprocal-sqrt, such as `vrsqrte` on NEON or `vrsqrt14` on AVX-512, `rsqrt`
/// uses a single Newton refinement to ~1 ULP — roughly 2-4× faster than computing 1 / √x
/// explicitly.
pub trait Roots: Sized {
    /// Non-negative square root of `self`, routed through NumKong's runtime-dispatched kernel.
    fn sqrt(self) -> Self;
    /// Reciprocal square root 1 / √x of `self` as a single primitive op where the hardware allows.
    fn rsqrt(self) -> Self;
}

impl Roots for f32 {
    fn sqrt(self) -> Self { unsafe { nk_f32_sqrt_best(self, enabled_cpu_capabilities_mask()) } }
    fn rsqrt(self) -> Self { unsafe { nk_f32_rsqrt_best(self, enabled_cpu_capabilities_mask()) } }
}

impl Roots for f64 {
    fn sqrt(self) -> Self { unsafe { nk_f64_sqrt_best(self, enabled_cpu_capabilities_mask()) } }
    fn rsqrt(self) -> Self { unsafe { nk_f64_rsqrt_best(self, enabled_cpu_capabilities_mask()) } }
}

impl Roots for f16 {
    fn sqrt(self) -> Self { f16(unsafe { nk_f16_sqrt_best(self.0, enabled_cpu_capabilities_mask()) }) }
    fn rsqrt(self) -> Self { f16(unsafe { nk_f16_rsqrt_best(self.0, enabled_cpu_capabilities_mask()) }) }
}
