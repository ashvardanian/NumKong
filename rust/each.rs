//! Elementwise operations — slice traits and tensor-shaped wrappers.
//!
//! This module provides:
//!
//! - Slice-level traits:
//!   - [`EachScale`]: Linear scaling (alpha * x + beta)
//!   - [`EachSum`]: Elementwise addition of two vectors
//!   - [`EachBlend`]: Weighted blend of two vectors
//!   - [`EachFMA`]: Fused multiply-add (a * alpha + b * beta)
//! - Tensor-shaped extension traits (auto-implemented on every [`crate::tensor::TensorRef`]):
//!   - [`ScaleOps`], [`SumOps`], [`BlendOps`], [`FmaOps`]: Tensor wrappers around the slice traits
//!   - [`AllCloseOps`]: Tolerance-based equality for any [`crate::tensor::TensorRef`]
//!
//! # In-Place vs Allocating Semantics
//!
//! Every operation in this module is **allocation-free**: the caller provides both
//! the input slices and a pre-sized output buffer. The kernels never grow a `Vec`
//! internally, never return a new allocation, and never call into the allocator —
//! perfect for warm inner loops and `no_std` contexts.
//!
//! Because the output buffer is always a separate `&mut [Self]`, a caller who
//! wants in-place update must explicitly alias the output to one of the inputs
//! (for example by passing the same slot for both):
//!
//! ```ignore
//! // Functionally equivalent to `x *= 2.0`.
//! f32::each_scale(&x.clone(), 2.0, 0.0, &mut x).unwrap();
//! ```
//!
//! # Stride Support
//!
//! Inputs are read contiguously in memory. For strided access, `slice::chunks_by`,
//! `step_by`, or a caller-side reshape is the right tool — this module keeps the
//! fast-path API simple and lets the SIMD kernels assume packed layout. See the
//! strided variants in [`reduce`](crate::reduce) when non-unit stride is needed.
//!
//! # Example
//!
//! ```
//! use numkong::EachScale;
//! let input = [1.0_f32, 2.0, 3.0, 4.0];
//! let mut output = [0.0_f32; 4];
//! // Compute output[i] = 2.0 * input[i] + 0.5
//! f32::each_scale(&input, 2.0, 0.5, &mut output).unwrap();
//! assert_eq!(output, [2.5, 4.5, 6.5, 8.5]);
//! ```

use crate::tensor::{Global, Tensor, TensorError, TensorMut, TensorRef};
use crate::types::{bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, StorageElement};

#[link(name = "numkong")]
extern "C" {
    // Elementwise operations
    fn nk_each_scale_f64(a: *const f64, n: usize, alpha: *const f64, beta: *const f64, result: *mut f64);
    fn nk_each_scale_f32(a: *const f32, n: usize, alpha: *const f32, beta: *const f32, result: *mut f32);
    fn nk_each_scale_f16(a: *const u16, n: usize, alpha: *const f32, beta: *const f32, result: *mut u16);
    fn nk_each_scale_bf16(a: *const u16, n: usize, alpha: *const f32, beta: *const f32, result: *mut u16);
    fn nk_each_scale_i8(a: *const i8, n: usize, alpha: *const f32, beta: *const f32, result: *mut i8);
    fn nk_each_scale_u8(a: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_each_scale_i16(a: *const i16, n: usize, alpha: *const f32, beta: *const f32, result: *mut i16);
    fn nk_each_scale_u16(a: *const u16, n: usize, alpha: *const f32, beta: *const f32, result: *mut u16);
    fn nk_each_scale_i32(a: *const i32, n: usize, alpha: *const f64, beta: *const f64, result: *mut i32);
    fn nk_each_scale_u32(a: *const u32, n: usize, alpha: *const f64, beta: *const f64, result: *mut u32);
    fn nk_each_scale_i64(a: *const i64, n: usize, alpha: *const f64, beta: *const f64, result: *mut i64);
    fn nk_each_scale_u64(a: *const u64, n: usize, alpha: *const f64, beta: *const f64, result: *mut u64);
    fn nk_each_scale_e4m3(a: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_each_scale_e5m2(a: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_each_scale_e2m3(a: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_each_scale_e3m2(a: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);

    fn nk_each_sum_f64(a: *const f64, b: *const f64, n: usize, result: *mut f64);
    fn nk_each_sum_f32(a: *const f32, b: *const f32, n: usize, result: *mut f32);
    fn nk_each_sum_f16(a: *const u16, b: *const u16, n: usize, result: *mut u16);
    fn nk_each_sum_bf16(a: *const u16, b: *const u16, n: usize, result: *mut u16);
    fn nk_each_sum_i8(a: *const i8, b: *const i8, n: usize, result: *mut i8);
    fn nk_each_sum_u8(a: *const u8, b: *const u8, n: usize, result: *mut u8);
    fn nk_each_sum_i16(a: *const i16, b: *const i16, n: usize, result: *mut i16);
    fn nk_each_sum_u16(a: *const u16, b: *const u16, n: usize, result: *mut u16);
    fn nk_each_sum_i32(a: *const i32, b: *const i32, n: usize, result: *mut i32);
    fn nk_each_sum_u32(a: *const u32, b: *const u32, n: usize, result: *mut u32);
    fn nk_each_sum_i64(a: *const i64, b: *const i64, n: usize, result: *mut i64);
    fn nk_each_sum_u64(a: *const u64, b: *const u64, n: usize, result: *mut u64);
    fn nk_each_sum_e4m3(a: *const u8, b: *const u8, n: usize, result: *mut u8);
    fn nk_each_sum_e5m2(a: *const u8, b: *const u8, n: usize, result: *mut u8);
    fn nk_each_sum_e2m3(a: *const u8, b: *const u8, n: usize, result: *mut u8);
    fn nk_each_sum_e3m2(a: *const u8, b: *const u8, n: usize, result: *mut u8);

    fn nk_each_blend_f64(a: *const f64, b: *const f64, n: usize, alpha: *const f64, beta: *const f64, result: *mut f64);
    fn nk_each_blend_f32(a: *const f32, b: *const f32, n: usize, alpha: *const f32, beta: *const f32, result: *mut f32);
    fn nk_each_blend_f16(a: *const u16, b: *const u16, n: usize, alpha: *const f32, beta: *const f32, result: *mut u16);
    fn nk_each_blend_bf16(
        a: *const u16,
        b: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_blend_i8(a: *const i8, b: *const i8, n: usize, alpha: *const f32, beta: *const f32, result: *mut i8);
    fn nk_each_blend_u8(a: *const u8, b: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_each_blend_i16(a: *const i16, b: *const i16, n: usize, alpha: *const f32, beta: *const f32, result: *mut i16);
    fn nk_each_blend_u16(a: *const u16, b: *const u16, n: usize, alpha: *const f32, beta: *const f32, result: *mut u16);
    fn nk_each_blend_i32(a: *const i32, b: *const i32, n: usize, alpha: *const f64, beta: *const f64, result: *mut i32);
    fn nk_each_blend_u32(a: *const u32, b: *const u32, n: usize, alpha: *const f64, beta: *const f64, result: *mut u32);
    fn nk_each_blend_i64(a: *const i64, b: *const i64, n: usize, alpha: *const f64, beta: *const f64, result: *mut i64);
    fn nk_each_blend_u64(a: *const u64, b: *const u64, n: usize, alpha: *const f64, beta: *const f64, result: *mut u64);
    fn nk_each_blend_e4m3(a: *const u8, b: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_each_blend_e5m2(a: *const u8, b: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_each_blend_e2m3(a: *const u8, b: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);
    fn nk_each_blend_e3m2(a: *const u8, b: *const u8, n: usize, alpha: *const f32, beta: *const f32, result: *mut u8);

    fn nk_each_fma_f64(
        a: *const f64,
        b: *const f64,
        c: *const f64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_fma_f32(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_fma_f16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_fma_bf16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_fma_i8(
        a: *const i8,
        b: *const i8,
        c: *const i8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_each_fma_u8(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_fma_e4m3(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_fma_e5m2(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_fma_e2m3(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_fma_e3m2(
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_fma_i16(
        a: *const i16,
        b: *const i16,
        c: *const i16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        r: *mut i16,
    );
    fn nk_each_fma_u16(
        a: *const u16,
        b: *const u16,
        c: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        r: *mut u16,
    );
    fn nk_each_fma_i32(
        a: *const i32,
        b: *const i32,
        c: *const i32,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        r: *mut i32,
    );
    fn nk_each_fma_u32(
        a: *const u32,
        b: *const u32,
        c: *const u32,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        r: *mut u32,
    );
    fn nk_each_fma_i64(
        a: *const i64,
        b: *const i64,
        c: *const i64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        r: *mut i64,
    );
    fn nk_each_fma_u64(
        a: *const u64,
        b: *const u64,
        c: *const u64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        r: *mut u64,
    );

    // Complex elementwise operations (interleaved real/imag layout, n = number of complex pairs)
    fn nk_each_sum_f32c(a: *const f32, b: *const f32, n: usize, result: *mut f32);
    fn nk_each_sum_f64c(a: *const f64, b: *const f64, n: usize, result: *mut f64);
    fn nk_each_scale_f32c(a: *const f32, n: usize, alpha: *const f32, beta: *const f32, result: *mut f32);
    fn nk_each_scale_f64c(a: *const f64, n: usize, alpha: *const f64, beta: *const f64, result: *mut f64);
    fn nk_each_blend_f32c(
        a: *const f32,
        b: *const f32,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_blend_f64c(
        a: *const f64,
        b: *const f64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_fma_f32c(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_fma_f64c(
        a: *const f64,
        b: *const f64,
        c: *const f64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_swiglu_f32(
        gate: *const f32,
        up: *const f32,
        y: *mut f32,
        rows: usize,
        cols: usize,
        gate_row_stride: usize,
        up_row_stride: usize,
        y_row_stride: usize,
        input_scale: f32,
    );
    fn nk_each_swiglu_bf16(
        gate: *const u16,
        up: *const u16,
        y: *mut u16,
        rows: usize,
        cols: usize,
        gate_row_stride: usize,
        up_row_stride: usize,
        y_row_stride: usize,
        input_scale: f32,
    );
    fn nk_each_swiglu_e4m3(
        gate: *const u8,
        up: *const u8,
        y: *mut u8,
        rows: usize,
        cols: usize,
        gate_row_stride: usize,
        up_row_stride: usize,
        y_row_stride: usize,
        input_scale: f32,
    );
}

// Complex fallback helpers

fn complex_each_sum_fallback<Scalar>(a: &[Scalar], b: &[Scalar], result: &mut [Scalar]) -> Option<()>
where
    Scalar: Copy + core::ops::Add<Output = Scalar>,
{
    if a.len() != b.len() || a.len() != result.len() {
        return None;
    }
    for ((left, right), out) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
        *out = *left + *right;
    }
    Some(())
}

fn complex_each_scale_fallback<Scalar>(a: &[Scalar], alpha: Scalar, beta: Scalar, result: &mut [Scalar]) -> Option<()>
where
    Scalar: Copy + core::ops::Add<Output = Scalar> + core::ops::Mul<Output = Scalar>,
{
    if a.len() != result.len() {
        return None;
    }
    for (value, out) in a.iter().zip(result.iter_mut()) {
        *out = alpha * *value + beta;
    }
    Some(())
}

fn complex_each_blend_fallback<Scalar>(
    a: &[Scalar],
    b: &[Scalar],
    alpha: Scalar,
    beta: Scalar,
    result: &mut [Scalar],
) -> Option<()>
where
    Scalar: Copy + core::ops::Add<Output = Scalar> + core::ops::Mul<Output = Scalar>,
{
    if a.len() != b.len() || a.len() != result.len() {
        return None;
    }
    for ((left, right), out) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
        *out = alpha * *left + beta * *right;
    }
    Some(())
}

fn complex_each_fma_fallback<Scalar>(
    a: &[Scalar],
    b: &[Scalar],
    c: &[Scalar],
    alpha: Scalar,
    beta: Scalar,
    result: &mut [Scalar],
) -> Option<()>
where
    Scalar: Copy + core::ops::Add<Output = Scalar> + core::ops::Mul<Output = Scalar>,
{
    if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() {
        return None;
    }
    for (((left, right), third), out) in a.iter().zip(b.iter()).zip(c.iter()).zip(result.iter_mut()) {
        *out = alpha * *left * *right + beta * *third;
    }
    Some(())
}

// In-place complex fallbacks: read-modify-write through a single `&mut` (sound — no
// aliased `&[T]` over the same storage). `data` is both the `a` operand and the result.

fn complex_each_sum_inplace_fallback<Scalar>(data: &mut [Scalar], other: &[Scalar]) -> Option<()>
where
    Scalar: Copy + core::ops::Add<Output = Scalar>,
{
    if data.len() != other.len() {
        return None;
    }
    for (out, right) in data.iter_mut().zip(other.iter()) {
        *out = *out + *right;
    }
    Some(())
}

fn complex_each_scale_inplace_fallback<Scalar>(data: &mut [Scalar], alpha: Scalar, beta: Scalar) -> Option<()>
where
    Scalar: Copy + core::ops::Add<Output = Scalar> + core::ops::Mul<Output = Scalar>,
{
    for out in data.iter_mut() {
        *out = alpha * *out + beta;
    }
    Some(())
}

fn complex_each_blend_inplace_fallback<Scalar>(
    data: &mut [Scalar],
    other: &[Scalar],
    alpha: Scalar,
    beta: Scalar,
) -> Option<()>
where
    Scalar: Copy + core::ops::Add<Output = Scalar> + core::ops::Mul<Output = Scalar>,
{
    if data.len() != other.len() {
        return None;
    }
    for (out, right) in data.iter_mut().zip(other.iter()) {
        *out = alpha * *out + beta * *right;
    }
    Some(())
}

fn complex_each_fma_inplace_fallback<Scalar>(
    data: &mut [Scalar],
    b: &[Scalar],
    alpha: Scalar,
    beta: Scalar,
) -> Option<()>
where
    Scalar: Copy + core::ops::Add<Output = Scalar> + core::ops::Mul<Output = Scalar>,
{
    if data.len() != b.len() {
        return None;
    }
    // In-place fused multiply-add matches `each_fma` with the `c` operand bound to `a`
    // (the same storage) — exactly how out-of-place `mul_tensor` wires `c = self`.
    for (out, right) in data.iter_mut().zip(b.iter()) {
        let value = *out;
        *out = alpha * value * *right + beta * value;
    }
    Some(())
}

// region: Scale

/// Applies an **element-wise affine transform** (scale and shift).
///
/// rᵢ = α × aᵢ + β
///
/// Returns `None` if `a` and `result` lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `e4m3`, `e5m2`, `e2m3`, `e3m2`.
pub trait EachScale: Sized + StorageElement {
    type Scalar;

    /// Writes `result[i] = alpha * a[i] + beta` into the pre-sized output slice.
    ///
    /// All three slices (`a`, `result`) must have identical length — the kernel
    /// does not allocate. Returns `None` on length mismatch.
    ///
    /// # Examples
    ///
    /// ```
    /// use numkong::EachScale;
    /// let input = [1.0_f32, 2.0, 3.0];
    /// let mut output = [0.0_f32; 3];
    /// // Rescale to [-1, 1]: alpha = 2/(max-min) = 1.0, beta = -1 - 1*min = -2.0
    /// f32::each_scale(&input, 1.0, -2.0, &mut output).unwrap();
    /// assert_eq!(output, [-1.0, 0.0, 1.0]);
    /// ```
    fn each_scale(a: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()>;

    /// In-place affine: `data[i] = alpha * data[i] + beta`.
    ///
    /// Both source and destination pointers are derived from the single `&mut`,
    /// so no aliased `&[Self]` + `&mut [Self]` over the same storage is formed.
    fn each_scale_inplace(data: &mut [Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()>;
}

impl EachScale for f64 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_f64(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f64, beta: f64) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_f64(p as *const f64, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for f32 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_f32(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_f32(p as *const f32, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for f16 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_f16(
                a.as_ptr() as *const u16,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_f16(p as *const u16, len, &alpha, &beta, p as *mut u16) };
        Some(())
    }
}

impl EachScale for bf16 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_bf16(
                a.as_ptr() as *const u16,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_bf16(p as *const u16, len, &alpha, &beta, p as *mut u16) };
        Some(())
    }
}

impl EachScale for i8 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_i8(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_i8(p as *const i8, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for u8 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_u8(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_u8(p as *const u8, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for i16 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_i16(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_i16(p as *const i16, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for u16 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_u16(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_u16(p as *const u16, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for i32 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_i32(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f64, beta: f64) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_i32(p as *const i32, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for u32 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_u32(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f64, beta: f64) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_u32(p as *const u32, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for i64 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_i64(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f64, beta: f64) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_i64(p as *const i64, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for u64 {
    type Scalar = f64;
    fn each_scale(a: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_scale_u64(a.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f64, beta: f64) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_u64(p as *const u64, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachScale for e4m3 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_e4m3(
                a.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_e4m3(p as *const u8, len, &alpha, &beta, p as *mut u8) };
        Some(())
    }
}

impl EachScale for e5m2 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_e5m2(
                a.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_e5m2(p as *const u8, len, &alpha, &beta, p as *mut u8) };
        Some(())
    }
}

impl EachScale for e2m3 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_e2m3(
                a.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_e2m3(p as *const u8, len, &alpha, &beta, p as *mut u8) };
        Some(())
    }
}

impl EachScale for e3m2 {
    type Scalar = f32;
    fn each_scale(a: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_e3m2(
                a.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: f32, beta: f32) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_e3m2(p as *const u8, len, &alpha, &beta, p as *mut u8) };
        Some(())
    }
}

impl EachScale for f64c {
    type Scalar = f64c;
    fn each_scale(a: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_f64c(
                a.as_ptr() as *const f64,
                a.len(),
                &alpha.re,
                &beta.re,
                result.as_mut_ptr() as *mut f64,
            )
        };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_f64c(p as *const f64, len, &alpha.re, &beta.re, p as *mut f64) };
        Some(())
    }
}

impl EachScale for f32c {
    type Scalar = f32c;
    fn each_scale(a: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()> {
        if a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_scale_f32c(
                a.as_ptr() as *const f32,
                a.len(),
                &alpha.re,
                &beta.re,
                result.as_mut_ptr() as *mut f32,
            )
        };
        Some(())
    }

    fn each_scale_inplace(data: &mut [Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_scale_f32c(p as *const f32, len, &alpha.re, &beta.re, p as *mut f32) };
        Some(())
    }
}

impl EachScale for f16c {
    type Scalar = f16c;
    fn each_scale(a: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()> {
        complex_each_scale_fallback(a, alpha, beta, result)
    }

    fn each_scale_inplace(data: &mut [Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        complex_each_scale_inplace_fallback(data, alpha, beta)
    }
}

impl EachScale for bf16c {
    type Scalar = bf16c;
    fn each_scale(a: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()> {
        complex_each_scale_fallback(a, alpha, beta, result)
    }

    fn each_scale_inplace(data: &mut [Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        complex_each_scale_inplace_fallback(data, alpha, beta)
    }
}

// endregion: Scale

// region: Sum

/// Applies **element-wise addition** of two vectors.
///
/// rᵢ = aᵢ + bᵢ
///
/// Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `e4m3`, `e5m2`, `e2m3`, `e3m2`.
pub trait EachSum: Sized + StorageElement {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()>;

    /// In-place sum: `data[i] = data[i] + other[i]`.
    ///
    /// `data` is both the `a` operand and the result; its source and destination
    /// pointers come from the single `&mut`, while `other` is disjoint storage —
    /// no aliased `&[Self]` + `&mut [Self]` over the same buffer is formed.
    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()>;
}

impl EachSum for f64 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_f64(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_f64(p as *const f64, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for f32 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_f32(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_f32(p as *const f32, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for f16 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_f16(p as *const u16, other.as_ptr() as *const u16, len, p as *mut u16) };
        Some(())
    }
}

impl EachSum for bf16 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_bf16(p as *const u16, other.as_ptr() as *const u16, len, p as *mut u16) };
        Some(())
    }
}

impl EachSum for i8 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_i8(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_i8(p as *const i8, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for u8 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_u8(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_u8(p as *const u8, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for i16 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_i16(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_i16(p as *const i16, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for u16 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_u16(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_u16(p as *const u16, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for i32 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_i32(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_i32(p as *const i32, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for u32 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_u32(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_u32(p as *const u32, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for i64 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_i64(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_i64(p as *const i64, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for u64 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_u64(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_u64(p as *const u64, other.as_ptr(), len, p) };
        Some(())
    }
}

impl EachSum for e4m3 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len(),
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_e4m3(p as *const u8, other.as_ptr() as *const u8, len, p as *mut u8) };
        Some(())
    }
}

impl EachSum for e5m2 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len(),
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_e5m2(p as *const u8, other.as_ptr() as *const u8, len, p as *mut u8) };
        Some(())
    }
}

impl EachSum for e2m3 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len(),
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_e2m3(p as *const u8, other.as_ptr() as *const u8, len, p as *mut u8) };
        Some(())
    }
}

impl EachSum for e3m2 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len(),
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_e3m2(p as *const u8, other.as_ptr() as *const u8, len, p as *mut u8) };
        Some(())
    }
}

impl EachSum for f64c {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_f64c(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len(),
                result.as_mut_ptr() as *mut f64,
            )
        };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_f64c(p as *const f64, other.as_ptr() as *const f64, len, p as *mut f64) };
        Some(())
    }
}

impl EachSum for f32c {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_sum_f32c(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len(),
                result.as_mut_ptr() as *mut f32,
            )
        };
        Some(())
    }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_sum_f32c(p as *const f32, other.as_ptr() as *const f32, len, p as *mut f32) };
        Some(())
    }
}

impl EachSum for f16c {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> { complex_each_sum_fallback(a, b, result) }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        complex_each_sum_inplace_fallback(data, other)
    }
}

impl EachSum for bf16c {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> { complex_each_sum_fallback(a, b, result) }

    fn each_sum_inplace(data: &mut [Self], other: &[Self]) -> Option<()> {
        complex_each_sum_inplace_fallback(data, other)
    }
}

// endregion: Sum

// region: Blend

/// Applies **element-wise weighted sum** (blend) of two vectors.
///
/// rᵢ = α × aᵢ + β × bᵢ
///
/// Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `e4m3`, `e5m2`, `e2m3`, `e3m2`.
pub trait EachBlend: Sized + StorageElement {
    type Scalar;
    fn each_blend(a: &[Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()>;

    /// In-place blend: `data[i] = alpha * data[i] + beta * other[i]`.
    ///
    /// `data` is both the `a` operand and the result; its source and destination
    /// pointers come from the single `&mut`, while `other` is disjoint storage —
    /// no aliased `&[Self]` + `&mut [Self]` over the same buffer is formed.
    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()>;
}

impl EachBlend for f64 {
    type Scalar = f64;
    fn each_blend(a: &[Self], b: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_f64(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_f64(p as *const f64, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for f32 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_f32(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_f32(p as *const f32, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for f16 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_blend_f16(
                p as *const u16,
                other.as_ptr() as *const u16,
                len,
                &alpha,
                &beta,
                p as *mut u16,
            )
        };
        Some(())
    }
}

impl EachBlend for bf16 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_blend_bf16(
                p as *const u16,
                other.as_ptr() as *const u16,
                len,
                &alpha,
                &beta,
                p as *mut u16,
            )
        };
        Some(())
    }
}

impl EachBlend for i8 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_i8(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_i8(p as *const i8, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for u8 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_u8(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_u8(p as *const u8, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for i16 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_i16(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_i16(p as *const i16, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for u16 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_u16(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_u16(p as *const u16, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for i32 {
    type Scalar = f64;
    fn each_blend(a: &[Self], b: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_i32(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_i32(p as *const i32, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for u32 {
    type Scalar = f64;
    fn each_blend(a: &[Self], b: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_u32(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_u32(p as *const u32, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for i64 {
    type Scalar = f64;
    fn each_blend(a: &[Self], b: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_i64(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_i64(p as *const i64, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for u64 {
    type Scalar = f64;
    fn each_blend(a: &[Self], b: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_blend_u64(a.as_ptr(), b.as_ptr(), a.len(), &alpha, &beta, result.as_mut_ptr()) };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_blend_u64(p as *const u64, other.as_ptr(), len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachBlend for e4m3 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_blend_e4m3(
                p as *const u8,
                other.as_ptr() as *const u8,
                len,
                &alpha,
                &beta,
                p as *mut u8,
            )
        };
        Some(())
    }
}

impl EachBlend for e5m2 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_blend_e5m2(
                p as *const u8,
                other.as_ptr() as *const u8,
                len,
                &alpha,
                &beta,
                p as *mut u8,
            )
        };
        Some(())
    }
}

impl EachBlend for e2m3 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_blend_e2m3(
                p as *const u8,
                other.as_ptr() as *const u8,
                len,
                &alpha,
                &beta,
                p as *mut u8,
            )
        };
        Some(())
    }
}

impl EachBlend for e3m2 {
    type Scalar = f32;
    fn each_blend(a: &[Self], b: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_blend_e3m2(
                p as *const u8,
                other.as_ptr() as *const u8,
                len,
                &alpha,
                &beta,
                p as *mut u8,
            )
        };
        Some(())
    }
}

impl EachBlend for f64c {
    type Scalar = f64c;
    fn each_blend(a: &[Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_f64c(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len(),
                &alpha.re,
                &beta.re,
                result.as_mut_ptr() as *mut f64,
            )
        };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_blend_f64c(
                p as *const f64,
                other.as_ptr() as *const f64,
                len,
                &alpha.re,
                &beta.re,
                p as *mut f64,
            )
        };
        Some(())
    }
}

impl EachBlend for f32c {
    type Scalar = f32c;
    fn each_blend(a: &[Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_f32c(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len(),
                &alpha.re,
                &beta.re,
                result.as_mut_ptr() as *mut f32,
            )
        };
        Some(())
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        if data.len() != other.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_blend_f32c(
                p as *const f32,
                other.as_ptr() as *const f32,
                len,
                &alpha.re,
                &beta.re,
                p as *mut f32,
            )
        };
        Some(())
    }
}

impl EachBlend for f16c {
    type Scalar = f16c;
    fn each_blend(a: &[Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()> {
        complex_each_blend_fallback(a, b, alpha, beta, result)
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        complex_each_blend_inplace_fallback(data, other, alpha, beta)
    }
}

impl EachBlend for bf16c {
    type Scalar = bf16c;
    fn each_blend(a: &[Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar, result: &mut [Self]) -> Option<()> {
        complex_each_blend_fallback(a, b, alpha, beta, result)
    }

    fn each_blend_inplace(data: &mut [Self], other: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        complex_each_blend_inplace_fallback(data, other, alpha, beta)
    }
}

// endregion: Blend

// region: FMA

/// Applies **fused multiply-add** element-wise across three vectors.
///
/// rᵢ = α × aᵢ × bᵢ + β × cᵢ
///
/// Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `e4m3`, `e5m2`, `e2m3`, `e3m2`.
pub trait EachFMA: Sized + StorageElement {
    type Scalar;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;

    /// In-place fused multiply-add with the `c` operand bound to `a`:
    /// `data[i] = alpha * data[i] * b[i] + beta * data[i]`.
    ///
    /// `data` is the `a` operand, the `c` operand, and the result — all derived
    /// from the single `&mut`, while `b` is disjoint storage. This matches
    /// out-of-place `mul_tensor`, which wires `c = self` and is the only in-place
    /// FMA caller. No aliased `&[Self]` + `&mut [Self]` over the same buffer is formed.
    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()>;
}

impl EachFMA for f64 {
    type Scalar = f64;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_f64(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_f64(p as *const f64, b.as_ptr(), p as *const f64, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for f32 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_f32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_f32(p as *const f32, b.as_ptr(), p as *const f32, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for f16 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_fma_f16(
                p as *const u16,
                b.as_ptr() as *const u16,
                p as *const u16,
                len,
                &alpha,
                &beta,
                p as *mut u16,
            )
        };
        Some(())
    }
}

impl EachFMA for bf16 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_fma_bf16(
                p as *const u16,
                b.as_ptr() as *const u16,
                p as *const u16,
                len,
                &alpha,
                &beta,
                p as *mut u16,
            )
        };
        Some(())
    }
}

impl EachFMA for i8 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_i8(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_i8(p as *const i8, b.as_ptr(), p as *const i8, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for u8 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_u8(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_u8(p as *const u8, b.as_ptr(), p as *const u8, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for e4m3 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_e4m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                c.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_fma_e4m3(
                p as *const u8,
                b.as_ptr() as *const u8,
                p as *const u8,
                len,
                &alpha,
                &beta,
                p as *mut u8,
            )
        };
        Some(())
    }
}

impl EachFMA for e5m2 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_e5m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                c.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_fma_e5m2(
                p as *const u8,
                b.as_ptr() as *const u8,
                p as *const u8,
                len,
                &alpha,
                &beta,
                p as *mut u8,
            )
        };
        Some(())
    }
}

impl EachFMA for e2m3 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_e2m3(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                c.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_fma_e2m3(
                p as *const u8,
                b.as_ptr() as *const u8,
                p as *const u8,
                len,
                &alpha,
                &beta,
                p as *mut u8,
            )
        };
        Some(())
    }
}

impl EachFMA for e3m2 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_e3m2(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                c.as_ptr() as *const u8,
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr() as *mut u8,
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_fma_e3m2(
                p as *const u8,
                b.as_ptr() as *const u8,
                p as *const u8,
                len,
                &alpha,
                &beta,
                p as *mut u8,
            )
        };
        Some(())
    }
}

impl EachFMA for i16 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_i16(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_i16(p as *const i16, b.as_ptr(), p as *const i16, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for u16 {
    type Scalar = f32;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f32, beta: f32, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_u16(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f32, beta: f32) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_u16(p as *const u16, b.as_ptr(), p as *const u16, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for i32 {
    type Scalar = f64;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_i32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_i32(p as *const i32, b.as_ptr(), p as *const i32, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for u32 {
    type Scalar = f64;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_u32(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_u32(p as *const u32, b.as_ptr(), p as *const u32, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for i64 {
    type Scalar = f64;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_i64(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_i64(p as *const i64, b.as_ptr(), p as *const i64, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for u64 {
    type Scalar = f64;
    fn each_fma(a: &[Self], b: &[Self], c: &[Self], alpha: f64, beta: f64, result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || b.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_u64(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: f64, beta: f64) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_each_fma_u64(p as *const u64, b.as_ptr(), p as *const u64, len, &alpha, &beta, p) };
        Some(())
    }
}

impl EachFMA for f64c {
    type Scalar = f64c;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_f64c(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                c.as_ptr() as *const f64,
                a.len(),
                &alpha.re,
                &beta.re,
                result.as_mut_ptr() as *mut f64,
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_fma_f64c(
                p as *const f64,
                b.as_ptr() as *const f64,
                p as *const f64,
                len,
                &alpha.re,
                &beta.re,
                p as *mut f64,
            )
        };
        Some(())
    }
}

impl EachFMA for f32c {
    type Scalar = f32c;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_fma_f32c(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                c.as_ptr() as *const f32,
                a.len(),
                &alpha.re,
                &beta.re,
                result.as_mut_ptr() as *mut f32,
            )
        };
        Some(())
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        if data.len() != b.len() {
            return None;
        }
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_each_fma_f32c(
                p as *const f32,
                b.as_ptr() as *const f32,
                p as *const f32,
                len,
                &alpha.re,
                &beta.re,
                p as *mut f32,
            )
        };
        Some(())
    }
}

impl EachFMA for f16c {
    type Scalar = f16c;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
        complex_each_fma_fallback(a, b, c, alpha, beta, result)
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        complex_each_fma_inplace_fallback(data, b, alpha, beta)
    }
}

impl EachFMA for bf16c {
    type Scalar = bf16c;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
        complex_each_fma_fallback(a, b, c, alpha, beta, result)
    }

    fn each_fma_inplace(data: &mut [Self], b: &[Self], alpha: Self::Scalar, beta: Self::Scalar) -> Option<()> {
        complex_each_fma_inplace_fallback(data, b, alpha, beta)
    }
}

// endregion: FMA

// region: Tensor-shaped tolerance equality

use crate::types::{is_close, FloatConvertible, NumberLike};

/// Extension trait: tolerance-based equality for any [`TensorRef`] implementor.
///
/// Uses the formula `|a - b| <= atol + rtol * |b|` per element.
/// Returns `false` if shapes differ.
pub trait AllCloseOps<Scalar: FloatConvertible, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar::DimScalar: NumberLike,
{
    fn allclose(&self, other: &(impl TensorRef<Scalar, MAX_RANK> + ?Sized), atol: f64, rtol: f64) -> bool {
        let a = self.view();
        let b = other.view();
        a.ndim() == b.ndim()
            && a.shape() == b.shape()
            && a.iter()
                .dims()
                .zip(b.iter().dims())
                .all(|(x, y)| is_close((*x).to_f64(), (*y).to_f64(), atol, rtol))
    }
}

impl<C, Scalar: FloatConvertible, const R: usize> AllCloseOps<Scalar, R> for C
where
    C: TensorRef<Scalar, R>,
    Scalar::DimScalar: NumberLike,
{
}

// endregion: Tensor-shaped tolerance equality

// region: Tensor-shaped scale / sum / blend / fma

/// Extension trait: scalar arithmetic for any [`TensorRef`] implementor.
pub trait ScaleOps<Scalar: Clone + EachScale, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    fn try_add_scalar(&self, scalar: Scalar::Scalar) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_add_scalar(scalar)
    }

    fn try_sub_scalar(&self, scalar: Scalar::Scalar) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_scalar(scalar)
    }

    fn try_mul_scalar(&self, scalar: Scalar::Scalar) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_scalar(scalar)
    }

    fn try_scale_tensor_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError> {
        self.view().try_scale_tensor_into(alpha, beta, out)
    }

    fn try_add_scalar_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        scalar: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError> {
        self.view().try_add_scalar_into(scalar, out)
    }

    fn try_sub_scalar_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        scalar: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError> {
        self.view().try_sub_scalar_into(scalar, out)
    }

    fn try_mul_scalar_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        scalar: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError> {
        self.view().try_mul_scalar_into(scalar, out)
    }
}

impl<Scalar: Clone + EachScale, const R: usize, C: TensorRef<Scalar, R> + ?Sized> ScaleOps<Scalar, R> for C where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy
{
}

impl<Scalar: Clone + EachScale, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    /// Element-wise add scalar in-place (infallible — self vs self always matches).
    pub fn add_scalar_inplace(&mut self, scalar: Scalar::Scalar) { self.span().add_scalar_inplace(scalar); }

    /// Element-wise subtract scalar in-place (infallible — self vs self always matches).
    pub fn sub_scalar_inplace(&mut self, scalar: Scalar::Scalar) { self.span().sub_scalar_inplace(scalar); }

    /// Element-wise multiply scalar in-place (infallible — self vs self always matches).
    pub fn mul_scalar_inplace(&mut self, scalar: Scalar::Scalar) { self.span().mul_scalar_inplace(scalar); }
}

impl<Scalar: Clone + EachScale, const MAX_RANK: usize> core::ops::AddAssign<Scalar::Scalar>
    for Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    fn add_assign(&mut self, scalar: Scalar::Scalar) { self.add_scalar_inplace(scalar); }
}

impl<Scalar: Clone + EachScale, const MAX_RANK: usize> core::ops::SubAssign<Scalar::Scalar>
    for Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    fn sub_assign(&mut self, scalar: Scalar::Scalar) { self.sub_scalar_inplace(scalar); }
}

impl<Scalar: Clone + EachScale, const MAX_RANK: usize> core::ops::MulAssign<Scalar::Scalar>
    for Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    fn mul_assign(&mut self, scalar: Scalar::Scalar) { self.mul_scalar_inplace(scalar); }
}

/// Extension trait: element-wise addition for any [`TensorRef`] implementor.
pub trait SumOps<Scalar: Clone + EachSum, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn try_add_tensor(
        &self,
        other: &(impl TensorRef<Scalar, MAX_RANK> + ?Sized),
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_add_tensor(&other.view())
    }

    fn try_add_tensor_into<OtherTensor, OutputTensor>(
        &self,
        other: &OtherTensor,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        OtherTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized,
    {
        self.view().try_add_tensor_into(&other.view(), out)
    }
}

impl<Scalar: Clone + EachSum, const R: usize, C: TensorRef<Scalar, R> + ?Sized> SumOps<Scalar, R> for C {}

impl<Scalar: Clone + EachSum, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    pub fn try_add_tensor_inplace(&mut self, other: &Tensor<Scalar, Global, MAX_RANK>) -> Result<(), TensorError> {
        let other_view = other.view();
        self.span().add_inplace(&other_view)
    }
}

/// Extension trait: element-wise subtraction for any [`TensorRef`] implementor.
pub trait BlendOps<Scalar: Clone + EachBlend, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    fn try_sub_tensor(
        &self,
        other: &(impl TensorRef<Scalar, MAX_RANK> + ?Sized),
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_tensor(&other.view())
    }

    fn try_blend_tensor_into<OtherTensor, OutputTensor>(
        &self,
        other: &OtherTensor,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        OtherTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized,
    {
        self.view().try_blend_tensor_into(&other.view(), alpha, beta, out)
    }

    fn try_sub_tensor_into<OtherTensor, OutputTensor>(
        &self,
        other: &OtherTensor,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        OtherTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized,
    {
        self.view().try_sub_tensor_into(&other.view(), out)
    }
}

impl<Scalar: Clone + EachBlend, const R: usize, C: TensorRef<Scalar, R> + ?Sized> BlendOps<Scalar, R> for C where
    Scalar::Scalar: From<f32> + Copy
{
}

impl<Scalar: Clone + EachBlend, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    pub fn try_sub_tensor_inplace(&mut self, other: &Tensor<Scalar, Global, MAX_RANK>) -> Result<(), TensorError> {
        let other_view = other.view();
        self.span().sub_inplace(&other_view)
    }
}

/// Extension trait: element-wise multiplication for any [`TensorRef`] implementor.
pub trait FmaOps<Scalar: Clone + EachFMA, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    fn try_mul_tensor(
        &self,
        other: &(impl TensorRef<Scalar, MAX_RANK> + ?Sized),
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_tensor(&other.view())
    }

    fn try_fma_tensors_into<BTensor, CTensor, OutputTensor>(
        &self,
        b: &BTensor,
        c: &CTensor,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        BTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        CTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized,
    {
        self.view().try_fma_tensors_into(&b.view(), &c.view(), alpha, beta, out)
    }

    fn try_mul_tensor_into<OtherTensor, OutputTensor>(
        &self,
        other: &OtherTensor,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError>
    where
        OtherTensor: TensorRef<Scalar, MAX_RANK> + ?Sized,
        OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized,
    {
        self.view().try_mul_tensor_into(&other.view(), out)
    }
}

impl<Scalar: Clone + EachFMA, const R: usize, C: TensorRef<Scalar, R> + ?Sized> FmaOps<Scalar, R> for C where
    Scalar::Scalar: From<f32> + Copy
{
}

impl<Scalar: Clone + EachFMA, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    pub fn try_mul_tensor_inplace(&mut self, other: &Tensor<Scalar, Global, MAX_RANK>) -> Result<(), TensorError> {
        let other_view = other.view();
        self.span().mul_inplace(&other_view)
    }
}

// endregion: Tensor-shaped scale / sum / blend / fma

// region: Fused SwiGLU

/// Fused SwiGLU over a row-major `[rows, cols]` slice: `y = silu(input_scale * gate) * (input_scale * up)`.
/// With `up = None` this reduces to plain SiLU (`cols = gate.len() / rows`).
pub trait EachSwiglu: Sized + StorageElement {
    /// Fused SwiGLU of 2D `[rows, cols]` tensors: `y = silu(input_scale*gate) * (input_scale*up)`.
    ///
    /// Strides are read from the tensors, so `gate`, `up`, and `y` may be independent strided
    /// sub-spans (e.g. the two column halves of a `[rows, 2*cols]` gate|up buffer). `up = None`
    /// reduces to plain SiLU. Returns `Err` on a shape mismatch.
    fn swiglu_into<GIn, UIn, YOut, const RG: usize, const RU: usize, const RY: usize>(
        gate: &GIn,
        up: Option<&UIn>,
        y: &mut YOut,
        input_scale: f32,
    ) -> Result<(), TensorError>
    where
        GIn: TensorRef<Self, RG> + ?Sized,
        UIn: TensorRef<Self, RU> + ?Sized,
        YOut: TensorMut<Self, RY> + ?Sized;
}

impl EachSwiglu for f32 {
    fn swiglu_into<GIn, UIn, YOut, const RG: usize, const RU: usize, const RY: usize>(
        gate: &GIn,
        up: Option<&UIn>,
        y: &mut YOut,
        input_scale: f32,
    ) -> Result<(), TensorError>
    where
        GIn: TensorRef<Self, RG> + ?Sized,
        UIn: TensorRef<Self, RU> + ?Sized,
        YOut: TensorMut<Self, RY> + ?Sized,
    {
        if gate.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: gate.ndim(),
            });
        }
        if y.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: y.ndim(),
            });
        }
        if gate.shape() != y.shape() {
            let axis = if gate.shape()[0] != y.shape()[0] { 0 } else { 1 };
            return Err(TensorError::ShapeMismatch {
                axis,
                expected: gate.shape()[axis],
                got: y.shape()[axis],
            });
        }
        let (rows, cols) = (gate.shape()[0], gate.shape()[1]);
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let gate_stride = gate.stride_bytes(0) as usize;
        let y_stride = y.stride_bytes(0) as usize;
        let (up_ptr, up_stride) = match up {
            Some(u) => {
                if u.ndim() != 2 {
                    return Err(TensorError::DimensionMismatch {
                        expected: 2,
                        got: u.ndim(),
                    });
                }
                if u.shape() != gate.shape() {
                    let axis = if u.shape()[0] != gate.shape()[0] { 0 } else { 1 };
                    return Err(TensorError::ShapeMismatch {
                        axis,
                        expected: gate.shape()[axis],
                        got: u.shape()[axis],
                    });
                }
                (u.as_ptr(), u.stride_bytes(0) as usize)
            }
            None => (core::ptr::null(), 0usize),
        };
        unsafe {
            nk_each_swiglu_f32(
                gate.as_ptr(),
                up_ptr,
                y.as_mut_ptr(),
                rows,
                cols,
                gate_stride,
                up_stride,
                y_stride,
                input_scale,
            );
        }
        Ok(())
    }
}

impl EachSwiglu for bf16 {
    fn swiglu_into<GIn, UIn, YOut, const RG: usize, const RU: usize, const RY: usize>(
        gate: &GIn,
        up: Option<&UIn>,
        y: &mut YOut,
        input_scale: f32,
    ) -> Result<(), TensorError>
    where
        GIn: TensorRef<Self, RG> + ?Sized,
        UIn: TensorRef<Self, RU> + ?Sized,
        YOut: TensorMut<Self, RY> + ?Sized,
    {
        if gate.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: gate.ndim(),
            });
        }
        if y.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: y.ndim(),
            });
        }
        if gate.shape() != y.shape() {
            let axis = if gate.shape()[0] != y.shape()[0] { 0 } else { 1 };
            return Err(TensorError::ShapeMismatch {
                axis,
                expected: gate.shape()[axis],
                got: y.shape()[axis],
            });
        }
        let (rows, cols) = (gate.shape()[0], gate.shape()[1]);
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let gate_stride = gate.stride_bytes(0) as usize;
        let y_stride = y.stride_bytes(0) as usize;
        let (up_ptr, up_stride) = match up {
            Some(u) => {
                if u.ndim() != 2 {
                    return Err(TensorError::DimensionMismatch {
                        expected: 2,
                        got: u.ndim(),
                    });
                }
                if u.shape() != gate.shape() {
                    let axis = if u.shape()[0] != gate.shape()[0] { 0 } else { 1 };
                    return Err(TensorError::ShapeMismatch {
                        axis,
                        expected: gate.shape()[axis],
                        got: u.shape()[axis],
                    });
                }
                (u.as_ptr() as *const u16, u.stride_bytes(0) as usize)
            }
            None => (core::ptr::null(), 0usize),
        };
        unsafe {
            nk_each_swiglu_bf16(
                gate.as_ptr() as *const u16,
                up_ptr,
                y.as_mut_ptr() as *mut u16,
                rows,
                cols,
                gate_stride,
                up_stride,
                y_stride,
                input_scale,
            );
        }
        Ok(())
    }
}

impl EachSwiglu for e4m3 {
    fn swiglu_into<GIn, UIn, YOut, const RG: usize, const RU: usize, const RY: usize>(
        gate: &GIn,
        up: Option<&UIn>,
        y: &mut YOut,
        input_scale: f32,
    ) -> Result<(), TensorError>
    where
        GIn: TensorRef<Self, RG> + ?Sized,
        UIn: TensorRef<Self, RU> + ?Sized,
        YOut: TensorMut<Self, RY> + ?Sized,
    {
        if gate.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: gate.ndim(),
            });
        }
        if y.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: y.ndim(),
            });
        }
        if gate.shape() != y.shape() {
            let axis = if gate.shape()[0] != y.shape()[0] { 0 } else { 1 };
            return Err(TensorError::ShapeMismatch {
                axis,
                expected: gate.shape()[axis],
                got: y.shape()[axis],
            });
        }
        let (rows, cols) = (gate.shape()[0], gate.shape()[1]);
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let gate_stride = gate.stride_bytes(0) as usize;
        let y_stride = y.stride_bytes(0) as usize;
        let (up_ptr, up_stride) = match up {
            Some(u) => {
                if u.ndim() != 2 {
                    return Err(TensorError::DimensionMismatch {
                        expected: 2,
                        got: u.ndim(),
                    });
                }
                if u.shape() != gate.shape() {
                    let axis = if u.shape()[0] != gate.shape()[0] { 0 } else { 1 };
                    return Err(TensorError::ShapeMismatch {
                        axis,
                        expected: gate.shape()[axis],
                        got: u.shape()[axis],
                    });
                }
                (u.as_ptr() as *const u8, u.stride_bytes(0) as usize)
            }
            None => (core::ptr::null(), 0usize),
        };
        unsafe {
            nk_each_swiglu_e4m3(
                gate.as_ptr() as *const u8,
                up_ptr,
                y.as_mut_ptr() as *mut u8,
                rows,
                cols,
                gate_stride,
                up_stride,
                y_stride,
                input_scale,
            );
        }
        Ok(())
    }
}

// endregion: Fused SwiGLU

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, bf16, e2m3, e3m2, e4m3, e5m2, f16, FloatLike, NumberLike, TestableType};
    /// Test a binary elementwise op: convert inputs, apply `op`, compare element-wise.
    pub(crate) fn check_each_binary<Scalar, F>(
        a_vals: &[f32],
        b_vals: &[f32],
        op: F,
        expected_fn: fn(f64, f64) -> f64,
        label: &str,
    ) where
        Scalar: FloatLike + TestableType,
        F: FnOnce(&[Scalar], &[Scalar], &mut [Scalar]) -> Option<()>,
    {
        let a: Vec<Scalar> = a_vals.iter().map(|&v| Scalar::from_f32(v)).collect();
        let b: Vec<Scalar> = b_vals.iter().map(|&v| Scalar::from_f32(v)).collect();
        let mut result = vec![Scalar::zero(); a.len()];
        op(&a, &b, &mut result).unwrap();
        for (i, &r) in result.iter().enumerate() {
            let expected = expected_fn(a_vals[i] as f64, b_vals[i] as f64);
            assert_close(
                r.to_f64(),
                expected,
                Scalar::atol(),
                Scalar::rtol(),
                &format!("{}<{}>[{i}]", label, core::any::type_name::<Scalar>()),
            );
        }
    }

    /// Test a unary elementwise op over generated values.

    // region: Elementwise Operations

    fn check_each_scale<Scalar>(values: &[f32], alpha: f32, beta: f32)
    where
        Scalar: FloatLike + TestableType + EachScale,
        <Scalar as EachScale>::Scalar: FloatLike,
    {
        let a: Vec<Scalar> = values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let mut result = vec![Scalar::zero(); a.len()];
        let alpha_s = <<Scalar as EachScale>::Scalar>::from_f32(alpha);
        let beta_s = <<Scalar as EachScale>::Scalar>::from_f32(beta);
        Scalar::each_scale(&a, alpha_s, beta_s, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = alpha as f64 * values[i] as f64 + beta as f64;
            assert_close(
                r.to_f64(),
                expected,
                Scalar::atol(),
                Scalar::rtol(),
                &format!("each_scale<{}>[{i}]", core::any::type_name::<Scalar>()),
            );
        }
    }

    fn check_each_sum<Scalar>(values_a: &[f32], values_b: &[f32])
    where
        Scalar: FloatLike + TestableType + EachSum,
    {
        check_each_binary::<Scalar, _>(values_a, values_b, Scalar::each_sum, |a, b| a + b, "each_sum");
    }

    fn check_each_blend<Scalar>(values_a: &[f32], values_b: &[f32], alpha: f32, beta: f32)
    where
        Scalar: FloatLike + TestableType + EachBlend,
        <Scalar as EachBlend>::Scalar: FloatLike,
    {
        let a: Vec<Scalar> = values_a.iter().map(|&v| Scalar::from_f32(v)).collect();
        let b: Vec<Scalar> = values_b.iter().map(|&v| Scalar::from_f32(v)).collect();
        let mut result = vec![Scalar::zero(); a.len()];
        let alpha_s = <<Scalar as EachBlend>::Scalar>::from_f32(alpha);
        let beta_s = <<Scalar as EachBlend>::Scalar>::from_f32(beta);
        Scalar::each_blend(&a, &b, alpha_s, beta_s, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = alpha as f64 * values_a[i] as f64 + beta as f64 * values_b[i] as f64;
            assert_close(
                r.to_f64(),
                expected,
                Scalar::atol(),
                Scalar::rtol(),
                &format!("each_blend<{}>[{i}]", core::any::type_name::<Scalar>()),
            );
        }
    }

    fn check_each_fma<Scalar>(values_a: &[f32], values_b: &[f32], values_c: &[f32], alpha: f32, beta: f32)
    where
        Scalar: FloatLike + TestableType + EachFMA,
        <Scalar as EachFMA>::Scalar: FloatLike,
    {
        let a: Vec<Scalar> = values_a.iter().map(|&v| Scalar::from_f32(v)).collect();
        let b: Vec<Scalar> = values_b.iter().map(|&v| Scalar::from_f32(v)).collect();
        let c: Vec<Scalar> = values_c.iter().map(|&v| Scalar::from_f32(v)).collect();
        let mut result = vec![Scalar::zero(); a.len()];
        let alpha_s = <<Scalar as EachFMA>::Scalar>::from_f32(alpha);
        let beta_s = <<Scalar as EachFMA>::Scalar>::from_f32(beta);
        Scalar::each_fma(&a, &b, &c, alpha_s, beta_s, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = alpha as f64 * values_a[i] as f64 * values_b[i] as f64 + beta as f64 * values_c[i] as f64;
            assert_close(
                r.to_f64(),
                expected,
                Scalar::atol(),
                Scalar::rtol(),
                &format!("each_fma<{}>[{i}]", core::any::type_name::<Scalar>()),
            );
        }
    }

    #[test]
    fn scale_elementwise() {
        check_each_scale::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<f16>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<bf16>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<e2m3>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<e4m3>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<e5m2>(&[1.0, 2.0], 2.0, 0.0);
        check_each_scale::<e3m2>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<i8>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<u8>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<i32>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<u32>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<i16>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<u16>(&[1.0, 2.0, 3.0], 2.0, 0.0);
        check_each_scale::<i64>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
        check_each_scale::<u64>(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 1.0);
    }

    #[test]
    fn sum_elementwise() {
        check_each_sum::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
        check_each_sum::<e4m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
        check_each_sum::<e5m2>(&[1.0, 2.0], &[1.0, 1.0]);
        check_each_sum::<e3m2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
        check_each_sum::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<i32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<u32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<i16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<u16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<i64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        check_each_sum::<u64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn sum_elementwise_length_mismatch() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0];
        let mut result = vec![0.0f32; a.len()];
        assert!(f32::each_sum(&a, &b, &mut result).is_none());
    }

    #[test]
    fn blend_elementwise() {
        check_each_blend::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 0.5, 0.5);
        check_each_blend::<e4m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 0.5, 0.5);
        check_each_blend::<e5m2>(&[1.0, 2.0], &[1.0, 1.0], 0.5, 0.5);
        check_each_blend::<e3m2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 0.5, 0.5);
        check_each_blend::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
        check_each_blend::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.5, 0.5);
    }

    #[test]
    fn fma_elementwise() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[2.0, 3.0, 4.0];
        let c = &[1.0, 1.0, 1.0];
        check_each_fma::<f32>(a, b, c, 1.0, 1.0);
        check_each_fma::<f64>(a, b, c, 1.0, 1.0);
        check_each_fma::<f16>(a, b, c, 1.0, 1.0);
        check_each_fma::<bf16>(a, b, c, 1.0, 1.0);
        // e2m3 max is 7.5, so use small inputs that stay in range: 1*1+1=2
        check_each_fma::<e2m3>(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], c, 1.0, 1.0);
        check_each_fma::<e4m3>(a, b, c, 1.0, 1.0);
        let a2 = &[1.0, 2.0];
        let b2 = &[2.0, 3.0];
        let c2 = &[1.0, 1.0];
        check_each_fma::<e5m2>(a2, b2, c2, 1.0, 1.0);
        check_each_fma::<e3m2>(a, b, c, 1.0, 1.0);
        check_each_fma::<i8>(a, b, c, 1.0, 1.0);
        check_each_fma::<u8>(a, b, c, 1.0, 1.0);
        check_each_fma::<i32>(a, b, c, 1.0, 1.0);
        check_each_fma::<u32>(a, b, c, 1.0, 1.0);
        check_each_fma::<i16>(a, b, c, 1.0, 1.0);
        check_each_fma::<u16>(a, b, c, 1.0, 1.0);
        check_each_fma::<i64>(a, b, c, 1.0, 1.0);
        check_each_fma::<u64>(a, b, c, 1.0, 1.0);
    }

    #[test]
    fn large_elementwise() {
        let values: Vec<f32> = (0..1536).map(|i| i as f32).collect();
        check_each_scale::<f32>(&values, 2.0, 0.5);

        let b: Vec<f32> = (0..1536).map(|i| (i as f32) * 2.0).collect();
        check_each_sum::<f32>(&values, &b);
    }

    // endregion: Elementwise Operations

    // region: tensor-shaped wrappers (ScaleOps / SumOps)

    #[test]
    fn tensor_add_tensor_via_sum_ops() {
        use crate::tensor::{SliceRange, Tensor};
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let left = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let right = Tensor::<f32>::try_full(&[3, 4], 2.0).unwrap();

        let left_even = left
            .try_slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();
        let right_even = right
            .try_slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        let added = left_even.try_add_tensor(&right_even).unwrap();
        assert_eq!(added.shape(), &[3, 2]);
        assert_eq!(added.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn tensor_add_tensor_into_owning_destination() {
        use crate::tensor::Tensor;
        use crate::SumOps;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let left = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let right = Tensor::<f32>::try_full(&[3, 4], 2.0).unwrap();

        let mut out = Tensor::<f32>::try_full(&[3, 4], 0.0).unwrap();
        left.try_add_tensor_into(&right, &mut out).unwrap();
        assert_eq!(out.as_slice()[0], 2.0);
        assert_eq!(out.as_slice()[11], 13.0);
    }

    #[test]
    fn tensor_mul_scalar_via_scale_ops() {
        use crate::tensor::{SliceRange, Tensor};
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let even = source
            .try_slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();
        let scaled = even.try_mul_scalar(0.5).unwrap();
        assert_eq!(scaled.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn tensor_add_scalar_inplace_via_scale_ops() {
        use crate::tensor::Tensor;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut tensor = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        tensor.add_scalar_inplace(1.0);
        assert_eq!(tensor.as_slice()[0], 1.0);
        assert_eq!(tensor.as_slice()[11], 12.0);
    }

    #[test]
    fn tensor_sin_into_via_trig_sin_ops() {
        use crate::tensor::{SliceRange, Tensor};
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let even = source
            .try_slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();
        let mut sin_out = Tensor::<f32>::try_full(&[3, 2], 0.0).unwrap();
        {
            let mut span = sin_out.span();
            even.try_sin_into(&mut span).unwrap();
        }
        assert_eq!(sin_out.shape(), &[3, 2]);
        // First element is sin(0) which is exactly 0.
        assert!((sin_out.as_slice()[0] - 0.0).abs() < 1e-6);
    }

    // endregion

    // region: in-place == out-of-place (TensorSpan ownership of in-place)

    #[test]
    fn inplace_scale_matches_out_of_place() {
        use crate::tensor::Tensor;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut inplace = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        inplace.scale_inplace(2.0, 1.0);
        let expected = source.view().try_scale_tensor(2.0, 1.0).unwrap();
        assert_eq!(inplace.as_slice(), expected.as_slice());
    }

    #[test]
    fn inplace_scale_matches_out_of_place_f64() {
        use crate::tensor::Tensor;
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let mut inplace = Tensor::<f64>::try_from_slice(&data, &[3, 4]).unwrap();
        let source = Tensor::<f64>::try_from_slice(&data, &[3, 4]).unwrap();
        inplace.scale_inplace(-0.5, 3.0);
        let expected = source.view().try_scale_tensor(-0.5, 3.0).unwrap();
        assert_eq!(inplace.as_slice(), expected.as_slice());
    }

    #[test]
    fn inplace_add_scalar_matches_out_of_place() {
        use crate::tensor::Tensor;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut inplace = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        inplace.add_scalar_inplace(2.5);
        let expected = source.view().try_add_scalar(2.5).unwrap();
        assert_eq!(inplace.as_slice(), expected.as_slice());
    }

    #[test]
    fn inplace_sin_matches_out_of_place() {
        use crate::tensor::Tensor;
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.25).collect();
        let mut inplace = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        inplace.sin_inplace();
        let expected = source.view().try_sin().unwrap();
        assert_eq!(inplace.as_slice(), expected.as_slice());
    }

    #[test]
    fn inplace_add_tensor_matches_out_of_place() {
        use crate::tensor::Tensor;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let other = Tensor::<f32>::try_full(&[3, 4], 2.0).unwrap();
        let mut inplace = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        inplace.try_add_tensor_inplace(&other).unwrap();
        let expected = source.view().try_add_tensor(&other.view()).unwrap();
        assert_eq!(inplace.as_slice(), expected.as_slice());
    }

    #[test]
    fn inplace_sub_tensor_matches_out_of_place() {
        use crate::tensor::Tensor;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let other = Tensor::<f32>::try_full(&[3, 4], 3.0).unwrap();
        let mut inplace = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        inplace.try_sub_tensor_inplace(&other).unwrap();
        let expected = source.view().try_sub_tensor(&other.view()).unwrap();
        assert_eq!(inplace.as_slice(), expected.as_slice());
    }

    #[test]
    fn inplace_mul_tensor_matches_out_of_place() {
        use crate::tensor::Tensor;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let other_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let other = Tensor::<f32>::try_from_slice(&other_data, &[3, 4]).unwrap();
        let mut inplace = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        inplace.try_mul_tensor_inplace(&other).unwrap();
        let expected = source.view().try_mul_tensor(&other.view()).unwrap();
        assert_eq!(inplace.as_slice(), expected.as_slice());
    }

    // endregion

    fn silu(x: f64) -> f64 { x / (1.0 + (-x).exp()) }

    fn check_swiglu<Scalar>(values: &[f32], with_up: bool)
    where
        Scalar: FloatLike + TestableType + EachSwiglu,
    {
        let rows = 2;
        let gate: Vec<Scalar> = values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let up: Vec<Scalar> = values.iter().map(|&v| Scalar::from_f32(0.5 - v)).collect();
        let cols = gate.len() / rows;
        let gate_t = crate::tensor::Tensor::<Scalar>::try_from_slice(&gate, &[rows, cols]).unwrap();
        let up_t = crate::tensor::Tensor::<Scalar>::try_from_slice(&up, &[rows, cols]).unwrap();
        let mut y_t = crate::tensor::Tensor::<Scalar>::try_full(&[rows, cols], Scalar::zero()).unwrap();
        let up_ref = if with_up { Some(&up_t) } else { None };
        Scalar::swiglu_into(&gate_t, up_ref, &mut y_t, 1.0).unwrap();
        let y = y_t.as_slice().to_vec();
        for i in 0..gate.len() {
            let g = Scalar::from_f32(values[i]).to_f64();
            let mut expected = silu(g);
            if with_up {
                expected *= Scalar::from_f32(0.5 - values[i]).to_f64();
            }
            let expected = Scalar::from_f32(expected as f32).to_f64();
            assert_close(
                y[i].to_f64(),
                expected,
                Scalar::atol() * 4.0,
                Scalar::rtol() * 4.0,
                &format!("swiglu<{}>[{i}]", core::any::type_name::<Scalar>()),
            );
        }
    }

    #[test]
    fn swiglu_and_silu() {
        let values: Vec<f32> = (0..64).map(|i| ((i % 13) as f32 - 6.0) * 0.4).collect();
        check_swiglu::<f32>(&values, true);
        check_swiglu::<f32>(&values, false);
        check_swiglu::<bf16>(&values, true);
        check_swiglu::<e4m3>(&values, true);
    }

    #[test]
    fn swiglu_gate_up_halves() {
        use crate::tensor::{SliceRange, Tensor};
        // gate|up are the two strided column halves of one `[rows, 2*cols]` buffer (row stride 2*cols).
        let rows = 3;
        let cols = 8;
        let buf: Vec<f32> = (0..rows * 2 * cols).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let wide = Tensor::<f32>::try_from_slice(&buf, &[rows, 2 * cols]).unwrap();
        let gate = wide
            .view()
            .try_slice(&[SliceRange::Full, SliceRange::range(0, cols)][..])
            .unwrap();
        let up = wide
            .view()
            .try_slice(&[SliceRange::Full, SliceRange::range(cols, 2 * cols)][..])
            .unwrap();
        let mut y = Tensor::<f32>::try_full(&[rows, cols], 0.0f32).unwrap();
        f32::swiglu_into(&gate, Some(&up), &mut y, 1.0).unwrap();

        // Contiguous reference over dense copies of the same two halves.
        let mut gate_c = vec![0.0f32; rows * cols];
        let mut up_c = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                gate_c[r * cols + c] = buf[r * 2 * cols + c];
                up_c[r * cols + c] = buf[r * 2 * cols + cols + c];
            }
        }
        let gate_ct = Tensor::<f32>::try_from_slice(&gate_c, &[rows, cols]).unwrap();
        let up_ct = Tensor::<f32>::try_from_slice(&up_c, &[rows, cols]).unwrap();
        let mut y_ref = Tensor::<f32>::try_full(&[rows, cols], 0.0f32).unwrap();
        f32::swiglu_into(&gate_ct, Some(&up_ct), &mut y_ref, 1.0).unwrap();

        for i in 0..rows * cols {
            assert!(
                (y.as_slice()[i] - y_ref.as_slice()[i]).abs() < 1e-5,
                "strided gate|up SwiGLU mismatch at {i}"
            );
        }
    }
}
