//! Elementwise operations and trigonometry — slice traits and tensor-shaped wrappers.
//!
//! This module provides:
//!
//! - Slice-level traits:
//!   - [`EachScale`]: Linear scaling (alpha * x + beta)
//!   - [`EachSum`]: Elementwise addition of two vectors
//!   - [`EachBlend`]: Weighted blend of two vectors
//!   - [`EachFMA`]: Fused multiply-add (a * alpha + b * beta)
//!   - [`EachSin`], [`EachCos`], [`EachATan`]: Trigonometric functions
//!   - [`Trigonometry`]: Blanket trait combining `EachSin + EachCos + EachATan`
//! - Tensor-shaped extension traits (auto-implemented on every [`crate::tensor::TensorRef`]):
//!   - [`ScaleOps`], [`SumOps`], [`BlendOps`], [`FmaOps`]: Tensor wrappers around the slice traits
//!   - [`TrigSinOps`], [`TrigCosOps`], [`TrigAtanOps`]: Tensor wrappers around the trig traits
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

use crate::tensor::{
    try_reborrow_tensor_inplace, try_reborrow_tensor_into, Global, Tensor, TensorError, TensorRef,
};
use crate::types::{bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, StorageElement};

#[link(name = "numkong")]
extern "C" {
    // Trigonometry
    fn nk_each_sin_f32(inputs: *const f32, n: usize, outputs: *mut f32);
    fn nk_each_sin_f64(inputs: *const f64, n: usize, outputs: *mut f64);
    fn nk_each_sin_f16(inputs: *const u16, n: usize, outputs: *mut u16);
    fn nk_each_cos_f32(inputs: *const f32, n: usize, outputs: *mut f32);
    fn nk_each_cos_f64(inputs: *const f64, n: usize, outputs: *mut f64);
    fn nk_each_cos_f16(inputs: *const u16, n: usize, outputs: *mut u16);
    fn nk_each_atan_f32(inputs: *const f32, n: usize, outputs: *mut f32);
    fn nk_each_atan_f64(inputs: *const f64, n: usize, outputs: *mut f64);
    fn nk_each_atan_f16(inputs: *const u16, n: usize, outputs: *mut u16);

    // Elementwise operations
    fn nk_each_scale_f64(
        a: *const f64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_scale_f32(
        a: *const f32,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_scale_f16(
        a: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_scale_bf16(
        a: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_scale_i8(
        a: *const i8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_each_scale_u8(
        a: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_scale_i16(
        a: *const i16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i16,
    );
    fn nk_each_scale_u16(
        a: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_scale_i32(
        a: *const i32,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i32,
    );
    fn nk_each_scale_u32(
        a: *const u32,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u32,
    );
    fn nk_each_scale_i64(
        a: *const i64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i64,
    );
    fn nk_each_scale_u64(
        a: *const u64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u64,
    );
    fn nk_each_scale_e4m3(
        a: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_scale_e5m2(
        a: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_scale_e2m3(
        a: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_scale_e3m2(
        a: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

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

    fn nk_each_blend_f64(
        a: *const f64,
        b: *const f64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
    fn nk_each_blend_f32(
        a: *const f32,
        b: *const f32,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_blend_f16(
        a: *const u16,
        b: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_blend_bf16(
        a: *const u16,
        b: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_blend_i8(
        a: *const i8,
        b: *const i8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i8,
    );
    fn nk_each_blend_u8(
        a: *const u8,
        b: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_blend_i16(
        a: *const i16,
        b: *const i16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut i16,
    );
    fn nk_each_blend_u16(
        a: *const u16,
        b: *const u16,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u16,
    );
    fn nk_each_blend_i32(
        a: *const i32,
        b: *const i32,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i32,
    );
    fn nk_each_blend_u32(
        a: *const u32,
        b: *const u32,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u32,
    );
    fn nk_each_blend_i64(
        a: *const i64,
        b: *const i64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut i64,
    );
    fn nk_each_blend_u64(
        a: *const u64,
        b: *const u64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut u64,
    );
    fn nk_each_blend_e4m3(
        a: *const u8,
        b: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_blend_e5m2(
        a: *const u8,
        b: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_blend_e2m3(
        a: *const u8,
        b: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );
    fn nk_each_blend_e3m2(
        a: *const u8,
        b: *const u8,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut u8,
    );

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
    fn nk_each_scale_f32c(
        a: *const f32,
        n: usize,
        alpha: *const f32,
        beta: *const f32,
        result: *mut f32,
    );
    fn nk_each_scale_f64c(
        a: *const f64,
        n: usize,
        alpha: *const f64,
        beta: *const f64,
        result: *mut f64,
    );
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
}

// Complex fallback helpers

fn complex_each_sum_fallback<Scalar>(
    a: &[Scalar],
    b: &[Scalar],
    result: &mut [Scalar],
) -> Option<()>
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

fn complex_each_scale_fallback<Scalar>(
    a: &[Scalar],
    alpha: Scalar,
    beta: Scalar,
    result: &mut [Scalar],
) -> Option<()>
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
    for (((left, right), third), out) in a.iter().zip(b.iter()).zip(c.iter()).zip(result.iter_mut())
    {
        *out = alpha * *left * *right + beta * *third;
    }
    Some(())
}

// region: EachSin

/// Computes **element-wise sine** of a vector.
pub trait EachSin: Sized + StorageElement {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl EachSin for f64 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_each_sin_f64(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }
}

impl EachSin for f32 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_each_sin_f32(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }
}

impl EachSin for f16 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_sin_f16(
                inputs.as_ptr() as *const u16,
                inputs.len(),
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

// endregion: EachSin

// region: EachCos

/// Computes **element-wise cosine** of a vector.
pub trait EachCos: Sized + StorageElement {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl EachCos for f64 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_each_cos_f64(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }
}

impl EachCos for f32 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_each_cos_f32(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }
}

impl EachCos for f16 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_cos_f16(
                inputs.as_ptr() as *const u16,
                inputs.len(),
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

// endregion: EachCos

// region: EachATan

/// Computes **element-wise arctangent** of a vector.
pub trait EachATan: Sized + StorageElement {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;
}

impl EachATan for f64 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_each_atan_f64(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }
}

impl EachATan for f32 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_each_atan_f32(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }
}

impl EachATan for f16 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_each_atan_f16(
                inputs.as_ptr() as *const u16,
                inputs.len(),
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }
}

// endregion: EachATan

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
    fn each_scale(
        a: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
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
}

impl EachScale for f64c {
    type Scalar = f64c;
    fn each_scale(
        a: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachScale for f32c {
    type Scalar = f32c;
    fn each_scale(
        a: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachScale for f16c {
    type Scalar = f16c;
    fn each_scale(
        a: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
        complex_each_scale_fallback(a, alpha, beta, result)
    }
}

impl EachScale for bf16c {
    type Scalar = bf16c;
    fn each_scale(
        a: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
        complex_each_scale_fallback(a, alpha, beta, result)
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
}

impl EachSum for f64 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_f64(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
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
}

impl EachSum for i8 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_i8(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
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
}

impl EachSum for i16 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_i16(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
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
}

impl EachSum for i32 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_i32(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
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
}

impl EachSum for i64 {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe { nk_each_sum_i64(a.as_ptr(), b.as_ptr(), a.len(), result.as_mut_ptr()) };
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
}

impl EachSum for f16c {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        complex_each_sum_fallback(a, b, result)
    }
}

impl EachSum for bf16c {
    fn each_sum(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<()> {
        complex_each_sum_fallback(a, b, result)
    }
}

// endregion: Sum

// region: WSum

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
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()>;
}

impl EachBlend for f64 {
    type Scalar = f64;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_f64(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for f32 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_f32(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for f16 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachBlend for bf16 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachBlend for i8 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_i8(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for u8 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_u8(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for i16 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_i16(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for u16 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for i32 {
    type Scalar = f64;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_i32(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for u32 {
    type Scalar = f64;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for i64 {
    type Scalar = f64;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_i64(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for u64 {
    type Scalar = f64;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return None;
        }
        unsafe {
            nk_each_blend_u64(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &alpha,
                &beta,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl EachBlend for e4m3 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachBlend for e5m2 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachBlend for e2m3 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachBlend for e3m2 {
    type Scalar = f32;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachBlend for f64c {
    type Scalar = f64c;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachBlend for f32c {
    type Scalar = f32c;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachBlend for f16c {
    type Scalar = f16c;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
        complex_each_blend_fallback(a, b, alpha, beta, result)
    }
}

impl EachBlend for bf16c {
    type Scalar = bf16c;
    fn each_blend(
        a: &[Self],
        b: &[Self],
        alpha: Self::Scalar,
        beta: Self::Scalar,
        result: &mut [Self],
    ) -> Option<()> {
        complex_each_blend_fallback(a, b, alpha, beta, result)
    }
}

// endregion: WSum

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
}

impl EachFMA for f64 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for f32 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for f16 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for bf16 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for i8 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for u8 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for e4m3 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for e5m2 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for e2m3 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for e3m2 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for i16 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for u16 {
    type Scalar = f32;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f32,
        beta: f32,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for i32 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for u32 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for i64 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
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
}

impl EachFMA for u64 {
    type Scalar = f64;
    fn each_fma(
        a: &[Self],
        b: &[Self],
        c: &[Self],
        alpha: f64,
        beta: f64,
        result: &mut [Self],
    ) -> Option<()> {
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
}

// endregion: FMA

/// `Trigonometry` bundles trigonometric functions: EachSin, EachCos, and EachATan.
pub trait Trigonometry: EachSin + EachCos + EachATan {}
impl<Scalar: EachSin + EachCos + EachATan> Trigonometry for Scalar {}

// region: Tensor-shaped trigonometry (moved from crate::tensor)

/// Extension trait: element-wise sine for any [`TensorRef`] implementor.
pub trait TrigSinOps<Scalar: Clone + EachSin, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
{
    fn try_sin(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_sin()
    }
}

impl<Scalar: Clone + EachSin, const R: usize, C: TensorRef<Scalar, R>> TrigSinOps<Scalar, R> for C {}

/// Extension trait: element-wise cosine for any [`TensorRef`] implementor.
pub trait TrigCosOps<Scalar: Clone + EachCos, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
{
    fn try_cos(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_cos()
    }
}

impl<Scalar: Clone + EachCos, const R: usize, C: TensorRef<Scalar, R>> TrigCosOps<Scalar, R> for C {}

/// Extension trait: element-wise arctangent for any [`TensorRef`] implementor.
pub trait TrigAtanOps<Scalar: Clone + EachATan, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
{
    fn try_atan(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_atan()
    }
}

impl<Scalar: Clone + EachATan, const R: usize, C: TensorRef<Scalar, R>> TrigAtanOps<Scalar, R>
    for C
{
}

impl<Scalar: Clone + EachSin, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    /// Element-wise sine: result\[i\] = sin(self\[i\])
    pub fn sin(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_sin()
    }

    /// Element-wise sine in-place (infallible — self vs self always matches).
    pub fn sin_inplace(&mut self) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_sin_into(span))
            .expect("inplace trig op on self cannot fail")
    }

    pub fn try_sin_inplace(&mut self) -> Result<(), TensorError> {
        self.sin_inplace();
        Ok(())
    }
}

impl<Scalar: Clone + EachCos, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    /// Element-wise cosine: result\[i\] = cos(self\[i\])
    pub fn cos(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_cos()
    }

    /// Element-wise cosine in-place (infallible — self vs self always matches).
    pub fn cos_inplace(&mut self) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_cos_into(span))
            .expect("inplace trig op on self cannot fail")
    }

    pub fn try_cos_inplace(&mut self) -> Result<(), TensorError> {
        self.cos_inplace();
        Ok(())
    }
}

impl<Scalar: Clone + EachATan, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    /// Element-wise arctangent: result\[i\] = atan(self\[i\])
    pub fn atan(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_atan()
    }

    /// Element-wise arctangent in-place (infallible — self vs self always matches).
    pub fn atan_inplace(&mut self) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_atan_into(span))
            .expect("inplace trig op on self cannot fail")
    }

    pub fn try_atan_inplace(&mut self) -> Result<(), TensorError> {
        self.atan_inplace();
        Ok(())
    }
}

// endregion: Tensor-shaped trigonometry

// region: Tensor-shaped tolerance equality (moved from crate::tensor)

use crate::types::{is_close, FloatConvertible, NumberLike};

/// Extension trait: tolerance-based equality for any [`TensorRef`] implementor.
///
/// Uses the formula `|a - b| <= atol + rtol * |b|` per element.
/// Returns `false` if shapes differ.
pub trait AllCloseOps<Scalar: FloatConvertible, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
where
    Scalar::DimScalar: NumberLike,
{
    fn allclose(
        &self,
        other: &(impl TensorRef<Scalar, MAX_RANK> + ?Sized),
        atol: f64,
        rtol: f64,
    ) -> bool {
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

// region: Tensor-shaped scale / sum / blend / fma (moved from crate::tensor)

/// Extension trait: scalar arithmetic for any [`TensorRef`] implementor.
pub trait ScaleOps<Scalar: Clone + EachScale, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    fn try_add_scalar(
        &self,
        scalar: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_add_scalar(scalar)
    }

    fn try_sub_scalar(
        &self,
        scalar: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_scalar(scalar)
    }

    fn try_mul_scalar(
        &self,
        scalar: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_scalar(scalar)
    }
}

impl<Scalar: Clone + EachScale, const R: usize, C: TensorRef<Scalar, R>> ScaleOps<Scalar, R> for C where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy
{
}

impl<Scalar: Clone + EachScale, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    pub fn try_add_scalar_into(
        &self,
        scalar: Scalar::Scalar,
        out: &mut Tensor<Scalar, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_add_scalar_into(scalar, span)
        })
    }

    pub fn try_sub_scalar_into(
        &self,
        scalar: Scalar::Scalar,
        out: &mut Tensor<Scalar, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_sub_scalar_into(scalar, span)
        })
    }

    pub fn try_mul_scalar_into(
        &self,
        scalar: Scalar::Scalar,
        out: &mut Tensor<Scalar, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_mul_scalar_into(scalar, span)
        })
    }

    pub fn try_add_scalar_inplace(&mut self, scalar: Scalar::Scalar) -> Result<(), TensorError> {
        self.add_scalar_inplace(scalar);
        Ok(())
    }

    pub fn try_sub_scalar_inplace(&mut self, scalar: Scalar::Scalar) -> Result<(), TensorError> {
        self.sub_scalar_inplace(scalar);
        Ok(())
    }

    pub fn try_mul_scalar_inplace(&mut self, scalar: Scalar::Scalar) -> Result<(), TensorError> {
        self.mul_scalar_inplace(scalar);
        Ok(())
    }

    /// Element-wise add scalar in-place (infallible — self vs self always matches).
    pub fn add_scalar_inplace(&mut self, scalar: Scalar::Scalar) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_add_scalar_into(scalar, span))
            .expect("inplace scalar op on self cannot fail")
    }

    /// Element-wise subtract scalar in-place (infallible — self vs self always matches).
    pub fn sub_scalar_inplace(&mut self, scalar: Scalar::Scalar) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_sub_scalar_into(scalar, span))
            .expect("inplace scalar op on self cannot fail")
    }

    /// Element-wise multiply scalar in-place (infallible — self vs self always matches).
    pub fn mul_scalar_inplace(&mut self, scalar: Scalar::Scalar) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_mul_scalar_into(scalar, span))
            .expect("inplace scalar op on self cannot fail")
    }
}

impl<Scalar: Clone + EachScale, const MAX_RANK: usize> core::ops::AddAssign<Scalar::Scalar>
    for Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    fn add_assign(&mut self, scalar: Scalar::Scalar) {
        self.add_scalar_inplace(scalar);
    }
}

impl<Scalar: Clone + EachScale, const MAX_RANK: usize> core::ops::SubAssign<Scalar::Scalar>
    for Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    fn sub_assign(&mut self, scalar: Scalar::Scalar) {
        self.sub_scalar_inplace(scalar);
    }
}

impl<Scalar: Clone + EachScale, const MAX_RANK: usize> core::ops::MulAssign<Scalar::Scalar>
    for Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    fn mul_assign(&mut self, scalar: Scalar::Scalar) {
        self.mul_scalar_inplace(scalar);
    }
}

/// Extension trait: element-wise addition for any [`TensorRef`] implementor.
pub trait SumOps<Scalar: Clone + EachSum, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
{
    fn try_add_tensor(
        &self,
        other: &(impl TensorRef<Scalar, MAX_RANK> + ?Sized),
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_add_tensor(&other.view())
    }
}

impl<Scalar: Clone + EachSum, const R: usize, C: TensorRef<Scalar, R>> SumOps<Scalar, R> for C {}

impl<Scalar: Clone + EachSum, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    pub fn try_add_tensor_into(
        &self,
        other: &Tensor<Scalar, Global, MAX_RANK>,
        out: &mut Tensor<Scalar, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_add_tensor_into(&other.view(), span)
        })
    }

    pub fn try_add_tensor_inplace(
        &mut self,
        other: &Tensor<Scalar, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_add_tensor_into(&other.view(), span)
        })
    }
}

/// Extension trait: element-wise subtraction for any [`TensorRef`] implementor.
pub trait BlendOps<Scalar: Clone + EachBlend, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    fn try_sub_tensor(
        &self,
        other: &(impl TensorRef<Scalar, MAX_RANK> + ?Sized),
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_tensor(&other.view())
    }
}

impl<Scalar: Clone + EachBlend, const R: usize, C: TensorRef<Scalar, R>> BlendOps<Scalar, R> for C where
    Scalar::Scalar: From<f32> + Copy
{
}

impl<Scalar: Clone + EachBlend, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    pub fn try_sub_tensor_into(
        &self,
        other: &Tensor<Scalar, Global, MAX_RANK>,
        out: &mut Tensor<Scalar, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_sub_tensor_into(&other.view(), span)
        })
    }

    pub fn try_sub_tensor_inplace(
        &mut self,
        other: &Tensor<Scalar, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_sub_tensor_into(&other.view(), span)
        })
    }
}

/// Extension trait: element-wise multiplication for any [`TensorRef`] implementor.
pub trait FmaOps<Scalar: Clone + EachFMA, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    fn try_mul_tensor(
        &self,
        other: &(impl TensorRef<Scalar, MAX_RANK> + ?Sized),
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_tensor(&other.view())
    }
}

impl<Scalar: Clone + EachFMA, const R: usize, C: TensorRef<Scalar, R>> FmaOps<Scalar, R> for C where
    Scalar::Scalar: From<f32> + Copy
{
}

impl<Scalar: Clone + EachFMA, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    pub fn try_mul_tensor_into(
        &self,
        other: &Tensor<Scalar, Global, MAX_RANK>,
        out: &mut Tensor<Scalar, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_mul_tensor_into(&other.view(), span)
        })
    }

    pub fn try_mul_tensor_inplace(
        &mut self,
        other: &Tensor<Scalar, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_mul_tensor_into(&other.view(), span)
        })
    }
}

// endregion: Tensor-shaped scale / sum / blend / fma

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        assert_close, bf16, e2m3, e3m2, e4m3, e5m2, f16, FloatLike, NumberLike, TestableType,
    };
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
    pub(crate) fn check_each_unary<Scalar, F>(
        count: usize,
        gen_fn: fn(usize, usize) -> f64,
        op: F,
        ref_fn: fn(f64) -> f64,
        label: &str,
    ) where
        Scalar: FloatLike + TestableType,
        F: FnOnce(&[Scalar], &mut [Scalar]) -> Option<()>,
    {
        let values: Vec<f64> = (0..count).map(|i| gen_fn(i, count)).collect();
        let a: Vec<Scalar> = values.iter().map(|&v| Scalar::from_f32(v as f32)).collect();
        let mut result = vec![Scalar::zero(); count];
        op(&a, &mut result).unwrap();
        for (i, r) in result.iter().enumerate() {
            let expected = ref_fn(values[i]);
            assert_close(
                r.to_f64(),
                expected,
                Scalar::atol() * 10000.0,
                Scalar::rtol() * 10000.0,
                &format!("{}<{}>[{}]", label, core::any::type_name::<Scalar>(), i),
            );
        }
    }

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
        check_each_binary::<Scalar, _>(
            values_a,
            values_b,
            Scalar::each_sum,
            |a, b| a + b,
            "each_sum",
        );
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

    fn check_each_fma<Scalar>(
        values_a: &[f32],
        values_b: &[f32],
        values_c: &[f32],
        alpha: f32,
        beta: f32,
    ) where
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
            let expected = alpha as f64 * values_a[i] as f64 * values_b[i] as f64
                + beta as f64 * values_c[i] as f64;
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

    // endregion

    // region: Trigonometry

    fn check_each_sin<Scalar>(count: usize)
    where
        Scalar: FloatLike + TestableType + EachSin,
    {
        use core::f64::consts::PI;
        check_each_unary::<Scalar, _>(
            count,
            |i, n| (i as f64) * 2.0 * PI / (n as f64),
            Scalar::sin,
            f64::sin,
            "sin",
        );
    }

    fn check_each_cos<Scalar>(count: usize)
    where
        Scalar: FloatLike + TestableType + EachCos,
    {
        use core::f64::consts::PI;
        check_each_unary::<Scalar, _>(
            count,
            |i, n| (i as f64) * 2.0 * PI / (n as f64),
            Scalar::cos,
            f64::cos,
            "cos",
        );
    }

    fn check_each_atan<Scalar>(count: usize)
    where
        Scalar: FloatLike + TestableType + EachATan,
    {
        check_each_unary::<Scalar, _>(
            count,
            |i, n| -5.0 + 10.0 * (i as f64) / (n as f64),
            Scalar::atan,
            f64::atan,
            "atan",
        );
    }

    #[test]
    fn sin_elementwise() {
        check_each_sin::<f32>(97);
        check_each_sin::<f64>(97);
        check_each_sin::<f16>(97);
    }

    #[test]
    fn cos_elementwise() {
        check_each_cos::<f32>(97);
        check_each_cos::<f64>(97);
        check_each_cos::<f16>(97);
    }

    #[test]
    fn atan_elementwise() {
        check_each_atan::<f32>(100);
        check_each_atan::<f64>(100);
        check_each_atan::<f16>(100);
    }

    // endregion

    // region: tensor-shaped wrappers (ScaleOps / SumOps / TrigSinOps)

    #[test]
    fn tensor_add_tensor_via_sum_ops() {
        use crate::tensor::{SliceRange, Tensor};
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let left = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let right = Tensor::<f32>::try_full(&[3, 4], 2.0).unwrap();

        let left_even = left
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();
        let right_even = right
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        let added = left_even.try_add_tensor(&right_even).unwrap();
        assert_eq!(added.shape(), &[3, 2]);
        assert_eq!(added.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn tensor_add_tensor_into_owning_destination() {
        use crate::tensor::Tensor;
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
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();
        let scaled = even.try_mul_scalar(0.5).unwrap();
        assert_eq!(scaled.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn tensor_add_scalar_inplace_via_scale_ops() {
        use crate::tensor::Tensor;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut tensor = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        tensor.try_add_scalar_inplace(1.0).unwrap();
        assert_eq!(tensor.as_slice()[0], 1.0);
        assert_eq!(tensor.as_slice()[11], 12.0);
    }

    #[test]
    fn tensor_sin_into_via_trig_sin_ops() {
        use crate::tensor::{SliceRange, Tensor};
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let even = source
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
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
}
