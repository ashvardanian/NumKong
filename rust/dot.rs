//! Dot products: real and complex inner products.
//!
//! This module provides:
//!
//! - [`Dot`]: Inner product of two vectors
//! - [`VDot`]: Complex-valued dot product
//!
//! # Accumulator Widening — The Core Value Proposition
//!
//! Every spatial kernel promotes its accumulator to a wider type than the inputs.
//! This is not cosmetic — on long vectors of low-precision data, naive same-width
//! accumulation overflows silently for integers or loses huge amounts of precision
//! for half-floats. NumKong widens systematically so every kernel in this module
//! returns a result that matches a textbook-accurate reference in the wide type:
//!
//! - **`f32` → `f64`**: single-precision inputs accumulate in double precision. A
//!   `dot` over 2048 `f32` values returns `f64` and uses Neumaier-compensated
//!   summation internally.
//! - **`f16` → `f32`** and **`bf16` → `f32`**: half-precision dots, norms, and
//!   distances accumulate in `f32` rather than clamping at `f16::MAX = 65 504`.
//! - **`i8` → `i32`** (unsigned `u8` → `u32`): byte-level quantised inputs widen
//!   into 32-bit integer accumulators. Two `i8` vectors of all `100`s and length
//!   2048 have a true dot of `20 480 000`, which overflows `i8` (saturates at
//!   `±127`) and `i16` (`±32 767`) but is exact in `i32`.
//! - **FP8 variants** — `e4m3` / `e5m2` / `e2m3` / `e3m2` — all accumulate in `f32`.
//! - **4-bit packed** `i4x2` / `u4x2` behave like `i8` / `u8` but compute over
//!   double the element count because each byte holds two logical values.
//! - **1-bit packed** `u1x8` counts bit coincidences over eight logical values
//!   per byte into a `u32`, so a 4096-bit embedding cannot saturate its result.
//!
//! This widening is the reason quantised retrieval pipelines — for example BFloat16 or
//! INT8 embeddings in vector search — can rely on NumKong without post-hoc rescaling.
//!
//! # Example — INT8 Dot Without Overflow
//!
//! ```
//! use numkong::Dot;
//!
//! // 2048-dim i8 vectors full of 100s: true dot = 2048 * 100 * 100 = 20_480_000,
//! // which would overflow i16 but fits exactly in i32.
//! let left = vec![100_i8; 2048];
//! let right = vec![100_i8; 2048];
//! let exact: i32 = i8::dot(&left, &right).unwrap();
//! assert_eq!(exact, 20_480_000);
//! ```

use crate::types::{bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, i4x2, u1x8, u4x2, StorageElement};

#[link(name = "numkong")]
extern "C" {

    // Vector dot products
    fn nk_dot_i8(a: *const i8, b: *const i8, c: usize, d: *mut i32);
    fn nk_dot_u8(a: *const u8, b: *const u8, c: usize, d: *mut u32);
    fn nk_dot_f16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_dot_bf16(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_dot_e4m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_dot_e5m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_dot_e2m3(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_dot_e3m2(a: *const u8, b: *const u8, c: usize, d: *mut f32);
    fn nk_dot_f32(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_dot_f64(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    fn nk_dot_f16c(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_dot_bf16c(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_dot_f32c(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_dot_f64c(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    fn nk_vdot_f16c(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_vdot_bf16c(a: *const u16, b: *const u16, c: usize, d: *mut f32);
    fn nk_vdot_f32c(a: *const f32, b: *const f32, c: usize, d: *mut f64);
    fn nk_vdot_f64c(a: *const f64, b: *const f64, c: usize, d: *mut f64);

    // Sub-byte integer kernels
    fn nk_dot_i4(a: *const u8, b: *const u8, n: usize, result: *mut i32);
    fn nk_dot_u4(a: *const u8, b: *const u8, n: usize, result: *mut u32);
    fn nk_dot_u1(a: *const u8, b: *const u8, n: usize, result: *mut u32);
}

// region: Dot

/// Computes the **dot product** (inner product) between two vectors.
///
/// d = ∑ᵢ aᵢ × bᵢ
///
/// Range: unbounded. Returns `None` if lengths differ.
///
/// Implemented for: `f64`, `f32`, `f16`, `bf16`, `i8`, `u8`,
/// `e4m3`, `e5m2`, `e2m3`, `e3m2`, `i4x2`, `u4x2`, `u1x8`.
///
/// On `u1x8` this is the binary inner product — the number of positions where
/// both bits are set — which is the intersection count that Tanimoto similarity
/// divides by the union. Use [`crate::Jaccard`] when you want that ratio.
///
/// # Example
/// ```
/// use numkong::Dot;
/// let a = vec![1.0_f32, 2.0, 3.0];
/// let b = vec![4.0_f32, 5.0, 6.0];
/// let result = f32::dot(&a, &b).unwrap();
/// assert!((result - 32.0).abs() < 1e-5);
/// ```
pub trait Dot: StorageElement {
    type Output;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output>;

    /// Alias for `dot`.
    fn inner(a: &[Self], b: &[Self]) -> Option<Self::Output> { Self::dot(a, b) }
}

impl Dot for f64 {
    type Output = f64;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_f64(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for f32 {
    type Output = f64;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for f16 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_f16(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for bf16 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_bf16(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for i8 {
    type Output = i32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        unsafe { nk_dot_i8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for u8 {
    type Output = u32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        unsafe { nk_dot_u8(a.as_ptr(), b.as_ptr(), a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for e4m3 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_e4m3(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for e5m2 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_e5m2(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for e2m3 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_e2m3(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for e3m2 {
    type Output = f32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0.0;
        unsafe { nk_dot_e3m2(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len(), &mut result) };
        Some(result)
    }
}

impl Dot for i4x2 {
    type Output = i32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        let element_count = a.len() * 2; // Each i4x2 contains 2 elements
        unsafe {
            nk_dot_i4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                element_count,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Dot for u4x2 {
    type Output = u32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        let element_count = a.len() * 2; // Each u4x2 contains 2 elements
        unsafe {
            nk_dot_u4(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                element_count,
                &mut result,
            )
        };
        Some(result)
    }
}

impl Dot for u1x8 {
    type Output = u32;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result: Self::Output = 0;
        let n = a.len() * 8; // Each u1x8 contains 8 bits
        unsafe { nk_dot_u1(a.as_ptr() as *const u8, b.as_ptr() as *const u8, n, &mut result) };
        Some(result)
    }
}

impl Dot for f16c {
    type Output = f32c;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_dot_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

impl Dot for bf16c {
    type Output = f32c;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_dot_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

impl Dot for f32c {
    type Output = f64c;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_dot_f32c(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

impl Dot for f64c {
    type Output = f64c;
    fn dot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_dot_f64c(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

// endregion: Dot

// region: VDot

/// Computes the conjugating (Hermitian) dot product.
///
/// For real-valued types this is identical to [`Dot::dot`]. For complex-valued
/// types it computes `∑ᵢ conj(aᵢ) × bᵢ`, which is the convention used in quantum
/// mechanics and most linear-algebra textbooks — BLAS `cdotc`, NumPy `vdot`.
///
/// Unlike an ordinary complex dot product, the Hermitian form produces a proper
/// inner product: `vdot(x, x)` is always a non-negative real number equal to
/// `‖x‖²`, and `vdot(x, y) == conj(vdot(y, x))`. This matters in retrieval and
/// signal-processing pipelines where cancellation between a vector and its
/// conjugate should leave the norm intact.
pub trait VDot: Dot {
    /// Hermitian inner product. On real-valued types this falls back to `Dot::dot`;
    /// on complex types it returns `∑ᵢ conj(aᵢ) × bᵢ` computed in the widened
    /// accumulator described by `Dot::Output`.
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> { Self::dot(a, b) }
}

impl VDot for f64 {}
impl VDot for f32 {}
impl VDot for f16 {}
impl VDot for bf16 {}
impl VDot for i8 {}
impl VDot for u8 {}
impl VDot for e4m3 {}
impl VDot for e5m2 {}
impl VDot for e2m3 {}
impl VDot for e3m2 {}
impl VDot for i4x2 {}
impl VDot for u4x2 {}
impl VDot for u1x8 {}

impl VDot for f16c {
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_vdot_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

impl VDot for bf16c {
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_vdot_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

impl VDot for f32c {
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_vdot_f32c(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

impl VDot for f64c {
    fn vdot(a: &[Self], b: &[Self]) -> Option<Self::Output> {
        if a.len() != b.len() {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_vdot_f64c(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len(),
                result.as_mut_ptr(),
            )
        };
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

// endregion: VDot

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curved::Bilinear;
    use crate::types::{
        assert_close, bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, i4x2, u1x8, u4x2, FloatLike,
        NumberLike, StorageElement, TestableType,
    };

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

    // region: Dot Products

    fn check_dot<Scalar>(a_vals: &[f32], b_vals: &[f32], expected: f64)
    where
        Scalar: FloatLike + TestableType + Dot,
        Scalar::Output: FloatLike,
    {
        check_binary::<Scalar, Scalar::Output, _>(a_vals, b_vals, Scalar::dot, expected, "dot");
    }

    #[test]
    fn dot() {
        check_dot::<f32>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<f64>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<f16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<bf16>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<i8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<u8>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<e4m3>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<e5m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<e2m3>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 6.0);
        check_dot::<e3m2>(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 32.0);
        check_dot::<i4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 12.0);
        check_dot::<u4x2>(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 12.0);
    }

    /// `u1x8` sits outside `check_dot`, which needs a float-convertible scalar,
    /// so the bit-packed inner product is checked against hand-counted overlaps.
    #[test]
    fn dot_u1x8() {
        // 0b1111_0000 & 0b1100_1100 → 2 bits; 0b1010_1010 & 0b0101_0101 → 0 bits.
        let left = vec![u1x8(0b1111_0000), u1x8(0b1010_1010)];
        let right = vec![u1x8(0b1100_1100), u1x8(0b0101_0101)];
        assert_eq!(u1x8::dot(&left, &right).unwrap(), 2);
        assert_eq!(u1x8::vdot(&left, &right).unwrap(), 2);

        // Self-overlap equals the population count, and length mismatch is rejected.
        assert_eq!(u1x8::dot(&left, &left).unwrap(), 8);
        assert!(u1x8::dot(&left, &right[..1]).is_none());
    }

    // endregion

    // region: Complex Products

    trait ComplexValue {
        fn real(&self) -> f64;
        fn imag(&self) -> f64;
    }
    impl ComplexValue for f32c {
        fn real(&self) -> f64 { self.re as f64 }
        fn imag(&self) -> f64 { self.im as f64 }
    }
    impl ComplexValue for f64c {
        fn real(&self) -> f64 { self.re }
        fn imag(&self) -> f64 { self.im }
    }

    trait ComplexSample: Copy + StorageElement + Dot + VDot + Bilinear {
        fn from_real_imag(re: f32, im: f32) -> Self;
        fn atol() -> f64;
        fn rtol() -> f64;
    }

    impl ComplexSample for f16c {
        fn from_real_imag(re: f32, im: f32) -> Self {
            Self {
                re: f16::from_f32(re),
                im: f16::from_f32(im),
            }
        }
        fn atol() -> f64 { 5e-2 }
        fn rtol() -> f64 { 5e-2 }
    }

    impl ComplexSample for bf16c {
        fn from_real_imag(re: f32, im: f32) -> Self {
            Self {
                re: bf16::from_f32(re),
                im: bf16::from_f32(im),
            }
        }
        fn atol() -> f64 { 5e-2 }
        fn rtol() -> f64 { 5e-2 }
    }

    impl ComplexSample for f32c {
        fn from_real_imag(re: f32, im: f32) -> Self { Self { re, im } }
        fn atol() -> f64 { 1e-6 }
        fn rtol() -> f64 { 1e-6 }
    }

    impl ComplexSample for f64c {
        fn from_real_imag(re: f32, im: f32) -> Self {
            Self {
                re: re as f64,
                im: im as f64,
            }
        }
        fn atol() -> f64 { 1e-12 }
        fn rtol() -> f64 { 1e-12 }
    }

    /// Test a complex two-input operation with real + imaginary expected outputs.
    fn check_complex<Scalar, R, F>(
        a: &[(f32, f32)],
        b: &[(f32, f32)],
        op: F,
        expected_re: f64,
        expected_im: f64,
        label: &str,
    ) where
        Scalar: ComplexSample,
        R: ComplexValue,
        F: FnOnce(&[Scalar], &[Scalar]) -> Option<R>,
    {
        let a_t: Vec<Scalar> = a.iter().map(|&(re, im)| Scalar::from_real_imag(re, im)).collect();
        let b_t: Vec<Scalar> = b.iter().map(|&(re, im)| Scalar::from_real_imag(re, im)).collect();
        let result = op(&a_t, &b_t).unwrap();
        let tol = Scalar::atol() + Scalar::rtol() * expected_re.abs().max(expected_im.abs());
        assert_close(
            result.real(),
            expected_re,
            tol,
            0.0,
            &format!("{}<{}> real", label, core::any::type_name::<Scalar>()),
        );
        assert_close(
            result.imag(),
            expected_im,
            tol,
            0.0,
            &format!("{}<{}> imag", label, core::any::type_name::<Scalar>()),
        );
    }

    fn check_complex_dot<Scalar>(a: &[f32], b: &[f32], expected_re: f64, expected_im: f64)
    where
        Scalar: ComplexSample,
        <Scalar as Dot>::Output: ComplexValue,
    {
        let a_pairs: Vec<(f32, f32)> = a.chunks_exact(2).map(|chunk| (chunk[0], chunk[1])).collect();
        let b_pairs: Vec<(f32, f32)> = b.chunks_exact(2).map(|chunk| (chunk[0], chunk[1])).collect();
        check_complex::<Scalar, <Scalar as Dot>::Output, _>(
            &a_pairs,
            &b_pairs,
            <Scalar as Dot>::dot,
            expected_re,
            expected_im,
            "complex_dot",
        );
    }

    fn check_complex_vdot<Scalar>(a: &[f32], b: &[f32], expected_re: f64, expected_im: f64)
    where
        Scalar: ComplexSample,
        <Scalar as Dot>::Output: ComplexValue,
    {
        let a_pairs: Vec<(f32, f32)> = a.chunks_exact(2).map(|chunk| (chunk[0], chunk[1])).collect();
        let b_pairs: Vec<(f32, f32)> = b.chunks_exact(2).map(|chunk| (chunk[0], chunk[1])).collect();
        check_complex::<Scalar, <Scalar as Dot>::Output, _>(
            &a_pairs,
            &b_pairs,
            Scalar::vdot,
            expected_re,
            expected_im,
            "complex_vdot",
        );
    }

    fn check_complex_bilinear_identity<Scalar>(element_count: usize)
    where
        Scalar: ComplexSample,
        <Scalar as Bilinear>::Output: ComplexValue,
    {
        let mut a = vec![Scalar::zero(); element_count];
        let mut b = vec![Scalar::zero(); element_count];
        a[0] = Scalar::one();
        b[0] = Scalar::one();
        let mut c = vec![Scalar::zero(); element_count * element_count];
        for i in 0..element_count {
            c[i * element_count + i] = Scalar::one();
        }
        let result = Scalar::bilinear(&a, &b, &c).unwrap();
        let tol = Scalar::atol() + Scalar::rtol();
        assert_close(
            result.real(),
            1.0,
            tol,
            0.0,
            &format!("complex_bilinear<{}> real", core::any::type_name::<Scalar>()),
        );
        assert_close(
            result.imag(),
            0.0,
            tol,
            0.0,
            &format!("complex_bilinear<{}> imag", core::any::type_name::<Scalar>()),
        );
    }

    #[test]
    fn complex_dot_vdot_bilinear() {
        // [1+2i, 3+4i] · [5+6i, 7+8i]
        let a = &[1.0_f32, 2.0, 3.0, 4.0];
        let b = &[5.0_f32, 6.0, 7.0, 8.0];

        // dot: (-18, 68)
        check_complex_dot::<f64c>(a, b, -18.0, 68.0);
        check_complex_dot::<f32c>(a, b, -18.0, 68.0);
        check_complex_dot::<f16c>(a, b, -18.0, 68.0);
        check_complex_dot::<bf16c>(a, b, -18.0, 68.0);

        // vdot (conjugate): (70, -8)
        check_complex_vdot::<f64c>(a, b, 70.0, -8.0);
        check_complex_vdot::<f32c>(a, b, 70.0, -8.0);
        check_complex_vdot::<f16c>(a, b, 70.0, -8.0);
        check_complex_vdot::<bf16c>(a, b, 70.0, -8.0);

        // bilinear: identity matrix, unit vector → (1, 0)
        check_complex_bilinear_identity::<f64c>(4);
        check_complex_bilinear_identity::<f32c>(4);
        check_complex_bilinear_identity::<f16c>(4);
        check_complex_bilinear_identity::<bf16c>(4);
    }

    // endregion

    // region: Denormal (subnormal) inputs
    //
    // Verify that distance kernels produce correct results when fed IEEE-754
    // denormal (subnormal) values. This guards against FTZ/DAZ silently
    // flushing tiny values to zero.

    #[test]
    fn dot_f32_denormals() {
        // Largest f32 denormal: exp=0, mantissa=0x7FFFFF → ≈ 1.1754942e-38
        let d = f32::from_bits(0x007F_FFFF);
        // dot([d,d,d], [d,d,d]) = 3 * d * d
        // F32 dot accumulates in F64 internally, so the result is precise.
        let expected = 3.0 * (d as f64) * (d as f64);
        let a = [d, d, d];
        let b = [d, d, d];
        let result = f32::dot(&a, &b).unwrap();
        assert_close(result as f64, expected, 1e-50, 1e-6, "dot<f32> denormal");
    }

    #[test]
    fn dot_f64_denormals() {
        // Largest f64 denormal: exp=0, mantissa all-ones → ≈ 2.225e-308
        let d = f64::from_bits(0x000F_FFFF_FFFF_FFFF);
        let expected = 3.0 * d * d;
        let a = [d, d, d];
        let b = [d, d, d];
        let result = f64::dot(&a, &b).unwrap();
        // F64 dot uses Neumaier compensation; the product d*d underflows to 0 in f64
        // since d ≈ 2.2e-308, d*d ≈ 5e-616, which is below f64 min denormal ~5e-324.
        // So expected = 0.0 and result should also be 0.0 — the key test is that
        // the kernel doesn't crash or produce NaN/Inf.
        assert!(result.is_finite(), "dot<f64> denormal produced non-finite: {result}");
        assert_close(result, expected, 1e-300, 1e-6, "dot<f64> denormal");
    }

    #[test]
    fn dot_f16_denormals() {
        // F16 denormal: exp=0, mantissa=0x0100 → 2^-14 * 256/1024 = 2^-16 ≈ 1.526e-5
        // F16 denormals upcast to normal f32 values, so no penalty or flushing.
        let d = f16(0x0100);
        let d_f64 = d.to_f64();
        let expected = 3.0 * d_f64 * d_f64;
        check_dot::<f16>(
            &[d.to_f32(), d.to_f32(), d.to_f32()],
            &[d.to_f32(), d.to_f32(), d.to_f32()],
            expected,
        );
    }

    #[test]
    fn dot_bf16_denormals() {
        // BF16 denormal: exp=0, mantissa=0x7F, the largest BF16 denormal
        // BF16 shares f32's exponent, so this IS a f32 denormal after upcast.
        let d = bf16(0x007F);
        let d_f64 = d.to_f64();
        let expected = 3.0 * d_f64 * d_f64;
        check_dot::<bf16>(
            &[d.to_f32(), d.to_f32(), d.to_f32()],
            &[d.to_f32(), d.to_f32(), d.to_f32()],
            expected,
        );
    }

    // endregion
}
