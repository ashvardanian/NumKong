//! Scalar types and conversion trait for mixed-precision computing.
//!
//! This module defines **15+ numeric types** spanning 1-bit packed storage to
//! 128-bit complex. All of them implement [`StorageElement`] (for containers) and
//! [`NumberLike`] (for conversion to / from `f32` / `f64`), so generic code can
//! treat them uniformly.
//!
//! ## IEEE / Brain Floats
//!
//! - [`struct@f16`]: IEEE 754 half-precision (16-bit) floating point
//! - [`bf16`]: Brain floating point (bfloat16) — truncated single precision
//! - The Rust primitives [`f32`](prim@f32) and [`f64`](prim@f64) also implement
//!   these traits
//!
//! ## 8-bit Floats (OCP / MX)
//!
//! - [`e4m3`]: 4-exponent, 3-mantissa FP8 (max ≈ 448)
//! - [`e5m2`]: 5-exponent, 2-mantissa FP8 (wider range, ~57 344 max)
//! - [`e2m3`]: 2-exponent, 3-mantissa 6-bit float padded to 8-bit (max 7.5)
//! - [`e3m2`]: 3-exponent, 2-mantissa 6-bit float padded to 8-bit (max 28)
//!
//! ## Signed / Unsigned Integers
//!
//! `i8` / `u8`, `i16` / `u16`, `i32` / `u32`, `i64` / `u64` all implement
//! [`StorageElement`] and [`NumberLike`] via the primitive blanket impls below.
//!
//! ## Sub-byte Packed Types
//!
//! - [`i4x2`]: Packed pair of signed 4-bit integers (one byte → two values)
//! - [`u4x2`]: Packed pair of unsigned 4-bit integers
//! - [`u1x8`]: Packed 8 binary values in a single byte
//!
//! ## Complex
//!
//! - [`f16c`], [`bf16c`], [`f32c`], [`f64c`] — generic `complex<Scalar>` wrappers
//!   that expose `re`, `im`, `conj`, and arithmetic operators.
//!
//! All types support standard arithmetic operations and conversion to / from
//! `f32` via the [`FloatLike`] trait.

#![allow(non_camel_case_types)]

#[link(name = "numkong")]
extern "C" {
    // Scalar square-root / reciprocal-square-root (backing the `Roots` trait).
    fn nk_f32_sqrt(x: f32) -> f32;
    fn nk_f32_rsqrt(x: f32) -> f32;
    fn nk_f64_sqrt(x: f64) -> f64;
    fn nk_f64_rsqrt(x: f64) -> f64;
    fn nk_f16_sqrt(x: u16) -> u16;
    fn nk_f16_rsqrt(x: u16) -> u16;
    fn nk_f32_to_f16(src: *const f32, dest: *mut u16);
    fn nk_f16_to_f32(src: *const u16, dest: *mut f32);
    fn nk_f32_to_bf16(src: *const f32, dest: *mut u16);
    fn nk_bf16_to_f32(src: *const u16, dest: *mut f32);
    fn nk_f32_to_e4m3(src: *const f32, dest: *mut u8);
    fn nk_e4m3_to_f32(src: *const u8, dest: *mut f32);
    fn nk_f32_to_e5m2(src: *const f32, dest: *mut u8);
    fn nk_e5m2_to_f32(src: *const u8, dest: *mut f32);
    fn nk_f32_to_e2m3(src: *const f32, dest: *mut u8);
    fn nk_e2m3_to_f32(src: *const u8, dest: *mut f32);
    fn nk_f32_to_e3m2(src: *const f32, dest: *mut u8);
    fn nk_e3m2_to_f32(src: *const u8, dest: *mut f32);
}

/// Scalar square-root and reciprocal-square-root operations backed by NumKong's exported kernels.
///
/// Unlike `f32::sqrt` / `f64::sqrt`, this routes into the hand-tuned NumKong C kernels; on ISAs with
/// a dedicated reciprocal-sqrt (`vrsqrte` on NEON, `vrsqrt14` on AVX-512) `rsqrt` uses a single
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

/// Compatibility function for pre 1.85 Rust versions lacking `f32::abs`.
#[inline(always)]
pub(crate) fn f32_abs_compat(x: f32) -> f32 { f32::from_bits(x.to_bits() & 0x7FFF_FFFF) }

/// Compatibility function for pre 1.85 Rust versions lacking `f32::round`.
#[inline(always)]
pub(crate) fn f32_round_compat(x: f32) -> f32 {
    let truncated_value = x as i32 as f32;
    let fractional_part = x - truncated_value;
    if fractional_part >= 0.5 {
        truncated_value + 1.0
    } else if fractional_part <= -0.5 {
        truncated_value - 1.0
    } else {
        truncated_value
    }
}

/// Compatibility function for pre 1.85 Rust versions lacking `f64::round`.
#[inline(always)]
pub(crate) fn f64_round_compat(x: f64) -> f64 {
    let truncated_value = x as i64 as f64;
    let fractional_part = x - truncated_value;
    if fractional_part >= 0.5 {
        truncated_value + 1.0
    } else if fractional_part <= -0.5 {
        truncated_value - 1.0
    } else {
        truncated_value
    }
}

/// Check whether two values are within tolerance: `|a - b| <= atol + rtol * |b|`.
#[inline]
pub fn is_close(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    let diff = if a > b { a - b } else { b - a };
    diff <= atol + rtol * (if b >= 0.0 { b } else { -b })
}

// region: f16 Type

/// Half-precision (16-bit) IEEE 754 floating-point number.
///
/// Layout: sign(1) + exponent(5) + mantissa(10), bias=15.
/// Range: ±65504, epsilon at 1.0 ≈ 9.77×10⁻⁴, subnormal min ≈ 5.96×10⁻⁸.
/// 30 722 of 63 488 finite values (48.4%) fall in [−1, +1].
/// All arithmetic via f32 upcast/downcast.
///
/// # Examples
///
/// ```
/// use numkong::f16;
///
/// // Create from f32
/// let half = f16::from_f32(3.14);
///
/// // Convert back to f32
/// let float = half.to_f32();
///
/// // Direct access to bits
/// let bits = half.0;
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct f16(pub u16);

impl f16 {
    /// Positive zero.
    pub const ZERO: Self = f16(0);
    /// Positive one.
    pub const ONE: Self = f16(0x3C00);
    /// Negative one.
    pub const NEG_ONE: Self = f16(0xBC00);
    /// Quiet NaN (Not a Number).
    pub const NAN: Self = f16(0x7E00);

    /// Converts an `f32` value to `f16`. Out-of-range values saturate.
    ///
    /// # Examples
    ///
    /// ```
    /// use numkong::f16;
    /// // Roundtrip a value well inside f16's ±65504 range.
    /// let half = f16::from_f32(3.14);
    /// assert!((half.to_f32() - 3.14).abs() < 1e-2);
    /// ```
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u16 = 0;
        unsafe { nk_f32_to_f16(&value, &mut result) };
        f16(result)
    }

    /// Converts this value back to `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use numkong::f16;
    /// let half = f16::from_f32(-2.5);
    /// assert_eq!(half.to_f32(), -2.5);
    /// ```
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_f16_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool { self.to_f32().is_nan() }

    /// Returns true if this value is ±∞.
    #[inline(always)]
    pub fn is_infinite(self) -> bool { self.to_f32().is_infinite() }

    /// Returns true if this number is neither infinite nor NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool { self.to_f32().is_finite() }

    /// Returns |self|.
    #[inline(always)]
    pub fn abs(self) -> Self { Self::from_f32(f32_abs_compat(self.to_f32())) }

    /// Returns ⌊self⌋. Requires `std`.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn floor(self) -> Self { Self::from_f32(self.to_f32().floor()) }

    /// Returns ⌈self⌉. Requires `std`.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn ceil(self) -> Self { Self::from_f32(self.to_f32().ceil()) }

    /// Rounds to the nearest integer; half-way cases go away from zero. Requires `std`.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn round(self) -> Self { Self::from_f32(self.to_f32().round()) }
}

impl core::fmt::Debug for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "f16({}, 0x{:04x})", self.to_f32(), self.0)
    }
}

impl core::fmt::Display for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            core::fmt::Display::fmt(&self.to_f32(), f)?;
            write!(f, " [0x{:04x}]", self.0)
        } else {
            core::fmt::Display::fmt(&self.to_f32(), f)
        }
    }
}

impl core::fmt::LowerHex for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::LowerHex::fmt(&self.0, f) }
}

impl core::fmt::UpperHex for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::UpperHex::fmt(&self.0, f) }
}

impl core::fmt::Binary for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::Binary::fmt(&self.0, f) }
}

impl core::ops::Add for f16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() + right.to_f32()) }
}

impl core::ops::Sub for f16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() - right.to_f32()) }
}

impl core::ops::Mul for f16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() * right.to_f32()) }
}

impl core::ops::Div for f16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() / right.to_f32()) }
}

impl core::ops::Neg for f16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output { Self::from_f32(-self.to_f32()) }
}

impl core::cmp::PartialOrd for f16 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> { self.to_f32().partial_cmp(&other.to_f32()) }
}

// endregion: f16 Type

// region: bf16 Type

/// BFloat16 (16-bit) floating-point number — truncated IEEE 754 single-precision.
///
/// Layout: sign(1) + exponent(8) + mantissa(7), bias=127.
/// Range: ±3.39×10³⁸ (same dynamic range as f32), epsilon at 1.0 ≈ 7.81×10⁻³.
/// 32 514 of 65 280 finite values (49.8%) fall in [−1, +1].
/// Wider dynamic range than f16 but lower precision (7 vs 10 mantissa bits).
///
/// # Examples
///
/// ```
/// use numkong::bf16;
///
/// let brain = bf16::from_f32(3.14);
/// let float = brain.to_f32();
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct bf16(pub u16);

impl bf16 {
    /// Positive zero.
    pub const ZERO: Self = bf16(0);

    /// Positive one.
    pub const ONE: Self = bf16(0x3F80);

    /// Negative one.
    pub const NEG_ONE: Self = bf16(0xBF80);
    /// Quiet NaN (Not a Number).
    pub const NAN: Self = bf16(0x7FC0);

    /// Converts an `f32` value to `bf16`. Out-of-range values saturate.
    ///
    /// # Examples
    ///
    /// ```
    /// use numkong::bf16;
    /// // bf16 matches f32's ±3.39e38 range but with only 7 bits of mantissa.
    /// let brain = bf16::from_f32(1.0e30);
    /// assert!((brain.to_f32() - 1.0e30).abs() < 1.0e28);
    /// ```
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u16 = 0;
        unsafe { nk_f32_to_bf16(&value, &mut result) };
        bf16(result)
    }

    /// Converts this value back to `f32`.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_bf16_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool { self.to_f32().is_nan() }

    /// Returns true if this value is positive or negative infinity.
    #[inline(always)]
    pub fn is_infinite(self) -> bool { self.to_f32().is_infinite() }

    /// Returns true if this number is neither infinite nor NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool { self.to_f32().is_finite() }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self { Self::from_f32(f32_abs_compat(self.to_f32())) }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn floor(self) -> Self { Self::from_f32(self.to_f32().floor()) }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn ceil(self) -> Self { Self::from_f32(self.to_f32().ceil()) }

    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn round(self) -> Self { Self::from_f32(self.to_f32().round()) }
}

impl core::fmt::Debug for bf16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "bf16({}, 0x{:04x})", self.to_f32(), self.0)
    }
}

impl core::fmt::Display for bf16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            core::fmt::Display::fmt(&self.to_f32(), f)?;
            write!(f, " [0x{:04x}]", self.0)
        } else {
            core::fmt::Display::fmt(&self.to_f32(), f)
        }
    }
}

impl core::fmt::LowerHex for bf16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::LowerHex::fmt(&self.0, f) }
}

impl core::fmt::UpperHex for bf16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::UpperHex::fmt(&self.0, f) }
}

impl core::fmt::Binary for bf16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::Binary::fmt(&self.0, f) }
}

impl core::ops::Add for bf16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() + right.to_f32()) }
}

impl core::ops::Sub for bf16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() - right.to_f32()) }
}

impl core::ops::Mul for bf16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() * right.to_f32()) }
}

impl core::ops::Div for bf16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() / right.to_f32()) }
}

impl core::ops::Neg for bf16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output { Self::from_f32(-self.to_f32()) }
}

impl core::cmp::PartialOrd for bf16 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> { self.to_f32().partial_cmp(&other.to_f32()) }
}

// endregion: bf16 Type

// region: e4m3 Type

/// 8-bit E4M3 floating-point number (OCP FP8).
///
/// Layout: sign(1) + exponent(4) + mantissa(3), bias=7.
/// Range: ±448, no infinities (all-ones exponent → NaN).
/// 114 of 254 finite values (44.9%) fall in [−1, +1].
/// Exact integer dot products via exponent-sum binning (29 bins).
///
/// # Examples
///
/// ```
/// use numkong::e4m3;
///
/// let fp8 = e4m3::from_f32(2.5);
/// let float = fp8.to_f32();
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct e4m3(pub u8);

impl e4m3 {
    pub const ZERO: Self = e4m3(0x00);
    pub const ONE: Self = e4m3(0x38);
    pub const NEG_ONE: Self = e4m3(0xB8);
    /// Quiet NaN (Not a Number).
    pub const NAN: Self = e4m3(0x7F);

    /// Converts an `f32` value to `e4m3`. Out-of-range values saturate.
    ///
    /// # Examples
    ///
    /// ```
    /// use numkong::e4m3;
    /// // e4m3 quantises to ~4 effective decimal digits inside ±448.
    /// let fp8 = e4m3::from_f32(3.5);
    /// assert_eq!(fp8.to_f32(), 3.5);
    /// ```
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u8 = 0;
        unsafe { nk_f32_to_e4m3(&value, &mut result) };
        e4m3(result)
    }

    /// Converts this value back to `f32`.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_e4m3_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool { (self.0 & 0x7F) == 0x7F }

    /// Returns true if this value is ±∞. Always false for E4M3 (no infinities).
    #[inline(always)]
    pub fn is_infinite(self) -> bool { false }

    /// Returns true if this number is neither infinite nor NaN.
    /// Note: E4M3 format has no infinities.
    #[inline(always)]
    pub fn is_finite(self) -> bool { !self.is_nan() }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self { Self::from_f32(f32_abs_compat(self.to_f32())) }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn floor(self) -> Self { Self::from_f32(self.to_f32().floor()) }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn ceil(self) -> Self { Self::from_f32(self.to_f32().ceil()) }

    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn round(self) -> Self { Self::from_f32(self.to_f32().round()) }
}

impl core::fmt::Debug for e4m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "e4m3({}, 0x{:02x})", self.to_f32(), self.0)
    }
}

impl core::fmt::Display for e4m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            core::fmt::Display::fmt(&self.to_f32(), f)?;
            write!(f, " [0x{:02x}]", self.0)
        } else {
            core::fmt::Display::fmt(&self.to_f32(), f)
        }
    }
}

impl core::fmt::LowerHex for e4m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::LowerHex::fmt(&self.0, f) }
}

impl core::fmt::UpperHex for e4m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::UpperHex::fmt(&self.0, f) }
}

impl core::fmt::Binary for e4m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::Binary::fmt(&self.0, f) }
}

impl core::ops::Add for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() + right.to_f32()) }
}

impl core::ops::Sub for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() - right.to_f32()) }
}

impl core::ops::Mul for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() * right.to_f32()) }
}

impl core::ops::Div for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() / right.to_f32()) }
}

impl core::ops::Neg for e4m3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output { Self::from_f32(-self.to_f32()) }
}

impl core::cmp::PartialOrd for e4m3 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> { self.to_f32().partial_cmp(&other.to_f32()) }
}

// endregion: e4m3 Type

// region: e5m2 Type

/// 8-bit E5M2 floating-point number (OCP FP8).
///
/// Layout: sign(1) + exponent(5) + mantissa(2), bias=15.
/// Range: ±57 344, supports infinities. Only 4 mantissa levels per exponent.
/// 122 of 248 finite values (49.2%) fall in [−1, +1].
/// High cancellation risk in dot products — consider compensated accumulation.
///
/// # Examples
///
/// ```
/// use numkong::e5m2;
///
/// let fp8 = e5m2::from_f32(2.5);
/// let float = fp8.to_f32();
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct e5m2(pub u8);

impl e5m2 {
    /// Positive zero.
    pub const ZERO: Self = e5m2(0x00);

    /// Positive one.
    pub const ONE: Self = e5m2(0x3C);

    /// Negative one.
    pub const NEG_ONE: Self = e5m2(0xBC);
    /// Quiet NaN (Not a Number).
    pub const NAN: Self = e5m2(0x7F);

    /// Convert an `f32` value to `e5m2`. Out-of-range values saturate.
    ///
    /// # Examples
    ///
    /// ```
    /// use numkong::e5m2;
    /// // e5m2 trades mantissa precision for wider range (up to ~57 344).
    /// let fp8 = e5m2::from_f32(1024.0);
    /// assert_eq!(fp8.to_f32(), 1024.0);
    /// ```
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u8 = 0;
        unsafe { nk_f32_to_e5m2(&value, &mut result) };
        e5m2(result)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_e5m2_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 2) & 0x1F;
        let mant = self.0 & 0x03;
        exp == 0x1F && mant != 0
    }

    /// Returns true if this value is positive or negative infinity.
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 2) & 0x1F;
        let mant = self.0 & 0x03;
        exp == 0x1F && mant == 0
    }

    /// Returns true if this number is neither infinite nor NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        let exp = (self.0 >> 2) & 0x1F;
        exp != 0x1F
    }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self { Self::from_f32(f32_abs_compat(self.to_f32())) }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn floor(self) -> Self { Self::from_f32(self.to_f32().floor()) }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn ceil(self) -> Self { Self::from_f32(self.to_f32().ceil()) }

    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn round(self) -> Self { Self::from_f32(self.to_f32().round()) }
}

impl core::fmt::Debug for e5m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "e5m2({}, 0x{:02x})", self.to_f32(), self.0)
    }
}

impl core::fmt::Display for e5m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            core::fmt::Display::fmt(&self.to_f32(), f)?;
            write!(f, " [0x{:02x}]", self.0)
        } else {
            core::fmt::Display::fmt(&self.to_f32(), f)
        }
    }
}

impl core::fmt::LowerHex for e5m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::LowerHex::fmt(&self.0, f) }
}

impl core::fmt::UpperHex for e5m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::UpperHex::fmt(&self.0, f) }
}

impl core::fmt::Binary for e5m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::Binary::fmt(&self.0, f) }
}

impl core::ops::Add for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() + right.to_f32()) }
}

impl core::ops::Sub for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() - right.to_f32()) }
}

impl core::ops::Mul for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() * right.to_f32()) }
}

impl core::ops::Div for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() / right.to_f32()) }
}

impl core::ops::Neg for e5m2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output { Self::from_f32(-self.to_f32()) }
}

impl core::cmp::PartialOrd for e5m2 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> { self.to_f32().partial_cmp(&other.to_f32()) }
}

// endregion: e5m2 Type

// region: e2m3 Type

/// 6-bit E2M3 micro-float (padded to 8-bit storage).
///
/// Layout: sign(1) + exponent(2) + mantissa(3), bias=1.
/// Range: ±7.5, no infinities. Only 64 total codes; 18 (28.1%) fall in [−1, +1].
/// 72% of codes lie outside [−1, +1] — poor resolution for normalized vectors.
/// Exact integer dot products via exponent-sum binning (15 bins).
///
/// # Examples
///
/// ```
/// use numkong::e2m3;
///
/// let fp6 = e2m3::from_f32(2.5);
/// let float = fp6.to_f32();
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct e2m3(pub u8);

impl e2m3 {
    /// Positive zero.
    pub const ZERO: Self = e2m3(0x00);

    /// Positive one.
    pub const ONE: Self = e2m3(0x08);

    /// Negative one.
    pub const NEG_ONE: Self = e2m3(0x28);

    /// Convert an `f32` value to `e2m3`. Out-of-range values saturate.
    ///
    /// # Examples
    ///
    /// ```
    /// use numkong::e2m3;
    /// // e2m3 covers ±7.5 with 3 mantissa bits — suitable for fine ratios
    /// // in a narrow dynamic range.
    /// let tiny = e2m3::from_f32(1.5);
    /// assert_eq!(tiny.to_f32(), 1.5);
    /// ```
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u8 = 0;
        unsafe { nk_f32_to_e2m3(&value, &mut result) };
        e2m3(result)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_e2m3_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    /// E2M3FN has no infinities - all special values are NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        false // E2M3FN has no NaN representation
    }

    /// Returns true if this value is positive or negative infinity.
    /// E2M3FN format has no infinities.
    #[inline(always)]
    pub fn is_infinite(self) -> bool { false }

    /// Returns true if this number is neither infinite nor NaN.
    /// Note: E2M3FN format has no infinities or NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool { true }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self {
        e2m3(self.0 & 0x1F) // Clear sign bit
    }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn floor(self) -> Self { Self::from_f32(self.to_f32().floor()) }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn ceil(self) -> Self { Self::from_f32(self.to_f32().ceil()) }

    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn round(self) -> Self { Self::from_f32(self.to_f32().round()) }
}

impl core::fmt::Debug for e2m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "e2m3({}, 0x{:02x})", self.to_f32(), self.0)
    }
}

impl core::fmt::Display for e2m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            core::fmt::Display::fmt(&self.to_f32(), f)?;
            write!(f, " [0x{:02x}]", self.0)
        } else {
            core::fmt::Display::fmt(&self.to_f32(), f)
        }
    }
}

impl core::fmt::LowerHex for e2m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::LowerHex::fmt(&self.0, f) }
}

impl core::fmt::UpperHex for e2m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::UpperHex::fmt(&self.0, f) }
}

impl core::fmt::Binary for e2m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::Binary::fmt(&self.0, f) }
}

impl core::ops::Add for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() + right.to_f32()) }
}

impl core::ops::Sub for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() - right.to_f32()) }
}

impl core::ops::Mul for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() * right.to_f32()) }
}

impl core::ops::Div for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() / right.to_f32()) }
}

impl core::ops::Neg for e2m3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output { Self::from_f32(-self.to_f32()) }
}

impl core::cmp::PartialOrd for e2m3 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> { self.to_f32().partial_cmp(&other.to_f32()) }
}

// endregion: e2m3 Type

// region: e3m2 Type

/// 6-bit E3M2 micro-float (padded to 8-bit storage).
///
/// Layout: sign(1) + exponent(3) + mantissa(2), bias=3.
/// Range: ±28, supports infinities. Only 64 total codes; 26 (40.6%) fall in [−1, +1].
/// Exact integer dot products via exponent-sum binning (15 bins).
///
/// # Examples
///
/// ```
/// use numkong::e3m2;
///
/// let fp6 = e3m2::from_f32(2.5);
/// let float = fp6.to_f32();
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct e3m2(pub u8);

impl e3m2 {
    /// Positive zero.
    pub const ZERO: Self = e3m2(0x00);

    /// Positive one.
    pub const ONE: Self = e3m2(0x0C);

    /// Negative one.
    pub const NEG_ONE: Self = e3m2(0x2C);

    /// Convert an `f32` value to `e3m2`. Out-of-range values saturate.
    ///
    /// # Examples
    ///
    /// ```
    /// use numkong::e3m2;
    /// // e3m2 trades mantissa bits for a larger range (±28).
    /// let tiny = e3m2::from_f32(16.0);
    /// assert_eq!(tiny.to_f32(), 16.0);
    /// ```
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let mut result: u8 = 0;
        unsafe { nk_f32_to_e3m2(&value, &mut result) };
        e3m2(result)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let mut result: f32 = 0.0;
        unsafe { nk_e3m2_to_f32(&self.0, &mut result) };
        result
    }

    /// Returns true if this value is NaN.
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 2) & 0x07;
        let mant = self.0 & 0x03;
        exp == 0x07 && mant != 0
    }

    /// Returns true if this value is positive or negative infinity.
    #[inline(always)]
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 2) & 0x07;
        let mant = self.0 & 0x03;
        exp == 0x07 && mant == 0
    }

    /// Returns true if this number is neither infinite nor NaN.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        let exp = (self.0 >> 2) & 0x07;
        exp != 0x07
    }

    /// Returns the absolute value of self.
    #[inline(always)]
    pub fn abs(self) -> Self {
        e3m2(self.0 & 0x1F) // Clear sign bit
    }

    /// Returns the largest integer less than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn floor(self) -> Self { Self::from_f32(self.to_f32().floor()) }

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn ceil(self) -> Self { Self::from_f32(self.to_f32().ceil()) }

    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    ///
    /// This method is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[inline(always)]
    pub fn round(self) -> Self { Self::from_f32(self.to_f32().round()) }
}

impl core::fmt::Debug for e3m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "e3m2({}, 0x{:02x})", self.to_f32(), self.0)
    }
}

impl core::fmt::Display for e3m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            core::fmt::Display::fmt(&self.to_f32(), f)?;
            write!(f, " [0x{:02x}]", self.0)
        } else {
            core::fmt::Display::fmt(&self.to_f32(), f)
        }
    }
}

impl core::fmt::LowerHex for e3m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::LowerHex::fmt(&self.0, f) }
}

impl core::fmt::UpperHex for e3m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::UpperHex::fmt(&self.0, f) }
}

impl core::fmt::Binary for e3m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::Binary::fmt(&self.0, f) }
}

impl core::ops::Add for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() + right.to_f32()) }
}

impl core::ops::Sub for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() - right.to_f32()) }
}

impl core::ops::Mul for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() * right.to_f32()) }
}

impl core::ops::Div for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, right: Self) -> Self::Output { Self::from_f32(self.to_f32() / right.to_f32()) }
}

impl core::ops::Neg for e3m2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output { Self::from_f32(-self.to_f32()) }
}

impl core::cmp::PartialOrd for e3m2 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> { self.to_f32().partial_cmp(&other.to_f32()) }
}

// endregion: e3m2 Type

// region: Block-Scale Bytes (Ue4m3, Ue8m0)

/// NVFP4 per-block scale byte: an unsigned E4M3 magnitude (sign bit forced to 0).
///
/// Stores one scale per 16-element NVFP4 block. Layout matches `nk_ue4m3_t` (1 byte).
/// Decodes to a positive `f32` multiplier; `from_f32` takes the absolute value.
///
/// # Examples
///
/// ```
/// use numkong::Ue4m3;
///
/// let scale = Ue4m3::from_f32(2.0);
/// assert_eq!(scale.to_f32(), 2.0);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Ue4m3(pub u8);

impl Ue4m3 {
    /// The all-zero byte (decodes to 0.0).
    pub const ZERO: Self = Ue4m3(0x00);

    /// Encodes an `f32` magnitude into a UE4M3 scale byte (absolute value, round-to-nearest).
    ///
    /// UE4M3 is E4M3 with the sign bit forced to 0, so this reuses the linkable E4M3 encoder
    /// on `|value|` and masks off the sign. NaN maps to the E4M3 NaN code (`0x7F`).
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        if value.is_nan() {
            return Ue4m3(0x7F);
        }
        let encoded = e4m3::from_f32(f32_abs_compat(value));
        Ue4m3(encoded.0 & 0x7F)
    }

    /// Decodes this UE4M3 scale byte to a non-negative `f32` multiplier.
    #[inline(always)]
    pub fn to_f32(self) -> f32 { e4m3(self.0 & 0x7F).to_f32() }
}

impl core::fmt::Debug for Ue4m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ue4m3({}, 0x{:02x})", self.to_f32(), self.0)
    }
}

impl StorageElement for Ue4m3 {
    fn zero() -> Self { Ue4m3(0) }
    fn one() -> Self { Ue4m3::from_f32(1.0) }
}

/// MX-family per-block scale byte: an unsigned power-of-two exponent (UE8M0).
///
/// Stores one scale per 32-element MX block. Layout matches `nk_ue8m0_t` (1 byte).
/// Decodes to a positive power-of-two `f32` multiplier; `from_f32` rounds the
/// magnitude UP to the smallest power of two that is ≥ `|value|`.
///
/// # Examples
///
/// ```
/// use numkong::Ue8m0;
///
/// let scale = Ue8m0::from_f32(4.0);
/// assert_eq!(scale.to_f32(), 4.0);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Ue8m0(pub u8);

impl Ue8m0 {
    /// Encodes an `f32` magnitude into a UE8M0 scale byte.
    ///
    /// Rounds the magnitude to the NEAREST power of two (round-to-nearest in log2 space, the
    /// OCP MX convention; the split point is the geometric midpoint √2·2ᵉ). NaN → `0xFF`
    /// (block-NaN sentinel), zero/subnormal → `0x00`, overflow/±∞ → `0xFE`.
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let abs_bits = value.to_bits() & 0x7FFF_FFFF;
        if abs_bits > 0x7F80_0000 {
            return Ue8m0(0xFF); // NaN
        }
        if abs_bits < 0x0080_0000 {
            return Ue8m0(0x00); // zero or subnormal
        }
        if abs_bits == 0x7F80_0000 {
            return Ue8m0(0xFE); // +Inf → saturate
        }
        let mut exp_biased = abs_bits >> 23;
        let mantissa = abs_bits & 0x7F_FFFF;
        // Round exponent to nearest: midpoint mantissa fraction √2-1 ≈ 0x3504F4.
        if mantissa >= 0x3504F4 {
            exp_biased += 1;
        }
        if exp_biased > 0xFE {
            return Ue8m0(0xFE); // saturate
        }
        Ue8m0(exp_biased as u8)
    }

    /// Decodes this UE8M0 scale byte to a power-of-two `f32` multiplier.
    ///
    /// `0x00` → `0.0`, `0xFF` → NaN, otherwise `2^(byte - 127)`.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let raw = self.0;
        if raw == 0 {
            return 0.0;
        }
        if raw == 0xFF {
            return f32::from_bits(0x7FC0_0000); // quiet NaN
        }
        // Power of two: f32 with biased exponent = raw, mantissa = 0.
        f32::from_bits((raw as u32) << 23)
    }
}

impl core::fmt::Debug for Ue8m0 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ue8m0({}, 0x{:02x})", self.to_f32(), self.0)
    }
}

impl StorageElement for Ue8m0 {
    fn zero() -> Self { Ue8m0(0) }
    fn one() -> Self { Ue8m0::from_f32(1.0) }
}

// endregion: Block-Scale Bytes (Ue4m3, Ue8m0)

// region: From<f32> Conversions

impl From<f32> for f16 {
    #[inline(always)]
    fn from(value: f32) -> Self { Self::from_f32(value) }
}
impl From<f16> for f32 {
    #[inline(always)]
    fn from(value: f16) -> Self { value.to_f32() }
}

impl From<f32> for bf16 {
    #[inline(always)]
    fn from(value: f32) -> Self { Self::from_f32(value) }
}
impl From<bf16> for f32 {
    #[inline(always)]
    fn from(value: bf16) -> Self { value.to_f32() }
}

impl From<f32> for e4m3 {
    #[inline(always)]
    fn from(value: f32) -> Self { Self::from_f32(value) }
}
impl From<e4m3> for f32 {
    #[inline(always)]
    fn from(value: e4m3) -> Self { value.to_f32() }
}

impl From<f32> for e5m2 {
    #[inline(always)]
    fn from(value: f32) -> Self { Self::from_f32(value) }
}
impl From<e5m2> for f32 {
    #[inline(always)]
    fn from(value: e5m2) -> Self { value.to_f32() }
}

impl From<f32> for e2m3 {
    #[inline(always)]
    fn from(value: f32) -> Self { Self::from_f32(value) }
}
impl From<e2m3> for f32 {
    #[inline(always)]
    fn from(value: e2m3) -> Self { value.to_f32() }
}

impl From<f32> for e3m2 {
    #[inline(always)]
    fn from(value: f32) -> Self { Self::from_f32(value) }
}
impl From<e3m2> for f32 {
    #[inline(always)]
    fn from(value: e3m2) -> Self { value.to_f32() }
}

// endregion: From<f32> Conversions

// region: u1x8 Type

/// Packed 8-bit bit-vector (8 booleans in one byte).
///
/// Layout: 8 bits packed into one byte, LSB = dimension 0.
/// Used for Hamming distance and Jaccard similarity via popcount.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct u1x8(pub u8);

impl core::fmt::Debug for u1x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "u1x8(0b{:08b}, 0x{:02x})", self.0, self.0)
    }
}

impl core::fmt::Display for u1x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            write!(f, "0b{:08b} [0x{:02x}]", self.0, self.0)
        } else {
            write!(f, "0b{:08b}", self.0)
        }
    }
}

impl core::fmt::LowerHex for u1x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::LowerHex::fmt(&self.0, f) }
}

impl core::fmt::UpperHex for u1x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::UpperHex::fmt(&self.0, f) }
}

impl core::fmt::Binary for u1x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::Binary::fmt(&self.0, f) }
}

impl u1x8 {
    /// Create from raw packed bits.
    #[inline(always)]
    pub const fn new(bits: u8) -> Self { u1x8(bits) }

    /// Get the raw packed bits.
    #[inline(always)]
    pub const fn bits(self) -> u8 { self.0 }

    /// Construct from 8 booleans (b0 = LSB, b7 = MSB).
    #[inline(always)]
    pub const fn from_bools(b0: bool, b1: bool, b2: bool, b3: bool, b4: bool, b5: bool, b6: bool, b7: bool) -> Self {
        u1x8(
            (b0 as u8)
                | ((b1 as u8) << 1)
                | ((b2 as u8) << 2)
                | ((b3 as u8) << 3)
                | ((b4 as u8) << 4)
                | ((b5 as u8) << 5)
                | ((b6 as u8) << 6)
                | ((b7 as u8) << 7),
        )
    }

    /// Extract to 8 booleans (b0 = LSB, b7 = MSB).
    #[inline(always)]
    pub const fn to_bools(self) -> (bool, bool, bool, bool, bool, bool, bool, bool) {
        (
            (self.0 & 1) != 0,
            (self.0 & 2) != 0,
            (self.0 & 4) != 0,
            (self.0 & 8) != 0,
            (self.0 & 16) != 0,
            (self.0 & 32) != 0,
            (self.0 & 64) != 0,
            (self.0 & 128) != 0,
        )
    }
}

impl From<(bool, bool, bool, bool, bool, bool, bool, bool)> for u1x8 {
    #[inline(always)]
    fn from(b: (bool, bool, bool, bool, bool, bool, bool, bool)) -> Self {
        u1x8::from_bools(b.0, b.1, b.2, b.3, b.4, b.5, b.6, b.7)
    }
}

impl From<u1x8> for (bool, bool, bool, bool, bool, bool, bool, bool) {
    #[inline(always)]
    fn from(v: u1x8) -> Self { v.to_bools() }
}

// endregion: u1x8 Type

// region: u4x2 Type

/// Packed 4-bit unsigned integer pair (2 × u4 in one byte).
///
/// Layout: low nibble = first element, high nibble = second element.
/// Range per element: [0, 15]. Elements zero-extended to u8 for arithmetic.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct u4x2(pub u8);

impl core::fmt::Debug for u4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (low, high) = self.to_u8s();
        write!(f, "u4x2({}, {}, 0x{:02x})", low, high, self.0)
    }
}

impl core::fmt::Display for u4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (low, high) = self.to_u8s();
        if f.alternate() {
            write!(f, "({}, {}) [0x{:02x}]", low, high, self.0)
        } else {
            write!(f, "({}, {})", low, high)
        }
    }
}

impl core::fmt::LowerHex for u4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::LowerHex::fmt(&self.0, f) }
}

impl core::fmt::UpperHex for u4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::UpperHex::fmt(&self.0, f) }
}

impl core::fmt::Binary for u4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::Binary::fmt(&self.0, f) }
}

impl u4x2 {
    /// Create from raw packed byte.
    #[inline(always)]
    pub const fn new(packed: u8) -> Self { u4x2(packed) }

    /// Get the raw packed byte.
    #[inline(always)]
    pub const fn packed(self) -> u8 { self.0 }

    /// Construct from two u8 values with saturation to 0..15.
    #[inline(always)]
    pub const fn from_u8s(low: u8, high: u8) -> Self {
        let low_sat = if low > 15 { 15 } else { low };
        let high_sat = if high > 15 { 15 } else { high };
        u4x2(low_sat | (high_sat << 4))
    }

    /// Extract to two u8 values (0..15 each).
    #[inline(always)]
    pub const fn to_u8s(self) -> (u8, u8) { (self.0 & 0x0F, self.0 >> 4) }
}

impl From<(u8, u8)> for u4x2 {
    #[inline(always)]
    fn from(v: (u8, u8)) -> Self { u4x2::from_u8s(v.0, v.1) }
}

impl From<u4x2> for (u8, u8) {
    #[inline(always)]
    fn from(v: u4x2) -> Self { v.to_u8s() }
}

// endregion: u4x2 Type

// region: i4x2 Type

/// Packed 4-bit signed integer pair (2 × i4 in one byte).
///
/// Layout: low nibble = first element, high nibble = second element (two's complement).
/// Range per element: [−8, +7]. Elements sign-extended to i8 for arithmetic.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct i4x2(pub u8);

impl core::fmt::Debug for i4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (low, high) = self.to_i8s();
        write!(f, "i4x2({}, {}, 0x{:02x})", low, high, self.0)
    }
}

impl core::fmt::Display for i4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (low, high) = self.to_i8s();
        if f.alternate() {
            write!(f, "({}, {}) [0x{:02x}]", low, high, self.0)
        } else {
            write!(f, "({}, {})", low, high)
        }
    }
}

impl core::fmt::LowerHex for i4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::LowerHex::fmt(&self.0, f) }
}

impl core::fmt::UpperHex for i4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::UpperHex::fmt(&self.0, f) }
}

impl core::fmt::Binary for i4x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { core::fmt::Binary::fmt(&self.0, f) }
}

impl i4x2 {
    /// Create from raw packed byte.
    #[inline(always)]
    pub const fn new(packed: u8) -> Self { i4x2(packed) }

    /// Get the raw packed byte.
    #[inline(always)]
    pub const fn packed(self) -> u8 { self.0 }

    /// Construct from two i8 values with saturation to -8..7.
    #[inline(always)]
    pub const fn from_i8s(low: i8, high: i8) -> Self {
        let low_sat = if low < -8 {
            -8
        } else if low > 7 {
            7
        } else {
            low
        };
        let high_sat = if high < -8 {
            -8
        } else if high > 7 {
            7
        } else {
            high
        };
        i4x2(((low_sat as u8) & 0x0F) | (((high_sat as u8) & 0x0F) << 4))
    }

    /// Extract to two i8 values (sign-extended from 4 bits).
    #[inline(always)]
    pub const fn to_i8s(self) -> (i8, i8) {
        let low = (self.0 & 0x0F) as i8;
        let high = ((self.0 >> 4) & 0x0F) as i8;
        // Sign extend from 4 bits: if bit 3 is set, fill upper bits with 1s
        let low = if low & 0x08 != 0 { low | (!0x0Fi8) } else { low };
        let high = if high & 0x08 != 0 { high | (!0x0Fi8) } else { high };
        (low, high)
    }
}

impl From<(i8, i8)> for i4x2 {
    #[inline(always)]
    fn from(v: (i8, i8)) -> Self { i4x2::from_i8s(v.0, v.1) }
}

impl From<i4x2> for (i8, i8) {
    #[inline(always)]
    fn from(v: i4x2) -> Self { v.to_i8s() }
}

// endregion: i4x2 Type

// region: StorageElement + NumberLike + FloatLike Traits

/// Minimal trait for types that can be stored in vectors and tensors.
///
/// Provides identity elements (`zero`, `one`) and sub-byte packing metadata.
/// Does not require numeric conversion — use [`NumberLike`] for that.
pub trait StorageElement: Sized + Copy + Clone + Default + core::fmt::Debug {
    /// The additive identity.
    fn zero() -> Self;
    /// The multiplicative identity.
    fn one() -> Self;
    /// Number of logical dimensions packed into one storage value.
    /// Default: 1 for all normal types. Override for sub-byte packed types.
    fn dimensions_per_value() -> usize { 1 }
}

/// Trait for types that support conversion to/from f32 with classification and constants.
///
/// Provides a unified interface for all numeric types used in NumKong,
/// including half-precision floats, mini-floats, integers, and packed types.
pub trait NumberLike: StorageElement {
    /// Convert from f32 to this type.
    fn from_f32(v: f32) -> Self;
    /// Convert from this type to f32.
    fn to_f32(self) -> f32;
    /// Convert from f64 to this type (default: via f32 roundtrip).
    fn from_f64(v: f64) -> Self { Self::from_f32(v as f32) }
    /// Convert from this type to f64 (default: via f32 roundtrip).
    fn to_f64(self) -> f64 { self.to_f32() as f64 }

    fn abs(self) -> Self { Self::from_f32(f32_abs_compat(self.to_f32())) }
    fn is_nan(self) -> bool { self.to_f32().is_nan() }
    fn is_finite(self) -> bool { self.to_f32().is_finite() }
    fn is_infinite(self) -> bool { self.to_f32().is_infinite() }

    fn has_infinity() -> bool { false }
    fn has_nan() -> bool { false }
    fn has_subnormals() -> bool { false }
    fn max_value() -> f32 { f32::MAX }
    fn min_positive() -> f32 { f32::MIN_POSITIVE }
}

/// Backward-compatible alias: any type implementing [`NumberLike`] also implements `FloatLike`.
pub trait FloatLike: NumberLike {}
impl<Scalar: NumberLike> FloatLike for Scalar {}

impl StorageElement for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl NumberLike for f32 {
    fn from_f32(v: f32) -> Self { v }
    fn to_f32(self) -> f32 { self }
    fn from_f64(v: f64) -> Self { v as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn abs(self) -> Self { f32_abs_compat(self) }
    fn is_nan(self) -> bool { f32::is_nan(self) }
    fn is_finite(self) -> bool { f32::is_finite(self) }
    fn is_infinite(self) -> bool { f32::is_infinite(self) }
    fn has_infinity() -> bool { true }
    fn has_nan() -> bool { true }
    fn has_subnormals() -> bool { true }
    fn max_value() -> f32 { f32::MAX }
    fn min_positive() -> f32 { f32::MIN_POSITIVE }
}

impl StorageElement for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl NumberLike for f64 {
    fn from_f32(v: f32) -> Self { v as f64 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f64(v: f64) -> Self { v }
    fn to_f64(self) -> f64 { self }
    fn abs(self) -> Self { f64::from_bits(self.to_bits() & 0x7FFF_FFFF_FFFF_FFFF) }
    fn is_nan(self) -> bool { f64::is_nan(self) }
    fn is_finite(self) -> bool { f64::is_finite(self) }
    fn is_infinite(self) -> bool { f64::is_infinite(self) }
    fn has_infinity() -> bool { true }
    fn has_nan() -> bool { true }
    fn has_subnormals() -> bool { true }
    fn max_value() -> f32 { f32::MAX }
    fn min_positive() -> f32 { f32::MIN_POSITIVE }
}

impl StorageElement for f16 {
    fn zero() -> Self { f16(0) }
    fn one() -> Self { f16::from_f32(1.0) }
}

impl NumberLike for f16 {
    fn from_f32(v: f32) -> Self { f16::from_f32(v) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn abs(self) -> Self { self.abs() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn is_infinite(self) -> bool { self.is_infinite() }
    fn has_infinity() -> bool { true }
    fn has_nan() -> bool { true }
    fn has_subnormals() -> bool { true }
    fn max_value() -> f32 { 65504.0 }
    fn min_positive() -> f32 { 6.1e-5 }
}

impl StorageElement for bf16 {
    fn zero() -> Self { bf16(0) }
    fn one() -> Self { bf16::from_f32(1.0) }
}

impl NumberLike for bf16 {
    fn from_f32(v: f32) -> Self { bf16::from_f32(v) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn abs(self) -> Self { self.abs() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn is_infinite(self) -> bool { self.is_infinite() }
    fn has_infinity() -> bool { true }
    fn has_nan() -> bool { true }
    fn has_subnormals() -> bool { true }
    fn max_value() -> f32 { 3.4e38 }
    fn min_positive() -> f32 { 1.2e-38 }
}

impl StorageElement for e4m3 {
    fn zero() -> Self { e4m3(0) }
    fn one() -> Self { e4m3::from_f32(1.0) }
}

impl NumberLike for e4m3 {
    fn from_f32(v: f32) -> Self { e4m3::from_f32(v) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn abs(self) -> Self { self.abs() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn is_infinite(self) -> bool { self.is_infinite() }
    fn has_nan() -> bool { true }
    fn has_subnormals() -> bool { true }
    fn max_value() -> f32 { 448.0 }
    fn min_positive() -> f32 { 0.001953125 }
}

impl StorageElement for e5m2 {
    fn zero() -> Self { e5m2(0) }
    fn one() -> Self { e5m2::from_f32(1.0) }
}

impl NumberLike for e5m2 {
    fn from_f32(v: f32) -> Self { e5m2::from_f32(v) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn abs(self) -> Self { self.abs() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn is_infinite(self) -> bool { self.is_infinite() }
    fn has_infinity() -> bool { true }
    fn has_nan() -> bool { true }
    fn has_subnormals() -> bool { true }
    fn max_value() -> f32 { 57344.0 }
    fn min_positive() -> f32 { 6.103_515_6e-5 }
}

impl StorageElement for e2m3 {
    fn zero() -> Self { e2m3(0) }
    fn one() -> Self { e2m3::from_f32(1.0) }
}

impl NumberLike for e2m3 {
    fn from_f32(v: f32) -> Self { e2m3::from_f32(v) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn abs(self) -> Self { self.abs() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn is_infinite(self) -> bool { self.is_infinite() }
    fn has_subnormals() -> bool { true }
    fn max_value() -> f32 { 7.5 }
    fn min_positive() -> f32 { 0.0625 }
}

impl StorageElement for e3m2 {
    fn zero() -> Self { e3m2(0) }
    fn one() -> Self { e3m2::from_f32(1.0) }
}

impl NumberLike for e3m2 {
    fn from_f32(v: f32) -> Self { e3m2::from_f32(v) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn abs(self) -> Self { self.abs() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn is_infinite(self) -> bool { self.is_infinite() }
    fn has_subnormals() -> bool { true }
    fn max_value() -> f32 { 28.0 }
    fn min_positive() -> f32 { 0.125 }
}

impl StorageElement for i8 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumberLike for i8 {
    fn from_f32(v: f32) -> Self { f32_round_compat(v) as i8 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f64(v: f64) -> Self { f64_round_compat(v) as i8 }
    fn to_f64(self) -> f64 { self as f64 }
}

impl StorageElement for u8 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumberLike for u8 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u8
        }
    }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f64(v: f64) -> Self {
        let r = f64_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u8
        }
    }
    fn to_f64(self) -> f64 { self as f64 }
}

impl StorageElement for i32 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumberLike for i32 {
    fn from_f32(v: f32) -> Self { f32_round_compat(v) as i32 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f64(v: f64) -> Self { f64_round_compat(v) as i32 }
    fn to_f64(self) -> f64 { self as f64 }
}

impl StorageElement for u32 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumberLike for u32 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u32
        }
    }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f64(v: f64) -> Self {
        let r = f64_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u32
        }
    }
    fn to_f64(self) -> f64 { self as f64 }
}

impl StorageElement for i16 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumberLike for i16 {
    fn from_f32(v: f32) -> Self { f32_round_compat(v) as i16 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f64(v: f64) -> Self { f64_round_compat(v) as i16 }
    fn to_f64(self) -> f64 { self as f64 }
}

impl StorageElement for u16 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumberLike for u16 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u16
        }
    }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f64(v: f64) -> Self {
        let r = f64_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u16
        }
    }
    fn to_f64(self) -> f64 { self as f64 }
}

impl StorageElement for i64 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumberLike for i64 {
    fn from_f32(v: f32) -> Self { f32_round_compat(v) as i64 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f64(v: f64) -> Self { f64_round_compat(v) as i64 }
    fn to_f64(self) -> f64 { self as f64 }
}

impl StorageElement for u64 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumberLike for u64 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u64
        }
    }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f64(v: f64) -> Self {
        let r = f64_round_compat(v);
        if r < 0.0 {
            0
        } else {
            r as u64
        }
    }
    fn to_f64(self) -> f64 { self as f64 }
}

impl StorageElement for usize {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl StorageElement for i4x2 {
    fn zero() -> Self { i4x2::from((0i8, 0i8)) }
    fn one() -> Self { i4x2::from((1i8, 1i8)) }
    fn dimensions_per_value() -> usize { 2 }
}

impl NumberLike for i4x2 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v) as i8;
        i4x2::from((r, r))
    }
    fn to_f32(self) -> f32 {
        let (a, _) = self.into();
        a as f32
    }
}

impl StorageElement for u4x2 {
    fn zero() -> Self { u4x2::from((0u8, 0u8)) }
    fn one() -> Self { u4x2::from((1u8, 1u8)) }
    fn dimensions_per_value() -> usize { 2 }
}

impl NumberLike for u4x2 {
    fn from_f32(v: f32) -> Self {
        let r = f32_round_compat(v);
        let r = if r < 0.0 { 0u8 } else { r as u8 };
        u4x2::from((r, r))
    }
    fn to_f32(self) -> f32 {
        let (a, _) = self.into();
        a as f32
    }
}

impl StorageElement for u1x8 {
    fn zero() -> Self { u1x8(0x00) }
    fn one() -> Self { u1x8(0xFF) }
    fn dimensions_per_value() -> usize { 8 }
}

impl NumberLike for u1x8 {
    fn from_f32(v: f32) -> Self {
        if v > 0.0 {
            u1x8(0xFF)
        } else {
            u1x8(0x00)
        }
    }
    fn to_f32(self) -> f32 { self.0.count_ones() as f32 }
}

// endregion: StorageElement + NumberLike + FloatLike Traits

// region: Complex Types

#[inline(always)]
fn complex_mul_components<Scalar>(
    left_re: Scalar,
    left_im: Scalar,
    right_re: Scalar,
    right_im: Scalar,
) -> (Scalar, Scalar)
where
    Scalar: Copy + core::ops::Add<Output = Scalar> + core::ops::Sub<Output = Scalar> + core::ops::Mul<Output = Scalar>,
{
    (
        left_re * right_re - left_im * right_im,
        left_re * right_im + left_im * right_re,
    )
}

#[inline(always)]
fn complex_div_components<Scalar>(
    left_re: Scalar,
    left_im: Scalar,
    right_re: Scalar,
    right_im: Scalar,
) -> (Scalar, Scalar)
where
    Scalar: Copy
        + core::ops::Add<Output = Scalar>
        + core::ops::Sub<Output = Scalar>
        + core::ops::Mul<Output = Scalar>
        + core::ops::Div<Output = Scalar>,
{
    let denom = right_re * right_re + right_im * right_im;
    (
        (left_re * right_re + left_im * right_im) / denom,
        (left_im * right_re - left_re * right_im) / denom,
    )
}

#[doc(hidden)]
pub trait ComplexComponent:
    NumberLike
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
    type Norm: Copy + core::ops::Add<Output = Self::Norm>;

    fn from_norm(value: Self::Norm) -> Self;
    fn norm_component(self) -> Self::Norm;
}

impl ComplexComponent for f16 {
    type Norm = f32;

    fn from_norm(value: Self::Norm) -> Self { Self::from_f32(value) }

    fn norm_component(self) -> Self::Norm { self.to_f32() * self.to_f32() }
}

impl ComplexComponent for bf16 {
    type Norm = f32;

    fn from_norm(value: Self::Norm) -> Self { Self::from_f32(value) }

    fn norm_component(self) -> Self::Norm { self.to_f32() * self.to_f32() }
}

impl ComplexComponent for f32 {
    type Norm = f32;

    fn from_norm(value: Self::Norm) -> Self { value }

    fn norm_component(self) -> Self::Norm { self * self }
}

impl ComplexComponent for f64 {
    type Norm = f64;

    fn from_norm(value: Self::Norm) -> Self { value }

    fn norm_component(self) -> Self::Norm { self * self }
}

/// Complex number with adjacent real and imaginary components of type `Scalar`.
///
/// Layout: `{re: Scalar, im: Scalar}`. Supports conjugate, norm², and component access.
/// Concrete aliases: [`f16c`], [`bf16c`], [`f32c`], [`f64c`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct complex<Scalar> {
    pub re: Scalar,
    pub im: Scalar,
}

impl<Scalar> complex<Scalar> {
    pub const fn from_real_imag(re: Scalar, im: Scalar) -> Self { Self { re, im } }

    pub const fn to_real_imag(self) -> (Scalar, Scalar)
    where
        Scalar: Copy,
    {
        (self.re, self.im)
    }
}

impl<Scalar: ComplexComponent> complex<Scalar> {
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn norm_sqr(self) -> Scalar::Norm { self.re.norm_component() + self.im.norm_component() }
}

/// Half-precision (32-bit) complex number — two [`f16`] components. Kernel outputs widened to f32c.
pub type f16c = complex<f16>;
/// BFloat16 (32-bit) complex number — two [`bf16`] components. Kernel outputs widened to f32c.
pub type bf16c = complex<bf16>;
/// Single-precision (64-bit) complex number — two `f32` components.
pub type f32c = complex<f32>;
/// Double-precision (128-bit) complex number — two `f64` components.
pub type f64c = complex<f64>;

impl<Scalar: ComplexComponent> From<f32> for complex<Scalar> {
    fn from(value: f32) -> Self {
        Self {
            re: Scalar::from_f32(value),
            im: Scalar::zero(),
        }
    }
}

impl<Scalar: ComplexComponent> core::ops::Add for complex<Scalar> {
    type Output = Self;
    fn add(self, right: Self) -> Self::Output {
        Self {
            re: self.re + right.re,
            im: self.im + right.im,
        }
    }
}

impl<Scalar: ComplexComponent> core::ops::Sub for complex<Scalar> {
    type Output = Self;
    fn sub(self, right: Self) -> Self::Output {
        Self {
            re: self.re - right.re,
            im: self.im - right.im,
        }
    }
}

impl<Scalar: ComplexComponent> core::ops::Mul for complex<Scalar> {
    type Output = Self;
    fn mul(self, right: Self) -> Self::Output {
        let (re, im) = complex_mul_components(self.re, self.im, right.re, right.im);
        Self { re, im }
    }
}

impl<Scalar: ComplexComponent> core::ops::Div for complex<Scalar> {
    type Output = Self;
    fn div(self, right: Self) -> Self::Output {
        let (re, im) = complex_div_components(self.re, self.im, right.re, right.im);
        Self { re, im }
    }
}

impl<Scalar: ComplexComponent> core::ops::Neg for complex<Scalar> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<Scalar: NumberLike> core::fmt::Display for complex<Scalar> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let re = self.re.to_f32();
        let im = self.im.to_f32();
        if let Some(prec) = f.precision() {
            if im < 0.0 {
                write!(f, "{:.prec$} - {:.prec$}i", re, -im)
            } else {
                write!(f, "{:.prec$} + {:.prec$}i", re, im)
            }
        } else if im < 0.0 {
            write!(f, "{} - {}i", re, -im)
        } else {
            write!(f, "{} + {}i", re, im)
        }
    }
}

impl<Scalar: ComplexComponent> StorageElement for complex<Scalar> {
    fn zero() -> Self {
        Self {
            re: Scalar::zero(),
            im: Scalar::zero(),
        }
    }
    fn one() -> Self {
        Self {
            re: Scalar::one(),
            im: Scalar::zero(),
        }
    }
}

impl<Scalar: ComplexComponent> NumberLike for complex<Scalar> {
    fn from_f32(v: f32) -> Self { Self::from(v) }
    fn to_f32(self) -> f32 { self.re.to_f32() }
    fn from_f64(v: f64) -> Self {
        Self {
            re: Scalar::from_f64(v),
            im: Scalar::zero(),
        }
    }
    fn to_f64(self) -> f64 { self.re.to_f64() }
    fn abs(self) -> Self {
        Self {
            re: Scalar::from_norm(self.norm_sqr()),
            im: Scalar::zero(),
        }
    }
    fn is_nan(self) -> bool { self.re.is_nan() || self.im.is_nan() }
    fn is_finite(self) -> bool { self.re.is_finite() && self.im.is_finite() }
    fn is_infinite(self) -> bool { self.re.is_infinite() || self.im.is_infinite() }
    fn has_infinity() -> bool { Scalar::has_infinity() }
    fn has_nan() -> bool { Scalar::has_nan() }
    fn has_subnormals() -> bool { Scalar::has_subnormals() }
    fn max_value() -> f32 { Scalar::max_value() }
    fn min_positive() -> f32 { Scalar::min_positive() }
}

impl<Scalar: ComplexComponent> FloatConvertible for complex<Scalar> {
    type DimScalar = complex<Scalar>;
    type Unpacked = [complex<Scalar>; 1];
    #[inline(always)]
    fn unpack(self) -> [complex<Scalar>; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [complex<Scalar>; 1]) -> Self { dims[0] }
}

// endregion: Complex Types

// region: FloatConvertible Trait

/// Trait for types that can unpack/pack logical sub-dimensions.
///
/// For normal scalar types (`dimensions_per_value() == 1`), `DimScalar = Self` and
/// `Unpacked = [Self; 1]`. For packed sub-byte types, `DimScalar` is the natural
/// scalar type for individual sub-dimensions (e.g., `i8` for `i4x2`).
pub trait FloatConvertible: NumberLike {
    /// Scalar type for individual sub-dimensions.
    type DimScalar: Copy + Default + NumberLike;

    /// Fixed-size array holding all unpacked sub-dimensions.
    type Unpacked: AsRef<[Self::DimScalar]> + AsMut<[Self::DimScalar]> + Copy + Default;

    /// Unpack all logical sub-dimensions from this packed value.
    fn unpack(self) -> Self::Unpacked;

    /// Pack sub-dimension scalars into a single storage value.
    fn pack(dims: Self::Unpacked) -> Self;
}

impl FloatConvertible for f32 {
    type DimScalar = f32;
    type Unpacked = [f32; 1];
    #[inline(always)]
    fn unpack(self) -> [f32; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [f32; 1]) -> Self { dims[0] }
}

impl FloatConvertible for f64 {
    type DimScalar = f64;
    type Unpacked = [f64; 1];
    #[inline(always)]
    fn unpack(self) -> [f64; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [f64; 1]) -> Self { dims[0] }
}

impl FloatConvertible for f16 {
    type DimScalar = f16;
    type Unpacked = [f16; 1];
    #[inline(always)]
    fn unpack(self) -> [f16; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [f16; 1]) -> Self { dims[0] }
}

impl FloatConvertible for bf16 {
    type DimScalar = bf16;
    type Unpacked = [bf16; 1];
    #[inline(always)]
    fn unpack(self) -> [bf16; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [bf16; 1]) -> Self { dims[0] }
}

impl FloatConvertible for e4m3 {
    type DimScalar = e4m3;
    type Unpacked = [e4m3; 1];
    #[inline(always)]
    fn unpack(self) -> [e4m3; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [e4m3; 1]) -> Self { dims[0] }
}

impl FloatConvertible for e5m2 {
    type DimScalar = e5m2;
    type Unpacked = [e5m2; 1];
    #[inline(always)]
    fn unpack(self) -> [e5m2; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [e5m2; 1]) -> Self { dims[0] }
}

impl FloatConvertible for e2m3 {
    type DimScalar = e2m3;
    type Unpacked = [e2m3; 1];
    #[inline(always)]
    fn unpack(self) -> [e2m3; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [e2m3; 1]) -> Self { dims[0] }
}

impl FloatConvertible for e3m2 {
    type DimScalar = e3m2;
    type Unpacked = [e3m2; 1];
    #[inline(always)]
    fn unpack(self) -> [e3m2; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [e3m2; 1]) -> Self { dims[0] }
}

impl FloatConvertible for i8 {
    type DimScalar = i8;
    type Unpacked = [i8; 1];
    #[inline(always)]
    fn unpack(self) -> [i8; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [i8; 1]) -> Self { dims[0] }
}

impl FloatConvertible for u8 {
    type DimScalar = u8;
    type Unpacked = [u8; 1];
    #[inline(always)]
    fn unpack(self) -> [u8; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [u8; 1]) -> Self { dims[0] }
}

impl FloatConvertible for i16 {
    type DimScalar = i16;
    type Unpacked = [i16; 1];
    #[inline(always)]
    fn unpack(self) -> [i16; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [i16; 1]) -> Self { dims[0] }
}

impl FloatConvertible for u16 {
    type DimScalar = u16;
    type Unpacked = [u16; 1];
    #[inline(always)]
    fn unpack(self) -> [u16; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [u16; 1]) -> Self { dims[0] }
}

impl FloatConvertible for i32 {
    type DimScalar = i32;
    type Unpacked = [i32; 1];
    #[inline(always)]
    fn unpack(self) -> [i32; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [i32; 1]) -> Self { dims[0] }
}

impl FloatConvertible for u32 {
    type DimScalar = u32;
    type Unpacked = [u32; 1];
    #[inline(always)]
    fn unpack(self) -> [u32; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [u32; 1]) -> Self { dims[0] }
}

impl FloatConvertible for i64 {
    type DimScalar = i64;
    type Unpacked = [i64; 1];
    #[inline(always)]
    fn unpack(self) -> [i64; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [i64; 1]) -> Self { dims[0] }
}

impl FloatConvertible for u64 {
    type DimScalar = u64;
    type Unpacked = [u64; 1];
    #[inline(always)]
    fn unpack(self) -> [u64; 1] { [self] }
    #[inline(always)]
    fn pack(dims: [u64; 1]) -> Self { dims[0] }
}

impl FloatConvertible for i4x2 {
    type DimScalar = i8;
    type Unpacked = [i8; 2];
    #[inline(always)]
    fn unpack(self) -> [i8; 2] {
        let (low, high) = self.to_i8s();
        [low, high]
    }
    #[inline(always)]
    fn pack(dims: [i8; 2]) -> Self { i4x2::from_i8s(dims[0], dims[1]) }
}

impl FloatConvertible for u4x2 {
    type DimScalar = u8;
    type Unpacked = [u8; 2];
    #[inline(always)]
    fn unpack(self) -> [u8; 2] {
        let (low, high) = self.to_u8s();
        [low, high]
    }
    #[inline(always)]
    fn pack(dims: [u8; 2]) -> Self { u4x2::from_u8s(dims[0], dims[1]) }
}

impl FloatConvertible for u1x8 {
    type DimScalar = u8;
    type Unpacked = [u8; 8];
    #[inline(always)]
    fn unpack(self) -> [u8; 8] {
        let mut out = [0u8; 8];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = (self.0 >> i) & 1;
        }
        out
    }
    #[inline(always)]
    fn pack(dims: [u8; 8]) -> Self {
        let mut byte = 0u8;
        for (i, &dim) in dims.iter().enumerate() {
            if dim != 0 {
                byte |= 1 << i;
            }
        }
        u1x8(byte)
    }
}

// endregion: FloatConvertible Trait

// region: Dimension Proxies

/// Immutable proxy for a single logical dimension within a packed storage value.
///
/// For normal types this is equivalent to a `Scalar::DimScalar` copy.
/// For sub-byte types it represents one unpacked dimension from a packed value.
/// The lifetime ensures the proxy does not outlive the container it came from.
pub struct DimRef<'a, Scalar: FloatConvertible> {
    value: Scalar::DimScalar,
    _marker: core::marker::PhantomData<&'a Scalar>,
}

impl<'a, Scalar: FloatConvertible> DimRef<'a, Scalar> {
    /// Create a new immutable dimension proxy.
    #[inline]
    pub fn new(value: Scalar::DimScalar) -> Self {
        Self {
            value,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<Scalar: FloatConvertible> Copy for DimRef<'_, Scalar> {}

impl<Scalar: FloatConvertible> Clone for DimRef<'_, Scalar> {
    #[inline]
    fn clone(&self) -> Self { *self }
}

impl<Scalar: FloatConvertible> core::ops::Deref for DimRef<'_, Scalar> {
    type Target = Scalar::DimScalar;
    #[inline]
    fn deref(&self) -> &Scalar::DimScalar { &self.value }
}

impl<Scalar: FloatConvertible> PartialEq for DimRef<'_, Scalar>
where
    Scalar::DimScalar: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool { self.value == other.value }
}

impl<Scalar: FloatConvertible> PartialOrd for DimRef<'_, Scalar>
where
    Scalar::DimScalar: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> { self.value.partial_cmp(&other.value) }
}

impl<Scalar: FloatConvertible> core::fmt::Debug for DimRef<'_, Scalar>
where
    Scalar::DimScalar: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { self.value.fmt(f) }
}

impl<Scalar: FloatConvertible> core::fmt::Display for DimRef<'_, Scalar>
where
    Scalar::DimScalar: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { self.value.fmt(f) }
}

/// Mutable proxy for a single logical dimension within a packed storage value.
///
/// On [`Drop`], performs a read-modify-write: reads the current storage value,
/// overwrites the sub-dimension at `sub_index`, and writes back.
///
/// For normal types (`dimensions_per_value() == 1`), this is equivalent to `&mut Scalar`.
/// For sub-byte types, this enables modifying individual nibbles/bits.
///
/// # Aliasing
///
/// In a standard `for` loop each proxy is dropped before the next is created.
/// Holding multiple proxies to sub-dimensions of the same storage value
/// simultaneously is safe but uses last-writer-wins semantics.
pub struct DimMut<'a, Scalar: FloatConvertible> {
    ptr: *mut Scalar,
    sub_index: usize,
    value: Scalar::DimScalar,
    _marker: core::marker::PhantomData<&'a mut Scalar>,
}

impl<'a, Scalar: FloatConvertible> DimMut<'a, Scalar> {
    /// Create a new mutable dimension proxy.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads and writes for the duration of `'a`.
    #[inline]
    pub unsafe fn new(ptr: *mut Scalar, sub_index: usize, value: Scalar::DimScalar) -> Self {
        Self {
            ptr,
            sub_index,
            value,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<Scalar: FloatConvertible> core::ops::Deref for DimMut<'_, Scalar> {
    type Target = Scalar::DimScalar;
    #[inline]
    fn deref(&self) -> &Scalar::DimScalar { &self.value }
}

impl<Scalar: FloatConvertible> core::ops::DerefMut for DimMut<'_, Scalar> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Scalar::DimScalar { &mut self.value }
}

impl<Scalar: FloatConvertible> Drop for DimMut<'_, Scalar> {
    fn drop(&mut self) {
        let mut unpacked = unsafe { *self.ptr }.unpack();
        unpacked.as_mut()[self.sub_index] = self.value;
        unsafe { self.ptr.write(Scalar::pack(unpacked)) };
    }
}

impl<Scalar: FloatConvertible> core::fmt::Debug for DimMut<'_, Scalar>
where
    Scalar::DimScalar: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { self.value.fmt(f) }
}

impl<Scalar: FloatConvertible> core::fmt::Display for DimMut<'_, Scalar>
where
    Scalar::DimScalar: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { self.value.fmt(f) }
}

// endregion: Dimension Proxies

// region: TestableType Trait (test-only)

#[cfg(test)]
pub(crate) trait TestableType: FloatLike {
    /// Base absolute tolerance for this type.
    fn atol() -> f64;
    /// Base relative tolerance for this type.
    fn rtol() -> f64;
}

#[cfg(test)]
pub(crate) fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, msg: &str) {
    let tol = atol + rtol * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "{}: expected {} but got {} (atol={}, rtol={}, tol={})",
        msg,
        expected,
        actual,
        atol,
        rtol,
        tol
    );
}

#[cfg(test)]
impl TestableType for f32 {
    fn atol() -> f64 { 1e-4 }
    fn rtol() -> f64 { 1e-4 }
}
#[cfg(test)]
impl TestableType for f64 {
    fn atol() -> f64 { 1e-9 }
    fn rtol() -> f64 { 1e-9 }
}
#[cfg(test)]
impl TestableType for f16 {
    fn atol() -> f64 { 0.05 }
    fn rtol() -> f64 { 0.05 }
}
#[cfg(test)]
impl TestableType for bf16 {
    fn atol() -> f64 { 0.1 }
    fn rtol() -> f64 { 0.1 }
}
#[cfg(test)]
impl TestableType for e4m3 {
    fn atol() -> f64 { 0.5 }
    fn rtol() -> f64 { 0.1 }
}
#[cfg(test)]
impl TestableType for e5m2 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.1 }
}
#[cfg(test)]
impl TestableType for e2m3 {
    fn atol() -> f64 { 0.5 }
    fn rtol() -> f64 { 0.1 }
}
#[cfg(test)]
impl TestableType for e3m2 {
    fn atol() -> f64 { 0.5 }
    fn rtol() -> f64 { 0.1 }
}
#[cfg(test)]
impl TestableType for i8 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for u8 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for i32 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for u32 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for i16 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for u16 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for i64 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for u64 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for i4x2 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for u4x2 {
    fn atol() -> f64 { 1.0 }
    fn rtol() -> f64 { 0.0 }
}
#[cfg(test)]
impl TestableType for u1x8 {
    fn atol() -> f64 { 0.0 }
    fn rtol() -> f64 { 0.0 }
}

// endregion: TestableType Trait

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_scalar_roundtrip<Scalar: FloatLike>(original: f32, abs_tol: f32, rel_tol: f32) {
        let converted = Scalar::from_f32(original);
        let roundtrip = NumberLike::to_f32(converted);
        if original == 0.0 {
            assert_eq!(roundtrip, 0.0, "Zero should roundtrip exactly");
            return;
        }
        let abs_error = f32_abs_compat(roundtrip - original);
        let rel_error = abs_error / f32_abs_compat(original);
        assert!(
            abs_error <= abs_tol || rel_error <= rel_tol,
            "Roundtrip failed for {}: got {} (abs_err={:.6}, rel_err={:.6})",
            original,
            roundtrip,
            abs_error,
            rel_error
        );
    }

    fn assert_scalar_almost_equal<Scalar: FloatLike>(
        actual: f32,
        expected: f32,
        abs_tol: f32,
        rel_tol: f32,
        context: &str,
    ) {
        let abs_error = f32_abs_compat(actual - expected);
        let rel_error = if expected != 0.0 {
            abs_error / f32_abs_compat(expected)
        } else {
            abs_error
        };
        assert!(
            abs_error <= abs_tol || rel_error <= rel_tol,
            "{}: expected {} but got {} (abs_err={:.6}, rel_err={:.6})",
            context,
            expected,
            actual,
            abs_error,
            rel_error
        );
    }

    fn check_arithmetic<Scalar>(a_val: f32, b_val: f32, abs_tol: f32, rel_tol: f32)
    where
        Scalar: FloatLike
            + PartialOrd
            + PartialEq
            + core::ops::Add<Output = Scalar>
            + core::ops::Sub<Output = Scalar>
            + core::ops::Mul<Output = Scalar>
            + core::ops::Div<Output = Scalar>
            + core::ops::Neg<Output = Scalar>,
    {
        let a = Scalar::from_f32(a_val);
        let b = Scalar::from_f32(b_val);
        assert_scalar_almost_equal::<Scalar>(NumberLike::to_f32(a + b), a_val + b_val, abs_tol, rel_tol, "add");
        assert_scalar_almost_equal::<Scalar>(NumberLike::to_f32(a - b), a_val - b_val, abs_tol, rel_tol, "sub");
        assert_scalar_almost_equal::<Scalar>(NumberLike::to_f32(a * b), a_val * b_val, abs_tol, rel_tol, "mul");
        assert_scalar_almost_equal::<Scalar>(NumberLike::to_f32(a / b), a_val / b_val, abs_tol, rel_tol, "div");
        assert_scalar_almost_equal::<Scalar>(NumberLike::to_f32(-a), -a_val, abs_tol, rel_tol, "neg");
        assert_eq!(NumberLike::to_f32(Scalar::zero()), 0.0);
        assert_scalar_almost_equal::<Scalar>(NumberLike::to_f32(Scalar::one()), 1.0, abs_tol, rel_tol, "ONE");
        assert_scalar_almost_equal::<Scalar>(
            NumberLike::to_f32(Scalar::from_f32(-1.0)),
            -1.0,
            abs_tol,
            rel_tol,
            "NEG_ONE",
        );
        assert!(a > b);
        assert!(a == a);
        assert_scalar_almost_equal::<Scalar>(NumberLike::to_f32(NumberLike::abs(-a)), a_val, abs_tol, rel_tol, "abs");
        assert!(NumberLike::is_finite(a));
        assert!(!NumberLike::is_nan(a));
        if Scalar::has_infinity() {
            assert!(!NumberLike::is_infinite(a));
        }
    }

    fn check_roundtrip<Scalar: FloatLike>(values: &[f32], abs_tol: f32, rel_tol: f32) {
        for &v in values {
            assert_scalar_roundtrip::<Scalar>(v, abs_tol, rel_tol);
        }
    }

    fn check_edge_cases<Scalar: FloatLike>() {
        assert_eq!(NumberLike::to_f32(Scalar::from_f32(0.0)), 0.0);
        assert_eq!(NumberLike::to_f32(Scalar::from_f32(-0.0)), 0.0);
        if Scalar::has_infinity() {
            assert!(NumberLike::to_f32(Scalar::from_f32(f32::INFINITY)).is_infinite());
            assert!(NumberLike::to_f32(Scalar::from_f32(f32::NEG_INFINITY)).is_infinite());
        } else {
            assert!(!NumberLike::to_f32(Scalar::from_f32(f32::INFINITY)).is_infinite());
        }
        if Scalar::has_nan() {
            assert!(NumberLike::to_f32(Scalar::from_f32(f32::NAN)).is_nan());
        } else {
            assert!(!NumberLike::to_f32(Scalar::from_f32(f32::NAN)).is_nan());
        }
        let big = Scalar::max_value() * 10.0;
        let overflow = Scalar::from_f32(big);
        if Scalar::has_infinity() {
            assert!(NumberLike::to_f32(overflow).is_infinite() || NumberLike::to_f32(overflow) >= Scalar::max_value());
        } else {
            let v = NumberLike::to_f32(overflow);
            assert!(!v.is_infinite() && !v.is_nan());
            assert!(v <= Scalar::max_value());
            let neg = NumberLike::to_f32(Scalar::from_f32(-big));
            assert!(!neg.is_infinite() && !neg.is_nan());
            assert!(neg >= -Scalar::max_value());
        }
    }

    fn check_subnormals<Scalar: FloatLike>(values: &[f32], upper_bound: f32) {
        for &val in values {
            let roundtrip = NumberLike::to_f32(Scalar::from_f32(val));
            assert!(
                roundtrip >= 0.0 && roundtrip < upper_bound,
                "{} subnormal test failed for {}: got {}",
                core::any::type_name::<Scalar>(),
                val,
                roundtrip
            );
        }
    }

    #[test]
    fn arithmetic_ieee_halfs() {
        check_arithmetic::<f16>(3.5, 2.0, 0.002, 0.001);
        check_arithmetic::<bf16>(3.5, 2.0, 0.016, 0.008);
    }

    #[test]
    fn arithmetic_minifloats() {
        check_arithmetic::<e4m3>(2.0, 1.5, 0.25, 0.125);
        check_arithmetic::<e5m2>(2.0, 1.5, 0.5, 0.25);
        check_arithmetic::<e2m3>(2.0, 1.5, 0.25, 0.125);
        check_arithmetic::<e3m2>(4.0, 2.0, 0.5, 0.25);
    }

    #[test]
    fn roundtrip() {
        check_roundtrip::<f16>(
            &[
                0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 100.0, 1000.0, 10000.0, 0.001, 0.0001, 0.00001, -100.0,
                -1000.0,
            ],
            0.002,
            0.001,
        );
        check_roundtrip::<bf16>(
            &[
                0.0, 1.0, -1.0, 0.5, 2.0, 10.0, 100.0, 1000.0, 1e6, 0.001, 1e-6, -100.0, -1000.0,
            ],
            0.016,
            0.008,
        );
        check_roundtrip::<e4m3>(
            &[0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 64.0, 128.0, 224.0],
            0.25,
            0.125,
        );
        check_roundtrip::<e5m2>(
            &[0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0],
            0.5,
            0.25,
        );
        check_roundtrip::<e2m3>(
            &[
                0.0, 1.0, -1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 0.25, -0.25, 0.125, -0.125, -4.0, -6.0, -7.5,
            ],
            0.25,
            0.125,
        );
        check_roundtrip::<e3m2>(
            &[
                0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 8.0, 16.0, 20.0, 24.0, 28.0, 0.25, -0.25, -20.0, -28.0,
            ],
            0.5,
            0.25,
        );
    }

    #[test]
    fn edge_cases() {
        check_edge_cases::<f16>();
        check_edge_cases::<bf16>();
        check_edge_cases::<e4m3>();
        check_edge_cases::<e5m2>();
        check_edge_cases::<e2m3>();
        check_edge_cases::<e3m2>();
    }

    #[test]
    fn subnormals() {
        check_subnormals::<f16>(&[1e-5, 1e-6, 1e-7, 5e-6, 5e-7], 1e-4);
        check_subnormals::<bf16>(&[1e-39, 1e-40, 1e-42], 1e-37);
        check_subnormals::<e4m3>(&[0.001, 0.0005], 0.002);
        check_subnormals::<e5m2>(&[0.00005, 0.00003, 0.00001], 0.0001);
        check_subnormals::<e2m3>(&[0.03, 0.015], 0.07);
        check_subnormals::<e3m2>(&[0.0625, 0.03], 0.15);
    }

    #[test]
    fn half_crate_interop() {
        use half::bf16 as HalfBF16;
        use half::f16 as HalfF16;

        // f16: all 65536 bit patterns
        for bits in 0u16..=u16::MAX {
            let half_val = HalfF16::from_bits(bits).to_f32();
            let nk_val = f16(bits).to_f32();
            assert!(
                half_val.to_bits() == nk_val.to_bits() || (half_val.is_nan() && nk_val.is_nan()),
                "f16 mismatch at bits 0x{bits:04X}: half={half_val}, numkong={nk_val}"
            );
        }

        // bf16: all 65536 bit patterns
        for bits in 0u16..=u16::MAX {
            let half_val = HalfBF16::from_bits(bits).to_f32();
            let nk_val = bf16(bits).to_f32();
            assert!(
                half_val.to_bits() == nk_val.to_bits() || (half_val.is_nan() && nk_val.is_nan()),
                "bf16 mismatch at bits 0x{bits:04X}: half={half_val}, numkong={nk_val}"
            );
        }
    }

    #[test]
    fn is_close_exact() {
        assert!(is_close(1.0, 1.0, 0.0, 0.0));
        assert!(is_close(0.0, 0.0, 0.0, 0.0));
        assert!(is_close(-5.0, -5.0, 0.0, 0.0));
    }

    #[test]
    fn is_close_within_atol() {
        assert!(is_close(1.0, 1.0 + 1e-7, 1e-6, 0.0));
        assert!(!is_close(1.0, 1.0 + 1e-5, 1e-6, 0.0));
    }

    #[test]
    fn is_close_within_rtol() {
        assert!(is_close(100.0, 100.01, 0.0, 1e-3));
        assert!(!is_close(100.0, 100.2, 0.0, 1e-3));
    }

    #[test]
    fn is_close_nan_inf() {
        // NaN is never close to anything (including itself).
        assert!(!is_close(f64::NAN, f64::NAN, 1e-6, 1e-6));
        assert!(!is_close(f64::NAN, 0.0, 1e-6, 1e-6));
        // INF - INF = NaN, so same-sign infinities are not "close" either.
        assert!(!is_close(f64::INFINITY, f64::INFINITY, 0.0, 0.0));
        assert!(!is_close(f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0));
    }
}
