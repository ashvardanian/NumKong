//! Binary set similarity: Hamming and Jaccard distances.
//!
//! This module provides:
//!
//! - [`Hamming`]: Bit-level or byte-level Hamming distance
//! - [`Jaccard`]: Jaccard distance (1 - intersection/union)
//! - [`BinarySimilarity`]: Blanket trait combining `Hamming + Jaccard`
//!
//! File: rust/set.rs
//! Author: Ash Vardanian

use core::{ffi::c_void, ptr::null_mut};

use crate::{
    capabilities::{nk_capability_t, nk_size_t, nk_status_t, Capabilities, StatusCode},
    tensor::{check_len, Error},
    types::{u1x8, StorageElement},
};

#[link(name = "numkong")]
extern "C" {
    fn nk_hamming_u1_best(
        a: *const u8,
        b: *const u8,
        c: nk_size_t,
        d: *mut u32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_jaccard_u1_best(
        a: *const u8,
        b: *const u8,
        c: nk_size_t,
        d: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_hamming_u8_best(
        a: *const u8,
        b: *const u8,
        n: nk_size_t,
        result: *mut u32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_jaccard_u16_best(
        a: *const u16,
        b: *const u16,
        n: nk_size_t,
        result: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_jaccard_u32_best(
        a: *const u32,
        b: *const u32,
        n: nk_size_t,
        result: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
}

// region: Hamming

/// Computes the __Hamming distance__ between two binary vectors.
///
/// Counts differing bits for `u1x8`, or differing bytes for `u8`.
///
/// Range: \[0, n\]. Fails with [`Error::ShapeMismatch`] if lengths differ, or
/// [`Error::KernelFailed`] carrying the kernel's status.
///
/// Implemented for: `u1x8`, `u8`.
pub trait Hamming: StorageElement {
    type Output;
    fn hamming(a: &[Self], b: &[Self]) -> Result<Self::Output, Error>;
}

impl Hamming for u1x8 {
    type Output = u32;
    fn hamming(a: &[Self], b: &[Self]) -> Result<Self::Output, Error> {
        check_len(a.len(), b.len())?;
        let mut result: Self::Output = 0;
        let n_bits = a.len() * Self::dimensions_per_value();
        unsafe {
            nk_hamming_u1_best(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n_bits,
                &mut result,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
        }
        .check()?;
        Ok(result)
    }
}

impl Hamming for u8 {
    type Output = u32;
    fn hamming(a: &[Self], b: &[Self]) -> Result<Self::Output, Error> {
        check_len(a.len(), b.len())?;
        let mut result: Self::Output = 0;
        unsafe {
            nk_hamming_u8_best(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &mut result,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
        }
        .check()?;
        Ok(result)
    }
}

// endregion: Hamming

// region: Jaccard

/// Computes the __Jaccard distance__ between two sets represented as bit/integer vectors.
///
/// d = 1 − |A ∩ B| / |A ∪ B|
///
/// Range: \[0, 1\]. Fails with [`Error::ShapeMismatch`] if lengths differ, or
/// [`Error::KernelFailed`] carrying the kernel's status.
///
/// Implemented for: `u1x8`, `u16`, `u32`.
pub trait Jaccard: StorageElement {
    type Output;
    fn jaccard(a: &[Self], b: &[Self]) -> Result<Self::Output, Error>;
}

impl Jaccard for u1x8 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Result<Self::Output, Error> {
        check_len(a.len(), b.len())?;
        let mut result: Self::Output = 0.0;
        let n_bits = a.len() * Self::dimensions_per_value();
        unsafe {
            nk_jaccard_u1_best(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                n_bits,
                &mut result,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
        }
        .check()?;
        Ok(result)
    }
}

impl Jaccard for u16 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Result<Self::Output, Error> {
        check_len(a.len(), b.len())?;
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_jaccard_u16_best(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &mut result,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
        }
        .check()?;
        Ok(result)
    }
}

impl Jaccard for u32 {
    type Output = f32;
    fn jaccard(a: &[Self], b: &[Self]) -> Result<Self::Output, Error> {
        check_len(a.len(), b.len())?;
        let mut result: Self::Output = 0.0;
        unsafe {
            nk_jaccard_u32_best(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                &mut result,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
        }
        .check()?;
        Ok(result)
    }
}

// endregion: Jaccard

/// `BinarySimilarity` bundles binary distance metrics: Hamming and Jaccard.
pub trait BinarySimilarity: Hamming + Jaccard {}
impl<Scalar: Hamming + Jaccard> BinarySimilarity for Scalar {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, u1x8};

    #[test]
    fn hamming() {
        // u1x8
        let left = vec![u1x8(0b11110000), u1x8(0b10101010)];
        let right = vec![u1x8(0b00001111), u1x8(0b01010101)];
        assert_eq!(u1x8::hamming(&left, &right).unwrap(), 16);

        // u8
        let left: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let right: Vec<u8> = vec![0, 1, 2, 3, 0, 0, 0, 0];
        assert_eq!(u8::hamming(&left, &right).unwrap(), 4);
    }

    #[test]
    fn jaccard() {
        // u1x8 — identical
        let left = vec![u1x8(0b11110000), u1x8(0b10101010)];
        let right = vec![u1x8(0b11110000), u1x8(0b10101010)];
        assert_close(
            u1x8::jaccard(&left, &right).unwrap() as f64,
            0.0,
            0.01,
            0.0,
            "jaccard_u1x8",
        );

        // u16 — identical
        let left: Vec<u16> = vec![1, 2, 3, 4];
        let right: Vec<u16> = vec![1, 2, 3, 4];
        assert_close(
            u16::jaccard(&left, &right).unwrap() as f64,
            0.0,
            0.01,
            0.0,
            "jaccard_u16 identical",
        );
        // u16 — disjoint
        let disjoint: Vec<u16> = vec![5, 6, 7, 8];
        assert_close(
            u16::jaccard(&left, &disjoint).unwrap() as f64,
            1.0,
            0.01,
            0.0,
            "jaccard_u16 disjoint",
        );

        // u32 — partial overlap
        let left: Vec<u32> = vec![1, 2, 3, 4];
        let right: Vec<u32> = vec![1, 2, 5, 6];
        assert_close(
            u32::jaccard(&left, &right).unwrap() as f64,
            0.5,
            0.01,
            0.0,
            "jaccard_u32",
        );
    }
}
