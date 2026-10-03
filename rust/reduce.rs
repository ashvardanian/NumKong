//! Statistical reductions: moments, min/max, and bit-population counts.
//!
//! This module provides:
//!
//! - Slice-level traits:
//!   - [`ReduceMoments`]: Sum and sum-of-squares over a strided slice
//!   - [`ReduceMinMax`]: Minimum and maximum over a strided slice
//!   - [`Reductions`]: Blanket trait combining `ReduceMoments + ReduceMinMax`
//! - Tensor-shaped extension traits, auto-implemented on every [`crate::tensor::TensorRef`]:
//!   - [`MomentsOps`]: Per-axis and full-tensor moments / sums / norms
//!   - [`MinMaxOps`]: Per-axis and full-tensor min / max / argmin / argmax
//!   - [`BitwiseReductionsOps`]: `popcount` / `any_set` / `none_set` / `all_set` for `u1x8`
//!     tensors; `Vector<u1x8>` / `VectorView<u1x8>` / `VectorSpan<u1x8>` expose the same method
//!     names as concrete impls defined in [`crate::vector`] until a `VectorRef` trait exists
//!
//! File: rust/reduce.rs
//! Author: Ash Vardanian

use core::{ffi::c_void, ptr::null_mut};

use crate::{
    capabilities::{nk_capability_t, nk_size_t, nk_status_t, Capabilities, StatusCode},
    tensor::{Error, MinMaxAxisResult, MinMaxResult, MomentsAxisResult, Tensor, TensorMut, TensorRef},
    types::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2, StorageElement},
    vector::VectorIndex,
};

#[link(name = "numkong")]
extern "C" {
    // Reductions: moments (sum + sum-of-squares)
    fn nk_reduce_moments_f64_best(
        data: *const f64,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut f64,
        sumsq: *mut f64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_f32_best(
        data: *const f32,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut f64,
        sumsq: *mut f64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_i8_best(
        data: *const i8,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut i64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_u8_best(
        data: *const u8,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut u64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_i16_best(
        data: *const i16,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut i64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_u16_best(
        data: *const u16,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut u64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_i32_best(
        data: *const i32,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut i64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_u32_best(
        data: *const u32,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut u64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_i64_best(
        data: *const i64,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut i64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_u64_best(
        data: *const u64,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut u64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_f16_best(
        data: *const f16,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut f32,
        sumsq: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_bf16_best(
        data: *const bf16,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut f32,
        sumsq: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_e4m3_best(
        data: *const e4m3,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut f32,
        sumsq: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_e5m2_best(
        data: *const e5m2,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut f32,
        sumsq: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_e2m3_best(
        data: *const e2m3,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut f32,
        sumsq: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_e3m2_best(
        data: *const e3m2,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut f32,
        sumsq: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_i4_best(
        data: *const i4x2,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut i64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_u4_best(
        data: *const u4x2,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut u64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_moments_u1_best(
        data: *const u1x8,
        count: nk_size_t,
        stride: nk_size_t,
        sum: *mut u64,
        sumsq: *mut u64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    // Reductions: minmax (min + max + argmin + argmax)
    fn nk_reduce_minmax_f64_best(
        data: *const f64,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut f64,
        min_idx: *mut nk_size_t,
        max_val: *mut f64,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_f32_best(
        data: *const f32,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut f32,
        min_idx: *mut nk_size_t,
        max_val: *mut f32,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_i8_best(
        data: *const i8,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut i8,
        min_idx: *mut nk_size_t,
        max_val: *mut i8,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_u8_best(
        data: *const u8,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut u8,
        min_idx: *mut nk_size_t,
        max_val: *mut u8,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_i16_best(
        data: *const i16,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut i16,
        min_idx: *mut nk_size_t,
        max_val: *mut i16,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_u16_best(
        data: *const u16,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut u16,
        min_idx: *mut nk_size_t,
        max_val: *mut u16,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_i32_best(
        data: *const i32,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut i32,
        min_idx: *mut nk_size_t,
        max_val: *mut i32,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_u32_best(
        data: *const u32,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut u32,
        min_idx: *mut nk_size_t,
        max_val: *mut u32,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_i64_best(
        data: *const i64,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut i64,
        min_idx: *mut nk_size_t,
        max_val: *mut i64,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_u64_best(
        data: *const u64,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut u64,
        min_idx: *mut nk_size_t,
        max_val: *mut u64,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_f16_best(
        data: *const f16,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut f16,
        min_idx: *mut nk_size_t,
        max_val: *mut f16,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_bf16_best(
        data: *const bf16,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut bf16,
        min_idx: *mut nk_size_t,
        max_val: *mut bf16,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_e4m3_best(
        data: *const e4m3,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut e4m3,
        min_idx: *mut nk_size_t,
        max_val: *mut e4m3,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_e5m2_best(
        data: *const e5m2,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut e5m2,
        min_idx: *mut nk_size_t,
        max_val: *mut e5m2,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_e2m3_best(
        data: *const e2m3,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut e2m3,
        min_idx: *mut nk_size_t,
        max_val: *mut e2m3,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_e3m2_best(
        data: *const e3m2,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut e3m2,
        min_idx: *mut nk_size_t,
        max_val: *mut e3m2,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_i4_best(
        data: *const i4x2,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut i8,
        min_idx: *mut nk_size_t,
        max_val: *mut i8,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_u4_best(
        data: *const u4x2,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut u8,
        min_idx: *mut nk_size_t,
        max_val: *mut u8,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_reduce_minmax_u1_best(
        data: *const u1x8,
        count: nk_size_t,
        stride: nk_size_t,
        min_val: *mut u8,
        min_idx: *mut nk_size_t,
        max_val: *mut u8,
        max_idx: *mut nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
}

/// Compute first and second moments — sum and sum-of-squares — with stride support.
///
/// Returns `(sum, sum_of_squares)` for all elements in a slice, with optional striding.
/// The output types may be wider than the input to avoid overflow.
///
/// # Errors
///
/// [`Error::KernelFailed`] when no capability of the current
/// [`Capabilities`](crate::Capabilities) has the kernel.
pub trait ReduceMoments: StorageElement {
    /// Type for the sum output.
    type SumOutput: StorageElement;
    /// Type for the sum-of-squares output.
    type SumSqOutput: StorageElement;
    /// Compute `(sum, sum_of_squares)` for `data` with the given stride in bytes.
    /// Use `stride = size_of::<Self>()` for contiguous data.
    ///
    /// Reads `data.len()` logical elements starting at `data.as_ptr()`, advancing by `stride`
    /// between each — pass `size_of::<Self>()` for contiguous storage.
    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error>;
}

unsafe fn reduce_moments_via_ffi<Scalar, Sum: Default, SumSq: Default>(
    data: *const Scalar,
    count: usize,
    stride: usize,
    ffi: unsafe extern "C" fn(
        *const Scalar,
        nk_size_t,
        nk_size_t,
        *mut Sum,
        *mut SumSq,
        nk_capability_t,
        *mut c_void,
    ) -> nk_status_t,
) -> Result<(Sum, SumSq), Error>
where
    Scalar: StorageElement,
{
    let mut sum: Sum = Default::default();
    let mut sumsq: SumSq = Default::default();
    ffi(
        data,
        count * Scalar::dimensions_per_value(),
        stride,
        &mut sum,
        &mut sumsq,
        Capabilities::CPUS.bits(),
        null_mut(),
    )
    .check()?;
    Ok((sum, sumsq))
}

impl ReduceMoments for f64 {
    type SumOutput = f64;
    type SumSqOutput = f64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_f64_best) }
    }
}

impl ReduceMoments for f32 {
    type SumOutput = f64;
    type SumSqOutput = f64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_f32_best) }
    }
}

impl ReduceMoments for i8 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_i8_best) }
    }
}

impl ReduceMoments for u8 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_u8_best) }
    }
}

impl ReduceMoments for i16 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_i16_best) }
    }
}

impl ReduceMoments for u16 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_u16_best) }
    }
}

impl ReduceMoments for i32 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_i32_best) }
    }
}

impl ReduceMoments for u32 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_u32_best) }
    }
}

impl ReduceMoments for i64 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_i64_best) }
    }
}

impl ReduceMoments for u64 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_u64_best) }
    }
}

impl ReduceMoments for f16 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_f16_best) }
    }
}

impl ReduceMoments for bf16 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_bf16_best) }
    }
}

impl ReduceMoments for e4m3 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_e4m3_best) }
    }
}

impl ReduceMoments for e5m2 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_e5m2_best) }
    }
}

impl ReduceMoments for e2m3 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_e2m3_best) }
    }
}

impl ReduceMoments for e3m2 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_e3m2_best) }
    }
}

impl ReduceMoments for i4x2 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_i4_best) }
    }
}

impl ReduceMoments for u4x2 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_u4_best) }
    }
}

impl ReduceMoments for u1x8 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    fn reduce_moments(data: &[Self], stride: usize) -> Result<(Self::SumOutput, Self::SumSqOutput), Error> {
        unsafe { reduce_moments_via_ffi(data.as_ptr(), data.len(), stride, nk_reduce_moments_u1_best) }
    }
}

/// Find minimum and maximum values with their indices, with stride support.
///
/// Returns `Ok(Some(MinMaxResult))` for all elements in a slice, or `Ok(None)` if all elements are
/// NaN, for NaN-masking formats. The output value type matches the logical reduced scalar type. A
/// refusing kernel fails with [`Error::KernelFailed`].
pub trait ReduceMinMax: StorageElement {
    /// Output type for the min/max values — matches the C layer's native type.
    type Output: StorageElement;
    /// Whether `NUMKONG_SIZE_MAX` indicates that the reduction produced no value.
    const NONE_ON_SENTINEL: bool;
    /// Returns `Ok(Some(MinMaxResult))` for the given data with the specified stride, or `Ok(None)`
    /// if all elements are NaN.
    ///
    /// Reads `data.len()` logical elements starting at `data.as_ptr()`, advancing by `stride`
    /// between each — pass `size_of::<Self>()` for contiguous storage.
    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error>;
}

unsafe fn reduce_minmax_via_ffi<Scalar, Out: Default>(
    data: *const Scalar,
    count: usize,
    stride: usize,
    none_on_sentinel: bool,
    ffi: unsafe extern "C" fn(
        *const Scalar,
        nk_size_t,
        nk_size_t,
        *mut Out,
        *mut nk_size_t,
        *mut Out,
        *mut nk_size_t,
        nk_capability_t,
        *mut c_void,
    ) -> nk_status_t,
) -> Result<Option<MinMaxResult<Out>>, Error>
where
    Scalar: StorageElement,
{
    let mut result = MinMaxResult {
        min_value: Out::default(),
        min_index: 0,
        max_value: Out::default(),
        max_index: 0,
    };
    ffi(
        data,
        count * Scalar::dimensions_per_value(),
        stride,
        &mut result.min_value,
        &mut result.min_index,
        &mut result.max_value,
        &mut result.max_index,
        Capabilities::CPUS.bits(),
        null_mut(),
    )
    .check()?;
    if none_on_sentinel && result.min_index == usize::MAX {
        return Ok(None);
    }
    Ok(Some(result))
}

impl ReduceMinMax for f64 {
    type Output = f64;
    const NONE_ON_SENTINEL: bool = true;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_f64_best,
            )
        }
    }
}

impl ReduceMinMax for f32 {
    type Output = f32;
    const NONE_ON_SENTINEL: bool = true;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_f32_best,
            )
        }
    }
}

impl ReduceMinMax for i8 {
    type Output = i8;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_i8_best,
            )
        }
    }
}

impl ReduceMinMax for u8 {
    type Output = u8;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_u8_best,
            )
        }
    }
}

impl ReduceMinMax for i16 {
    type Output = i16;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_i16_best,
            )
        }
    }
}

impl ReduceMinMax for u16 {
    type Output = u16;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_u16_best,
            )
        }
    }
}

impl ReduceMinMax for i32 {
    type Output = i32;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_i32_best,
            )
        }
    }
}

impl ReduceMinMax for u32 {
    type Output = u32;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_u32_best,
            )
        }
    }
}

impl ReduceMinMax for i64 {
    type Output = i64;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_i64_best,
            )
        }
    }
}

impl ReduceMinMax for u64 {
    type Output = u64;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_u64_best,
            )
        }
    }
}

impl ReduceMinMax for f16 {
    type Output = f16;
    const NONE_ON_SENTINEL: bool = true;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_f16_best,
            )
        }
    }
}

impl ReduceMinMax for bf16 {
    type Output = bf16;
    const NONE_ON_SENTINEL: bool = true;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_bf16_best,
            )
        }
    }
}

impl ReduceMinMax for e4m3 {
    type Output = e4m3;
    const NONE_ON_SENTINEL: bool = true;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_e4m3_best,
            )
        }
    }
}

impl ReduceMinMax for e5m2 {
    type Output = e5m2;
    const NONE_ON_SENTINEL: bool = true;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_e5m2_best,
            )
        }
    }
}

impl ReduceMinMax for e2m3 {
    type Output = e2m3;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_e2m3_best,
            )
        }
    }
}

impl ReduceMinMax for e3m2 {
    type Output = e3m2;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_e3m2_best,
            )
        }
    }
}

impl ReduceMinMax for i4x2 {
    type Output = i8;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_i4_best,
            )
        }
    }
}

impl ReduceMinMax for u4x2 {
    type Output = u8;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_u4_best,
            )
        }
    }
}

impl ReduceMinMax for u1x8 {
    type Output = u8;
    const NONE_ON_SENTINEL: bool = false;

    fn reduce_minmax(data: &[Self], stride: usize) -> Result<Option<MinMaxResult<Self::Output>>, Error> {
        unsafe {
            reduce_minmax_via_ffi(
                data.as_ptr(),
                data.len(),
                stride,
                Self::NONE_ON_SENTINEL,
                nk_reduce_minmax_u1_best,
            )
        }
    }
}

/// `Reductions` bundles reduction operations: ReduceMoments and ReduceMinMax.
pub trait Reductions: ReduceMoments + ReduceMinMax {}
impl<Scalar: ReduceMoments + ReduceMinMax> Reductions for Scalar {}

/// Walks a `u1x8` tensor view, accumulating set-bit counts via [`u1x8::reduce_moments`] on each
/// rank-1 contiguous lane.
///
/// At rank ≤ 1 the view's logical bit count divides cleanly into a contiguous storage slice —
/// sub-byte stride is always one byte per `u1x8`, so rank-1 is trivially contiguous. At higher
/// ranks we recurse on the leading axis via the `TensorRef`-supplied structural accessors so that
/// matrices and higher-rank tensors of bits work without flattening first. Non-contiguous outer
/// strides fall out of the recursion naturally — only the leaf storage slice has to be dense, and
/// sub-byte storage always is.
fn popcount_via_tensor_ref<View, const MAX_RANK: usize>(view: &View) -> Result<u64, Error>
where
    View: TensorRef<u1x8, MAX_RANK> + ?Sized,
{
    let dims_per_value = u1x8::dimensions_per_value();
    let logical_count: usize = view.shape().iter().product();
    if logical_count == 0 {
        return Ok(0);
    }
    if view.ndim() <= 1 {
        let storage_count = logical_count / dims_per_value;
        if storage_count == 0 {
            return Ok(0);
        }
        let storage_slice = unsafe { core::slice::from_raw_parts(view.as_ptr(), storage_count) };
        let (sum, _sum_of_squares) = u1x8::reduce_moments(storage_slice, core::mem::size_of::<u1x8>())?;
        return Ok(sum);
    }
    let mut total: u64 = 0;
    for sub_view in view.view().axis_views(0usize)? {
        total += popcount_via_tensor_ref::<_, MAX_RANK>(&sub_view)?;
    }
    Ok(total)
}

/// Population count, any/none/all reductions over packed-bit (`u1x8`) tensors.
///
/// Blanket-impl'd on every [`TensorRef<u1x8, R>`], so `Tensor<u1x8>`, `TensorView<u1x8>`, and
/// `TensorSpan<u1x8>` all expose `.popcount()` / `.any_set()` / `.none_set()` / `.all_set()` with
/// identical spellings. Mirrors `numkong::popcount(tensor_view)` in `include/numkong/reduce.hpp`.
///
/// Method names use the `_set` suffix to leave room for a future generic `any` / `none` / `all`
/// over arbitrary scalar predicates and to avoid colliding with the `all` slice marker exported
/// from [`crate::vector`].
pub trait BitwiseReductionsOps<const MAX_RANK: usize>: TensorRef<u1x8, MAX_RANK> {
    /// Number of set bits across the entire tensor.
    fn popcount(&self) -> Result<u64, Error> { popcount_via_tensor_ref::<_, MAX_RANK>(self) }

    /// `true` if at least one bit in the tensor is set.
    fn any_set(&self) -> Result<bool, Error> { Ok(self.popcount()? != 0) }

    /// `true` if no bit in the tensor is set.
    fn none_set(&self) -> Result<bool, Error> { Ok(self.popcount()? == 0) }

    /// `true` if every bit in the tensor is set.
    fn all_set(&self) -> Result<bool, Error> { Ok(self.popcount()? == self.numel() as u64) }
}

impl<Container, const MAX_RANK: usize> BitwiseReductionsOps<MAX_RANK> for Container where
    Container: TensorRef<u1x8, MAX_RANK> + ?Sized
{
}

// region: SumSqToF64 helper trait

#[doc(hidden)]
pub trait SumSqToF64 {
    fn to_f64(self) -> f64;
}

impl SumSqToF64 for f32 {
    fn to_f64(self) -> f64 { self as f64 }
}
impl SumSqToF64 for f64 {
    fn to_f64(self) -> f64 { self }
}
impl SumSqToF64 for u64 {
    fn to_f64(self) -> f64 { self as f64 }
}
impl SumSqToF64 for i64 {
    fn to_f64(self) -> f64 { self as f64 }
}

// endregion: SumSqToF64

// region: MomentsOps / MinMaxOps

/// Extension trait: statistical reductions for any [`TensorRef`] implementor.
pub trait MomentsOps<Scalar: Clone + ReduceMoments, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar::SumOutput: Clone + Default + core::ops::AddAssign,
    Scalar::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
    fn moments_all(&self) -> Result<(Scalar::SumOutput, Scalar::SumSqOutput), Error> { self.view().moments_all() }

    fn moments_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> MomentsAxisResult<Scalar, MAX_RANK, Self::Alloc>
    where
        Self::Alloc: Clone,
    {
        self.view().moments_axis(axis, keep_dims)
    }

    fn moments_axis_into<AnyIndex, SumTensor, SumSqTensor>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        sum_out: &mut SumTensor,
        sumsq_out: &mut SumSqTensor,
    ) -> Result<(), Error>
    where
        AnyIndex: VectorIndex,
        SumTensor: TensorMut<Scalar::SumOutput, MAX_RANK> + ?Sized,
        SumSqTensor: TensorMut<Scalar::SumSqOutput, MAX_RANK> + ?Sized,
    {
        self.view().moments_axis_into(axis, keep_dims, sum_out, sumsq_out)
    }

    fn sum_all(&self) -> Result<Scalar::SumOutput, Error> { self.view().sum_all() }

    fn sum_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<Scalar::SumOutput, Self::Alloc, MAX_RANK>, Error>
    where
        Self::Alloc: Clone,
    {
        self.view().sum_axis(axis, keep_dims)
    }

    fn sum_axis_into<AnyIndex, SumTensor>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        out: &mut SumTensor,
    ) -> Result<(), Error>
    where
        AnyIndex: VectorIndex,
        SumTensor: TensorMut<Scalar::SumOutput, MAX_RANK> + ?Sized,
    {
        self.view().sum_axis_into(axis, keep_dims, out)
    }

    fn norm_all(&self) -> Result<f64, Error> { self.view().norm_all() }

    fn norm_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<f64, Self::Alloc, MAX_RANK>, Error>
    where
        Self::Alloc: Clone,
    {
        self.view().norm_axis(axis, keep_dims)
    }

    fn norm_axis_into<AnyIndex, NormTensor>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        out: &mut NormTensor,
    ) -> Result<(), Error>
    where
        AnyIndex: VectorIndex,
        NormTensor: TensorMut<f64, MAX_RANK> + ?Sized,
    {
        self.view().norm_axis_into(axis, keep_dims, out)
    }
}

impl<Scalar: Clone + ReduceMoments, const R: usize, C: TensorRef<Scalar, R>> MomentsOps<Scalar, R> for C
where
    Scalar::SumOutput: Clone + Default + core::ops::AddAssign,
    Scalar::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
}

/// Extension trait: min/max reductions for any [`TensorRef`] implementor.
pub trait MinMaxOps<Scalar: Clone + ReduceMinMax, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK>
where
    Scalar::Output: Clone + Default + PartialOrd,
{
    fn minmax_all(&self) -> Result<MinMaxResult<Scalar::Output>, Error> { self.view().minmax_all() }

    fn minmax_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> MinMaxAxisResult<Scalar, MAX_RANK, Self::Alloc>
    where
        Self::Alloc: Clone,
    {
        self.view().minmax_axis(axis, keep_dims)
    }

    fn minmax_axis_into<AnyIndex, ValueTensor, IndexTensor>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        min_out: &mut ValueTensor,
        argmin_out: &mut IndexTensor,
        max_out: &mut ValueTensor,
        argmax_out: &mut IndexTensor,
    ) -> Result<(), Error>
    where
        AnyIndex: VectorIndex,
        ValueTensor: TensorMut<Scalar::Output, MAX_RANK> + ?Sized,
        IndexTensor: TensorMut<usize, MAX_RANK> + ?Sized,
    {
        self.view()
            .minmax_axis_into(axis, keep_dims, min_out, argmin_out, max_out, argmax_out)
    }

    fn min_all(&self) -> Result<Scalar::Output, Error> { self.view().min_all() }

    fn argmin_all(&self) -> Result<usize, Error> { self.view().argmin_all() }

    fn max_all(&self) -> Result<Scalar::Output, Error> { self.view().max_all() }

    fn argmax_all(&self) -> Result<usize, Error> { self.view().argmax_all() }

    fn min_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<Scalar::Output, Self::Alloc, MAX_RANK>, Error>
    where
        Self::Alloc: Clone,
    {
        self.view().min_axis(axis, keep_dims)
    }

    fn argmin_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Self::Alloc, MAX_RANK>, Error>
    where
        Self::Alloc: Clone,
    {
        self.view().argmin_axis(axis, keep_dims)
    }

    fn max_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<Scalar::Output, Self::Alloc, MAX_RANK>, Error>
    where
        Self::Alloc: Clone,
    {
        self.view().max_axis(axis, keep_dims)
    }

    fn argmax_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Self::Alloc, MAX_RANK>, Error>
    where
        Self::Alloc: Clone,
    {
        self.view().argmax_axis(axis, keep_dims)
    }
}

impl<Scalar: Clone + ReduceMinMax, const R: usize, C: TensorRef<Scalar, R>> MinMaxOps<Scalar, R> for C where
    Scalar::Output: Clone + Default + PartialOrd
{
}

// endregion: MomentsOps / MinMaxOps

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, bf16, e2m3, e3m2, e4m3, e5m2, f16, FloatLike, NumberLike, TestableType};

    // region: ReduceMoments

    fn check_reduce_moments<Scalar>(input_values: &[f32])
    where
        Scalar: FloatLike + TestableType + ReduceMoments,
        Scalar::SumOutput: FloatLike,
        Scalar::SumSqOutput: FloatLike,
    {
        let data: Vec<Scalar> = input_values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let stride = core::mem::size_of::<Scalar>();
        let (actual_sum, actual_sumsq) = Scalar::reduce_moments(&data, stride).unwrap();
        let expected_sum: f64 = input_values.iter().map(|&v| v as f64).sum();
        let expected_sumsq: f64 = input_values.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let sample_count = input_values.len() as f64;
        assert_close(
            actual_sum.to_f64(),
            expected_sum,
            Scalar::atol() * sample_count,
            Scalar::rtol(),
            &format!("reduce_moments<{}> sum", core::any::type_name::<Scalar>()),
        );
        assert_close(
            actual_sumsq.to_f64(),
            expected_sumsq,
            Scalar::atol() * sample_count,
            Scalar::rtol(),
            &format!("reduce_moments<{}> sumsq", core::any::type_name::<Scalar>()),
        );
    }

    #[test]
    fn moments_reduction() {
        // Float types — SumOutput/SumSqOutput are FloatLike, either f32 or f64
        let input_values = &[1.0, 2.0, 3.0, 4.0, 5.0];
        check_reduce_moments::<f64>(input_values);
        check_reduce_moments::<f32>(input_values);
        check_reduce_moments::<f16>(input_values);
        check_reduce_moments::<bf16>(input_values);
        check_reduce_moments::<e4m3>(input_values);
        check_reduce_moments::<e5m2>(&[1.0, 2.0, 3.0]);
        check_reduce_moments::<e2m3>(&[1.0, 2.0, 3.0]);
        check_reduce_moments::<e3m2>(&[1.0, 2.0, 3.0]);

        // Integer types — now also go through generics with FloatLike for i64/u64
        let signed = &[1.0_f32, -2.0, 3.0, -4.0, 5.0];
        let unsigned = &[1.0_f32, 2.0, 3.0, 4.0, 5.0];
        check_reduce_moments::<i8>(signed);
        check_reduce_moments::<u8>(unsigned);
        check_reduce_moments::<i16>(signed);
        check_reduce_moments::<u16>(unsigned);
        check_reduce_moments::<i32>(signed);
        check_reduce_moments::<u32>(unsigned);
        check_reduce_moments::<i64>(signed);
        check_reduce_moments::<u64>(unsigned);
    }

    // endregion

    // region: ReduceMinMax

    fn check_reduce_minmax<Scalar>(input_values: &[f32])
    where
        Scalar: FloatLike + TestableType + ReduceMinMax,
        Scalar::Output: FloatLike,
    {
        let data: Vec<Scalar> = input_values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let stride = core::mem::size_of::<Scalar>();
        let MinMaxResult {
            min_value: actual_min,
            min_index: actual_min_index,
            max_value: actual_max,
            max_index: actual_max_index,
        } = Scalar::reduce_minmax(&data, stride)
            .unwrap()
            .expect("Expected Some for non-NaN input");
        let (expected_min_index, expected_min) = input_values
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .unwrap();
        let (expected_max_index, expected_max) = input_values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .unwrap();
        assert_close(
            actual_min.to_f64(),
            *expected_min as f64,
            Scalar::atol(),
            0.0,
            &format!("reduce_minmax<{}> min", core::any::type_name::<Scalar>()),
        );
        assert_eq!(
            actual_min_index,
            expected_min_index,
            "reduce_minmax<{}> min_index",
            core::any::type_name::<Scalar>()
        );
        assert_close(
            actual_max.to_f64(),
            *expected_max as f64,
            Scalar::atol(),
            0.0,
            &format!("reduce_minmax<{}> max", core::any::type_name::<Scalar>()),
        );
        assert_eq!(
            actual_max_index,
            expected_max_index,
            "reduce_minmax<{}> max_index",
            core::any::type_name::<Scalar>()
        );
    }

    #[test]
    fn minmax_reduction() {
        // All FloatLike types — Output is also FloatLike
        let input_values = &[3.0, 1.0, 4.0, 1.5, 5.0, 2.0];
        check_reduce_minmax::<f64>(input_values);
        check_reduce_minmax::<f32>(input_values);
        check_reduce_minmax::<f16>(input_values);
        check_reduce_minmax::<bf16>(input_values);
        check_reduce_minmax::<e4m3>(input_values);
        check_reduce_minmax::<e5m2>(input_values);
        check_reduce_minmax::<e2m3>(&[3.0, 1.0, 4.0, 1.5, 5.0, 2.0]);
        check_reduce_minmax::<e3m2>(input_values);
        check_reduce_minmax::<i8>(input_values);
        check_reduce_minmax::<u8>(input_values);
        check_reduce_minmax::<i32>(input_values);
        check_reduce_minmax::<u32>(input_values);

        // i16, u16, i64, u64 — now also go through generics with FloatLike
        check_reduce_minmax::<i16>(&[3.0, -1.0, 4.0, -5.0, 2.0]);
        check_reduce_minmax::<u16>(&[3.0, 1.0, 4.0, 5.0, 2.0]);
        check_reduce_minmax::<i64>(&[3.0, -1.0, 4.0, -5.0, 2.0]);
        check_reduce_minmax::<u64>(&[3.0, 1.0, 4.0, 5.0, 2.0]);
    }

    #[test]
    fn minmax_reduction_all_nan() {
        let nan_f64: Vec<f64> = vec![f64::NAN; 16];
        assert_eq!(f64::reduce_minmax(&nan_f64, core::mem::size_of::<f64>()), Ok(None));

        let nan_f32: Vec<f32> = vec![f32::NAN; 16];
        assert_eq!(f32::reduce_minmax(&nan_f32, core::mem::size_of::<f32>()), Ok(None));

        let nan_f16: Vec<f16> = vec![f16::NAN; 16];
        assert_eq!(f16::reduce_minmax(&nan_f16, core::mem::size_of::<f16>()), Ok(None));

        let nan_bf16: Vec<bf16> = vec![bf16::NAN; 16];
        assert_eq!(bf16::reduce_minmax(&nan_bf16, core::mem::size_of::<bf16>()), Ok(None));

        let nan_e4m3: Vec<e4m3> = vec![e4m3::NAN; 16];
        assert_eq!(e4m3::reduce_minmax(&nan_e4m3, core::mem::size_of::<e4m3>()), Ok(None));

        let nan_e5m2: Vec<e5m2> = vec![e5m2::NAN; 16];
        assert_eq!(e5m2::reduce_minmax(&nan_e5m2, core::mem::size_of::<e5m2>()), Ok(None));
    }

    #[test]
    fn minmax_reduction_mixed_nan() {
        let data = vec![f64::NAN, 3.0, f64::NAN, 1.0, f64::NAN, 5.0];
        let MinMaxResult {
            min_value,
            min_index,
            max_value,
            max_index,
        } = f64::reduce_minmax(&data, core::mem::size_of::<f64>()).unwrap().unwrap();
        assert_close(min_value, 1.0, 1e-10, 0.0, "mixed min");
        assert_eq!(min_index, 3);
        assert_close(max_value, 5.0, 1e-10, 0.0, "mixed max");
        assert_eq!(max_index, 5);

        let data_f32: Vec<f32> = vec![f32::NAN, 3.0, f32::NAN, 1.0, f32::NAN, 5.0];
        let MinMaxResult {
            min_value,
            min_index,
            max_value,
            max_index,
        } = f32::reduce_minmax(&data_f32, core::mem::size_of::<f32>())
            .unwrap()
            .unwrap();
        assert_close(min_value as f64, 1.0, 1e-5, 0.0, "mixed f32 min");
        assert_eq!(min_index, 3);
        assert_close(max_value as f64, 5.0, 1e-5, 0.0, "mixed f32 max");
        assert_eq!(max_index, 5);
    }

    // endregion

    // region: BitwiseReductionsOps across containers

    #[test]
    fn bitwise_popcount_across_tensor_shapes() {
        use crate::tensor::Tensor;

        // 16 bytes × 8 bits = 128 bits total. Alternating fully-set/zero bytes → popcount = 64.
        let bits_storage: Vec<u1x8> = (0..16u8)
            .map(|byte_index| {
                if byte_index % 2 == 0 {
                    u1x8(0xFFu8)
                } else {
                    u1x8(0x00u8)
                }
            })
            .collect();
        let mut tensor = Tensor::<u1x8>::zeros(&[4, 32]).unwrap();
        for (slot, value) in tensor.as_mut_slice().iter_mut().zip(bits_storage.iter()) {
            *slot = *value;
        }

        // Same `.popcount()` spelling on Tensor and TensorView.
        assert_eq!(tensor.popcount().unwrap(), 64);
        assert_eq!(tensor.view().popcount().unwrap(), 64);

        assert!(tensor.any_set().unwrap());
        assert!(tensor.view().any_set().unwrap());

        assert!(!tensor.none_set().unwrap());
        assert!(!tensor.view().none_set().unwrap());

        assert!(!tensor.all_set().unwrap());
        assert!(!tensor.view().all_set().unwrap());

        // All-zeros tensor → none_set, !any_set, !all_set.
        let zero_tensor = Tensor::<u1x8>::zeros(&[4, 32]).unwrap();
        assert_eq!(zero_tensor.popcount().unwrap(), 0);
        assert!(!zero_tensor.any_set().unwrap());
        assert!(zero_tensor.none_set().unwrap());
        assert!(!zero_tensor.all_set().unwrap());

        // All-ones tensor → all_set.
        let mut ones_tensor = Tensor::<u1x8>::zeros(&[4, 32]).unwrap();
        for slot in ones_tensor.as_mut_slice().iter_mut() {
            *slot = u1x8(0xFFu8);
        }
        assert_eq!(ones_tensor.popcount().unwrap(), 128);
        assert!(ones_tensor.any_set().unwrap());
        assert!(!ones_tensor.none_set().unwrap());
        assert!(ones_tensor.all_set().unwrap());
    }

    #[test]
    fn bitwise_reductions_on_vector_containers() {
        use crate::vector::{Vector, VectorSpan, VectorView};

        // Three set bytes — 24 bits — one zero byte → popcount = 24.
        let mut storage = [u1x8(0xFFu8), u1x8(0xFFu8), u1x8(0x00u8), u1x8(0xFFu8)];
        let mut vector = Vector::<u1x8>::zeros(32).unwrap();
        for (slot, value) in vector.as_mut_slice().iter_mut().zip(storage.iter()) {
            *slot = *value;
        }
        assert_eq!(vector.popcount().unwrap(), 24);
        assert!(vector.any_set().unwrap());
        assert!(!vector.none_set().unwrap());
        assert!(!vector.all_set().unwrap());

        let view =
            unsafe { VectorView::<u1x8>::from_raw_parts(storage.as_ptr(), 32, core::mem::size_of::<u1x8>() as isize) };
        assert_eq!(view.popcount().unwrap(), 24);
        assert!(view.any_set().unwrap());

        let span = unsafe {
            VectorSpan::<u1x8>::from_raw_parts(storage.as_mut_ptr(), 32, core::mem::size_of::<u1x8>() as isize)
        };
        assert_eq!(span.popcount().unwrap(), 24);
        assert!(span.any_set().unwrap());
    }

    // endregion: BitwiseReductionsOps across containers

    // region: tensor-shaped MomentsOps / MinMaxOps wrappers

    #[test]
    fn reductions_axis_and_strided_views() {
        use crate::tensor::{MinMaxResult, SliceRange, Tensor};

        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::<f32>::from_slice(&data, &[3, 4]).unwrap();
        let a_even = a.slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)]).unwrap();

        let sum_all = a_even.sum_all().unwrap();
        assert!((sum_all - 30.0).abs() < 1e-6);

        let norm_all = a_even.norm_all().unwrap();
        assert!((norm_all - 14.832396974191326).abs() < 1e-9);

        let (sum_axis0, sumsq_axis0) = a_even.moments_axis(0, false).unwrap();
        assert_eq!(sum_axis0.shape(), &[2]);
        assert!((sum_axis0.as_slice()[0] - 12.0).abs() < 1e-6);
        assert!((sum_axis0.as_slice()[1] - 18.0).abs() < 1e-6);
        assert!((sumsq_axis0.as_slice()[0] - 80.0).abs() < 1e-6);
        assert!((sumsq_axis0.as_slice()[1] - 140.0).abs() < 1e-6);

        let sum_axis1_keep = a_even.sum_axis(-1_i32, true).unwrap();
        assert_eq!(sum_axis1_keep.shape(), &[3, 1]);
        assert!((sum_axis1_keep.as_slice()[0] - 2.0).abs() < 1e-6);
        assert!((sum_axis1_keep.as_slice()[1] - 10.0).abs() < 1e-6);
        assert!((sum_axis1_keep.as_slice()[2] - 18.0).abs() < 1e-6);

        let MinMaxResult {
            min_value: min_axis0,
            min_index: argmin_axis0,
            max_value: max_axis0,
            max_index: argmax_axis0,
        } = a_even.minmax_axis(0, false).unwrap();
        assert_eq!(min_axis0.as_slice(), &[0.0, 2.0]);
        assert_eq!(max_axis0.as_slice(), &[8.0, 10.0]);
        assert_eq!(argmin_axis0.as_slice(), &[0, 0]);
        assert_eq!(argmax_axis0.as_slice(), &[2, 2]);

        let reversed = a
            .slice(&[SliceRange::full(), SliceRange::range_step(3, 0, -1)])
            .unwrap();
        let reversed_sum = reversed.sum_axis(-1_i32, false).unwrap();
        assert_eq!(reversed_sum.shape(), &[3]);
        assert_eq!(reversed_sum.as_slice(), &[6.0, 18.0, 30.0]);

        let reversed_argmin = reversed.argmin_axis(-1_i32, false).unwrap();
        let reversed_argmax = reversed.argmax_axis(-1_i32, false).unwrap();
        assert_eq!(reversed_argmin.as_slice(), &[2, 2, 2]);
        assert_eq!(reversed_argmax.as_slice(), &[0, 0, 0]);
    }

    // endregion: tensor-shaped MomentsOps / MinMaxOps wrappers
}
