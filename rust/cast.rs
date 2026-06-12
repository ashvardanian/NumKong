//! Type casting between scalar formats — slice traits and tensor-shaped wrappers.
//!
//! This module provides:
//!
//! - [`CastDtype`]: Trait marking types eligible for bulk casting
//! - [`cast`]: Bulk-converts a slice from one scalar format to another
//! - [`CastOps`]: Tensor-shaped extension trait — auto-implemented on every
//!   [`crate::tensor::TensorRef`] so any container can do `tensor.try_cast_dtype::<Destination>()`

extern crate alloc;

use crate::types::{bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, StorageElement};
use alloc::vec::Vec;
use core::ffi::c_void;

#[link(name = "numkong")]
extern "C" {
    fn nk_cast(from: *const c_void, from_type: u32, n: usize, to: *mut c_void, to_type: u32);

    fn nk_cast_block_scaled(
        from: *const c_void,
        from_scales: *const c_void,
        from_tensor_scale: *const ScalarBuffer,
        from_format: *const BlockScaledDescriptor,
        to: *mut c_void,
        to_scales: *mut c_void,
        to_tensor_scale: *mut ScalarBuffer,
        to_format: *const BlockScaledDescriptor,
        count: usize,
    );
}

/// Internal dtype codes matching `nk_dtype_t` from C.
/// Not exposed to users.
pub(crate) mod dtype {
    pub(crate) const F64: u32 = 1 << 10;
    pub(crate) const F32: u32 = 1 << 11;
    pub(crate) const BF16: u32 = 1 << 13;
    pub(crate) const F16: u32 = 1 << 12;
    pub(crate) const E5M2: u32 = 1 << 15;
    pub(crate) const E4M3: u32 = 1 << 14;
    pub(crate) const E3M2: u32 = 1 << 19;
    pub(crate) const E2M3: u32 = 1 << 18;

    pub(crate) const F64C: u32 = 1 << 20;
    pub(crate) const F32C: u32 = 1 << 21;
    pub(crate) const BF16C: u32 = 1 << 23;
    pub(crate) const F16C: u32 = 1 << 22;

    pub(crate) const I64: u32 = 1 << 5;
    pub(crate) const I32: u32 = 1 << 4;
    pub(crate) const I16: u32 = 1 << 3;
    pub(crate) const I8: u32 = 1 << 2;

    pub(crate) const U64: u32 = 1 << 9;
    pub(crate) const U32: u32 = 1 << 8;
    pub(crate) const U16: u32 = 1 << 7;
    pub(crate) const U8: u32 = 1 << 6;

    // Block-scaled element / scale dtype codes (matching `nk_dtype_t`).
    // Consumed by `crate::block_scaled` to build format descriptors.
    pub(crate) const E2M1: u32 = 1 << 24;
    pub(crate) const UE8M0: u32 = 1 << 25;
    pub(crate) const UE4M3: u32 = 1 << 26;

    /// Sentinel for "no dtype" (plain buffers, absent scale / tensor-scale).
    pub(crate) const UNKNOWN: u32 = 0;
}

// Sealed trait pattern to prevent external implementations
mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for super::f16 {}
    impl Sealed for super::bf16 {}
    impl Sealed for super::e4m3 {}
    impl Sealed for super::e5m2 {}
    impl Sealed for super::e2m3 {}
    impl Sealed for super::e3m2 {}
    impl Sealed for super::f64c {}
    impl Sealed for super::f32c {}
    impl Sealed for super::f16c {}
    impl Sealed for super::bf16c {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Trait for types that can participate in cast operations.
///
/// This trait is sealed - users cannot implement it for their own types.
pub trait CastDtype: private::Sealed + StorageElement {
    #[doc(hidden)]
    fn dtype_code() -> u32;
}

impl CastDtype for f64 {
    fn dtype_code() -> u32 { dtype::F64 }
}
impl CastDtype for f32 {
    fn dtype_code() -> u32 { dtype::F32 }
}
impl CastDtype for f16 {
    fn dtype_code() -> u32 { dtype::F16 }
}
impl CastDtype for bf16 {
    fn dtype_code() -> u32 { dtype::BF16 }
}
impl CastDtype for e4m3 {
    fn dtype_code() -> u32 { dtype::E4M3 }
}
impl CastDtype for e5m2 {
    fn dtype_code() -> u32 { dtype::E5M2 }
}
impl CastDtype for e2m3 {
    fn dtype_code() -> u32 { dtype::E2M3 }
}
impl CastDtype for e3m2 {
    fn dtype_code() -> u32 { dtype::E3M2 }
}
impl CastDtype for f64c {
    fn dtype_code() -> u32 { dtype::F64C }
}
impl CastDtype for f32c {
    fn dtype_code() -> u32 { dtype::F32C }
}
impl CastDtype for f16c {
    fn dtype_code() -> u32 { dtype::F16C }
}
impl CastDtype for bf16c {
    fn dtype_code() -> u32 { dtype::BF16C }
}
impl CastDtype for i8 {
    fn dtype_code() -> u32 { dtype::I8 }
}
impl CastDtype for i16 {
    fn dtype_code() -> u32 { dtype::I16 }
}
impl CastDtype for i32 {
    fn dtype_code() -> u32 { dtype::I32 }
}
impl CastDtype for i64 {
    fn dtype_code() -> u32 { dtype::I64 }
}
impl CastDtype for u8 {
    fn dtype_code() -> u32 { dtype::U8 }
}
impl CastDtype for u16 {
    fn dtype_code() -> u32 { dtype::U16 }
}
impl CastDtype for u32 {
    fn dtype_code() -> u32 { dtype::U32 }
}
impl CastDtype for u64 {
    fn dtype_code() -> u32 { dtype::U64 }
}

/// Cast source slice elements to destination slice.
///
/// Converts elements from source type `S` to destination type `D` using
/// hardware-accelerated SIMD operations when available.
///
/// # Arguments
/// * `source` - Source slice of elements to cast
/// * `dest` - Destination slice to receive cast elements (must be same length as source)
///
/// # Returns
/// * `Some(())` if successful
/// * `None` if slices have different lengths
///
/// # Example
/// ```ignore
/// use numkong::{f16, cast};
///
/// let f16_data: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
/// let mut f32_data: Vec<f32> = vec![0.0; f16_data.len()];
/// cast(&f16_data, &mut f32_data);
/// ```
pub fn cast<S: CastDtype, D: CastDtype>(source: &[S], dest: &mut [D]) -> Option<()> {
    if source.len() != dest.len() {
        return None;
    }
    unsafe {
        nk_cast(
            source.as_ptr() as *const c_void,
            S::dtype_code(),
            source.len(),
            dest.as_mut_ptr() as *mut c_void,
            D::dtype_code(),
        );
    }
    Some(())
}

// region: Tensor-shaped cast (moved from crate::tensor)

use crate::tensor::{try_reborrow_tensor_into, Global, Tensor, TensorError, TensorRef};

/// Extension trait: type casting for any [`TensorRef`] implementor.
pub trait CastOps<Source: Clone + CastDtype, const MAX_RANK: usize>: TensorRef<Source, MAX_RANK> {
    fn try_cast_dtype<Destination: Clone + CastDtype>(
        &self,
    ) -> Result<Tensor<Destination, Global, MAX_RANK>, TensorError> {
        self.view().try_cast_dtype()
    }
}

impl<Source: Clone + CastDtype, const R: usize, C: TensorRef<Source, R>> CastOps<Source, R> for C {}

impl<Source: Clone + CastDtype, const MAX_RANK: usize> Tensor<Source, Global, MAX_RANK> {
    pub fn try_cast_dtype_into<Destination: Clone + CastDtype>(
        &self,
        out: &mut Tensor<Destination, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| view.try_cast_dtype_into(span))
    }
}

// endregion: Tensor-shaped cast

// region: Block-Scaled Formats (OCP MX family + NVIDIA NVFP4)

use crate::tensor::{ScaledTensor, ScaledTensorView};
use crate::types::{Ue4m3, Ue8m0};

/// Packed FP4 (E2M1) element pair: two 4-bit elements share one byte.
///
/// This is the storage scalar for the `elements` tensor of NVFP4 / MXFP4. Like the
/// other sub-byte packers ([`crate::types::u4x2`]), it reports
/// `dimensions_per_value() == 2` so a tensor of logical shape `(rows, cols)` allocates
/// `rows * cols / 2` bytes — exactly `nk_block_scaled_elements_size`. Bytes are produced
/// and consumed by the C kernel (`element_dtype = nk_e2m1_k`); Rust never unpacks nibbles.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct e2m1x2(pub u8);

impl core::fmt::Debug for e2m1x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { write!(f, "e2m1x2(0x{:02x})", self.0) }
}

impl StorageElement for e2m1x2 {
    fn zero() -> Self { e2m1x2(0) }
    fn one() -> Self {
        const E2M1_ONE_NIBBLE: u8 = 0x2; // E2M1 encoding of +1.0
        e2m1x2((E2M1_ONE_NIBBLE << 4) | E2M1_ONE_NIBBLE)
    }
    fn dimensions_per_value() -> usize { 2 }
}

/// `#[repr(C)]` mirror of `nk_block_scaled_format_t`.
///
/// Field order and types match the C struct exactly so a `*const` can be handed to
/// `nk_cast_block_scaled`. The `tensor_scale_dtype` field is `nk_f32_k` for NVFP4 and
/// `nk_dtype_unknown_k` (0) for the MX family; `block_size` is `0` for plain buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockScaledDescriptor {
    pub element_dtype: u32,
    pub scale_dtype: u32,
    pub tensor_scale_dtype: u32,
    pub block_size: usize,
}

impl BlockScaledDescriptor {
    /// `nk_plain(dtype)` — a non-block-scaled buffer of `element_dtype`.
    #[inline]
    pub fn plain(element_dtype: u32) -> Self {
        BlockScaledDescriptor {
            element_dtype,
            scale_dtype: dtype::UNKNOWN,
            tensor_scale_dtype: dtype::UNKNOWN,
            block_size: 0,
        }
    }

    /// Element storage bytes for `count` logical elements — mirrors
    /// `nk_block_scaled_elements_size`: `round_up(count * element_bits, 8) / 8`.
    #[inline]
    pub fn elements_size(&self, count: usize) -> usize { (count * dtype_bits(self.element_dtype)).div_ceil(8) }

    /// Scale storage bytes for `count` logical elements — mirrors
    /// `nk_block_scaled_scales_size`: `0` when plain, else `round_up(count, block_size)`.
    #[inline]
    pub fn scales_size(&self, count: usize) -> usize {
        if self.scale_dtype == dtype::UNKNOWN || self.block_size == 0 {
            return 0;
        }
        count.div_ceil(self.block_size)
    }
}

/// Bits-per-element for the element dtypes used by block-scaled formats.
/// Mirrors the relevant arms of `nk_dtype_bits`.
#[inline]
fn dtype_bits(code: u32) -> usize {
    match code {
        dtype::E2M1 => 4,
        dtype::E2M3 | dtype::E3M2 | dtype::E4M3 | dtype::E5M2 | dtype::I8 => 8,
        dtype::F16 | dtype::BF16 => 16,
        dtype::F32 => 32,
        dtype::F64 => 64,
        _ => 0,
    }
}

/// `#[repr(C)]` mirror of `nk_scalar_buffer_t` — a 16-byte union. The only field the
/// block-scaled kernel reads or writes is the leading `f32` (NVFP4 per-tensor multiplier).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct ScalarBuffer {
    pub bytes: [u8; 16],
}

impl ScalarBuffer {
    #[inline]
    fn from_f32(value: f32) -> Self {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&value.to_ne_bytes());
        ScalarBuffer { bytes }
    }

    #[inline]
    fn to_f32(self) -> f32 {
        let mut four = [0u8; 4];
        four.copy_from_slice(&self.bytes[..4]);
        f32::from_ne_bytes(four)
    }
}

/// Compile-time description of a block-scaled tensor layout.
///
/// Each implementor is a zero-sized tag (`Nvfp4`, `Mxfp4`, …). The associated types name
/// the storage scalars of the two composed tensors: `Element` for the packed values and
/// `Scale` for the per-block scale bytes. There are no `macro_rules!` here — every format
/// is an explicit `impl`.
pub trait BlockScaledFormat {
    /// Storage scalar of the `elements` tensor (e.g. [`e2m1x2`], `e4m3`, `i8`).
    type Element: StorageElement + Clone;
    /// Storage scalar of the `block_scales` tensor ([`Ue4m3`] or [`Ue8m0`]).
    type Scale: StorageElement + Clone;
    /// Logical elements per block (quantization is on the last axis).
    const BLOCK_SIZE: usize;
    /// Whether this format carries a per-tensor f32 multiplier (`tensor_scale`).
    const HAS_TENSOR_SCALE: bool;
    /// The C descriptor literal for this format.
    fn descriptor() -> BlockScaledDescriptor;
}

/// NVIDIA NVFP4: E2M1 elements, UE4M3 block scale (block=16), per-tensor f32 scale.
#[derive(Clone, Copy, Debug, Default)]
pub struct Nvfp4;
impl BlockScaledFormat for Nvfp4 {
    type Element = e2m1x2;
    type Scale = Ue4m3;
    const BLOCK_SIZE: usize = 16;
    const HAS_TENSOR_SCALE: bool = true;
    fn descriptor() -> BlockScaledDescriptor {
        BlockScaledDescriptor {
            element_dtype: dtype::E2M1,
            scale_dtype: dtype::UE4M3,
            tensor_scale_dtype: dtype::F32,
            block_size: 16,
        }
    }
}

/// OCP MXFP4: E2M1 elements, UE8M0 block scale (block=32), no tensor scale.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mxfp4;
impl BlockScaledFormat for Mxfp4 {
    type Element = e2m1x2;
    type Scale = Ue8m0;
    const BLOCK_SIZE: usize = 32;
    const HAS_TENSOR_SCALE: bool = false;
    fn descriptor() -> BlockScaledDescriptor {
        BlockScaledDescriptor {
            element_dtype: dtype::E2M1,
            scale_dtype: dtype::UE8M0,
            tensor_scale_dtype: dtype::UNKNOWN,
            block_size: 32,
        }
    }
}

/// OCP MXFP6 (E2M3 variant): E2M3 elements, UE8M0 block scale (block=32).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mxfp6E2m3;
impl BlockScaledFormat for Mxfp6E2m3 {
    type Element = e2m3;
    type Scale = Ue8m0;
    const BLOCK_SIZE: usize = 32;
    const HAS_TENSOR_SCALE: bool = false;
    fn descriptor() -> BlockScaledDescriptor {
        BlockScaledDescriptor {
            element_dtype: dtype::E2M3,
            scale_dtype: dtype::UE8M0,
            tensor_scale_dtype: dtype::UNKNOWN,
            block_size: 32,
        }
    }
}

/// OCP MXFP6 (E3M2 variant): E3M2 elements, UE8M0 block scale (block=32).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mxfp6E3m2;
impl BlockScaledFormat for Mxfp6E3m2 {
    type Element = e3m2;
    type Scale = Ue8m0;
    const BLOCK_SIZE: usize = 32;
    const HAS_TENSOR_SCALE: bool = false;
    fn descriptor() -> BlockScaledDescriptor {
        BlockScaledDescriptor {
            element_dtype: dtype::E3M2,
            scale_dtype: dtype::UE8M0,
            tensor_scale_dtype: dtype::UNKNOWN,
            block_size: 32,
        }
    }
}

/// OCP MXFP8 (E4M3 variant): E4M3 elements, UE8M0 block scale (block=32).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mxfp8E4m3;
impl BlockScaledFormat for Mxfp8E4m3 {
    type Element = e4m3;
    type Scale = Ue8m0;
    const BLOCK_SIZE: usize = 32;
    const HAS_TENSOR_SCALE: bool = false;
    fn descriptor() -> BlockScaledDescriptor {
        BlockScaledDescriptor {
            element_dtype: dtype::E4M3,
            scale_dtype: dtype::UE8M0,
            tensor_scale_dtype: dtype::UNKNOWN,
            block_size: 32,
        }
    }
}

/// OCP MXFP8 (E5M2 variant): E5M2 elements, UE8M0 block scale (block=32).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mxfp8E5m2;
impl BlockScaledFormat for Mxfp8E5m2 {
    type Element = e5m2;
    type Scale = Ue8m0;
    const BLOCK_SIZE: usize = 32;
    const HAS_TENSOR_SCALE: bool = false;
    fn descriptor() -> BlockScaledDescriptor {
        BlockScaledDescriptor {
            element_dtype: dtype::E5M2,
            scale_dtype: dtype::UE8M0,
            tensor_scale_dtype: dtype::UNKNOWN,
            block_size: 32,
        }
    }
}

/// OCP MXINT8: i8 elements, UE8M0 block scale (block=32).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mxint8;
impl BlockScaledFormat for Mxint8 {
    type Element = i8;
    type Scale = Ue8m0;
    const BLOCK_SIZE: usize = 32;
    const HAS_TENSOR_SCALE: bool = false;
    fn descriptor() -> BlockScaledDescriptor {
        BlockScaledDescriptor {
            element_dtype: dtype::I8,
            scale_dtype: dtype::UE8M0,
            tensor_scale_dtype: dtype::UNKNOWN,
            block_size: 32,
        }
    }
}

// endregion: Block-Scaled Formats

// region: Block-Scaled Casts

use crate::tensor::TensorView;

/// Validate a block-scaled shape and return the per-block scales shape: the input shape with its
/// last extent divided by `block_size`. Quantization runs along the last axis, which must split
/// evenly into blocks; any rank with at least one axis is accepted.
fn blocked_scales_shape(shape: &[usize], block_size: usize) -> Result<Vec<usize>, TensorError> {
    let Some((&last_extent, leading)) = shape.split_last() else {
        return Err(TensorError::DimensionMismatch { expected: 1, got: 0 });
    };
    if last_extent % block_size != 0 {
        return Err(TensorError::InvalidShape {
            axis: leading.len(),
            size: last_extent,
            reason: "last axis must be divisible by the format block size",
        });
    }
    let mut scales = shape.to_vec();
    *scales.last_mut().expect("shape has a last axis") = last_extent / block_size;
    Ok(scales)
}

/// The one place the block-scaled C kernel is called. Marshals the per-tensor scale — seeded from
/// the source, or derived into a fresh buffer for the destination — and returns the derived
/// destination scale when `to_derives_scale` is set. `count` is the logical element total (`numel`).
#[allow(clippy::too_many_arguments)]
fn block_scaled_cast_(
    from_elements: *const c_void,
    from_scales: *const c_void,
    from_tensor_scale: Option<f32>,
    from_format: &BlockScaledDescriptor,
    to_elements: *mut c_void,
    to_scales: *mut c_void,
    to_derives_scale: bool,
    to_format: &BlockScaledDescriptor,
    count: usize,
) -> Option<f32> {
    let from_scale_buf = from_tensor_scale.map(ScalarBuffer::from_f32);
    let from_scale_ptr = from_scale_buf
        .as_ref()
        .map_or(core::ptr::null(), |buf| buf as *const ScalarBuffer);
    let mut to_scale_buf = ScalarBuffer::from_f32(0.0);
    let to_scale_ptr = if to_derives_scale {
        &mut to_scale_buf as *mut ScalarBuffer
    } else {
        core::ptr::null_mut()
    };

    // SAFETY: the caller sizes the source slices and the freshly-allocated destination tensors from
    // the same shape, so both buffers cover `count` logical elements; the scale buffers live across
    // the call; the kernel reads the source and writes only the destination.
    unsafe {
        nk_cast_block_scaled(
            from_elements,
            from_scales,
            from_scale_ptr,
            from_format,
            to_elements,
            to_scales,
            to_scale_ptr,
            to_format,
            count,
        );
    }

    to_derives_scale.then(|| to_scale_buf.to_f32())
}

/// Encode a dense `f32` matrix into a [`ScaledTensor`].
///
/// The source must be a contiguous 2D `(rows, cols)` view with `cols` divisible by
/// `F::BLOCK_SIZE`. For formats with a per-tensor scale (NVFP4), the multiplier is derived
/// from the tensor amax by the kernel (we seed the buffer with `0.0` and read it back).
impl<'a, const MAX_RANK: usize> TensorView<'a, f32, MAX_RANK> {
    pub fn try_cast_to_scaled<F: BlockScaledFormat>(&self) -> Result<ScaledTensor<F>, TensorError> {
        let shape = self.shape();
        let scales_shape = blocked_scales_shape(shape, F::BLOCK_SIZE)?;
        let count: usize = shape.iter().product();
        let source = self.as_packed_slice().ok_or(TensorError::NonContiguousRows)?;

        let mut elements = Tensor::<F::Element>::try_zeros(shape)?;
        let mut block_scales = Tensor::<F::Scale>::try_zeros(&scales_shape)?;
        let tensor_scale = block_scaled_cast_(
            source.as_ptr() as *const c_void,
            core::ptr::null(),
            None,
            &BlockScaledDescriptor::plain(dtype::F32),
            elements.as_mut_ptr() as *mut c_void,
            block_scales.as_mut_ptr() as *mut c_void,
            F::HAS_TENSOR_SCALE,
            &F::descriptor(),
            count,
        );
        Ok(ScaledTensor::from_parts(elements, block_scales, tensor_scale))
    }
}

/// Decode / materialize: a [`ScaledTensorView`] → a dense `Tensor<T>` (e.g. `f32`).
/// Transcode: a [`ScaledTensorView`] → another [`ScaledTensor`].
impl<'a, F: BlockScaledFormat> ScaledTensorView<'a, F> {
    /// Materialize this block-scaled view into a dense `Tensor<T>` of the same shape.
    pub fn try_cast_dense<T: Clone + CastDtype>(&self) -> Result<Tensor<T>, TensorError> {
        let shape = self.shape();
        let count: usize = shape.iter().product();
        let elements_view = self.elements();
        let scales_view = self.block_scales();
        let elements = elements_view.as_packed_slice().ok_or(TensorError::NonContiguousRows)?;
        let scales = scales_view.as_packed_slice().ok_or(TensorError::NonContiguousRows)?;

        let mut out = Tensor::<T>::try_zeros(shape)?;
        block_scaled_cast_(
            elements.as_ptr() as *const c_void,
            scales.as_ptr() as *const c_void,
            self.tensor_scale(),
            &F::descriptor(),
            out.as_mut_ptr() as *mut c_void,
            core::ptr::null_mut(),
            false,
            &BlockScaledDescriptor::plain(T::dtype_code()),
            count,
        );
        Ok(out)
    }

    /// Transcode this block-scaled view into a different block-scaled format.
    pub fn try_cast_to_scaled<G: BlockScaledFormat>(&self) -> Result<ScaledTensor<G>, TensorError> {
        let shape = self.shape();
        let scales_shape = blocked_scales_shape(shape, G::BLOCK_SIZE)?;
        let count: usize = shape.iter().product();

        let src_elements_view = self.elements();
        let src_scales_view = self.block_scales();
        let src_elements = src_elements_view
            .as_packed_slice()
            .ok_or(TensorError::NonContiguousRows)?;
        let src_scales = src_scales_view
            .as_packed_slice()
            .ok_or(TensorError::NonContiguousRows)?;

        let mut elements = Tensor::<G::Element>::try_zeros(shape)?;
        let mut block_scales = Tensor::<G::Scale>::try_zeros(&scales_shape)?;
        let tensor_scale = block_scaled_cast_(
            src_elements.as_ptr() as *const c_void,
            src_scales.as_ptr() as *const c_void,
            self.tensor_scale(),
            &F::descriptor(),
            elements.as_mut_ptr() as *mut c_void,
            block_scales.as_mut_ptr() as *mut c_void,
            G::HAS_TENSOR_SCALE,
            &G::descriptor(),
            count,
        );
        Ok(ScaledTensor::from_parts(elements, block_scales, tensor_scale))
    }
}

// endregion: Block-Scaled Casts

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        assert_close, bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, FloatLike, StorageElement,
        TestableType,
    };

    fn check_cast_roundtrip<T: FloatLike + TestableType + CastDtype>(values: &[f32]) {
        let src: Vec<T> = values.iter().map(|&v| T::from_f32(v)).collect();
        let mut dst = vec![0.0f32; src.len()];
        cast(&src, &mut dst).unwrap();
        for (i, (&expected, &actual)) in values.iter().zip(dst.iter()).enumerate() {
            assert_close(
                actual as f64,
                expected as f64,
                T::atol(),
                T::rtol(),
                &format!("cast_roundtrip<{}>[{i}]", core::any::type_name::<T>()),
            );
        }
    }

    #[test]
    fn cast_roundtrip() {
        check_cast_roundtrip::<f16>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<bf16>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<e4m3>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<e5m2>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<e2m3>(&[1.0, 0.5, -1.0]);
        check_cast_roundtrip::<e3m2>(&[1.0, 0.5, -1.0]);
    }

    fn check_special_roundtrip<T: FloatLike + CastDtype>() {
        // NaN, +Inf, -Inf, then a few finite values.
        let specials = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.5, -2.0];
        let src: Vec<T> = specials.iter().map(|&v| T::from_f32(v)).collect();
        let mut back = vec![0.0f32; src.len()];
        cast(&src, &mut back).unwrap();
        let ty = core::any::type_name::<T>();
        assert!(back[0].is_nan(), "{ty}: NaN not preserved");
        assert_eq!(back[1], f32::INFINITY, "{ty}: +Inf not preserved");
        assert_eq!(back[2], f32::NEG_INFINITY, "{ty}: -Inf not preserved");
        for (&expected, &actual) in specials[3..].iter().zip(&back[3..]) {
            assert!((expected - actual).abs() <= 0.05, "{ty}: finite {expected} -> {actual}");
        }
    }

    #[test]
    fn cast_preserves_nan_inf() {
        // f16/bf16 carry IEEE specials end-to-end; the narrower fp8/fp6 formats saturate instead.
        check_special_roundtrip::<f16>();
        check_special_roundtrip::<bf16>();
    }

    #[test]
    fn cast_f32_to_f16() {
        let src = [1.0f32, -1.0];
        let mut dst = [f16(0); 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!([dst[0].0, dst[1].0], [0x3C00, 0xBC00]);
    }

    #[test]
    fn cast_length_mismatch() {
        let src = [f16(0x3C00)];
        let mut dst = [0.0f32; 2];
        assert!(cast(&src, &mut dst).is_none());
    }

    #[test]
    fn cast_real_to_complex() {
        let src = [1.25f32, -2.5];
        let mut dst = [f32c::zero(); 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!(dst[0], f32c::from_real_imag(1.25, 0.0));
        assert_eq!(dst[1], f32c::from_real_imag(-2.5, 0.0));

        let src = [f16::from_f32(3.0), f16::from_f32(-4.0)];
        let mut dst = [f16c::zero(); 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!(
            dst,
            [
                f16c::from_real_imag(f16::from_f32(3.0), f16::ZERO),
                f16c::from_real_imag(f16::from_f32(-4.0), f16::ZERO),
            ]
        );
    }

    #[test]
    fn cast_complex_to_real() {
        let src = [f64c::from_real_imag(1.25, 9.0), f64c::from_real_imag(-2.5, -7.0)];
        let mut dst = [0.0f64; 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!(dst, [1.25, -2.5]);
    }

    #[test]
    fn cast_complex_to_complex() {
        let src = [f32c::from_real_imag(1.25, -2.5), f32c::from_real_imag(-3.5, 4.25)];
        let mut widened = [f64c::zero(); 2];
        cast(&src, &mut widened).unwrap();
        assert_eq!(widened[0], f64c::from_real_imag(1.25, -2.5));
        assert_eq!(widened[1], f64c::from_real_imag(-3.5, 4.25));

        let mut narrowed = [bf16c::zero(); 2];
        cast(&widened, &mut narrowed).unwrap();
        assert_eq!(narrowed[0].re.to_f32(), bf16::from_f32(1.25).to_f32());
        assert_eq!(narrowed[0].im.to_f32(), bf16::from_f32(-2.5).to_f32());
        assert_eq!(narrowed[1].re.to_f32(), bf16::from_f32(-3.5).to_f32());
        assert_eq!(narrowed[1].im.to_f32(), bf16::from_f32(4.25).to_f32());
    }

    #[test]
    fn cast_via_tensor_view_round_trip() {
        // Exercises `CastOps::try_cast_dtype` on a strided `TensorView`,
        // mirroring how callers reach the trait through the tensor-shaped wrapper.
        use crate::tensor::{SliceRange, Tensor};
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let source = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let even_columns = source
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        let widened = even_columns.try_cast_dtype::<f64>().unwrap();
        assert_eq!(widened.shape(), &[3, 2]);
        assert_eq!(widened.as_slice(), &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);

        let complexified = even_columns.try_cast_dtype::<f32c>().unwrap();
        assert_eq!(complexified.shape(), &[3, 2]);
        assert_eq!(complexified.as_slice()[0], f32c::from_real_imag(0.0, 0.0));
        assert_eq!(complexified.as_slice()[5], f32c::from_real_imag(10.0, 0.0));
    }

    // region: Block-scaled tests

    use crate::tensor::Tensor;

    /// Sample matrix: 2 rows × 32 columns (two NVFP4 blocks of 16 per row, one MX block of 32).
    fn sample_matrix() -> (Vec<f32>, [usize; 2]) {
        let cols = 32usize;
        let rows = 2usize;
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                // A smooth, signed ramp that exercises both NVFP4 blocks per row.
                let v = (c as f32 - 16.0) * 0.25 + (r as f32) * 2.0;
                data.push(v);
            }
        }
        (data, [rows, cols])
    }

    /// Encode then decode `sample_matrix` through format `F`; assert the dense round-trip error
    /// stays within `rel_bound` × amax (the per-format element resolution).
    fn check_roundtrip<F: BlockScaledFormat>(rel_bound: f32) {
        let (data, shape) = sample_matrix();
        let dense = Tensor::<f32>::try_from_slice(&data, &shape).unwrap();
        let scaled = dense.view().try_cast_to_scaled::<F>().unwrap();
        assert_eq!(scaled.shape(), &shape[..]);
        assert_eq!(scaled.block_scales().shape(), &[shape[0], shape[1] / F::BLOCK_SIZE]);

        let decoded = scaled.view().try_cast_dense::<f32>().unwrap();
        let max_abs = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for (i, (&expected, &actual)) in data.iter().zip(decoded.as_slice()).enumerate() {
            assert!(
                (expected - actual).abs() <= rel_bound * max_abs + 1e-3,
                "{} roundtrip[{i}]: expected {expected}, got {actual}",
                core::any::type_name::<F>()
            );
        }
    }

    #[test]
    fn roundtrip_bounded_error_all_formats() {
        check_roundtrip::<Nvfp4>(0.30);
        check_roundtrip::<Mxfp4>(0.30);
        check_roundtrip::<Mxfp6E2m3>(0.15);
        check_roundtrip::<Mxfp6E3m2>(0.25);
        check_roundtrip::<Mxfp8E4m3>(0.15);
        check_roundtrip::<Mxfp8E5m2>(0.25);
        check_roundtrip::<Mxint8>(0.05);
    }

    #[test]
    fn degenerate_blocks() {
        let block = Mxfp8E4m3::BLOCK_SIZE;
        // An all-zero block has zero amax → zero scale → all-zero decode (no division by zero / NaN).
        let zeros = Tensor::<f32>::try_zeros(&[1, block]).unwrap();
        let scaled = zeros.view().try_cast_to_scaled::<Mxfp8E4m3>().unwrap();
        let decoded = scaled.view().try_cast_dense::<f32>().unwrap();
        assert!(
            decoded.as_slice().iter().all(|&x| x == 0.0),
            "all-zero block did not decode to zero"
        );

        // A NaN in row 0 poisons only that block; row 1 stays clean.
        let mut data = vec![1.5f32; 2 * block];
        data[3] = f32::NAN;
        let dense = Tensor::<f32>::try_from_slice(&data, &[2, block]).unwrap();
        let scaled = dense.view().try_cast_to_scaled::<Mxfp8E4m3>().unwrap();
        let decoded = scaled.view().try_cast_dense::<f32>().unwrap();
        let clean_row = &decoded.as_slice()[block..];
        assert!(
            clean_row.iter().all(|&x| (x - 1.5).abs() <= 0.2),
            "clean block corrupted by a NaN in another block"
        );
    }

    #[test]
    fn tensor_scale_present_for_nvfp4_absent_for_mx() {
        let (data, shape) = sample_matrix();
        let dense = Tensor::<f32>::try_from_slice(&data, &shape).unwrap();

        let nvfp4 = dense.view().try_cast_to_scaled::<Nvfp4>().unwrap();
        let ts = nvfp4.tensor_scale();
        assert!(ts.is_some(), "NVFP4 must carry a per-tensor scale");
        assert!(ts.unwrap() > 0.0, "derived tensor_scale must be positive");

        let mxfp8 = dense.view().try_cast_to_scaled::<Mxfp8E4m3>().unwrap();
        assert!(
            mxfp8.tensor_scale().is_none(),
            "MX formats must not carry a per-tensor scale"
        );
    }

    #[test]
    fn slice_row_then_materialize() {
        let (data, shape) = sample_matrix();
        let dense = Tensor::<f32>::try_from_slice(&data, &shape).unwrap();
        let scaled = dense.view().try_cast_to_scaled::<Nvfp4>().unwrap();

        let view = scaled.view();
        let row1 = view.row(1).unwrap();
        assert_eq!(row1.shape(), &[1, 32]);
        assert_eq!(row1.block_scales().shape(), &[1, 2]);

        let dense_row = row1.try_cast_dense::<f32>().unwrap();
        assert_eq!(dense_row.shape(), &[1, 32]);

        // Row 1 of the source should round-trip within FP4 tolerance.
        let max_abs = data[32..].iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for (c, &actual) in dense_row.as_slice().iter().enumerate() {
            let expected = data[32 + c];
            assert!(
                (expected - actual).abs() <= 0.30 * max_abs + 1e-3,
                "sliced row[{c}]: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn transcode_mxfp8_to_nvfp4() {
        let (data, shape) = sample_matrix();
        let dense = Tensor::<f32>::try_from_slice(&data, &shape).unwrap();

        let mxfp8 = dense.view().try_cast_to_scaled::<Mxfp8E4m3>().unwrap();
        assert!(mxfp8.tensor_scale().is_none());

        let nvfp4 = mxfp8.view().try_cast_to_scaled::<Nvfp4>().unwrap();
        assert_eq!(nvfp4.shape(), &[2, 32]);
        assert!(nvfp4.tensor_scale().is_some());

        // Transcode then decode: still bounded by the coarsest format (NVFP4).
        let decoded = nvfp4.view().try_cast_dense::<f32>().unwrap();
        let max_abs = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for (i, (&expected, &actual)) in data.iter().zip(decoded.as_slice()).enumerate() {
            assert!(
                (expected - actual).abs() <= 0.35 * max_abs + 1e-3,
                "transcode roundtrip[{i}]: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn transcode_equals_decode_then_encode() {
        // Transcoding a block-scaled tensor must equal decoding it to dense and re-encoding (the
        // kernel does exactly that). Destination is MX (no per-tensor scale) so it's byte-exact.
        let (data, shape) = sample_matrix();
        let dense = Tensor::<f32>::try_from_slice(&data, &shape).unwrap();
        let nvfp4 = dense.view().try_cast_to_scaled::<Nvfp4>().unwrap();

        let direct = nvfp4.view().try_cast_to_scaled::<Mxfp8E4m3>().unwrap();
        let decoded = nvfp4.view().try_cast_dense::<f32>().unwrap();
        let two_step = decoded.view().try_cast_to_scaled::<Mxfp8E4m3>().unwrap();

        let direct_elements = direct.elements();
        let two_step_elements = two_step.elements();
        assert_eq!(
            direct_elements.as_packed_slice().unwrap(),
            two_step_elements.as_packed_slice().unwrap(),
            "transcode elements differ from decode-then-encode"
        );
        let direct_scales = direct.block_scales();
        let two_step_scales = two_step.block_scales();
        assert_eq!(
            direct_scales.as_packed_slice().unwrap(),
            two_step_scales.as_packed_slice().unwrap(),
            "transcode scales differ from decode-then-encode"
        );
    }

    #[test]
    fn rank1_vector_roundtrips() {
        // A bare 1-D vector (no leading axis): the verbs block the last axis only.
        let cols = 32usize;
        let data: Vec<f32> = (0..cols).map(|c| (c as f32 - 16.0) * 0.25).collect();
        let dense = Tensor::<f32>::try_from_slice(&data, &[cols]).unwrap();

        let scaled = dense.view().try_cast_to_scaled::<Nvfp4>().unwrap();
        assert_eq!(scaled.shape(), &[cols]);
        assert_eq!(scaled.block_scales().shape(), &[cols / 16]);

        let decoded = scaled.view().try_cast_dense::<f32>().unwrap();
        assert_eq!(decoded.shape(), &[cols]);
        let max_abs = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for (i, (&expected, &actual)) in data.iter().zip(decoded.as_slice()).enumerate() {
            assert!(
                (expected - actual).abs() <= 0.30 * max_abs + 1e-3,
                "rank-1 roundtrip[{i}]: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn rank3_batch_roundtrips() {
        // (batch, rows, cols): the rank-general path blocks only the last axis. The old rank-2-only
        // verbs indexed `shape[1]` and errored/panicked on this shape.
        let (batch, rows, cols) = (2usize, 2usize, 32usize);
        let data: Vec<f32> = (0..batch * rows * cols).map(|i| (i % 31) as f32 - 15.0).collect();
        let dense = Tensor::<f32>::try_from_slice(&data, &[batch, rows, cols]).unwrap();

        let scaled = dense.view().try_cast_to_scaled::<Mxfp8E4m3>().unwrap();
        assert_eq!(scaled.shape(), &[batch, rows, cols]);
        assert_eq!(scaled.block_scales().shape(), &[batch, rows, cols / 32]);

        let decoded = scaled.view().try_cast_dense::<f32>().unwrap();
        assert_eq!(decoded.shape(), &[batch, rows, cols]);
        let max_abs = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for (i, (&expected, &actual)) in data.iter().zip(decoded.as_slice()).enumerate() {
            assert!(
                (expected - actual).abs() <= 0.20 * max_abs + 1e-3,
                "rank-3 roundtrip[{i}]: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn index_block_scales_element() {
        let (data, shape) = sample_matrix();
        let dense = Tensor::<f32>::try_from_slice(&data, &shape).unwrap();
        let scaled = dense.view().try_cast_to_scaled::<Nvfp4>().unwrap();

        let scales = scaled.block_scales();
        // (2, 2) UE4M3 scale bytes; each must decode to a positive multiplier for a nonzero block.
        let first: Ue4m3 = scales[(0usize, 0usize)];
        assert!(first.to_f32() >= 0.0);
        // The block straddling the largest magnitudes should have a non-zero scale.
        let last: Ue4m3 = scales[(1usize, 1usize)];
        assert!(last.to_f32() > 0.0, "non-empty block must have a positive scale");
    }

    #[test]
    fn descriptor_size_arithmetic_matches_c_contract() {
        // Mirror nk_block_scaled_elements_size / nk_block_scaled_scales_size for 64 elements.
        let nvfp4 = Nvfp4::descriptor();
        assert_eq!(nvfp4.elements_size(64), 32); // 64 * 4 bits / 8
        assert_eq!(nvfp4.scales_size(64), 4); // 64 / 16

        let mxfp8 = Mxfp8E4m3::descriptor();
        assert_eq!(mxfp8.elements_size(64), 64); // 64 * 8 bits / 8
        assert_eq!(mxfp8.scales_size(64), 2); // 64 / 32

        let plain = BlockScaledDescriptor::plain(dtype::F32);
        assert_eq!(plain.scales_size(64), 0);
        assert_eq!(plain.elements_size(64), 256); // 64 * 32 bits / 8
    }

    /// Reinterpret a slice of single-byte storage elements as raw bytes (block-scaled elements and
    /// scales are all one byte wide), so the comparison works regardless of the element newtype.
    fn raw_bytes<T>(slice: &[T]) -> &[u8] {
        assert_eq!(
            core::mem::size_of::<T>(),
            1,
            "block-scaled storage element must be one byte"
        );
        unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len()) }
    }

    fn check_byte_match_c<F: BlockScaledFormat>() {
        // Byte-match the C kernel: encode the same f32 buffer through the FFI directly with this
        // format's descriptor and compare element + scale bytes against the Rust verb's output.
        let ty = core::any::type_name::<F>();
        let (data, shape) = sample_matrix();
        let count = shape[0] * shape[1];
        let dense = Tensor::<f32>::try_from_slice(&data, &shape).unwrap();
        let scaled = dense.view().try_cast_to_scaled::<F>().unwrap();
        let has_tensor_scale = scaled.tensor_scale().is_some();

        let to_format = F::descriptor();
        let from_format = BlockScaledDescriptor::plain(dtype::F32);
        let mut ref_elements = vec![0u8; to_format.elements_size(count)];
        let mut ref_scales = vec![0u8; to_format.scales_size(count)];
        let mut ref_scale_buf = ScalarBuffer::from_f32(0.0);
        unsafe {
            nk_cast_block_scaled(
                data.as_ptr() as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                &from_format,
                ref_elements.as_mut_ptr() as *mut c_void,
                ref_scales.as_mut_ptr() as *mut c_void,
                if has_tensor_scale {
                    &mut ref_scale_buf
                } else {
                    core::ptr::null_mut()
                },
                &to_format,
                count,
            );
        }

        let elements_view = scaled.elements();
        let scales_view = scaled.block_scales();
        assert_eq!(
            raw_bytes(elements_view.as_packed_slice().unwrap()),
            &ref_elements[..],
            "{ty} element bytes"
        );
        assert_eq!(
            raw_bytes(scales_view.as_packed_slice().unwrap()),
            &ref_scales[..],
            "{ty} scale bytes"
        );
        if has_tensor_scale {
            assert_eq!(
                scaled.tensor_scale().unwrap(),
                ref_scale_buf.to_f32(),
                "{ty} tensor scale"
            );
        }
    }

    #[test]
    fn byte_match_c_all_formats() {
        check_byte_match_c::<Nvfp4>();
        check_byte_match_c::<Mxfp4>();
        check_byte_match_c::<Mxfp6E2m3>();
        check_byte_match_c::<Mxfp6E3m2>();
        check_byte_match_c::<Mxfp8E4m3>();
        check_byte_match_c::<Mxfp8E5m2>();
        check_byte_match_c::<Mxint8>();
    }

    fn check_encode_idempotent<F: BlockScaledFormat>() {
        // Re-encoding decoded values reproduces them exactly: the quantizer is a projection, so a
        // second encode/decode pass over already-representable values must not drift.
        let (data, shape) = sample_matrix();
        let dense = Tensor::<f32>::try_from_slice(&data, &shape).unwrap();
        let decoded_once = dense
            .view()
            .try_cast_to_scaled::<F>()
            .unwrap()
            .view()
            .try_cast_dense::<f32>()
            .unwrap();
        let decoded_twice = decoded_once
            .view()
            .try_cast_to_scaled::<F>()
            .unwrap()
            .view()
            .try_cast_dense::<f32>()
            .unwrap();
        assert_eq!(
            decoded_twice.as_slice(),
            decoded_once.as_slice(),
            "{} encode is not idempotent",
            core::any::type_name::<F>()
        );
    }

    #[test]
    fn encode_is_idempotent_all_formats() {
        check_encode_idempotent::<Nvfp4>();
        check_encode_idempotent::<Mxfp4>();
        check_encode_idempotent::<Mxfp6E2m3>();
        check_encode_idempotent::<Mxfp6E3m2>();
        check_encode_idempotent::<Mxfp8E4m3>();
        check_encode_idempotent::<Mxfp8E5m2>();
        check_encode_idempotent::<Mxint8>();
    }

    // endregion: Block-scaled tests
}
