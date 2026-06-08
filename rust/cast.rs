//! Type casting between scalar formats — slice traits and tensor-shaped wrappers.
//!
//! This module provides:
//!
//! - [`CastDtype`]: Trait marking types eligible for bulk casting
//! - [`cast`]: Bulk-converts a slice from one scalar format to another
//! - [`CastOps`]: Tensor-shaped extension trait — auto-implemented on every
//!   [`crate::tensor::TensorRef`] so any container can do `tensor.try_cast_dtype::<Destination>()`

use crate::types::{bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, StorageElement};

#[link(name = "numkong")]
extern "C" {
    fn nk_cast(
        from: *const core::ffi::c_void,
        from_type: u32,
        n: usize,
        to: *mut core::ffi::c_void,
        to_type: u32,
    );
}

/// Internal dtype codes matching `nk_dtype_t` from C.
/// Not exposed to users.
mod dtype {
    pub(crate) const F64: u32 = 1 << 10;
    pub(crate) const F32: u32 = 1 << 11;
    pub(crate) const F16: u32 = 1 << 12;
    pub(crate) const BF16: u32 = 1 << 13;
    pub(crate) const E4M3: u32 = 1 << 14;
    pub(crate) const E5M2: u32 = 1 << 15;
    pub(crate) const E2M3: u32 = 1 << 18;
    pub(crate) const E3M2: u32 = 1 << 19;
    pub(crate) const F64C: u32 = 1 << 20;
    pub(crate) const F32C: u32 = 1 << 21;
    pub(crate) const F16C: u32 = 1 << 22;
    pub(crate) const BF16C: u32 = 1 << 23;
    pub(crate) const I8: u32 = 1 << 2;
    pub(crate) const I16: u32 = 1 << 3;
    pub(crate) const I32: u32 = 1 << 4;
    pub(crate) const I64: u32 = 1 << 5;
    pub(crate) const U8: u32 = 1 << 6;
    pub(crate) const U16: u32 = 1 << 7;
    pub(crate) const U32: u32 = 1 << 8;
    pub(crate) const U64: u32 = 1 << 9;
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
    fn dtype_code() -> u32 {
        dtype::F64
    }
}
impl CastDtype for f32 {
    fn dtype_code() -> u32 {
        dtype::F32
    }
}
impl CastDtype for f16 {
    fn dtype_code() -> u32 {
        dtype::F16
    }
}
impl CastDtype for bf16 {
    fn dtype_code() -> u32 {
        dtype::BF16
    }
}
impl CastDtype for e4m3 {
    fn dtype_code() -> u32 {
        dtype::E4M3
    }
}
impl CastDtype for e5m2 {
    fn dtype_code() -> u32 {
        dtype::E5M2
    }
}
impl CastDtype for e2m3 {
    fn dtype_code() -> u32 {
        dtype::E2M3
    }
}
impl CastDtype for e3m2 {
    fn dtype_code() -> u32 {
        dtype::E3M2
    }
}
impl CastDtype for f64c {
    fn dtype_code() -> u32 {
        dtype::F64C
    }
}
impl CastDtype for f32c {
    fn dtype_code() -> u32 {
        dtype::F32C
    }
}
impl CastDtype for f16c {
    fn dtype_code() -> u32 {
        dtype::F16C
    }
}
impl CastDtype for bf16c {
    fn dtype_code() -> u32 {
        dtype::BF16C
    }
}
impl CastDtype for i8 {
    fn dtype_code() -> u32 {
        dtype::I8
    }
}
impl CastDtype for i16 {
    fn dtype_code() -> u32 {
        dtype::I16
    }
}
impl CastDtype for i32 {
    fn dtype_code() -> u32 {
        dtype::I32
    }
}
impl CastDtype for i64 {
    fn dtype_code() -> u32 {
        dtype::I64
    }
}
impl CastDtype for u8 {
    fn dtype_code() -> u32 {
        dtype::U8
    }
}
impl CastDtype for u16 {
    fn dtype_code() -> u32 {
        dtype::U16
    }
}
impl CastDtype for u32 {
    fn dtype_code() -> u32 {
        dtype::U32
    }
}
impl CastDtype for u64 {
    fn dtype_code() -> u32 {
        dtype::U64
    }
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
            source.as_ptr() as *const core::ffi::c_void,
            S::dtype_code(),
            source.len(),
            dest.as_mut_ptr() as *mut core::ffi::c_void,
            D::dtype_code(),
        );
    }
    Some(())
}

// region: Tensor-shaped cast (moved from crate::tensor)

use crate::tensor::{try_reborrow_tensor_into, Global, Tensor, TensorError, TensorRef};

/// Extension trait: type casting for any [`TensorRef`] implementor.
pub trait CastOps<Source: Clone + CastDtype, const MAX_RANK: usize>:
    TensorRef<Source, MAX_RANK>
{
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        assert_close, bf16, bf16c, e2m3, e3m2, e4m3, e5m2, f16, f16c, f32c, f64c, FloatLike,
        StorageElement, TestableType,
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
        let src = [
            f64c::from_real_imag(1.25, 9.0),
            f64c::from_real_imag(-2.5, -7.0),
        ];
        let mut dst = [0.0f64; 2];
        cast(&src, &mut dst).unwrap();
        assert_eq!(dst, [1.25, -2.5]);
    }

    #[test]
    fn cast_complex_to_complex() {
        let src = [
            f32c::from_real_imag(1.25, -2.5),
            f32c::from_real_imag(-3.5, 4.25),
        ];
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
}
