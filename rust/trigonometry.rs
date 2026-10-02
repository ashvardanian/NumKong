//! Trigonometry — element-wise sine, cosine and arctangent.
//!
//! File: rust/trigonometry.rs
//! Author: Ash Vardanian

use core::{ffi::c_void, ptr::null_mut};

use crate::{
    capabilities::{enabled_cpu_capabilities_mask, nk_capability_t, nk_size_t, nk_status_t, StatusCode},
    tensor::{check_len, Allocator, Error, Tensor, TensorMut, TensorRef},
    types::{f16, StorageElement},
};

#[link(name = "numkong")]
extern "C" {
    fn nk_trig_sin_f32_best(
        inputs: *const f32,
        n: nk_size_t,
        outputs: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_trig_sin_f64_best(
        inputs: *const f64,
        n: nk_size_t,
        outputs: *mut f64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_trig_sin_f16_best(
        inputs: *const u16,
        n: nk_size_t,
        outputs: *mut u16,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_trig_cos_f32_best(
        inputs: *const f32,
        n: nk_size_t,
        outputs: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_trig_cos_f64_best(
        inputs: *const f64,
        n: nk_size_t,
        outputs: *mut f64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_trig_cos_f16_best(
        inputs: *const u16,
        n: nk_size_t,
        outputs: *mut u16,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_trig_atan_f32_best(
        inputs: *const f32,
        n: nk_size_t,
        outputs: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_trig_atan_f64_best(
        inputs: *const f64,
        n: nk_size_t,
        outputs: *mut f64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_trig_atan_f16_best(
        inputs: *const u16,
        n: nk_size_t,
        outputs: *mut u16,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
}

// region: TrigSin

/// Computes __element-wise sine__ of a vector.
///
/// Fails with [`Error::ShapeMismatch`] if lengths differ, or [`Error::KernelFailed`]
/// carrying the kernel's status.
pub trait TrigSin: Sized + StorageElement {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error>;

    /// In-place sine: `data[i] = sin(data[i])`.
    ///
    /// Both source and destination pointers are derived from the single `&mut`, so no aliased
    /// `&[Self]` + `&mut [Self]` over the same storage is formed.
    fn sin_inplace(data: &mut [Self]) -> Result<(), Error>;
}

impl TrigSin for f64 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error> {
        check_len(inputs.len(), outputs.len())?;
        unsafe {
            nk_trig_sin_f64_best(
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_mut_ptr(),
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }

    fn sin_inplace(data: &mut [Self]) -> Result<(), Error> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_sin_f64_best(p as *const f64, len, p, enabled_cpu_capabilities_mask(), null_mut()) }.check()
    }
}

impl TrigSin for f32 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error> {
        check_len(inputs.len(), outputs.len())?;
        unsafe {
            nk_trig_sin_f32_best(
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_mut_ptr(),
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }

    fn sin_inplace(data: &mut [Self]) -> Result<(), Error> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_sin_f32_best(p as *const f32, len, p, enabled_cpu_capabilities_mask(), null_mut()) }.check()
    }
}

impl TrigSin for f16 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error> {
        check_len(inputs.len(), outputs.len())?;
        unsafe {
            nk_trig_sin_f16_best(
                inputs.as_ptr() as *const u16,
                inputs.len(),
                outputs.as_mut_ptr() as *mut u16,
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }

    fn sin_inplace(data: &mut [Self]) -> Result<(), Error> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_trig_sin_f16_best(
                p as *const u16,
                len,
                p as *mut u16,
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }
}

// endregion: TrigSin

// region: TrigCos

/// Computes __element-wise cosine__ of a vector.
///
/// Fails with [`Error::ShapeMismatch`] if lengths differ, or [`Error::KernelFailed`]
/// carrying the kernel's status.
pub trait TrigCos: Sized + StorageElement {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error>;

    /// In-place cosine: `data[i] = cos(data[i])`.
    ///
    /// Both source and destination pointers are derived from the single `&mut`, so no aliased
    /// `&[Self]` + `&mut [Self]` over the same storage is formed.
    fn cos_inplace(data: &mut [Self]) -> Result<(), Error>;
}

impl TrigCos for f64 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error> {
        check_len(inputs.len(), outputs.len())?;
        unsafe {
            nk_trig_cos_f64_best(
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_mut_ptr(),
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }

    fn cos_inplace(data: &mut [Self]) -> Result<(), Error> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_cos_f64_best(p as *const f64, len, p, enabled_cpu_capabilities_mask(), null_mut()) }.check()
    }
}

impl TrigCos for f32 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error> {
        check_len(inputs.len(), outputs.len())?;
        unsafe {
            nk_trig_cos_f32_best(
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_mut_ptr(),
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }

    fn cos_inplace(data: &mut [Self]) -> Result<(), Error> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_cos_f32_best(p as *const f32, len, p, enabled_cpu_capabilities_mask(), null_mut()) }.check()
    }
}

impl TrigCos for f16 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error> {
        check_len(inputs.len(), outputs.len())?;
        unsafe {
            nk_trig_cos_f16_best(
                inputs.as_ptr() as *const u16,
                inputs.len(),
                outputs.as_mut_ptr() as *mut u16,
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }

    fn cos_inplace(data: &mut [Self]) -> Result<(), Error> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_trig_cos_f16_best(
                p as *const u16,
                len,
                p as *mut u16,
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }
}

// endregion: TrigCos

// region: TrigAtan

/// Computes __element-wise arctangent__ of a vector.
///
/// Fails with [`Error::ShapeMismatch`] if lengths differ, or [`Error::KernelFailed`]
/// carrying the kernel's status.
pub trait TrigAtan: Sized + StorageElement {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error>;

    /// In-place arctangent: `data[i] = atan(data[i])`.
    ///
    /// Both source and destination pointers are derived from the single `&mut`, so no aliased
    /// `&[Self]` + `&mut [Self]` over the same storage is formed.
    fn atan_inplace(data: &mut [Self]) -> Result<(), Error>;
}

impl TrigAtan for f64 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error> {
        check_len(inputs.len(), outputs.len())?;
        unsafe {
            nk_trig_atan_f64_best(
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_mut_ptr(),
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }

    fn atan_inplace(data: &mut [Self]) -> Result<(), Error> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_atan_f64_best(p as *const f64, len, p, enabled_cpu_capabilities_mask(), null_mut()) }.check()
    }
}

impl TrigAtan for f32 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error> {
        check_len(inputs.len(), outputs.len())?;
        unsafe {
            nk_trig_atan_f32_best(
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_mut_ptr(),
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }

    fn atan_inplace(data: &mut [Self]) -> Result<(), Error> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_atan_f32_best(p as *const f32, len, p, enabled_cpu_capabilities_mask(), null_mut()) }.check()
    }
}

impl TrigAtan for f16 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Result<(), Error> {
        check_len(inputs.len(), outputs.len())?;
        unsafe {
            nk_trig_atan_f16_best(
                inputs.as_ptr() as *const u16,
                inputs.len(),
                outputs.as_mut_ptr() as *mut u16,
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }

    fn atan_inplace(data: &mut [Self]) -> Result<(), Error> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe {
            nk_trig_atan_f16_best(
                p as *const u16,
                len,
                p as *mut u16,
                enabled_cpu_capabilities_mask(),
                null_mut(),
            )
        }
        .check()
    }
}

// endregion: TrigAtan

/// `Trigonometry` bundles trigonometric functions: TrigSin, TrigCos, and TrigAtan.
pub trait Trigonometry: TrigSin + TrigCos + TrigAtan {}
impl<Scalar: TrigSin + TrigCos + TrigAtan> Trigonometry for Scalar {}

// region: Tensor-shaped trigonometry

/// Extension trait: element-wise sine for any [`TensorRef`] implementor.
pub trait TrigSinOps<Scalar: Clone + TrigSin, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn sin(&self) -> Result<Tensor<Scalar, Self::Alloc, MAX_RANK>, Error>
    where
        Self::Alloc: Clone,
    {
        self.view().sin()
    }

    fn sin_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        self.view().sin_into(out)
    }
}

impl<Scalar: Clone + TrigSin, const R: usize, C: TensorRef<Scalar, R> + ?Sized> TrigSinOps<Scalar, R> for C {}

/// Extension trait: element-wise cosine for any [`TensorRef`] implementor.
pub trait TrigCosOps<Scalar: Clone + TrigCos, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn cos(&self) -> Result<Tensor<Scalar, Self::Alloc, MAX_RANK>, Error>
    where
        Self::Alloc: Clone,
    {
        self.view().cos()
    }

    fn cos_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        self.view().cos_into(out)
    }
}

impl<Scalar: Clone + TrigCos, const R: usize, C: TensorRef<Scalar, R> + ?Sized> TrigCosOps<Scalar, R> for C {}

/// Extension trait: element-wise arctangent for any [`TensorRef`] implementor.
pub trait TrigAtanOps<Scalar: Clone + TrigAtan, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn atan(&self) -> Result<Tensor<Scalar, Self::Alloc, MAX_RANK>, Error>
    where
        Self::Alloc: Clone,
    {
        self.view().atan()
    }

    fn atan_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        self.view().atan_into(out)
    }
}

impl<Scalar: Clone + TrigAtan, const R: usize, C: TensorRef<Scalar, R> + ?Sized> TrigAtanOps<Scalar, R> for C {}

impl<Scalar: Clone + TrigSin, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Element-wise sine in-place.
    pub fn sin_inplace(&mut self) -> Result<(), Error> { self.span().sin_inplace() }
}

impl<Scalar: Clone + TrigCos, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Element-wise cosine in-place.
    pub fn cos_inplace(&mut self) -> Result<(), Error> { self.span().cos_inplace() }
}

impl<Scalar: Clone + TrigAtan, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Element-wise arctangent in-place.
    pub fn atan_inplace(&mut self) -> Result<(), Error> { self.span().atan_inplace() }
}

// endregion: Tensor-shaped trigonometry

#[cfg(test)]
mod tests {
    use super::{TrigAtan, TrigCos, TrigSin};
    use crate::{
        tensor::Error,
        types::{assert_close, f16, FloatLike, TestableType},
    };

    pub(crate) fn check_trig_unary<Scalar, F>(
        count: usize,
        gen_fn: fn(usize, usize) -> f64,
        op: F,
        ref_fn: fn(f64) -> f64,
        label: &str,
    ) where
        Scalar: FloatLike + TestableType,
        F: FnOnce(&[Scalar], &mut [Scalar]) -> Result<(), Error>,
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

    fn check_trig_sin<Scalar>(count: usize)
    where
        Scalar: FloatLike + TestableType + TrigSin,
    {
        use core::f64::consts::PI;

        check_trig_unary::<Scalar, _>(
            count,
            |i, n| (i as f64) * 2.0 * PI / (n as f64),
            Scalar::sin,
            f64::sin,
            "sin",
        );
    }

    fn check_trig_cos<Scalar>(count: usize)
    where
        Scalar: FloatLike + TestableType + TrigCos,
    {
        use core::f64::consts::PI;

        check_trig_unary::<Scalar, _>(
            count,
            |i, n| (i as f64) * 2.0 * PI / (n as f64),
            Scalar::cos,
            f64::cos,
            "cos",
        );
    }

    fn check_trig_atan<Scalar>(count: usize)
    where
        Scalar: FloatLike + TestableType + TrigAtan,
    {
        check_trig_unary::<Scalar, _>(
            count,
            |i, n| -5.0 + 10.0 * (i as f64) / (n as f64),
            Scalar::atan,
            f64::atan,
            "atan",
        );
    }

    #[test]
    fn sin() {
        check_trig_sin::<f32>(97);
        check_trig_sin::<f64>(97);
        check_trig_sin::<f16>(97);
    }

    #[test]
    fn cos() {
        check_trig_cos::<f32>(97);
        check_trig_cos::<f64>(97);
        check_trig_cos::<f16>(97);
    }

    #[test]
    fn atan() {
        check_trig_atan::<f32>(100);
        check_trig_atan::<f64>(100);
        check_trig_atan::<f16>(100);
    }
}
