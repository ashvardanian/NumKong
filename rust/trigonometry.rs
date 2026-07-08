//! Trigonometry — element-wise sin/cos/atan plus NeoX split-half rotary position embedding (RoPE).
//!
//! Rotates channel pairs `(i, i + half_dim)` of every head by per-token angle grids. The caller bakes
//! position lookup and multi-axis (M-RoPE) assignment into the `[rows, half_dim]` cos/sin grids, so a
//! single call rotates the whole head. The rotation writes every channel it is given, so it is done in
//! place over the `[rows, heads * 2 * half_dim]` slice.

use crate::tensor::{Global, Tensor, TensorError, TensorMut, TensorRef};
use crate::types::{bf16, e4m3, f16, StorageElement};

/// Precision of the RoPE `cos`/`sin` rotation coefficients — always `f32`, deliberately decoupled
/// from the rotated element dtype (BF16/E4M3 inputs rotate through f32 angles, since a lower-precision
/// angle would corrupt the rotation). Mirrors the C `nk_rope_angle_t` typedef and the `rope_angle_t`
/// element-trait alias in `types.hpp`.
pub type RopeAngle = f32;

#[link(name = "numkong")]
extern "C" {
    fn nk_trig_rope_f32(
        x: *const f32,
        y: *mut f32,
        cos: *const RopeAngle,
        sin: *const RopeAngle,
        rows: usize,
        heads: usize,
        half_dim: usize,
        x_row_stride: usize,
        y_row_stride: usize,
        input_scale: f32,
    );
    fn nk_trig_rope_bf16(
        x: *const u16,
        y: *mut u16,
        cos: *const RopeAngle,
        sin: *const RopeAngle,
        rows: usize,
        heads: usize,
        half_dim: usize,
        x_row_stride: usize,
        y_row_stride: usize,
        input_scale: f32,
    );
    fn nk_trig_rope_e4m3(
        x: *const u8,
        y: *mut u8,
        cos: *const RopeAngle,
        sin: *const RopeAngle,
        rows: usize,
        heads: usize,
        half_dim: usize,
        x_row_stride: usize,
        y_row_stride: usize,
        input_scale: f32,
    );
    fn nk_trig_sin_f32(inputs: *const f32, n: usize, outputs: *mut f32);
    fn nk_trig_sin_f64(inputs: *const f64, n: usize, outputs: *mut f64);
    fn nk_trig_sin_f16(inputs: *const u16, n: usize, outputs: *mut u16);
    fn nk_trig_cos_f32(inputs: *const f32, n: usize, outputs: *mut f32);
    fn nk_trig_cos_f64(inputs: *const f64, n: usize, outputs: *mut f64);
    fn nk_trig_cos_f16(inputs: *const u16, n: usize, outputs: *mut u16);
    fn nk_trig_atan_f32(inputs: *const f32, n: usize, outputs: *mut f32);
    fn nk_trig_atan_f64(inputs: *const f64, n: usize, outputs: *mut f64);
    fn nk_trig_atan_f16(inputs: *const u16, n: usize, outputs: *mut u16);
}

/// In-place NeoX split-half RoPE over a row-major `[rows, heads * 2 * half_dim]` tensor.
pub trait TrigRope: Sized + StorageElement {
    /// Rotates a 2D `[rows, heads * 2 * half_dim]` tensor in place using the `[rows, half_dim]`
    /// `cos`/`sin` angle grids (row `r` at `r * half_dim`), shared across heads.
    ///
    /// The row stride is read from the tensor, so `x` may be a non-contiguous sub-span (e.g. the Q
    /// or K column-section of a fused QKV buffer). Returns `Err` on a shape mismatch.
    fn rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[RopeAngle],
        sin: &[RopeAngle],
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Result<(), TensorError>
    where
        XMut: TensorMut<Self, RX> + ?Sized;
}

impl TrigRope for f32 {
    fn rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[RopeAngle],
        sin: &[RopeAngle],
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Result<(), TensorError>
    where
        XMut: TensorMut<Self, RX> + ?Sized,
    {
        if x.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: x.ndim(),
            });
        }
        let (rows, width) = (x.shape()[0], x.shape()[1]);
        if width < heads * 2 * half_dim {
            return Err(TensorError::ShapeMismatch {
                axis: 1,
                expected: heads * 2 * half_dim,
                got: width,
            });
        }
        if cos.len() < rows * half_dim || sin.len() < rows * half_dim {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: rows * half_dim,
                got: cos.len().min(sin.len()),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let stride = x.stride_bytes(0) as usize;
        let yp = x.as_mut_ptr();
        unsafe {
            nk_trig_rope_f32(
                yp,
                yp,
                cos.as_ptr(),
                sin.as_ptr(),
                rows,
                heads,
                half_dim,
                stride,
                stride,
                input_scale,
            );
        }
        Ok(())
    }
}

impl TrigRope for bf16 {
    fn rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[RopeAngle],
        sin: &[RopeAngle],
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Result<(), TensorError>
    where
        XMut: TensorMut<Self, RX> + ?Sized,
    {
        if x.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: x.ndim(),
            });
        }
        let (rows, width) = (x.shape()[0], x.shape()[1]);
        if width < heads * 2 * half_dim {
            return Err(TensorError::ShapeMismatch {
                axis: 1,
                expected: heads * 2 * half_dim,
                got: width,
            });
        }
        if cos.len() < rows * half_dim || sin.len() < rows * half_dim {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: rows * half_dim,
                got: cos.len().min(sin.len()),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let stride = x.stride_bytes(0) as usize;
        let yp = x.as_mut_ptr() as *mut u16;
        unsafe {
            nk_trig_rope_bf16(
                yp,
                yp,
                cos.as_ptr(),
                sin.as_ptr(),
                rows,
                heads,
                half_dim,
                stride,
                stride,
                input_scale,
            );
        }
        Ok(())
    }
}

impl TrigRope for e4m3 {
    fn rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[RopeAngle],
        sin: &[RopeAngle],
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Result<(), TensorError>
    where
        XMut: TensorMut<Self, RX> + ?Sized,
    {
        if x.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: x.ndim(),
            });
        }
        let (rows, width) = (x.shape()[0], x.shape()[1]);
        if width < heads * 2 * half_dim {
            return Err(TensorError::ShapeMismatch {
                axis: 1,
                expected: heads * 2 * half_dim,
                got: width,
            });
        }
        if cos.len() < rows * half_dim || sin.len() < rows * half_dim {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: rows * half_dim,
                got: cos.len().min(sin.len()),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let stride = x.stride_bytes(0) as usize;
        let yp = x.as_mut_ptr() as *mut u8;
        unsafe {
            nk_trig_rope_e4m3(
                yp,
                yp,
                cos.as_ptr(),
                sin.as_ptr(),
                rows,
                heads,
                half_dim,
                stride,
                stride,
                input_scale,
            );
        }
        Ok(())
    }
}

// region: TrigSin

/// Computes **element-wise sine** of a vector.
pub trait TrigSin: Sized + StorageElement {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;

    /// In-place sine: `data[i] = sin(data[i])`.
    ///
    /// Both source and destination pointers are derived from the single `&mut`,
    /// so no aliased `&[Self]` + `&mut [Self]` over the same storage is formed.
    fn sin_inplace(data: &mut [Self]) -> Option<()>;
}

impl TrigSin for f64 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_trig_sin_f64(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }

    fn sin_inplace(data: &mut [Self]) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_sin_f64(p as *const f64, len, p) };
        Some(())
    }
}

impl TrigSin for f32 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_trig_sin_f32(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }

    fn sin_inplace(data: &mut [Self]) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_sin_f32(p as *const f32, len, p) };
        Some(())
    }
}

impl TrigSin for f16 {
    fn sin(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_trig_sin_f16(
                inputs.as_ptr() as *const u16,
                inputs.len(),
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn sin_inplace(data: &mut [Self]) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_sin_f16(p as *const u16, len, p as *mut u16) };
        Some(())
    }
}

// endregion: TrigSin

// region: TrigCos

/// Computes **element-wise cosine** of a vector.
pub trait TrigCos: Sized + StorageElement {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;

    /// In-place cosine: `data[i] = cos(data[i])`.
    ///
    /// Both source and destination pointers are derived from the single `&mut`,
    /// so no aliased `&[Self]` + `&mut [Self]` over the same storage is formed.
    fn cos_inplace(data: &mut [Self]) -> Option<()>;
}

impl TrigCos for f64 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_trig_cos_f64(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }

    fn cos_inplace(data: &mut [Self]) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_cos_f64(p as *const f64, len, p) };
        Some(())
    }
}

impl TrigCos for f32 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_trig_cos_f32(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }

    fn cos_inplace(data: &mut [Self]) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_cos_f32(p as *const f32, len, p) };
        Some(())
    }
}

impl TrigCos for f16 {
    fn cos(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_trig_cos_f16(
                inputs.as_ptr() as *const u16,
                inputs.len(),
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn cos_inplace(data: &mut [Self]) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_cos_f16(p as *const u16, len, p as *mut u16) };
        Some(())
    }
}

// endregion: TrigCos

// region: TrigAtan

/// Computes **element-wise arctangent** of a vector.
pub trait TrigAtan: Sized + StorageElement {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()>;

    /// In-place arctangent: `data[i] = atan(data[i])`.
    ///
    /// Both source and destination pointers are derived from the single `&mut`,
    /// so no aliased `&[Self]` + `&mut [Self]` over the same storage is formed.
    fn atan_inplace(data: &mut [Self]) -> Option<()>;
}

impl TrigAtan for f64 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_trig_atan_f64(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }

    fn atan_inplace(data: &mut [Self]) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_atan_f64(p as *const f64, len, p) };
        Some(())
    }
}

impl TrigAtan for f32 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe { nk_trig_atan_f32(inputs.as_ptr(), inputs.len(), outputs.as_mut_ptr()) };
        Some(())
    }

    fn atan_inplace(data: &mut [Self]) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_atan_f32(p as *const f32, len, p) };
        Some(())
    }
}

impl TrigAtan for f16 {
    fn atan(inputs: &[Self], outputs: &mut [Self]) -> Option<()> {
        if inputs.len() != outputs.len() {
            return None;
        }
        unsafe {
            nk_trig_atan_f16(
                inputs.as_ptr() as *const u16,
                inputs.len(),
                outputs.as_mut_ptr() as *mut u16,
            )
        };
        Some(())
    }

    fn atan_inplace(data: &mut [Self]) -> Option<()> {
        let len = data.len();
        let p = data.as_mut_ptr();
        unsafe { nk_trig_atan_f16(p as *const u16, len, p as *mut u16) };
        Some(())
    }
}

// endregion: TrigAtan

/// `Trigonometry` bundles trigonometric functions: TrigSin, TrigCos, and TrigAtan.
pub trait Trigonometry: TrigSin + TrigCos + TrigAtan {}
impl<Scalar: TrigSin + TrigCos + TrigAtan> Trigonometry for Scalar {}

// region: Tensor-shaped trigonometry

/// Extension trait: element-wise sine for any [`TensorRef`] implementor.
pub trait TrigSinOps<Scalar: Clone + TrigSin, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn try_sin(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> { self.view().try_sin() }

    fn try_sin_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError> {
        self.view().try_sin_into(out)
    }
}

impl<Scalar: Clone + TrigSin, const R: usize, C: TensorRef<Scalar, R> + ?Sized> TrigSinOps<Scalar, R> for C {}

/// Extension trait: element-wise cosine for any [`TensorRef`] implementor.
pub trait TrigCosOps<Scalar: Clone + TrigCos, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn try_cos(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> { self.view().try_cos() }

    fn try_cos_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError> {
        self.view().try_cos_into(out)
    }
}

impl<Scalar: Clone + TrigCos, const R: usize, C: TensorRef<Scalar, R> + ?Sized> TrigCosOps<Scalar, R> for C {}

/// Extension trait: element-wise arctangent for any [`TensorRef`] implementor.
pub trait TrigAtanOps<Scalar: Clone + TrigAtan, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    fn try_atan(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> { self.view().try_atan() }

    fn try_atan_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        out: &mut OutputTensor,
    ) -> Result<(), TensorError> {
        self.view().try_atan_into(out)
    }
}

impl<Scalar: Clone + TrigAtan, const R: usize, C: TensorRef<Scalar, R> + ?Sized> TrigAtanOps<Scalar, R> for C {}

impl<Scalar: Clone + TrigSin, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    /// Element-wise sine in-place (infallible — self vs self always matches).
    pub fn sin_inplace(&mut self) { self.span().sin_inplace(); }
}

impl<Scalar: Clone + TrigCos, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    /// Element-wise cosine in-place (infallible — self vs self always matches).
    pub fn cos_inplace(&mut self) { self.span().cos_inplace(); }
}

impl<Scalar: Clone + TrigAtan, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    /// Element-wise arctangent in-place (infallible — self vs self always matches).
    pub fn atan_inplace(&mut self) { self.span().atan_inplace(); }
}

// endregion: Tensor-shaped trigonometry

#[cfg(test)]
mod tests {
    use super::{TrigAtan, TrigCos, TrigRope, TrigSin};
    use crate::types::{assert_close, bf16, e4m3, f16, FloatLike, TestableType};

    fn check_rope<Scalar>(values: &[f32], heads: usize, half_dim: usize)
    where
        Scalar: FloatLike + TestableType + TrigRope,
    {
        let rows = 2;
        let width = heads * 2 * half_dim;
        assert_eq!(values.len(), rows * width);
        let x: Vec<Scalar> = values.iter().map(|&v| Scalar::from_f32(v)).collect();
        // Per-token angle grids [rows, half_dim].
        let cos: Vec<f32> = (0..rows * half_dim).map(|k| (0.1 * k as f32).cos()).collect();
        let sin: Vec<f32> = (0..rows * half_dim).map(|k| (0.1 * k as f32).sin()).collect();
        let reference = x.clone();
        let mut x_t = crate::tensor::Tensor::<Scalar>::try_from_slice(&x, &[rows, width]).unwrap();
        Scalar::rope_into(&mut x_t, &cos, &sin, heads, half_dim, 1.0).unwrap();
        let x = x_t.as_slice().to_vec();
        for r in 0..rows {
            for h in 0..heads {
                let base = r * width + h * 2 * half_dim;
                for i in 0..half_dim {
                    let low = reference[base + i].to_f64();
                    let high = reference[base + i + half_dim].to_f64();
                    let cosine = cos[r * half_dim + i] as f64;
                    let sine = sin[r * half_dim + i] as f64;
                    let expected_low = Scalar::from_f32((low * cosine - high * sine) as f32).to_f64();
                    let expected_high = Scalar::from_f32((low * sine + high * cosine) as f32).to_f64();
                    assert_close(
                        x[base + i].to_f64(),
                        expected_low,
                        Scalar::atol() * 4.0,
                        Scalar::rtol() * 4.0,
                        "rope low",
                    );
                    assert_close(
                        x[base + i + half_dim].to_f64(),
                        expected_high,
                        Scalar::atol() * 4.0,
                        Scalar::rtol() * 4.0,
                        "rope high",
                    );
                }
            }
        }
    }

    #[test]
    fn rope_split_half() {
        let values: Vec<f32> = (0..2 * 2 * 2 * 8).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        check_rope::<f32>(&values, 2, 8);
        check_rope::<bf16>(&values, 2, 8);
        check_rope::<e4m3>(&values, 2, 8);
    }

    #[test]
    fn rope_strided_section() {
        use crate::tensor::{SliceRange, Tensor};
        // Rotate the left `[rows, width]` column-section of a `[rows, 2*width]` buffer in place
        // (row stride 2*width, not width) — the Q/K-section-of-a-fused-QKV case.
        let (rows, heads, half_dim) = (3, 2, 4);
        let width = heads * 2 * half_dim; // 16
        let full = 2 * width;
        let section: Vec<f32> = (0..rows * width).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let mut wide_vec = vec![999.0f32; rows * full]; // right half is a sentinel
        for r in 0..rows {
            for c in 0..width {
                wide_vec[r * full + c] = section[r * width + c];
            }
        }
        let cos: Vec<f32> = (0..rows * half_dim).map(|k| (0.1 * k as f32).cos()).collect();
        let sin: Vec<f32> = (0..rows * half_dim).map(|k| (0.1 * k as f32).sin()).collect();

        let mut wide = Tensor::<f32>::try_from_slice(&wide_vec, &[rows, full]).unwrap();
        {
            let mut span = wide.span();
            let mut sec = span
                .slice_mut(&[SliceRange::Full, SliceRange::range(0, width)][..])
                .unwrap();
            f32::rope_into(&mut sec, &cos, &sin, heads, half_dim, 1.0).unwrap();
        }

        let mut contig = Tensor::<f32>::try_from_slice(&section, &[rows, width]).unwrap();
        f32::rope_into(&mut contig, &cos, &sin, heads, half_dim, 1.0).unwrap();

        let wide_after = wide.as_slice();
        let contig_after = contig.as_slice();
        for r in 0..rows {
            for c in 0..width {
                assert!(
                    (wide_after[r * full + c] - contig_after[r * width + c]).abs() < 1e-5,
                    "strided RoPE section mismatch at [{r},{c}]"
                );
            }
            for c in width..full {
                assert_eq!(
                    wide_after[r * full + c],
                    999.0,
                    "RoPE wrote outside its strided section"
                );
            }
        }
    }

    pub(crate) fn check_trig_unary<Scalar, F>(
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
