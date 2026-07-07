//! NeoX split-half rotary position embedding (RoPE).
//!
//! Rotates channel pairs `(i, i + half_dim)` of every head by per-token angle grids. The caller bakes
//! position lookup and multi-axis (M-RoPE) assignment into the `[rows, half_dim]` cos/sin grids, so a
//! single call rotates the whole head. The rotation writes every channel it is given, so it is done in
//! place over the `[rows, heads * 2 * half_dim]` slice.

use crate::tensor::TensorMut;
use crate::types::{bf16, e4m3, StorageElement};

/// Precision of the RoPE `cos`/`sin` rotation coefficients — always `f32`, deliberately decoupled
/// from the rotated element dtype (BF16/E4M3 inputs rotate through f32 angles, since a lower-precision
/// angle would corrupt the rotation). Mirrors the C `nk_rope_angle_t` typedef and the `rope_angle_t`
/// element-trait alias in `types.hpp`.
pub type RopeAngle = f32;

#[link(name = "numkong")]
extern "C" {
    fn nk_each_rope_f32(
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
    fn nk_each_rope_bf16(
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
    fn nk_each_rope_e4m3(
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
}

/// In-place NeoX split-half RoPE over a row-major `[rows, heads * 2 * half_dim]` tensor.
pub trait EachRope: Sized + StorageElement {
    /// Rotates a 2D `[rows, heads * 2 * half_dim]` tensor in place using the `[rows, half_dim]`
    /// `cos`/`sin` angle grids (row `r` at `r * half_dim`), shared across heads.
    ///
    /// The row stride is read from the tensor, so `x` may be a non-contiguous sub-span (e.g. the Q
    /// or K column-section of a fused QKV buffer). Returns `None` on a shape mismatch.
    fn rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[RopeAngle],
        sin: &[RopeAngle],
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Option<()>
    where
        XMut: TensorMut<Self, RX> + ?Sized;
}

impl EachRope for f32 {
    fn rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[RopeAngle],
        sin: &[RopeAngle],
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Option<()>
    where
        XMut: TensorMut<Self, RX> + ?Sized,
    {
        if x.ndim() != 2 {
            return None;
        }
        let (rows, width) = (x.shape()[0], x.shape()[1]);
        if rows == 0 || width < heads * 2 * half_dim || cos.len() < rows * half_dim || sin.len() < rows * half_dim {
            return None;
        }
        let stride = x.stride_bytes(0) as usize;
        let yp = x.as_mut_ptr();
        unsafe {
            nk_each_rope_f32(
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
        Some(())
    }
}

impl EachRope for bf16 {
    fn rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[RopeAngle],
        sin: &[RopeAngle],
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Option<()>
    where
        XMut: TensorMut<Self, RX> + ?Sized,
    {
        if x.ndim() != 2 {
            return None;
        }
        let (rows, width) = (x.shape()[0], x.shape()[1]);
        if rows == 0 || width < heads * 2 * half_dim || cos.len() < rows * half_dim || sin.len() < rows * half_dim {
            return None;
        }
        let stride = x.stride_bytes(0) as usize;
        let yp = x.as_mut_ptr() as *mut u16;
        unsafe {
            nk_each_rope_bf16(
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
        Some(())
    }
}

impl EachRope for e4m3 {
    fn rope_into<XMut, const RX: usize>(
        x: &mut XMut,
        cos: &[RopeAngle],
        sin: &[RopeAngle],
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Option<()>
    where
        XMut: TensorMut<Self, RX> + ?Sized,
    {
        if x.ndim() != 2 {
            return None;
        }
        let (rows, width) = (x.shape()[0], x.shape()[1]);
        if rows == 0 || width < heads * 2 * half_dim || cos.len() < rows * half_dim || sin.len() < rows * half_dim {
            return None;
        }
        let stride = x.stride_bytes(0) as usize;
        let yp = x.as_mut_ptr() as *mut u8;
        unsafe {
            nk_each_rope_e4m3(
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
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::EachRope;
    use crate::types::{assert_close, bf16, e4m3, FloatLike, TestableType};

    fn check_rope<Scalar>(values: &[f32], heads: usize, half_dim: usize)
    where
        Scalar: FloatLike + TestableType + EachRope,
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
}
