//! NeoX split-half rotary position embedding (RoPE).
//!
//! Rotates channel pairs `(i, i + half_dim)` of every head by per-token angle grids. The caller bakes
//! position lookup and multi-axis (M-RoPE) assignment into the `[rows, half_dim]` cos/sin grids, so a
//! single call rotates the whole head. The rotation writes every channel it is given, so it is done in
//! place over the `[rows, heads * 2 * half_dim]` slice.

use crate::types::{bf16, e4m3, StorageElement};

#[link(name = "numkong")]
extern "C" {
    fn nk_each_rope_f32(
        x: *const f32,
        y: *mut f32,
        cos: *const f32,
        sin: *const f32,
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
        cos: *const f32,
        sin: *const f32,
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
        cos: *const f32,
        sin: *const f32,
        rows: usize,
        heads: usize,
        half_dim: usize,
        x_row_stride: usize,
        y_row_stride: usize,
        input_scale: f32,
    );
}

/// In-place NeoX split-half RoPE over a row-major `[rows, heads * 2 * half_dim]` slice.
pub trait EachRope: Sized + StorageElement {
    /// Rotates `x` in place using the `[rows, half_dim]` `cos`/`sin` angle grids (row `r` at
    /// `r * half_dim`), shared across heads. Returns `None` on a shape mismatch.
    fn rope(
        x: &mut [Self],
        cos: &[f32],
        sin: &[f32],
        rows: usize,
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Option<()>;
}

impl EachRope for f32 {
    fn rope(
        x: &mut [Self],
        cos: &[f32],
        sin: &[f32],
        rows: usize,
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Option<()> {
        if rows == 0 || x.len() % rows != 0 {
            return None;
        }
        let width = x.len() / rows;
        if width < heads * 2 * half_dim || cos.len() < rows * half_dim || sin.len() < rows * half_dim {
            return None;
        }
        let stride = width * core::mem::size_of::<f32>();
        let xp = x.as_ptr();
        let yp = x.as_mut_ptr();
        unsafe {
            nk_each_rope_f32(
                xp,
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
    fn rope(
        x: &mut [Self],
        cos: &[f32],
        sin: &[f32],
        rows: usize,
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Option<()> {
        if rows == 0 || x.len() % rows != 0 {
            return None;
        }
        let width = x.len() / rows;
        if width < heads * 2 * half_dim || cos.len() < rows * half_dim || sin.len() < rows * half_dim {
            return None;
        }
        let stride = width * core::mem::size_of::<bf16>();
        let xp = x.as_ptr() as *const u16;
        let yp = x.as_mut_ptr() as *mut u16;
        unsafe {
            nk_each_rope_bf16(
                xp,
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
    fn rope(
        x: &mut [Self],
        cos: &[f32],
        sin: &[f32],
        rows: usize,
        heads: usize,
        half_dim: usize,
        input_scale: f32,
    ) -> Option<()> {
        if rows == 0 || x.len() % rows != 0 {
            return None;
        }
        let width = x.len() / rows;
        if width < heads * 2 * half_dim || cos.len() < rows * half_dim || sin.len() < rows * half_dim {
            return None;
        }
        let stride = width * core::mem::size_of::<e4m3>();
        let xp = x.as_ptr() as *const u8;
        let yp = x.as_mut_ptr() as *mut u8;
        unsafe {
            nk_each_rope_e4m3(
                xp,
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
        let mut x: Vec<Scalar> = values.iter().map(|&v| Scalar::from_f32(v)).collect();
        // Per-token angle grids [rows, half_dim].
        let cos: Vec<f32> = (0..rows * half_dim).map(|k| (0.1 * k as f32).cos()).collect();
        let sin: Vec<f32> = (0..rows * half_dim).map(|k| (0.1 * k as f32).sin()).collect();
        let reference = x.clone();
        Scalar::rope(&mut x, &cos, &sin, rows, heads, half_dim, 1.0).unwrap();
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
}
