//! Curved metric spaces: Bilinear forms and Mahalanobis distance.
//!
//! This module provides:
//!
//! - [`Bilinear`]: Bilinear form aᵀ × C × b with a metric tensor
//! - [`Mahalanobis`]: Mahalanobis distance √((a−b)ᵀ × C × (a−b))

use crate::types::{bf16, bf16c, f16, f16c, f32c, f64c, StorageElement};

#[link(name = "numkong")]
extern "C" {
    fn nk_bilinear_f64(a: *const f64, b: *const f64, c: *const f64, n: usize, result: *mut f64);
    fn nk_bilinear_f32(a: *const f32, b: *const f32, c: *const f32, n: usize, result: *mut f64);
    fn nk_bilinear_f16(a: *const u16, b: *const u16, c: *const u16, n: usize, result: *mut f32);
    fn nk_bilinear_bf16(a: *const u16, b: *const u16, c: *const u16, n: usize, result: *mut f32);
    fn nk_bilinear_f64c(a: *const f64, b: *const f64, c: *const f64, n: usize, results: *mut f64);
    fn nk_bilinear_f32c(a: *const f32, b: *const f32, c: *const f32, n: usize, results: *mut f64);
    fn nk_bilinear_f16c(a: *const u16, b: *const u16, c: *const u16, n: usize, results: *mut f32);
    fn nk_bilinear_bf16c(a: *const u16, b: *const u16, c: *const u16, n: usize, results: *mut f32);

    // Mahalanobis distance
    fn nk_mahalanobis_f64(a: *const f64, b: *const f64, c: *const f64, n: usize, result: *mut f64);
    fn nk_mahalanobis_f32(a: *const f32, b: *const f32, c: *const f32, n: usize, result: *mut f64);
    fn nk_mahalanobis_f16(a: *const u16, b: *const u16, c: *const u16, n: usize, result: *mut f32);
    fn nk_mahalanobis_bf16(a: *const u16, b: *const u16, c: *const u16, n: usize, result: *mut f32);
}

/// Bilinear form computation: aᵀ × C × b where C is a metric tensor.
///
/// Computes the bilinear form of two vectors `a` and `b` with respect to
/// a symmetric matrix `C`, given in row-major order as a flat slice of length n².
pub trait Bilinear: StorageElement {
    /// Output type for results. `f32` widens to `f64`; f16/bf16 use f32.
    type Output;

    /// Computes the bilinear form aᵀ × C × b.
    ///
    /// # Arguments
    /// * `a` - First vector of length n
    /// * `b` - Second vector of length n
    /// * `c` - Metric tensor — n×n matrix in row-major order, length n²
    ///
    /// # Returns
    /// `Some(result)` if inputs are valid, `None` if lengths are incompatible.
    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output>;
}

impl Bilinear for f64 {
    type Output = f64;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result: f64 = 0.0;
        unsafe {
            nk_bilinear_f64(a.as_ptr(), b.as_ptr(), c.as_ptr(), point_count, &mut result);
        }
        Some(result)
    }
}

impl Bilinear for f32 {
    type Output = f64;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result: f64 = 0.0;
        unsafe {
            nk_bilinear_f32(a.as_ptr(), b.as_ptr(), c.as_ptr(), point_count, &mut result);
        }
        Some(result)
    }
}

impl Bilinear for f16 {
    type Output = f32;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_bilinear_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                point_count,
                &mut result,
            );
        }
        Some(result)
    }
}

impl Bilinear for bf16 {
    type Output = f32;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_bilinear_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                point_count,
                &mut result,
            );
        }
        Some(result)
    }
}

impl Bilinear for f64c {
    type Output = f64c;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_bilinear_f64c(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                c.as_ptr() as *const f64,
                point_count,
                result.as_mut_ptr(),
            );
        }
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

impl Bilinear for f32c {
    type Output = f64c;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result = [0.0f64; 2];
        unsafe {
            nk_bilinear_f32c(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                c.as_ptr() as *const f32,
                point_count,
                result.as_mut_ptr(),
            );
        }
        Some(f64c {
            re: result[0],
            im: result[1],
        })
    }
}

impl Bilinear for f16c {
    type Output = f32c;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_bilinear_f16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                point_count,
                result.as_mut_ptr(),
            );
        }
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

impl Bilinear for bf16c {
    type Output = f32c;

    fn bilinear(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result = [0.0f32; 2];
        unsafe {
            nk_bilinear_bf16c(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                point_count,
                result.as_mut_ptr(),
            );
        }
        Some(f32c {
            re: result[0],
            im: result[1],
        })
    }
}

/// Mahalanobis distance: √((a−b)ᵀ × C × (a−b)).
///
/// Computes the Mahalanobis distance between two vectors `a` and `b` with respect
/// to an inverse covariance matrix `C`, given in row-major order as a flat slice of length n².
pub trait Mahalanobis: StorageElement {
    /// Output type for results. `f32` widens to `f64`; f16/bf16 use f32.
    type Output;

    /// Computes the Mahalanobis distance √((a−b)ᵀ × C × (a−b)).
    ///
    /// # Arguments
    /// * `a` - First vector of length n
    /// * `b` - Second vector of length n
    /// * `c` - Inverse covariance matrix — n×n matrix in row-major order, length n²
    ///
    /// # Returns
    /// `Some(result)` if inputs are valid, `None` if lengths are incompatible.
    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output>;
}

impl Mahalanobis for f64 {
    type Output = f64;

    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result: f64 = 0.0;
        unsafe {
            nk_mahalanobis_f64(a.as_ptr(), b.as_ptr(), c.as_ptr(), point_count, &mut result);
        }
        Some(result)
    }
}

impl Mahalanobis for f32 {
    type Output = f64;

    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result: f64 = 0.0;
        unsafe {
            nk_mahalanobis_f32(a.as_ptr(), b.as_ptr(), c.as_ptr(), point_count, &mut result);
        }
        Some(result)
    }
}

impl Mahalanobis for f16 {
    type Output = f32;

    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_mahalanobis_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                point_count,
                &mut result,
            );
        }
        Some(result)
    }
}

impl Mahalanobis for bf16 {
    type Output = f32;

    fn mahalanobis(a: &[Self], b: &[Self], c: &[Self]) -> Option<Self::Output> {
        let point_count = a.len();
        if point_count == 0 || b.len() != point_count || c.len() != point_count * point_count {
            return None;
        }
        let mut result: f32 = 0.0;
        unsafe {
            nk_mahalanobis_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_ptr() as *const u16,
                point_count,
                &mut result,
            );
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, bf16, f16, FloatLike, NumberLike, TestableType};
    /// Build an identity matrix of size point_count × point_count.
    pub(crate) fn make_identity<Scalar: FloatLike>(point_count: usize) -> Vec<Scalar> {
        let mut v = vec![Scalar::zero(); point_count * point_count];
        for i in 0..point_count {
            v[i * point_count + i] = Scalar::one();
        }
        v
    }

    fn check_bilinear<Scalar>(first_values: &[f32], second_values: &[f32], expected: f64)
    where
        Scalar: FloatLike + TestableType + Bilinear,
        Scalar::Output: FloatLike,
    {
        let first: Vec<Scalar> = first_values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let second: Vec<Scalar> = second_values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let identity = make_identity::<Scalar>(first.len());
        let result = Scalar::bilinear(&first, &second, &identity).unwrap();
        assert_close(
            result.to_f64(),
            expected,
            Scalar::atol(),
            Scalar::rtol(),
            &format!("bilinear<{}>", core::any::type_name::<Scalar>()),
        );
    }

    #[test]
    fn bilinear() {
        // first=[1,2,3], second=[4,5,6], identity matrix → dot = 32
        let first_values = &[1.0, 2.0, 3.0];
        let second_values = &[4.0, 5.0, 6.0];
        check_bilinear::<f64>(first_values, second_values, 32.0);
        check_bilinear::<f32>(first_values, second_values, 32.0);
        check_bilinear::<f16>(first_values, second_values, 32.0);
        check_bilinear::<bf16>(first_values, second_values, 32.0);
    }

    fn check_mahalanobis<Scalar>(first_values: &[f32], second_values: &[f32], expected: f64)
    where
        Scalar: FloatLike + TestableType + Mahalanobis,
        Scalar::Output: FloatLike,
    {
        let first: Vec<Scalar> = first_values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let second: Vec<Scalar> = second_values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let identity = make_identity::<Scalar>(first.len());
        let result = Scalar::mahalanobis(&first, &second, &identity).unwrap();
        assert_close(
            result.to_f64(),
            expected,
            Scalar::atol(),
            Scalar::rtol(),
            &format!("mahalanobis<{}>", core::any::type_name::<Scalar>()),
        );
    }

    #[test]
    fn mahalanobis() {
        // first=[1,2,3], second=[4,5,6], identity → sqrt(27)
        let first_values = &[1.0, 2.0, 3.0];
        let second_values = &[4.0, 5.0, 6.0];
        let expected = (27.0_f64).sqrt();
        check_mahalanobis::<f64>(first_values, second_values, expected);
        check_mahalanobis::<f32>(first_values, second_values, expected);
        check_mahalanobis::<f16>(first_values, second_values, expected);
        check_mahalanobis::<bf16>(first_values, second_values, expected);
    }
}
