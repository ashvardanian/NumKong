//! Sparse set intersection and weighted dot products.
//!
//! This module provides:
//!
//! - [`SparseIntersect`]: Sorted-set intersection of index arrays
//! - [`SparseDot`]: Weighted dot product over sparse index-value pairs
//!
//! # Sorted-Index Assumption
//!
//! Every sparse routine in this module assumes that the index arrays are **strictly
//! ascending**, sorted with no duplicates — for example `[1, 3, 5, 7]`. The
//! underlying SIMD kernels use galloping / merge-style advancement that is only
//! correct on sorted input; feeding unsorted indices silently produces wrong results,
//! typically an undercounted intersection.
//!
//! For [`SparseDot`], the paired `weights` slice must mirror the index layout: entry
//! `weights[i]` is the weight at `indices[i]`. The dot product sums
//! `a_weights[i] × b_weights[j]` over all pairs `i, j` where
//! `a_indices[i] == b_indices[j]` — i.e., the intersection's co-indexed weights.
//!
//! Callers with unsorted sparse vectors should sort, for example via `slice::sort_unstable`,
//! and deduplicate before calling these kernels.

use crate::types::bf16;

#[link(name = "numkong")]
extern "C" {
    fn nk_sparse_intersect_u16(
        a: *const u16,
        b: *const u16,
        a_length: usize,
        b_length: usize,
        result: *mut u16,
        count: *mut usize,
    );
    fn nk_sparse_intersect_u32(
        a: *const u32,
        b: *const u32,
        a_length: usize,
        b_length: usize,
        result: *mut u32,
        count: *mut usize,
    );
    fn nk_sparse_intersect_u64(
        a: *const u64,
        b: *const u64,
        a_length: usize,
        b_length: usize,
        result: *mut u64,
        count: *mut usize,
    );
    fn nk_sparse_dot_u16bf16(
        a: *const u16,
        b: *const u16,
        a_weights: *const u16,
        b_weights: *const u16,
        a_length: usize,
        b_length: usize,
        product: *mut f32,
    );
    fn nk_sparse_dot_u32f32(
        a: *const u32,
        b: *const u32,
        a_weights: *const f32,
        b_weights: *const f32,
        a_length: usize,
        b_length: usize,
        product: *mut f64,
    );
}

// region: SparseIntersect

/// Computes set operations on sorted sparse vectors.
pub trait SparseIntersect: Sized {
    /// Returns the intersection size between two sorted sparse vectors.
    fn sparse_intersection_size(a: &[Self], b: &[Self]) -> usize;

    /// Computes intersection and writes matching elements to output buffer.
    /// Buffer must be at least `min(a.len(), b.len())` in size.
    /// Returns `Some(count)` with number of elements written, or `None` if buffer too small.
    fn sparse_intersect_into(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<usize>;
}

impl SparseIntersect for u16 {
    fn sparse_intersection_size(a: &[Self], b: &[Self]) -> usize {
        let mut count: usize = 0;
        unsafe {
            nk_sparse_intersect_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                b.len(),
                core::ptr::null_mut(),
                &mut count,
            )
        };
        count
    }

    fn sparse_intersect_into(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<usize> {
        let min_len = a.len().min(b.len());
        if result.len() < min_len {
            return None;
        }
        let mut count: usize = 0;
        unsafe {
            nk_sparse_intersect_u16(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                b.len(),
                result.as_mut_ptr(),
                &mut count,
            )
        };
        Some(count)
    }
}

impl SparseIntersect for u32 {
    fn sparse_intersection_size(a: &[Self], b: &[Self]) -> usize {
        let mut count: usize = 0;
        unsafe {
            nk_sparse_intersect_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                b.len(),
                core::ptr::null_mut(),
                &mut count,
            )
        };
        count
    }

    fn sparse_intersect_into(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<usize> {
        let min_len = a.len().min(b.len());
        if result.len() < min_len {
            return None;
        }
        let mut count: usize = 0;
        unsafe {
            nk_sparse_intersect_u32(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                b.len(),
                result.as_mut_ptr(),
                &mut count,
            )
        };
        Some(count)
    }
}

impl SparseIntersect for u64 {
    fn sparse_intersection_size(a: &[Self], b: &[Self]) -> usize {
        let mut count: usize = 0;
        unsafe {
            nk_sparse_intersect_u64(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                b.len(),
                core::ptr::null_mut(),
                &mut count,
            )
        };
        count
    }

    fn sparse_intersect_into(a: &[Self], b: &[Self], result: &mut [Self]) -> Option<usize> {
        let min_len = a.len().min(b.len());
        if result.len() < min_len {
            return None;
        }
        let mut count: usize = 0;
        unsafe {
            nk_sparse_intersect_u64(
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
                b.len(),
                result.as_mut_ptr(),
                &mut count,
            )
        };
        Some(count)
    }
}

// endregion: SparseIntersect

// region: SparseDot

/// Computes sparse dot product between two sorted sparse vectors with weights.
///
/// Each vector consists of sorted indices and corresponding weights. The dot product
/// is computed over the intersection of indices, summing the products of weights.
pub trait SparseDot: Sized {
    /// Weight type for this sparse dot product.
    type Weight;
    /// Output type for this sparse dot product.
    type Output;

    /// Computes sparse dot product.
    ///
    /// Returns the sum of `a_weights[i] × b_weights[j]` for all pairs where `a_indices[i] == b_indices[j]`.
    fn sparse_dot(
        a_indices: &[Self],
        b_indices: &[Self],
        a_weights: &[Self::Weight],
        b_weights: &[Self::Weight],
    ) -> Self::Output;
}

impl SparseDot for u16 {
    type Weight = bf16;
    type Output = f32;

    fn sparse_dot(a_indices: &[Self], b_indices: &[Self], a_weights: &[bf16], b_weights: &[bf16]) -> Self::Output {
        let mut product: f32 = 0.0;
        unsafe {
            nk_sparse_dot_u16bf16(
                a_indices.as_ptr(),
                b_indices.as_ptr(),
                a_weights.as_ptr() as *const u16,
                b_weights.as_ptr() as *const u16,
                a_indices.len(),
                b_indices.len(),
                &mut product,
            );
        }
        product
    }
}

impl SparseDot for u32 {
    type Weight = f32;
    type Output = f64;

    fn sparse_dot(a_indices: &[Self], b_indices: &[Self], a_weights: &[f32], b_weights: &[f32]) -> Self::Output {
        let mut product: f64 = 0.0;
        unsafe {
            nk_sparse_dot_u32f32(
                a_indices.as_ptr(),
                b_indices.as_ptr(),
                a_weights.as_ptr(),
                b_weights.as_ptr(),
                a_indices.len(),
                b_indices.len(),
                &mut product,
            );
        }
        product
    }
}

// endregion: SparseDot

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::bf16;

    // region: Sparse Intersections

    #[test]
    fn sparse_intersection() {
        // u16 — intersection size
        let left: Vec<u16> = vec![1, 3, 5, 7, 9];
        let right: Vec<u16> = vec![2, 3, 5, 8, 9];
        assert_eq!(u16::sparse_intersection_size(&left, &right), 3);

        // u16 — intersect into buffer
        let mut result: Vec<u16> = vec![0; 5];
        let count = u16::sparse_intersect_into(&left, &right, &mut result).unwrap();
        assert_eq!(count, 3);
        assert_eq!(&result[..count], &[3, 5, 9]);

        // u32 — intersect into buffer
        let left: Vec<u32> = vec![10, 20, 30, 40];
        let right: Vec<u32> = vec![15, 20, 30, 45];
        let mut result: Vec<u32> = vec![0; 4];
        let count = u32::sparse_intersect_into(&left, &right, &mut result).unwrap();
        assert_eq!(count, 2);
        assert_eq!(&result[..count], &[20, 30]);

        // u64 — intersection size
        let left: Vec<u64> = vec![100, 200, 300];
        let right: Vec<u64> = vec![200, 300, 400];
        assert_eq!(u64::sparse_intersection_size(&left, &right), 2);
    }

    #[test]
    fn sparse_intersect_into_buffer_too_small() {
        let left: Vec<u16> = vec![1, 2, 3, 4, 5];
        let right: Vec<u16> = vec![3, 4, 5, 6, 7];
        let mut result: Vec<u16> = vec![0; 2];
        assert!(u16::sparse_intersect_into(&left, &right, &mut result).is_none());
    }

    // endregion

    // region: Intersection Tests

    fn reference_intersect<Scalar: Ord>(a: &[Scalar], b: &[Scalar]) -> usize {
        let mut a_iter = a.iter();
        let mut b_iter = b.iter();
        let mut a_current = a_iter.next();
        let mut b_current = b_iter.next();
        let mut count = 0;
        while let (Some(a_val), Some(b_val)) = (a_current, b_current) {
            match a_val.cmp(b_val) {
                core::cmp::Ordering::Less => a_current = a_iter.next(),
                core::cmp::Ordering::Greater => b_current = b_iter.next(),
                core::cmp::Ordering::Equal => {
                    count += 1;
                    a_current = a_iter.next();
                    b_current = b_iter.next();
                }
            }
        }
        count
    }

    fn generate_intersection_test_arrays<Scalar>() -> Vec<Vec<Scalar>>
    where
        Scalar: core::convert::TryFrom<u32> + Copy,
        <Scalar as core::convert::TryFrom<u32>>::Error: core::fmt::Debug,
    {
        vec![
            vec![],
            vec![Scalar::try_from(42).unwrap()],
            vec![
                Scalar::try_from(1).unwrap(),
                Scalar::try_from(5).unwrap(),
                Scalar::try_from(10).unwrap(),
            ],
            vec![
                Scalar::try_from(2).unwrap(),
                Scalar::try_from(4).unwrap(),
                Scalar::try_from(6).unwrap(),
                Scalar::try_from(8).unwrap(),
                Scalar::try_from(10).unwrap(),
                Scalar::try_from(12).unwrap(),
                Scalar::try_from(14).unwrap(),
            ],
            (0..14).map(|x| Scalar::try_from(x * 10).unwrap()).collect(),
            (5..20).map(|x| Scalar::try_from(x * 10).unwrap()).collect(),
            (0..40).map(|x| Scalar::try_from(x * 2).unwrap()).collect(),
            (10..50).map(|x| Scalar::try_from(x * 2).unwrap()).collect(),
            (0..45).map(|x| Scalar::try_from(x * 3).unwrap()).collect(),
            (0..100).map(|x| Scalar::try_from(x * 2).unwrap()).collect(),
            (50..150).map(|x| Scalar::try_from(x * 2).unwrap()).collect(),
            (0..100).map(|x| Scalar::try_from(x * 5).unwrap()).collect(),
            (0..150)
                .filter(|x| x % 7 == 0)
                .map(|x| Scalar::try_from(x).unwrap())
                .collect(),
            (0..500).map(|x| Scalar::try_from(x * 3).unwrap()).collect(),
            (100..600).map(|x| Scalar::try_from(x * 3).unwrap()).collect(),
            (0..600).map(|x| Scalar::try_from(x * 7).unwrap()).collect(),
            (0..50).map(|x| Scalar::try_from(x * 2).unwrap()).collect(),
            (1000..1050).map(|x| Scalar::try_from(x * 2).unwrap()).collect(),
            (0..16).map(|x| Scalar::try_from(x).unwrap()).collect(),
            (0..32).map(|x| Scalar::try_from(x).unwrap()).collect(),
            (0..64).map(|x| Scalar::try_from(x).unwrap()).collect(),
        ]
    }

    #[test]
    fn intersect_u32_comprehensive() {
        let test_arrays: Vec<Vec<u32>> = generate_intersection_test_arrays();
        for (i, array_a) in test_arrays.iter().enumerate() {
            for (j, array_b) in test_arrays.iter().enumerate() {
                let expected = reference_intersect(array_a, array_b);
                let result = u32::sparse_intersection_size(array_a.as_slice(), array_b.as_slice());
                assert_eq!(
                    expected,
                    result,
                    "Intersection mismatch for arrays[{}] (len={}) and arrays[{}] (len={})",
                    i,
                    array_a.len(),
                    j,
                    array_b.len()
                );
            }
        }
    }

    #[test]
    fn intersect_u16_comprehensive() {
        let test_arrays: Vec<Vec<u16>> = generate_intersection_test_arrays();
        for (i, array_a) in test_arrays.iter().enumerate() {
            for (j, array_b) in test_arrays.iter().enumerate() {
                let expected = reference_intersect(array_a, array_b);
                let result = u16::sparse_intersection_size(array_a.as_slice(), array_b.as_slice());
                assert_eq!(
                    expected,
                    result,
                    "Intersection mismatch for arrays[{}] (len={}) and arrays[{}] (len={})",
                    i,
                    array_a.len(),
                    j,
                    array_b.len()
                );
            }
        }
    }

    #[test]
    fn intersect_edge_cases() {
        let empty: &[u32] = &[];
        let non_empty: &[u32] = &[1, 2, 3];
        assert_eq!(u32::sparse_intersection_size(empty, empty), 0);
        assert_eq!(u32::sparse_intersection_size(empty, non_empty), 0);
        assert_eq!(u32::sparse_intersection_size(non_empty, empty), 0);

        assert_eq!(u32::sparse_intersection_size(&[42u32], &[42u32]), 1);
        assert_eq!(u32::sparse_intersection_size(&[42u32], &[43u32]), 0);

        let a: &[u32] = &[1, 2, 3, 4, 5];
        let b: &[u32] = &[10, 20, 30, 40, 50];
        assert_eq!(u32::sparse_intersection_size(a, b), 0);

        let c: &[u32] = &[10, 20, 30, 40, 50];
        assert_eq!(u32::sparse_intersection_size(c, c), 5);

        let boundary_16: Vec<u32> = (0..16).collect();
        let boundary_32: Vec<u32> = (0..32).collect();
        let boundary_64: Vec<u32> = (0..64).collect();
        assert_eq!(u32::sparse_intersection_size(&boundary_16, &boundary_16), 16);
        assert_eq!(u32::sparse_intersection_size(&boundary_32, &boundary_32), 32);
        assert_eq!(u32::sparse_intersection_size(&boundary_64, &boundary_64), 64);

        let first_half: Vec<u32> = (0..32).collect();
        let second_half: Vec<u32> = (16..48).collect();
        assert_eq!(u32::sparse_intersection_size(&first_half, &second_half), 16);
    }

    // endregion

    // region: SparseDot

    #[test]
    fn sparse_dot() {
        // u32 indices with f32 weights
        let first_indices: Vec<u32> = vec![1, 3, 5];
        let second_indices: Vec<u32> = vec![2, 3, 5, 7];
        let first_weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        let second_weights: Vec<f32> = vec![4.0, 5.0, 6.0, 7.0];
        // Overlap at indices 3 and 5: 2.0*5.0 + 3.0*6.0 = 28.0
        let result = u32::sparse_dot(&first_indices, &second_indices, &first_weights, &second_weights);
        assert!((result - 28.0).abs() < 0.01, "sparse_dot u32f32: {result}");

        // u16 indices with bf16 weights
        let first_indices_u16: Vec<u16> = vec![1, 3, 5];
        let second_indices_u16: Vec<u16> = vec![2, 3, 5, 7];
        let first_weights_bf16: Vec<bf16> = vec![1.0, 2.0, 3.0].iter().map(|&value| bf16::from_f32(value)).collect();
        let second_weights_bf16: Vec<bf16> = vec![4.0, 5.0, 6.0, 7.0]
            .iter()
            .map(|&value| bf16::from_f32(value))
            .collect();
        let result = u16::sparse_dot(
            &first_indices_u16,
            &second_indices_u16,
            &first_weights_bf16,
            &second_weights_bf16,
        );
        assert!((result - 28.0).abs() < 1.0, "sparse_dot u16bf16: {result}");

        // Disjoint sets → 0
        let result = u32::sparse_dot(&[1, 2], &[3, 4], &[1.0, 1.0], &[1.0, 1.0]);
        assert!(result.abs() < 0.01, "sparse_dot disjoint: {result}");
    }

    // endregion
}
