//! Core N-dimensional tensor types and the `TensorRef` / `TensorMut` structural traits.
//!
//! This module provides:
//!
//! - [`Tensor`]: N-dimensional array with customizable rank and allocator
//! - [`TensorView`]: Immutable view into a tensor
//! - [`TensorSpan`]: Mutable view into a tensor
//! - [`Matrix`]: Type alias for 2D tensors
//! - [`TensorRef`] / [`TensorMut`]: Structural traits implemented by all three above
//! - [`Fill`] / [`CopyFrom`]: Container-level traits for in-place fill and slice/view-to-span copies
//!
//! Operation extension traits live in their respective domain modules so each `<op>.rs`
//! mirrors the C++ `<op>.hpp` layout:
//! - [`crate::each`]: `ScaleOps` / `SumOps` / `BlendOps` / `FmaOps` / `TrigSinOps` /
//!   `TrigCosOps` / `TrigAtanOps` / `AllCloseOps`
//! - [`crate::reduce`]: `MomentsOps` / `MinMaxOps` / `BitwiseReductions`
//! - [`crate::cast`]: `CastOps`
//!
//! Batch matrix operations (GEMM, packed spatial distances) live in [`crate::matrix`].
//!
//! # Custom allocators
//!
//! [`Tensor`] is generic over any type that implements the [`Allocator`] trait.
//! The default [`Global`] allocator forwards to the system allocator, but arena
//! allocators, pool allocators, or pinned-memory allocators can be plugged in
//! via the `try_*_in` constructors:
//!
//! ```rust,ignore
//! use numkong::tensor::{Tensor, Allocator};
//!
//! struct MyArena { /* ... */ }
//! unsafe impl Allocator for MyArena { /* ... */ }
//!
//! let arena = MyArena::new();
//! let tensor = Tensor::<f32, _>::try_full_in(&[1024, 1024], 0.0, arena).unwrap();
//! ```
//!
//! # Slicing and views
//!
//! Tensors support zero-copy slicing with NumPy-style tuple syntax or the
//! [`SliceRange`] enum. Views and spans share memory with the parent tensor
//! and may have non-contiguous strides:
//!
//! ```rust,ignore
//! let matrix = Tensor::<f32>::try_full(&[4, 5], 1.0).unwrap();
//! let row = matrix.slice((1_usize, ..)).unwrap();       // t[1, :]
//! let block = matrix.slice((0..2_usize, 1..4_usize)).unwrap();
//! ```
//!
//! # Sub-byte types
//!
//! Sub-byte element types (`i4x2`, `u4x2`, `u1x8`) pack multiple logical
//! elements per storage byte. The [`Index`](core::ops::Index) trait cannot
//! return a `&Scalar` reference to an individual nibble, so sub-byte tensors
//! must be accessed via [`Tensor::try_flat`], [`Tensor::try_coords`], and the
//! iterator APIs that yield [`crate::types::DimRef`] / [`crate::types::DimMut`]
//! proxies.

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use core::ptr::NonNull;

use crate::cast::{cast, CastDtype};
use crate::each::{EachATan, EachBlend, EachCos, EachFMA, EachScale, EachSin, EachSum};
use crate::reduce::{MomentsOps, ReduceMinMax, ReduceMoments, SumSqToF64};
use crate::spatial::{Dot, Roots};
use crate::types::{DimMut, DimRef, FloatConvertible, StorageElement};
use crate::vector::{Vector, VectorIndex};

// region: Constants and Allocator

/// Default maximum rank for tensors.
pub const DEFAULT_MAX_RANK: usize = 8;

/// Alignment for SIMD-friendly allocations (64 bytes for AVX-512).
pub const SIMD_ALIGNMENT: usize = 64;

/// Memory allocator trait for custom allocation strategies.
///
/// Implement this trait to use custom allocators (arena, pool, etc.) with
/// [`Tensor`] and [`PackedMatrix`].
///
/// # Safety
///
/// Implementations must ensure:
/// - `allocate` returns a valid, properly aligned pointer on success
/// - `deallocate` is called with the same `Layout` used in `allocate`
/// - The returned memory is not aliased
pub unsafe trait Allocator {
    /// Allocates memory with the given layout.
    ///
    /// # Contract
    ///
    /// On success returns `Some(ptr)` where `ptr` satisfies the requested
    /// size and alignment (at least [`SIMD_ALIGNMENT`] when called from
    /// [`Tensor`] constructors). Returns `None` when the allocator cannot
    /// fulfil the request — typical causes are out-of-memory conditions or
    /// an alignment the backing allocator cannot provide. Callers must
    /// propagate this `None` as [`TensorError::AllocationFailed`].
    fn allocate(&self, layout: alloc::alloc::Layout) -> Option<NonNull<u8>>;

    /// Deallocates memory previously allocated with `allocate`.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by a previous call to `allocate` on
    ///   the *same* allocator instance (or an allocator that shares its
    ///   backing storage).
    /// - `layout` must match the `Layout` used at allocation time: same
    ///   size and same alignment.
    /// - The memory must not be accessed after deallocation.
    /// - `ptr` must not be deallocated twice.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: alloc::alloc::Layout);
}

/// Global allocator using the system allocator.
#[derive(Debug, Clone, Copy, Default)]
pub struct Global;

unsafe impl Allocator for Global {
    #[inline]
    fn allocate(&self, layout: alloc::alloc::Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return Some(NonNull::dangling());
        }
        unsafe {
            let ptr = alloc::alloc::alloc(layout);
            NonNull::new(ptr)
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: alloc::alloc::Layout) {
        if layout.size() > 0 {
            alloc::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }
}

// endregion: Constants and Allocator

// region: Error Types

/// Error type for Tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Memory allocation failed.
    AllocationFailed,
    /// Shape mismatch: axis `axis` has size `expected` on one side and `got` on the other.
    ShapeMismatch {
        axis: usize,
        expected: usize,
        got: usize,
    },
    /// Invalid shape specification.
    InvalidShape {
        axis: usize,
        size: usize,
        reason: &'static str,
    },
    /// Operation requires contiguous rows but array has non-contiguous rows.
    NonContiguousRows,
    /// Expected a specific number of dimensions.
    DimensionMismatch { expected: usize, got: usize },
    /// Index out of bounds.
    IndexOutOfBounds { index: usize, size: usize },
    /// Too many dimensions (exceeds MAX_RANK).
    TooManyRanks { got: usize },
    /// Operation not supported for sub-byte types (i4x2, u4x2, u1x8).
    SubByteUnsupported,
}

#[cfg(feature = "std")]
impl std::error::Error for TensorError {}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorError::AllocationFailed => write!(f, "memory allocation failed"),
            TensorError::ShapeMismatch {
                axis,
                expected,
                got,
            } => {
                write!(
                    f,
                    "shape mismatch on axis {axis}: expected {expected}, got {got}"
                )
            }
            TensorError::InvalidShape { axis, size, reason } => {
                write!(f, "invalid shape: axis {axis} has size {size}: {reason}")
            }
            TensorError::NonContiguousRows => {
                write!(f, "operation requires contiguous rows")
            }
            TensorError::DimensionMismatch { expected, got } => {
                write!(f, "expected {} dimensions, got {}", expected, got)
            }
            TensorError::IndexOutOfBounds { index, size } => {
                write!(f, "index {} out of bounds for size {}", index, size)
            }
            TensorError::TooManyRanks { got } => {
                write!(f, "too many ranks: {}", got)
            }
            TensorError::SubByteUnsupported => {
                write!(f, "operation not supported for sub-byte types")
            }
        }
    }
}

// endregion: Error Types

// region: MinMaxResult

/// Named result from min/max reduction operations.
///
/// For scalar reductions (`try_minmax_all`), `Value` is the scalar output type
/// and `AnyIndex` defaults to `usize`.
/// For axis reductions (`try_minmax_axis`), `Value` and `AnyIndex` are tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinMaxResult<Value, AnyIndex = usize> {
    pub min_value: Value,
    pub min_index: AnyIndex,
    pub max_value: Value,
    pub max_index: AnyIndex,
}

// endregion: MinMaxResult

// region: Tensor

/// N-dimensional array with NumKong-accelerated operations.
///
/// Uses raw memory allocation (no std::Vec) for maximum control.
///
/// Supports:
/// - Slicing and subviews (zero-copy)
/// - Dot-product multiplication with [`PackedMatrix`]
/// - Reductions (sum, min, max)
/// - Elementwise ops (scale, sum, blend, fma)
/// - Trigonometry (sin, cos, atan)
///
/// # Example
///
/// ```rust,ignore
/// // Requires linking against libnumkong C library
/// use numkong::{Tensor, PackedMatrix};
///
/// let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
/// let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();
///
/// // Pack B once, multiply many times
/// let b_packed = PackedMatrix::try_pack(&b).unwrap();
/// let c = a.dots_packed(&b_packed);  // Returns (1024 × 256)
/// ```
pub struct Tensor<
    Scalar: StorageElement,
    Alloc: Allocator = Global,
    const MAX_RANK: usize = DEFAULT_MAX_RANK,
> {
    /// Raw pointer to data buffer.
    data: NonNull<Scalar>,
    /// Shape dimensions (always logical).
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Allocator instance.
    pub(crate) alloc: Alloc,
}

// Safety: Tensor owns its data and Scalar: Send implies the array is Send
unsafe impl<Scalar: StorageElement + Send, Alloc: Allocator + Send, const MAX_RANK: usize> Send
    for Tensor<Scalar, Alloc, MAX_RANK>
{
}
// Safety: Tensor has no interior mutability, &Tensor<Scalar> is safe to share if Scalar: Sync
unsafe impl<Scalar: StorageElement + Sync, Alloc: Allocator + Sync, const MAX_RANK: usize> Sync
    for Tensor<Scalar, Alloc, MAX_RANK>
{
}

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Drop
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn drop(&mut self) {
        let storage_count = if self.ndim == 0 {
            0
        } else {
            let total: usize = self.shape[..self.ndim].iter().product();
            total / Scalar::dimensions_per_value()
        };
        if storage_count > 0 {
            unsafe {
                core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                    self.data.as_ptr(),
                    storage_count,
                ));
                let layout = alloc::alloc::Layout::from_size_align(
                    storage_count * core::mem::size_of::<Scalar>(),
                    SIMD_ALIGNMENT,
                )
                .unwrap();
                self.alloc.deallocate(
                    NonNull::new_unchecked(self.data.as_ptr() as *mut u8),
                    layout,
                );
            }
        }
    }
}

impl<Scalar: StorageElement, Alloc: Allocator + Clone, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Try to clone this tensor, returning an error on allocation failure.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        Self::try_from_slice_in(self.as_slice(), self.shape(), self.alloc.clone())
    }
}

impl<Scalar: StorageElement, Alloc: Allocator + Clone, const MAX_RANK: usize> Clone
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn clone(&self) -> Self {
        self.try_clone().expect("tensor clone allocation failed")
    }
}

// Generic allocator-aware methods
impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Creates a new Tensor filled with a value using a custom allocator.
    ///
    /// The `shape` specifies logical dimensions. For sub-byte types (e.g.
    /// `i4x2` with `dimensions_per_value() == 2`), the innermost dimension
    /// must be divisible by the packing factor. Storage is allocated for
    /// `total / dimensions_per_value()` packed values.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_full_in(shape: &[usize], value: Scalar, alloc: Alloc) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if total == 0 && !shape.is_empty() {
            if let Some(i) = shape.iter().position(|&d| d == 0) {
                return Err(TensorError::InvalidShape {
                    axis: i,
                    size: 0,
                    reason: "zero-sized dimension",
                });
            }
        }

        let dims_per_value = Scalar::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && shape[shape.len() - 1] % dims_per_value != 0 {
            return Err(TensorError::InvalidShape {
                axis: shape.len() - 1,
                size: shape[shape.len() - 1],
                reason: "innermost dimension must be divisible by dimensions_per_value()",
            });
        }
        let storage_count = if dims_per_value == 1 {
            total
        } else {
            total / dims_per_value
        };

        // Allocate SIMD-aligned raw buffer using our allocator
        let data = if storage_count == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                storage_count * core::mem::size_of::<Scalar>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Initialize all storage elements
            unsafe {
                let ptr = ptr.as_ptr() as *mut Scalar;
                for i in 0..storage_count {
                    core::ptr::write(ptr.add(i), value);
                }
                NonNull::new_unchecked(ptr)
            }
        };

        // Build shape and strides arrays
        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, dims_per_value, &mut strides_arr);

        Ok(Self {
            data,
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    /// Creates a zero-initialized Tensor using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_zeros_in(shape: &[usize], alloc: Alloc) -> Result<Self, TensorError>
    where
        Scalar: Default,
    {
        Self::try_full_in(shape, Scalar::default(), alloc)
    }

    /// Creates a Tensor filled with ones using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_ones_in(shape: &[usize], alloc: Alloc) -> Result<Self, TensorError>
    where
        Scalar: crate::types::NumberLike,
    {
        Self::try_full_in(shape, Scalar::one(), alloc)
    }

    /// Creates an uninitialized Tensor using a custom allocator.
    ///
    /// # Safety
    /// The returned tensor's contents are uninitialized. Reading before writing
    /// is undefined behavior.
    pub unsafe fn try_empty_in(shape: &[usize], alloc: Alloc) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if total == 0 && !shape.is_empty() {
            if let Some(i) = shape.iter().position(|&d| d == 0) {
                return Err(TensorError::InvalidShape {
                    axis: i,
                    size: 0,
                    reason: "zero-sized dimension",
                });
            }
        }

        let dims_per_value = Scalar::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && shape[shape.len() - 1] % dims_per_value != 0 {
            return Err(TensorError::InvalidShape {
                axis: shape.len() - 1,
                size: shape[shape.len() - 1],
                reason: "innermost dimension must be divisible by dimensions_per_value()",
            });
        }
        let storage_count = if dims_per_value == 1 {
            total
        } else {
            total / dims_per_value
        };

        let data = if storage_count == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                storage_count * core::mem::size_of::<Scalar>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut Scalar) }
        };

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, dims_per_value, &mut strides_arr);

        Ok(Self {
            data,
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    /// Creates a Tensor from existing storage data using a custom allocator.
    ///
    /// The `shape` specifies logical dimensions. For sub-byte types, the
    /// `data` slice has `shape.product() / dimensions_per_value()` storage
    /// values. For normal types, `data.len() == shape.product()`.
    pub fn try_from_slice_in(
        data: &[Scalar],
        shape: &[usize],
        alloc: Alloc,
    ) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        let dims_per_value = Scalar::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && shape[shape.len() - 1] % dims_per_value != 0 {
            return Err(TensorError::InvalidShape {
                axis: shape.len() - 1,
                size: shape[shape.len() - 1],
                reason: "innermost dimension must be divisible by dimensions_per_value()",
            });
        }
        let expected_storage = if dims_per_value == 1 {
            total
        } else {
            total / dims_per_value
        };
        if data.len() != expected_storage {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: expected_storage,
                got: data.len(),
            });
        }

        // Allocate SIMD-aligned buffer and copy using our allocator
        let ptr = if expected_storage == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                expected_storage * core::mem::size_of::<Scalar>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Clone all storage elements
            unsafe {
                let ptr = ptr.as_ptr() as *mut Scalar;
                for (i, item) in data[..expected_storage].iter().enumerate() {
                    core::ptr::write(ptr.add(i), *item);
                }
                NonNull::new_unchecked(ptr)
            }
        };

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, dims_per_value, &mut strides_arr);

        Ok(Self {
            data: ptr,
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    /// Creates a Tensor from per-dimension `f32` values using a custom allocator.
    ///
    /// Each `f32` is converted through [`FloatConvertible::DimScalar::from_f32`] before
    /// storage, so this works for full-byte types (`f16`, `bf16`, `i8`, …) and
    /// sub-byte types (`i4x2`, `u4x2`, `u1x8`) alike. The length of `scalars`
    /// must equal the product of `shape`.
    pub fn try_from_scalars_in(
        scalars: &[f32],
        shape: &[usize],
        alloc: Alloc,
    ) -> Result<Self, TensorError>
    where
        Scalar: FloatConvertible,
        Alloc: Clone,
    {
        let total: usize = shape.iter().product();
        if scalars.len() != total {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: total,
                got: scalars.len(),
            });
        }
        // Pack through a Vector so sub-byte types round-trip via `try_set`.
        let flat = Vector::<Scalar, Alloc>::try_from_scalars_in(scalars, alloc.clone())?;
        let mut tensor = Self::try_zeros_in(shape, alloc)?;
        unsafe {
            core::ptr::copy_nonoverlapping(flat.as_ptr(), tensor.as_mut_ptr(), flat.size_values());
        }
        Ok(tensor)
    }

    /// Creates a Tensor from per-dimension `DimScalar` values using a custom allocator.
    ///
    /// Each element of `dim_values` represents one logical dimension; for sub-byte
    /// types the values are packed into their storage representation. The length
    /// of `dim_values` must equal the product of `shape`.
    pub fn try_from_dims_in(
        dim_values: &[<Scalar as FloatConvertible>::DimScalar],
        shape: &[usize],
        alloc: Alloc,
    ) -> Result<Self, TensorError>
    where
        Scalar: FloatConvertible,
        Alloc: Clone,
    {
        let total: usize = shape.iter().product();
        if dim_values.len() != total {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: total,
                got: dim_values.len(),
            });
        }
        let flat = Vector::<Scalar, Alloc>::try_from_dims_in(dim_values, alloc.clone())?;
        let mut tensor = Self::try_zeros_in(shape, alloc)?;
        unsafe {
            core::ptr::copy_nonoverlapping(flat.as_ptr(), tensor.as_mut_ptr(), flat.size_values());
        }
        Ok(tensor)
    }

    /// Compute byte strides from a logical shape.
    ///
    /// For sub-byte types (`dims_per_value > 1`), the innermost dimension is
    /// divided by `dims_per_value` before computing strides, so the innermost
    /// stride is `size_of::<Scalar>()` and covers `dims_per_value` logical elements
    /// per step.
    fn compute_strides_into(
        shape: &[usize],
        dims_per_value: usize,
        strides: &mut [isize; MAX_RANK],
    ) {
        let elem_size = core::mem::size_of::<Scalar>();
        if shape.is_empty() {
            return;
        }

        let mut stride = elem_size as isize;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            let dim_storage = if i == shape.len() - 1 && dims_per_value > 1 {
                shape[i] / dims_per_value
            } else {
                shape[i]
            };
            stride *= dim_storage as isize;
        }
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc {
        &self.alloc
    }

    /// Number of storage values (for sub-byte types, less than numel).
    pub fn storage_len(&self) -> usize {
        self.numel().div_ceil(Scalar::dimensions_per_value())
    }

    /// Convert a 1D contiguous tensor into a [`Vector`], transferring ownership without copying.
    ///
    /// Returns an error if the tensor is not 1D or not contiguous.
    pub fn try_into_vector(self) -> Result<Vector<Scalar, Alloc>, TensorError> {
        if self.ndim != 1 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: self.ndim,
            });
        }
        let expected_stride = core::mem::size_of::<Scalar>() as isize;
        if self.strides[0] != expected_stride {
            return Err(TensorError::NonContiguousRows);
        }
        let dims = self.shape[0];
        let data = self.data;
        let alloc = unsafe { core::ptr::read(&self.alloc) };
        core::mem::forget(self);
        // SAFETY: storage size derived from `dims` matches the original allocation.
        Ok(unsafe { Vector::from_raw_parts(data, dims, alloc) })
    }
}

// Methods that don't require Clone
impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Construct a tensor from raw parts, taking ownership of the allocation.
    ///
    /// # Safety
    /// - `data` must point to an allocation obtained from `alloc`, aligned to
    ///   [`SIMD_ALIGNMENT`], whose backing buffer is sized for the storage count
    ///   implied by `shape` (`product(shape) / Scalar::dimensions_per_value()`
    ///   slots of `Scalar`).
    /// - `shape`, `strides`, and `ndim` must be consistent with the data layout.
    /// - The caller must not free the memory (this tensor takes ownership).
    pub unsafe fn from_raw_parts(
        data: NonNull<Scalar>,
        shape: [usize; MAX_RANK],
        strides: [isize; MAX_RANK],
        ndim: usize,
        alloc: Alloc,
    ) -> Self {
        Self {
            data,
            shape,
            strides,
            ndim,
            alloc,
        }
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of logical elements.
    ///
    /// For sub-byte types, this is the logical count (e.g. 6 nibbles for
    /// shape `[6]` of `i4x2`), not the storage count (3 packed values).
    pub fn numel(&self) -> usize {
        self.shape[..self.ndim].iter().product()
    }

    /// Returns true if the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize {
        self.strides[dim]
    }

    /// Returns a pointer to the data.
    pub fn as_ptr(&self) -> *const Scalar {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut Scalar {
        self.data.as_ptr()
    }

    /// Returns the underlying storage data as a slice.
    ///
    /// For sub-byte types, the slice contains packed storage values, not
    /// individual logical elements.
    pub fn as_slice(&self) -> &[Scalar] {
        let count = self.numel() / Scalar::dimensions_per_value();
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), count) }
    }

    /// Returns the underlying storage data as a mutable slice.
    ///
    /// For sub-byte types, the slice contains packed storage values, not
    /// individual logical elements.
    pub fn as_mut_slice(&mut self) -> &mut [Scalar] {
        let count = self.numel() / Scalar::dimensions_per_value();
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), count) }
    }

    /// Check if rows are contiguous (required for GEMM A matrix).
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        // Last dimension stride should be element size
        self.strides[1] == core::mem::size_of::<Scalar>() as isize
    }

    /// Returns a row of a 2D array.
    pub fn row(&self, i: usize) -> Option<&[Scalar]> {
        if self.ndim != 2 {
            return None;
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if i >= rows {
            return None;
        }
        let start = i * cols;
        Some(&self.as_slice()[start..start + cols])
    }

    /// Returns a mutable row of a 2D array.
    pub fn row_mut(&mut self, i: usize) -> Option<&mut [Scalar]> {
        if self.ndim != 2 {
            return None;
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if i >= rows {
            return None;
        }
        let start = i * cols;
        Some(&mut self.as_mut_slice()[start..start + cols])
    }
}

// Convenience methods using Global allocator
impl<Scalar: StorageElement + Clone, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    /// Creates a new Tensor filled with a value using the global allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use numkong::tensor::Tensor;
    ///
    /// let zeros = Tensor::<f32>::try_full(&[2, 3], 0.0).unwrap();
    /// assert_eq!(zeros.shape(), &[2, 3]);
    /// assert_eq!(zeros.numel(), 6);
    /// assert!(zeros.as_slice().iter().all(|&v| v == 0.0));
    /// ```
    pub fn try_full(shape: &[usize], value: Scalar) -> Result<Self, TensorError> {
        Self::try_full_in(shape, value, Global)
    }

    /// Creates a zero-initialized Tensor using the global allocator.
    pub fn try_zeros(shape: &[usize]) -> Result<Self, TensorError>
    where
        Scalar: Default,
    {
        Self::try_zeros_in(shape, Global)
    }

    /// Creates a Tensor filled with ones using the global allocator.
    pub fn try_ones(shape: &[usize]) -> Result<Self, TensorError>
    where
        Scalar: crate::types::NumberLike,
    {
        Self::try_ones_in(shape, Global)
    }

    /// Creates an uninitialized Tensor using the global allocator.
    ///
    /// # Safety
    /// The returned tensor's contents are uninitialized. Reading before writing
    /// is undefined behavior.
    pub unsafe fn try_empty(shape: &[usize]) -> Result<Self, TensorError> {
        unsafe { Self::try_empty_in(shape, Global) }
    }

    /// Creates a Tensor from existing slice data using the global allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn try_from_slice(data: &[Scalar], shape: &[usize]) -> Result<Self, TensorError> {
        Self::try_from_slice_in(data, shape, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn from_slice(data: &[Scalar], shape: &[usize]) -> Self {
        Self::try_from_slice(data, shape).expect("Tensor::from_slice failed")
    }

    /// Creates a Tensor from per-dimension `f32` values using the global allocator.
    ///
    /// Each `f32` is converted through [`FloatConvertible::DimScalar::from_f32`] before
    /// storage. The length of `scalars` must equal the product of `shape`.
    pub fn try_from_scalars(scalars: &[f32], shape: &[usize]) -> Result<Self, TensorError>
    where
        Scalar: FloatConvertible,
    {
        Self::try_from_scalars_in(scalars, shape, Global)
    }

    /// Creates a Tensor from per-dimension `DimScalar` values using the global allocator.
    ///
    /// Each element of `dim_values` represents one logical dimension. The length
    /// of `dim_values` must equal the product of `shape`.
    pub fn try_from_dims(
        dim_values: &[<Scalar as FloatConvertible>::DimScalar],
        shape: &[usize],
    ) -> Result<Self, TensorError>
    where
        Scalar: FloatConvertible,
    {
        Self::try_from_dims_in(dim_values, shape, Global)
    }
}

// endregion: Tensor

// region: SliceRange

/// Represents a range specification for slicing along one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceRange {
    /// Full range (equivalent to `..`)
    Full,
    /// Single index (reduces dimension)
    Index(usize),
    /// Range from start to end exclusive (equivalent to `start..end`)
    Range { start: usize, end: usize },
    /// Range from start to end with step (equivalent to `start..end;step`)
    RangeStep {
        start: usize,
        end: usize,
        step: isize,
    },
}

impl SliceRange {
    /// Create a full range.
    pub fn full() -> Self {
        Self::Full
    }

    /// Create a single index.
    pub fn index(i: usize) -> Self {
        Self::Index(i)
    }

    /// Create a range from start to end.
    pub fn range(start: usize, end: usize) -> Self {
        Self::Range { start, end }
    }

    /// Create a range with step.
    pub fn range_step(start: usize, end: usize, step: isize) -> Self {
        Self::RangeStep { start, end, step }
    }
}

/// A stepped range for use in tuple-based slicing (compile-time dispatch).
///
/// Rust has no built-in literal for stepped ranges, so this struct fills that
/// gap.  Use it inside `.slice()` tuples:
/// ```ignore
/// t.slice((.., RangeStep::new(0, 6, 2))).unwrap();  // t[:, 0:6:2]
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RangeStep {
    pub start: usize,
    pub end: usize,
    pub step: isize,
}

impl RangeStep {
    pub fn new(start: usize, end: usize, step: isize) -> Self {
        Self { start, end, step }
    }
}

// endregion: SliceRange

// region: SliceArg + SliceSpec

/// Resolve a signed index against a dimension size.  Negative values wrap from
/// the end: `-1` → `dim_size - 1`, `-2` → `dim_size - 2`, etc.
#[inline(always)]
fn resolve_signed_(index: isize, dim_size: usize) -> Result<usize, TensorError> {
    if index >= 0 {
        Ok(index as usize)
    } else {
        dim_size
            .checked_sub(index.unsigned_abs())
            .ok_or(TensorError::IndexOutOfBounds {
                index: index.unsigned_abs(),
                size: dim_size,
            })
    }
}

/// Processes one axis of a slice operation, directly computing the effect on
/// output shape, strides, and byte offset — no intermediate enum dispatch.
///
/// Each impl is monomorphized and fully inlined, so the compiler sees concrete
/// types at every call site with zero runtime branching overhead.
///
/// Unsigned types: `RangeFull`, `usize`, `Range<usize>`, `RangeTo<usize>`,
/// `RangeFrom<usize>`, `RangeInclusive<usize>`.
///
/// Signed types (negative wraps from end): `isize`, `Range<isize>`,
/// `RangeTo<isize>`, `RangeFrom<isize>`, `RangeInclusive<isize>`.
///
/// Stepped: `RangeStep` (Rust has no built-in stepped range literal).
///
/// Note: integer literals default to `i32`, so write `0_usize` not `0`.
pub trait SliceArg {
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError>;
}

impl SliceArg for core::ops::RangeFull {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        _offset: &mut isize,
    ) -> Result<(), TensorError> {
        out_shape[*new_ndim] = dim_size;
        out_strides[*new_ndim] = dim_stride;
        *new_ndim += 1;
        Ok(())
    }
}

impl SliceArg for usize {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        _out_shape: &mut [usize],
        _out_strides: &mut [isize],
        _new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        if self >= dim_size {
            return Err(TensorError::IndexOutOfBounds {
                index: self,
                size: dim_size,
            });
        }
        *offset += self as isize * dim_stride;
        Ok(())
    }
}

impl SliceArg for isize {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        resolve_signed_(self, dim_size)?.apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::Range<usize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        if self.start > self.end || self.end > dim_size {
            return Err(TensorError::IndexOutOfBounds {
                index: self.end,
                size: dim_size,
            });
        }
        out_shape[*new_ndim] = self.end - self.start;
        out_strides[*new_ndim] = dim_stride;
        *new_ndim += 1;
        *offset += self.start as isize * dim_stride;
        Ok(())
    }
}

impl SliceArg for core::ops::Range<isize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let start = resolve_signed_(self.start, dim_size)?;
        let end = resolve_signed_(self.end, dim_size)?;
        (start..end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeTo<usize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        (0..self.end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeTo<isize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let end = resolve_signed_(self.end, dim_size)?;
        (0..end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeFrom<usize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        (self.start..dim_size).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeFrom<isize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let start = resolve_signed_(self.start, dim_size)?;
        (start..dim_size).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeInclusive<usize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let start = *self.start();
        let end = self.end().saturating_add(1);
        (start..end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeInclusive<isize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let start = resolve_signed_(*self.start(), dim_size)?;
        let end = resolve_signed_(*self.end(), dim_size)?.saturating_add(1);
        (start..end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for RangeStep {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        if self.start >= dim_size || (self.end > dim_size && self.step > 0) {
            return Err(TensorError::IndexOutOfBounds {
                index: if self.start >= dim_size {
                    self.start
                } else {
                    self.end
                },
                size: dim_size,
            });
        }
        if self.step == 0 {
            return Err(TensorError::InvalidShape {
                axis: 0,
                size: 0,
                reason: "step cannot be zero",
            });
        }
        let count = if self.step > 0 {
            (self.end.saturating_sub(self.start) + self.step as usize - 1) / self.step as usize
        } else {
            let abs_step = (-self.step) as usize;
            (self.start.saturating_sub(self.end) + abs_step - 1) / abs_step
        };
        out_shape[*new_ndim] = count;
        out_strides[*new_ndim] = dim_stride * self.step;
        *new_ndim += 1;
        *offset += self.start as isize * dim_stride;
        Ok(())
    }
}

type LayoutResult<const MAX_RANK: usize> =
    Result<([usize; MAX_RANK], [isize; MAX_RANK], usize, isize, usize), TensorError>;

/// Computes the full slice layout from shape, strides, and ndim.
///
/// Implemented for `&[SliceRange]` / `&[SliceRange; N]` (backward compat, runtime
/// dispatch) and tuples of [`SliceArg`] types from arity 1-8 (compile-time
/// dispatch, fully inlined with zero branching overhead).
pub trait SliceSpec {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK>;
}

impl SliceSpec for &[SliceRange] {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        slice_layout_(shape, strides, ndim, self)
    }
}

impl<const N: usize> SliceSpec for &[SliceRange; N] {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        slice_layout_(shape, strides, ndim, self.as_slice())
    }
}

impl<A0: SliceArg> SliceSpec for (A0,) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 1 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg> SliceSpec for (A0, A1) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg, A2: SliceArg> SliceSpec for (A0, A1, A2) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 3 {
            return Err(TensorError::DimensionMismatch {
                expected: 3,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg, A2: SliceArg, A3: SliceArg> SliceSpec for (A0, A1, A2, A3) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 4 {
            return Err(TensorError::DimensionMismatch {
                expected: 4,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg, A2: SliceArg, A3: SliceArg, A4: SliceArg> SliceSpec
    for (A0, A1, A2, A3, A4)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 5 {
            return Err(TensorError::DimensionMismatch {
                expected: 5,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        self.4
            .apply(shape[4], strides[4], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg, A2: SliceArg, A3: SliceArg, A4: SliceArg, A5: SliceArg> SliceSpec
    for (A0, A1, A2, A3, A4, A5)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 6 {
            return Err(TensorError::DimensionMismatch {
                expected: 6,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        self.4
            .apply(shape[4], strides[4], &mut s, &mut st, &mut nd, &mut off)?;
        self.5
            .apply(shape[5], strides[5], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<
        A0: SliceArg,
        A1: SliceArg,
        A2: SliceArg,
        A3: SliceArg,
        A4: SliceArg,
        A5: SliceArg,
        A6: SliceArg,
    > SliceSpec for (A0, A1, A2, A3, A4, A5, A6)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 7 {
            return Err(TensorError::DimensionMismatch {
                expected: 7,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        self.4
            .apply(shape[4], strides[4], &mut s, &mut st, &mut nd, &mut off)?;
        self.5
            .apply(shape[5], strides[5], &mut s, &mut st, &mut nd, &mut off)?;
        self.6
            .apply(shape[6], strides[6], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<
        A0: SliceArg,
        A1: SliceArg,
        A2: SliceArg,
        A3: SliceArg,
        A4: SliceArg,
        A5: SliceArg,
        A6: SliceArg,
        A7: SliceArg,
    > SliceSpec for (A0, A1, A2, A3, A4, A5, A6, A7)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 8 {
            return Err(TensorError::DimensionMismatch {
                expected: 8,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        self.4
            .apply(shape[4], strides[4], &mut s, &mut st, &mut nd, &mut off)?;
        self.5
            .apply(shape[5], strides[5], &mut s, &mut st, &mut nd, &mut off)?;
        self.6
            .apply(shape[6], strides[6], &mut s, &mut st, &mut nd, &mut off)?;
        self.7
            .apply(shape[7], strides[7], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

// endregion: SliceArg + SliceSpec

#[doc(hidden)]
pub trait TensorCoordinates {
    const ARITY: usize;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError>;
}

impl<I0: VectorIndex, I1: VectorIndex> TensorCoordinates for (I0, I1) {
    const ARITY: usize = 2;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        Ok(resolved)
    }
}

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex> TensorCoordinates for (I0, I1, I2) {
    const ARITY: usize = 3;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        Ok(resolved)
    }
}

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, I3: VectorIndex> TensorCoordinates
    for (I0, I1, I2, I3)
{
    const ARITY: usize = 4;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        Ok(resolved)
    }
}

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, I3: VectorIndex, I4: VectorIndex>
    TensorCoordinates for (I0, I1, I2, I3, I4)
{
    const ARITY: usize = 5;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        resolved[4] = resolve_index_for_size_(self.4, shape[4])?;
        Ok(resolved)
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
    > TensorCoordinates for (I0, I1, I2, I3, I4, I5)
{
    const ARITY: usize = 6;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        resolved[4] = resolve_index_for_size_(self.4, shape[4])?;
        resolved[5] = resolve_index_for_size_(self.5, shape[5])?;
        Ok(resolved)
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
    > TensorCoordinates for (I0, I1, I2, I3, I4, I5, I6)
{
    const ARITY: usize = 7;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        resolved[4] = resolve_index_for_size_(self.4, shape[4])?;
        resolved[5] = resolve_index_for_size_(self.5, shape[5])?;
        resolved[6] = resolve_index_for_size_(self.6, shape[6])?;
        Ok(resolved)
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
    > TensorCoordinates for (I0, I1, I2, I3, I4, I5, I6, I7)
{
    const ARITY: usize = 8;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        resolved[4] = resolve_index_for_size_(self.4, shape[4])?;
        resolved[5] = resolve_index_for_size_(self.5, shape[5])?;
        resolved[6] = resolve_index_for_size_(self.6, shape[6])?;
        resolved[7] = resolve_index_for_size_(self.7, shape[7])?;
        Ok(resolved)
    }
}

// region: TensorView

/// A read-only, zero-copy view into a [`Tensor`].
///
/// `TensorView` borrows the parent tensor's memory for the duration of `'a`
/// without owning or copying any data. Views may have arbitrary byte strides,
/// so they transparently represent slicing, transposition, or step-sampling
/// of the underlying storage — iteration and indexing honour those strides.
///
/// A view is normally obtained by calling [`Tensor::view`] or by slicing:
/// `tensor.slice((0..4_usize, ..))`. For lower-level construction from a
/// raw pointer plus shape/stride arrays, see [`TensorView::from_raw_parts`].
///
/// `TensorView` is the immutable counterpart of [`TensorSpan`]. Both share
/// the same layout fields, but a view cannot be used to mutate the backing
/// storage. Multiple views into the same tensor may coexist, subject to
/// Rust's borrow rules; a mutable span excludes all other references.
///
/// The `'a` lifetime ties the view to the source tensor (or outer view),
/// ensuring the referenced memory outlives the view itself.
pub struct TensorView<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Pointer to first element of view.
    data: *const Scalar,
    /// Shape of the view (always logical).
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a Scalar>,
}

impl<'a, Scalar, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Create a view from a raw pointer, shape, and byte strides.
    ///
    /// The `shape` specifies logical dimensions. For sub-byte types, the
    /// storage count is inferred as `shape.product() / dimensions_per_value()`.
    /// For normal types the two are equal.
    ///
    /// # Safety
    /// - `data` must be valid for reads over the region described by `shape` and `strides_bytes`.
    /// - The pointed-to memory must outlive `'a`.
    /// - `shape.len()` must be `<= MAX_RANK`.
    /// - `shape.len()` must equal `strides_bytes.len()`.
    ///
    /// # Panics
    /// Panics if `shape.len() > MAX_RANK` or `shape.len() != strides_bytes.len()`.
    pub unsafe fn from_raw_parts(
        data: *const Scalar,
        shape: &[usize],
        strides_bytes: &[isize],
    ) -> Self {
        assert!(
            shape.len() <= MAX_RANK,
            "ndim {} exceeds MAX_RANK {}",
            shape.len(),
            MAX_RANK
        );
        assert_eq!(
            shape.len(),
            strides_bytes.len(),
            "shape and strides must have the same length"
        );
        let ndim = shape.len();
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        s[..ndim].copy_from_slice(shape);
        st[..ndim].copy_from_slice(strides_bytes);
        Self {
            data,
            shape: s,
            strides: st,
            ndim,
            _marker: PhantomData,
        }
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of logical elements (computed from shape).
    pub fn numel(&self) -> usize {
        self.shape[..self.ndim].iter().product()
    }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize {
        self.strides[dim]
    }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const Scalar {
        self.data
    }

    /// Check if the view has contiguous rows.
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        self.strides[1] == core::mem::size_of::<Scalar>() as isize
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<Scalar>() as isize;
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Get element at flat index (only valid for contiguous views).
    ///
    /// # Safety
    /// Caller must ensure the view is contiguous and index is in bounds.
    pub unsafe fn get_unchecked(&self, index: usize) -> &Scalar {
        &*self.data.add(index)
    }

    /// Try to get an element by flat logical row-major index.
    ///
    /// Negative `isize` indices wrap from the end (`-1` is the last element).
    /// Returns [`TensorError::DimensionMismatch`] on a rank-0 view and
    /// [`TensorError::IndexOutOfBounds`] when the index is outside the
    /// logical element count.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use numkong::tensor::Tensor;
    ///
    /// let t = Tensor::<f32>::try_from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let view = t.view();
    /// assert_eq!(*view.try_flat(0_usize).unwrap(), 1.0);
    /// assert_eq!(*view.try_flat(-1_i32).unwrap(), 4.0);
    /// ```
    pub fn try_flat<AnyIndex: VectorIndex>(&self, index: AnyIndex) -> Result<&Scalar, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const Scalar) })
    }

    /// Try to get an element by exact coordinates.
    pub fn try_coords<C: TensorCoordinates>(&self, coords: C) -> Result<&Scalar, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const Scalar) })
    }

    /// Try to access the scalar value of a rank-0 tensor view.
    pub fn try_scalar(&self) -> Result<&Scalar, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &*self.data })
    }

    /// Slice the leading axis by one index, reducing rank by one.
    pub fn slice_leading<AnyIndex: VectorIndex>(
        &self,
        index: AnyIndex,
    ) -> Result<TensorView<'a, Scalar, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the view along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`. Indexing with a
    /// scalar reduces the rank, while range arguments preserve it.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use numkong::tensor::{Tensor, SliceRange};
    ///
    /// let t = Tensor::<f32>::try_full(&[4, 5], 1.0).unwrap();
    /// let view = t.view();
    ///
    /// // Rust-native tuple syntax
    /// let row = view.slice((1_usize, ..)).unwrap();            // t[1, :]
    /// let block = view.slice((1..3_usize, 0..4_usize)).unwrap();// t[1:3, 0:4]
    ///
    /// // Enum-based syntax for programmatic construction
    /// let same_row = view.slice(&[SliceRange::index(1), SliceRange::full()]).unwrap();
    /// assert_eq!(row.shape(), same_row.shape());
    /// ```
    pub fn slice(
        &self,
        spec: impl SliceSpec,
    ) -> Result<TensorView<'a, Scalar, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }
}

impl<'a, Scalar: Clone + StorageElement, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Copy the view contents to a new owned Tensor.
    pub fn to_owned(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        if self.is_contiguous() {
            let slice = unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) };
            Tensor::try_from_slice(slice, self.shape())
        } else {
            // For non-contiguous views, we need to copy element by element
            let mut result = Tensor::try_full(self.shape(), unsafe { *self.data })?;
            self.copy_to_contiguous(result.as_mut_slice());
            Ok(result)
        }
    }

    /// Number of storage values (for sub-byte types, less than numel).
    pub fn storage_len(&self) -> usize {
        self.numel().div_ceil(Scalar::dimensions_per_value())
    }

    /// Convert to slice (only valid for contiguous views).
    pub fn as_contiguous_slice(&self) -> Option<&[Scalar]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) })
        } else {
            None
        }
    }

    fn copy_to_contiguous(&self, dest: &mut [Scalar]) {
        // For 2D case, optimize the copy
        if self.ndim == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let row_stride = self.strides[0];
            let col_stride = self.strides[1];
            let mut dest_idx = 0;
            for r in 0..rows {
                let row_ptr = unsafe {
                    (self.data as *const u8).offset(r as isize * row_stride) as *const Scalar
                };
                for c in 0..cols {
                    let elem_ptr = unsafe {
                        (row_ptr as *const u8).offset(c as isize * col_stride) as *const Scalar
                    };
                    dest[dest_idx] = unsafe { *elem_ptr };
                    dest_idx += 1;
                }
            }
        } else {
            // General N-dimensional case: iterate in row-major order
            let mut indices = [0usize; MAX_RANK];
            for dest_slot in dest[..self.numel()].iter_mut() {
                let mut offset = 0isize;
                for (index_val, stride_val) in indices[..self.ndim]
                    .iter()
                    .zip(self.strides[..self.ndim].iter())
                {
                    offset += *index_val as isize * stride_val;
                }
                let elem_ptr = unsafe { (self.data as *const u8).offset(offset) as *const Scalar };
                *dest_slot = unsafe { *elem_ptr };

                // Increment indices (row-major order)
                for d in (0..self.ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < self.shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
        }
    }
}

// region: TensorSpan

/// A mutable, zero-copy view into a [`Tensor`].
///
/// `TensorSpan` borrows the parent tensor's memory exclusively for the
/// duration of `'a`, giving write access without taking ownership. As with
/// [`TensorView`], a span may carry non-contiguous byte strides so slicing,
/// transposition, and stepped sub-views all remain free of data copies.
///
/// Spans are typically produced by [`Tensor::span`] or by a mutable slicing
/// method such as `tensor.slice_mut((..,0_usize))`. Lower-level construction
/// from a raw pointer is available through [`TensorSpan::from_raw_parts`].
///
/// A span is the mutable counterpart of [`TensorView`]. At most one span to
/// a given region may exist at a time; reborrow via [`TensorSpan::as_view`]
/// to hand out immutable sub-views without surrendering the span.
///
/// The `'a` lifetime ties the span to the owning tensor, ensuring the
/// referenced memory remains valid.
pub struct TensorSpan<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Pointer to first element of view.
    data: *mut Scalar,
    /// Shape of the view (always logical).
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a mut Scalar>,
}

impl<'a, Scalar, const MAX_RANK: usize> TensorSpan<'a, Scalar, MAX_RANK> {
    /// Create a mutable view from a raw pointer, shape, and byte strides.
    ///
    /// # Safety
    /// - `data` must be valid for reads and writes over the region described by `shape` and `strides_bytes`.
    /// - The pointed-to memory must outlive `'a`.
    /// - `shape.len()` must be `<= MAX_RANK`.
    /// - `shape.len()` must equal `strides_bytes.len()`.
    /// - No other references to the memory may exist for the duration of `'a`.
    ///
    /// # Panics
    /// Panics if `shape.len() > MAX_RANK` or `shape.len() != strides_bytes.len()`.
    pub unsafe fn from_raw_parts(
        data: *mut Scalar,
        shape: &[usize],
        strides_bytes: &[isize],
    ) -> Self {
        assert!(
            shape.len() <= MAX_RANK,
            "ndim {} exceeds MAX_RANK {}",
            shape.len(),
            MAX_RANK
        );
        assert_eq!(
            shape.len(),
            strides_bytes.len(),
            "shape and strides must have the same length"
        );
        let ndim = shape.len();
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        s[..ndim].copy_from_slice(shape);
        st[..ndim].copy_from_slice(strides_bytes);
        Self {
            data,
            shape: s,
            strides: st,
            ndim,
            _marker: PhantomData,
        }
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of logical elements (computed from shape).
    pub fn numel(&self) -> usize {
        self.shape[..self.ndim].iter().product()
    }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize {
        self.strides[dim]
    }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const Scalar {
        self.data
    }

    /// Returns a mutable pointer to the first element.
    pub fn as_mut_ptr(&mut self) -> *mut Scalar {
        self.data
    }

    /// Check if the view has contiguous rows.
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        self.strides[1] == core::mem::size_of::<Scalar>() as isize
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<Scalar>() as isize;
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Reborrow as immutable view.
    pub fn as_view(&self) -> TensorView<'_, Scalar, MAX_RANK> {
        TensorView {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Try to get an element by flat logical row-major index.
    pub fn try_flat<AnyIndex: VectorIndex>(&self, index: AnyIndex) -> Result<&Scalar, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const Scalar) })
    }

    /// Try to get a mutable element by flat logical row-major index.
    ///
    /// Negative `isize` indices wrap from the end. Returns
    /// [`TensorError::DimensionMismatch`] on a rank-0 span and
    /// [`TensorError::IndexOutOfBounds`] when the index is out of range.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use numkong::tensor::Tensor;
    ///
    /// let mut t = Tensor::<f32>::try_full(&[3], 0.0).unwrap();
    /// let mut span = t.span();
    /// *span.try_flat_mut(-1_i32).unwrap() = 9.0;
    /// assert_eq!(*span.try_flat(2_usize).unwrap(), 9.0);
    /// ```
    pub fn try_flat_mut<AnyIndex: VectorIndex>(
        &mut self,
        index: AnyIndex,
    ) -> Result<&mut Scalar, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &mut *((self.data as *mut u8).offset(offset) as *mut Scalar) })
    }

    /// Try to get an element by exact coordinates.
    pub fn try_coords<C: TensorCoordinates>(&self, coords: C) -> Result<&Scalar, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const Scalar) })
    }

    /// Try to get a mutable element by exact coordinates.
    pub fn try_coords_mut<C: TensorCoordinates>(
        &mut self,
        coords: C,
    ) -> Result<&mut Scalar, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &mut *((self.data as *mut u8).offset(offset) as *mut Scalar) })
    }

    /// Try to access the scalar value of a rank-0 tensor span.
    pub fn try_scalar(&self) -> Result<&Scalar, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &*self.data })
    }

    /// Try to access the mutable scalar value of a rank-0 tensor span.
    pub fn try_scalar_mut(&mut self) -> Result<&mut Scalar, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &mut *self.data })
    }

    /// Slice the leading axis by one index, reducing rank by one.
    pub fn slice_leading<AnyIndex: VectorIndex>(
        &self,
        index: AnyIndex,
    ) -> Result<TensorView<'_, Scalar, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the leading axis mutably by one index, reducing rank by one.
    pub fn slice_leading_mut<AnyIndex: VectorIndex>(
        &mut self,
        index: AnyIndex,
    ) -> Result<TensorSpan<'_, Scalar, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorSpan {
            data: unsafe { (self.data as *mut u8).offset(offset) as *mut Scalar },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the span along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`.
    pub fn slice(
        &self,
        spec: impl SliceSpec,
    ) -> Result<TensorView<'_, Scalar, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the span mutably along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`.
    pub fn slice_mut(
        &mut self,
        spec: impl SliceSpec,
    ) -> Result<TensorSpan<'_, Scalar, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim)?;
        Ok(TensorSpan {
            data: unsafe { (self.data as *mut u8).offset(offset) as *mut Scalar },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }
}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize> TensorSpan<'a, Scalar, MAX_RANK> {
    /// Number of storage values (for sub-byte types, less than numel).
    pub fn storage_len(&self) -> usize {
        self.numel().div_ceil(Scalar::dimensions_per_value())
    }

    /// Convert to slice (only valid for contiguous views).
    pub fn as_contiguous_slice(&self) -> Option<&[Scalar]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) })
        } else {
            None
        }
    }

    /// Convert to mutable slice (only valid for contiguous views).
    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [Scalar]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts_mut(self.data, self.storage_len()) })
        } else {
            None
        }
    }
}

// endregion: TensorSpan

// region: TensorRef / TensorMut Traits

/// Read-only structural access to N-dimensional tensor containers.
///
/// Implemented by [`Tensor`], [`TensorView`], and [`TensorSpan`], enabling
/// generic code over any tensor-like container. Most operations in this
/// crate accept `&(impl TensorRef<Scalar, MAX_RANK> + ?Sized)` so you can
/// mix owned tensors, views, and spans freely without copying.
///
/// The trait only exposes *structural* accessors — shape, strides, pointer,
/// and a borrow-as-view constructor. Element reads happen through the
/// resulting [`TensorView`]. See [`TensorMut`] for the mutable counterpart.
pub trait TensorRef<Scalar: StorageElement, const MAX_RANK: usize> {
    /// Logical shape as a slice of length `ndim()`.
    ///
    /// For sub-byte types this is the number of individual elements, not the
    /// packed storage count.
    fn shape(&self) -> &[usize];

    /// Number of dimensions currently in use.
    ///
    /// Always satisfies `ndim() <= MAX_RANK`. A rank-0 tensor reports `0`.
    fn ndim(&self) -> usize;

    /// Byte stride along dimension `dim`.
    ///
    /// Negative values indicate reverse traversal (e.g. from a stepped
    /// slice). Only indices in `0..ndim()` are valid.
    ///
    /// # Example
    /// ```rust,ignore
    /// let t = Tensor::<f32>::try_full(&[3, 4], 0.0).unwrap();
    /// assert_eq!(t.stride_bytes(1), 4);  // innermost f32 stride
    /// ```
    fn stride_bytes(&self, dim: usize) -> isize;

    /// Raw pointer to the first storage element.
    ///
    /// The pointer is valid for `shape().iter().product()` *logical* reads
    /// (divided by `Scalar::dimensions_per_value()` for sub-byte types).
    fn as_ptr(&self) -> *const Scalar;

    /// Borrow as an immutable [`TensorView`] with the same shape and strides.
    ///
    /// This is a zero-cost reborrow and is the normal entry point for
    /// extension-trait methods that want to delegate to view-level code.
    fn view(&self) -> TensorView<'_, Scalar, MAX_RANK>;

    /// Total number of logical elements (product of shape dimensions).
    ///
    /// # Example
    /// ```rust,ignore
    /// let t = Tensor::<f32>::try_full(&[2, 3, 4], 0.0).unwrap();
    /// assert_eq!(t.numel(), 24);
    /// ```
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Alias for [`ndim`](TensorRef::ndim) — number of dimensions.
    fn rank(&self) -> usize {
        self.ndim()
    }

    /// Returns `true` if the tensor contains zero logical elements.
    ///
    /// A rank-0 (scalar) tensor is *not* empty — it has one element.
    fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Returns `true` for rank-2 tensors whose innermost stride equals one
    /// element — the layout required by GEMM's left-hand matrix.
    fn has_contiguous_rows(&self) -> bool {
        self.ndim() == 2 && self.stride_bytes(1) == core::mem::size_of::<Scalar>() as isize
    }

    /// Returns `true` if the entire tensor is stored in row-major contiguous
    /// order with no gaps.
    ///
    /// A contiguous tensor can be reinterpreted as a flat slice via
    /// [`TensorView::as_contiguous_slice`]; a non-contiguous one cannot.
    fn is_contiguous(&self) -> bool {
        let dims_per_value = Scalar::dimensions_per_value();
        let elem_size = core::mem::size_of::<Scalar>() as isize;
        let mut expected = elem_size;
        let ndim = self.ndim();
        for i in (0..ndim).rev() {
            if self.stride_bytes(i) != expected {
                return false;
            }
            let dim_storage = if i == ndim - 1 && dims_per_value > 1 {
                self.shape()[i] / dims_per_value
            } else {
                self.shape()[i]
            };
            expected *= dim_storage as isize;
        }
        true
    }
}

/// Mutable structural access to N-dimensional tensor containers.
///
/// `TensorMut` is a supertrait of [`TensorRef`] that adds a single method —
/// a mutable raw pointer to the first element. It is implemented by
/// [`Tensor`] (which owns its memory) and by [`TensorSpan`] (which borrows
/// a mutable sub-region). Immutable views ([`TensorView`]) deliberately do
/// not implement `TensorMut`.
///
/// Generic write-access helpers accept `&mut (impl TensorMut<Scalar, R>)`
/// so they work uniformly against owned tensors and span reborrows.
pub trait TensorMut<Scalar: StorageElement, const MAX_RANK: usize>:
    TensorRef<Scalar, MAX_RANK>
{
    /// Raw mutable pointer to the first storage element.
    ///
    /// The pointer is valid for `numel() / dimensions_per_value()` writes
    /// and preserves whatever stride layout [`TensorRef::stride_bytes`]
    /// reports — callers must honour the per-axis strides when striding
    /// through non-contiguous memory.
    fn as_mut_ptr(&mut self) -> *mut Scalar;
}

/// In-place fill operations shared by every mutable container of `Scalar`.
///
/// Implemented for [`Tensor`], [`TensorSpan`], [`crate::vector::Vector`], and
/// [`crate::vector::VectorSpan`]. Mirrors `tensor_span::fill_zeros / fill` and
/// `vector_span::fill_zeros / fill` from the C++ side. Both methods are
/// infallible and operate on every storage byte the container owns or borrows.
pub trait Fill<Scalar: StorageElement> {
    /// Set every storage byte to zero. Works for every `StorageElement` since
    /// the binary-zero of all NumKong scalar dtypes is the value-zero.
    fn fill_zeros(&mut self);

    /// Set every storage element to `value`. For 1-byte storage (including
    /// sub-byte packings) this becomes a single byte-pattern memset; for
    /// multi-byte storage it falls through to a typed broadcast loop on the
    /// pre-zeroed buffer (so float NaN/inf patterns from uninitialised memory
    /// cannot leak through).
    fn fill(&mut self, value: Scalar);
}

/// Copy storage from a `Source` into this container in place.
///
/// `Source` is the natural read-side counterpart for the implementor:
/// `&[Scalar]` (raw storage slice) for owning `Tensor` / `Vector`, or the
/// matching `View` type for span implementors. Mirrors `tensor_span::copy_from`
/// and `vector_span::copy_from` on the C++ side.
pub trait CopyFrom<Source> {
    /// Copy from `source` into self. Returns an error on a shape or storage
    /// mismatch and does not modify the destination on error.
    fn copy_from(&mut self, source: Source) -> Result<(), TensorError>;
}

impl<Scalar: StorageElement, Alloc: Allocator, const R: usize> TensorRef<Scalar, R>
    for Tensor<Scalar, Alloc, R>
{
    fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }
    fn ndim(&self) -> usize {
        self.ndim
    }
    fn stride_bytes(&self, dim: usize) -> isize {
        self.strides[dim]
    }
    fn as_ptr(&self) -> *const Scalar {
        self.data.as_ptr()
    }
    fn view(&self) -> TensorView<'_, Scalar, R> {
        TensorView {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }
}

impl<Scalar: StorageElement, Alloc: Allocator, const R: usize> TensorMut<Scalar, R>
    for Tensor<Scalar, Alloc, R>
{
    fn as_mut_ptr(&mut self) -> *mut Scalar {
        self.data.as_ptr()
    }
}

impl<'a, Scalar: StorageElement, const R: usize> TensorRef<Scalar, R>
    for TensorView<'a, Scalar, R>
{
    fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }
    fn ndim(&self) -> usize {
        self.ndim
    }
    fn stride_bytes(&self, dim: usize) -> isize {
        self.strides[dim]
    }
    fn as_ptr(&self) -> *const Scalar {
        self.data
    }
    fn view(&self) -> TensorView<'_, Scalar, R> {
        TensorView {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }
}

impl<'a, Scalar: StorageElement, const R: usize> TensorRef<Scalar, R>
    for TensorSpan<'a, Scalar, R>
{
    fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }
    fn ndim(&self) -> usize {
        self.ndim
    }
    fn stride_bytes(&self, dim: usize) -> isize {
        self.strides[dim]
    }
    fn as_ptr(&self) -> *const Scalar {
        self.data
    }
    fn view(&self) -> TensorView<'_, Scalar, R> {
        self.as_view()
    }
}

impl<'a, Scalar: StorageElement, const R: usize> TensorMut<Scalar, R>
    for TensorSpan<'a, Scalar, R>
{
    fn as_mut_ptr(&mut self) -> *mut Scalar {
        self.data
    }
}

// endregion: TensorRef / TensorMut Traits

// region: Fill / CopyFrom Implementations

/// Compares the bit pattern of a fresh `Scalar::default()` to `value` to decide
/// whether the post-`fill_zeros` overlay can be skipped. Treats `+0.0` and the
/// default `Scalar::default()` representation as identical so a `fill(0.0)` on a
/// freshly-defaulted buffer is a single memset.
#[inline]
fn value_is_default_bit_pattern<Scalar: StorageElement>(value: Scalar) -> bool {
    let default_value = Scalar::default();
    let value_bytes = unsafe {
        core::slice::from_raw_parts(
            (&value as *const Scalar) as *const u8,
            core::mem::size_of::<Scalar>(),
        )
    };
    let default_bytes = unsafe {
        core::slice::from_raw_parts(
            (&default_value as *const Scalar) as *const u8,
            core::mem::size_of::<Scalar>(),
        )
    };
    value_bytes == default_bytes
}

/// Body shared by every `Fill::fill` impl: zero the buffer, then byte-pattern
/// memset (1-byte storage) or typed broadcast (multi-byte storage).
#[inline]
unsafe fn overlay_value_into_storage<Scalar: StorageElement>(
    storage_ptr: *mut Scalar,
    storage_count: usize,
    value: Scalar,
) {
    if storage_count == 0 {
        return;
    }
    if value_is_default_bit_pattern(value) {
        return;
    }
    if core::mem::size_of::<Scalar>() == 1 {
        let value_byte = *((&value as *const Scalar) as *const u8);
        core::ptr::write_bytes(storage_ptr as *mut u8, value_byte, storage_count);
    } else {
        for storage_index in 0..storage_count {
            core::ptr::write(storage_ptr.add(storage_index), value);
        }
    }
}

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Fill<Scalar>
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn fill_zeros(&mut self) {
        let storage_count = self.numel() / Scalar::dimensions_per_value();
        if storage_count == 0 {
            return;
        }
        unsafe {
            core::ptr::write_bytes(self.data.as_ptr(), 0, storage_count);
        }
    }

    fn fill(&mut self, value: Scalar) {
        self.fill_zeros();
        let storage_count = self.numel() / Scalar::dimensions_per_value();
        unsafe {
            overlay_value_into_storage(self.data.as_ptr(), storage_count, value);
        }
    }
}

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> CopyFrom<&[Scalar]>
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn copy_from(&mut self, source: &[Scalar]) -> Result<(), TensorError> {
        let storage_count = self.storage_len();
        if source.len() != storage_count {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: storage_count,
                got: source.len(),
            });
        }
        if storage_count == 0 {
            return Ok(());
        }
        unsafe {
            core::ptr::copy_nonoverlapping(source.as_ptr(), self.data.as_ptr(), storage_count);
        }
        Ok(())
    }
}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize> Fill<Scalar>
    for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn fill_zeros(&mut self) {
        // Contiguous fast path: one write_bytes for the whole span.
        if self.is_contiguous() {
            let storage_count = self.numel().div_ceil(Scalar::dimensions_per_value());
            if storage_count == 0 {
                return;
            }
            unsafe {
                core::ptr::write_bytes(self.data, 0, storage_count);
            }
            return;
        }
        // Strided path: walk the leading axis and recurse.
        if self.ndim == 0 {
            return;
        }
        if let Ok(axis_iter) = self.axis_spans(0usize) {
            for mut sub_span in axis_iter {
                sub_span.fill_zeros();
            }
        }
    }

    fn fill(&mut self, value: Scalar) {
        if self.is_contiguous() {
            let storage_count = self.numel().div_ceil(Scalar::dimensions_per_value());
            if storage_count == 0 {
                return;
            }
            unsafe {
                core::ptr::write_bytes(self.data, 0, storage_count);
                overlay_value_into_storage(self.data, storage_count, value);
            }
            return;
        }
        if self.ndim == 0 {
            return;
        }
        if let Ok(axis_iter) = self.axis_spans(0usize) {
            for mut sub_span in axis_iter {
                sub_span.fill(value);
            }
        }
    }
}

impl<'a, 'b, Scalar: StorageElement, const MAX_RANK: usize>
    CopyFrom<&'b TensorView<'_, Scalar, MAX_RANK>> for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn copy_from(
        &mut self,
        source: &'b TensorView<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        if self.shape() != source.shape() {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: source.numel(),
                got: self.numel(),
            });
        }
        if self.is_empty() {
            return Ok(());
        }
        // Contiguous fast path: memcpy the whole storage at once.
        if self.is_contiguous() && source.is_contiguous() {
            let storage_count = self.numel().div_ceil(Scalar::dimensions_per_value());
            unsafe {
                core::ptr::copy_nonoverlapping(source.as_ptr(), self.data, storage_count);
            }
            return Ok(());
        }
        // Strided: walk the leading axis on both source and destination, recurse.
        if self.ndim == 0 {
            return Ok(());
        }
        let source_axis = source.axis_views(0usize)?;
        let mut destination_axis = self.axis_spans(0usize)?;
        for source_sub in source_axis {
            let mut destination_sub = match destination_axis.next() {
                Some(span) => span,
                None => break,
            };
            destination_sub.copy_from(&source_sub)?;
        }
        Ok(())
    }
}

// endregion: Fill / CopyFrom Implementations

// region: AxisIterator

/// Iterator over sub-tensor views along a given axis.
///
/// Each item is a `TensorView` with the iterated dimension removed (rank - 1).
/// For a rank-2 matrix, `axis_views(0)` yields row views.
pub struct AxisIterator<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    data: *const Scalar,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    axis: usize,
    axis_size: usize,
    axis_stride: isize,
    current: usize,
    _marker: PhantomData<&'a Scalar>,
}

impl<'a, Scalar, const MAX_RANK: usize> Iterator for AxisIterator<'a, Scalar, MAX_RANK> {
    type Item = TensorView<'a, Scalar, MAX_RANK>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.axis_size {
            return None;
        }
        let offset = self.current as isize * self.axis_stride;
        let sub_ptr = unsafe { (self.data as *const u8).offset(offset) as *const Scalar };

        // Build sub-shape/strides with the axis dimension removed
        let mut sub_shape = [0usize; MAX_RANK];
        let mut sub_strides = [0isize; MAX_RANK];
        let mut j = 0;
        for i in 0..self.ndim {
            if i != self.axis {
                sub_shape[j] = self.shape[i];
                sub_strides[j] = self.strides[i];
                j += 1;
            }
        }
        let sub_ndim = self.ndim - 1;

        self.current += 1;
        Some(TensorView {
            data: sub_ptr,
            shape: sub_shape,
            strides: sub_strides,
            ndim: sub_ndim,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.axis_size - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, Scalar, const MAX_RANK: usize> ExactSizeIterator for AxisIterator<'a, Scalar, MAX_RANK> {}
impl<'a, Scalar, const MAX_RANK: usize> core::iter::FusedIterator
    for AxisIterator<'a, Scalar, MAX_RANK>
{
}

/// Mutable iterator over sub-tensor spans along a given axis.
///
/// Each item is a `TensorSpan` with the iterated dimension removed (rank - 1).
/// For a rank-2 matrix, `axis_spans(0)` yields mutable row spans.
pub struct AxisIteratorMut<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    data: *mut Scalar,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    axis: usize,
    axis_size: usize,
    axis_stride: isize,
    current: usize,
    _marker: PhantomData<&'a mut Scalar>,
}

impl<'a, Scalar, const MAX_RANK: usize> Iterator for AxisIteratorMut<'a, Scalar, MAX_RANK> {
    type Item = TensorSpan<'a, Scalar, MAX_RANK>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.axis_size {
            return None;
        }
        let offset = self.current as isize * self.axis_stride;
        let sub_ptr = unsafe { (self.data as *mut u8).offset(offset) as *mut Scalar };

        let mut sub_shape = [0usize; MAX_RANK];
        let mut sub_strides = [0isize; MAX_RANK];
        let mut j = 0;
        for i in 0..self.ndim {
            if i != self.axis {
                sub_shape[j] = self.shape[i];
                sub_strides[j] = self.strides[i];
                j += 1;
            }
        }
        let sub_ndim = self.ndim - 1;

        self.current += 1;
        Some(TensorSpan {
            data: sub_ptr,
            shape: sub_shape,
            strides: sub_strides,
            ndim: sub_ndim,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.axis_size - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, Scalar, const MAX_RANK: usize> ExactSizeIterator
    for AxisIteratorMut<'a, Scalar, MAX_RANK>
{
}
impl<'a, Scalar, const MAX_RANK: usize> core::iter::FusedIterator
    for AxisIteratorMut<'a, Scalar, MAX_RANK>
{
}

impl<'a, Scalar, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Iterate along the given axis, yielding sub-tensor views with rank-1.
    pub fn axis_views<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
    ) -> Result<AxisIterator<'a, Scalar, MAX_RANK>, TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.ndim,
            });
        }
        Ok(AxisIterator {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            axis,
            axis_size: self.shape[axis],
            axis_stride: self.strides[axis],
            current: 0,
            _marker: PhantomData,
        })
    }
}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Transpose (reverse all dimensions, no data copy).
    ///
    /// Returns an error for sub-byte types with ndim >= 2, since transposing
    /// would produce non-contiguous strides that break packed element addressing.
    pub fn transpose(&self) -> Result<TensorView<'a, Scalar, MAX_RANK>, TensorError> {
        if self.ndim < 2 {
            return Ok(TensorView {
                data: self.data,
                shape: self.shape,
                strides: self.strides,
                ndim: self.ndim,
                _marker: PhantomData,
            });
        }
        if Scalar::dimensions_per_value() > 1 {
            return Err(TensorError::SubByteUnsupported);
        }
        let (shape, strides) = transpose_layout(&self.shape, &self.strides, self.ndim);
        Ok(TensorView {
            data: self.data,
            shape,
            strides,
            ndim: self.ndim,
            _marker: PhantomData,
        })
    }

    /// Reshape the view (must have same total elements, contiguous only).
    ///
    /// Returns an error for sub-byte types, since reshape would invalidate
    /// the packed element layout.
    pub fn reshape(
        &self,
        new_shape: &[usize],
    ) -> Result<TensorView<'a, Scalar, MAX_RANK>, TensorError> {
        if Scalar::dimensions_per_value() > 1 {
            return Err(TensorError::SubByteUnsupported);
        }
        let (shape, strides, ndim) = reshape_layout::<Scalar, MAX_RANK>(
            &self.shape,
            &self.strides,
            self.ndim,
            self.numel(),
            new_shape,
        )?;
        Ok(TensorView {
            data: self.data,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Flatten to 1D (requires contiguous layout).
    pub fn flatten(&self) -> Result<TensorView<'a, Scalar, MAX_RANK>, TensorError> {
        self.reshape(&[self.numel()])
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorView<'a, Scalar, MAX_RANK> {
        let (shape, strides, ndim) =
            squeeze_layout::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim);
        TensorView {
            data: self.data,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        }
    }
}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize> TensorSpan<'a, Scalar, MAX_RANK> {
    /// Transpose (reverse all dimensions, no data copy).
    ///
    /// Returns an error for sub-byte types with ndim >= 2.
    pub fn transpose(&self) -> Result<TensorSpan<'a, Scalar, MAX_RANK>, TensorError> {
        if self.ndim < 2 {
            return Ok(TensorSpan {
                data: self.data,
                shape: self.shape,
                strides: self.strides,
                ndim: self.ndim,
                _marker: PhantomData,
            });
        }
        if Scalar::dimensions_per_value() > 1 {
            return Err(TensorError::SubByteUnsupported);
        }
        let (shape, strides) = transpose_layout(&self.shape, &self.strides, self.ndim);
        Ok(TensorSpan {
            data: self.data,
            shape,
            strides,
            ndim: self.ndim,
            _marker: PhantomData,
        })
    }

    /// Reshape the span (must have same total elements, contiguous only).
    ///
    /// Returns an error for sub-byte types.
    pub fn reshape(
        &self,
        new_shape: &[usize],
    ) -> Result<TensorSpan<'a, Scalar, MAX_RANK>, TensorError> {
        if Scalar::dimensions_per_value() > 1 {
            return Err(TensorError::SubByteUnsupported);
        }
        let (shape, strides, ndim) = reshape_layout::<Scalar, MAX_RANK>(
            &self.shape,
            &self.strides,
            self.ndim,
            self.numel(),
            new_shape,
        )?;
        Ok(TensorSpan {
            data: self.data,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Flatten to 1D (requires contiguous layout).
    pub fn flatten(&self) -> Result<TensorSpan<'a, Scalar, MAX_RANK>, TensorError> {
        self.reshape(&[self.numel()])
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorSpan<'a, Scalar, MAX_RANK> {
        let (shape, strides, ndim) =
            squeeze_layout::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim);
        TensorSpan {
            data: self.data,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        }
    }
}

impl<'a, Scalar: Clone + StorageElement, const MAX_RANK: usize> TensorSpan<'a, Scalar, MAX_RANK> {
    /// Copy the span contents to a new owned Tensor.
    pub fn to_owned(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.as_view().to_owned()
    }
}

impl<'a, Scalar, const MAX_RANK: usize> TensorSpan<'a, Scalar, MAX_RANK> {
    /// Iterate mutably along the given axis, yielding sub-tensor spans with rank-1.
    pub fn axis_spans<AnyIndex: VectorIndex>(
        &mut self,
        axis: AnyIndex,
    ) -> Result<AxisIteratorMut<'a, Scalar, MAX_RANK>, TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.ndim,
            });
        }
        Ok(AxisIteratorMut {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            axis,
            axis_size: self.shape[axis],
            axis_stride: self.strides[axis],
            current: 0,
            _marker: PhantomData,
        })
    }
}

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Iterate along the given axis, yielding sub-tensor views with rank-1.
    pub fn axis_views<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
    ) -> Result<AxisIterator<'_, Scalar, MAX_RANK>, TensorError> {
        self.view().axis_views(axis)
    }

    /// Iterate mutably along the given axis, yielding sub-tensor spans with rank-1.
    pub fn axis_spans<AnyIndex: VectorIndex>(
        &mut self,
        axis: AnyIndex,
    ) -> Result<AxisIteratorMut<'_, Scalar, MAX_RANK>, TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.ndim,
            });
        }
        Ok(AxisIteratorMut {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            axis,
            axis_size: self.shape[axis],
            axis_stride: self.strides[axis],
            current: 0,
            _marker: PhantomData,
        })
    }
}

// endregion: AxisIterator

// region: Debug

impl<Scalar: StorageElement + core::fmt::Debug, Alloc: Allocator, const MAX_RANK: usize> core::fmt::Debug
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Tensor(shape={:?}, [", &self.shape[..self.ndim])?;
        let slice = self.as_slice();
        for (i, value) in slice.iter().enumerate() {
            if i >= 8 {
                write!(f, ", ...")?;
                break;
            }
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", value)?;
        }
        write!(f, "])")
    }
}

// endregion: Debug

// region: TensorViewIterator

/// Lazy element iterator over possibly non-contiguous tensor data.
///
/// Yields `(position, DimRef<'a, Scalar>)` pairs in row-major order at the logical-scalar
/// level. For sub-byte types, the innermost dimension is expanded by `dimensions_per_value`.
/// Use [`.dims()`](TensorViewIterator::dims) when only the dimension proxies are needed.
///
/// Cost per `next()` is O(1) amortized (O(ndim) only on carry propagation).
pub struct TensorViewIterator<
    'a,
    Scalar: FloatConvertible,
    const MAX_RANK: usize = DEFAULT_MAX_RANK,
> {
    data: *const Scalar,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    dims_per_value: usize,
    indices: [usize; MAX_RANK],
    remaining: usize,
    _marker: PhantomData<&'a Scalar>,
}

/// Backward-compatible alias for [`TensorViewIterator`].
pub type TensorIterator<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK> =
    TensorViewIterator<'a, Scalar, MAX_RANK>;

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> Iterator
    for TensorViewIterator<'a, Scalar, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, Scalar>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let pos = self.indices;

        // Compute byte offset: outer dims use strides directly,
        // innermost logical index splits into storage_inner / sub_index.
        let mut offset = 0isize;
        if self.ndim > 0 {
            for d in 0..self.ndim - 1 {
                offset += self.indices[d] as isize * self.strides[d];
            }
            let inner_logical = self.indices[self.ndim - 1];
            let storage_inner = inner_logical / self.dims_per_value;
            offset += storage_inner as isize * self.strides[self.ndim - 1];
            let sub_index = inner_logical % self.dims_per_value;
            let ptr = unsafe { (self.data as *const u8).offset(offset) as *const Scalar };
            let scalar = unsafe { *ptr }.unpack().as_ref()[sub_index];

            // Advance indices
            self.remaining -= 1;
            for d in (0..self.ndim).rev() {
                self.indices[d] += 1;
                if self.indices[d] < self.shape[d] {
                    break;
                }
                self.indices[d] = 0;
            }
            Some((pos, DimRef::new(scalar)))
        } else {
            // Scalar tensor (ndim == 0)
            self.remaining -= 1;
            let ptr = self.data;
            let scalar = unsafe { *ptr }.unpack().as_ref()[0];
            Some((pos, DimRef::new(scalar)))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator
    for TensorViewIterator<'a, Scalar, MAX_RANK>
{
}
impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorViewIterator<'a, Scalar, MAX_RANK>
{
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> TensorViewIterator<'a, Scalar, MAX_RANK> {
    /// Adapt this iterator to yield only dimension proxies, discarding positions.
    pub fn dims(self) -> TensorViewDims<'a, Scalar, MAX_RANK> {
        TensorViewDims { inner: self }
    }
}

/// Dimension-only adapter over [`TensorViewIterator`], yielding [`DimRef`] without positions.
///
/// Created by [`TensorViewIterator::dims()`].
pub struct TensorViewDims<'a, Scalar: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    inner: TensorViewIterator<'a, Scalar, MAX_RANK>,
}

/// Backward-compatible alias for [`TensorViewDims`].
pub type TensorDims<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK> =
    TensorViewDims<'a, Scalar, MAX_RANK>;
/// Backward-compatible alias for [`TensorViewDims`].
pub type TensorValues<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK> =
    TensorViewDims<'a, Scalar, MAX_RANK>;

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> Iterator
    for TensorViewDims<'a, Scalar, MAX_RANK>
{
    type Item = DimRef<'a, Scalar>;

    #[inline]
    fn next(&mut self) -> Option<DimRef<'a, Scalar>> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator
    for TensorViewDims<'a, Scalar, MAX_RANK>
{
}
impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorViewDims<'a, Scalar, MAX_RANK>
{
}

// endregion: TensorViewIterator

// region: TensorSpanIterator

/// Mutable element iterator over possibly non-contiguous tensor data.
///
/// Yields `(position, DimMut<'a, Scalar>)` pairs in row-major order at the logical-scalar
/// level. For sub-byte types, each proxy performs a read-modify-write on drop.
pub struct TensorSpanIterator<
    'a,
    Scalar: FloatConvertible,
    const MAX_RANK: usize = DEFAULT_MAX_RANK,
> {
    data: *mut Scalar,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    dims_per_value: usize,
    indices: [usize; MAX_RANK],
    remaining: usize,
    _marker: PhantomData<&'a mut Scalar>,
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> Iterator
    for TensorSpanIterator<'a, Scalar, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimMut<'a, Scalar>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let pos = self.indices;

        let mut offset = 0isize;
        if self.ndim > 0 {
            for d in 0..self.ndim - 1 {
                offset += self.indices[d] as isize * self.strides[d];
            }
            let inner_logical = self.indices[self.ndim - 1];
            let storage_inner = inner_logical / self.dims_per_value;
            offset += storage_inner as isize * self.strides[self.ndim - 1];
            let sub_index = inner_logical % self.dims_per_value;
            let ptr = unsafe { (self.data as *mut u8).offset(offset) as *mut Scalar };
            let scalar = unsafe { *ptr }.unpack().as_ref()[sub_index];

            self.remaining -= 1;
            for d in (0..self.ndim).rev() {
                self.indices[d] += 1;
                if self.indices[d] < self.shape[d] {
                    break;
                }
                self.indices[d] = 0;
            }
            Some((pos, unsafe { DimMut::new(ptr, sub_index, scalar) }))
        } else {
            self.remaining -= 1;
            let ptr = self.data;
            let scalar = unsafe { *ptr }.unpack().as_ref()[0];
            Some((pos, unsafe { DimMut::new(ptr, 0, scalar) }))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator
    for TensorSpanIterator<'a, Scalar, MAX_RANK>
{
}
impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorSpanIterator<'a, Scalar, MAX_RANK>
{
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> TensorSpanIterator<'a, Scalar, MAX_RANK> {
    /// Adapt this iterator to yield only mutable dimension proxies, discarding positions.
    pub fn dims(self) -> TensorSpanDims<'a, Scalar, MAX_RANK> {
        TensorSpanDims { inner: self }
    }
}

/// Mutable dimension-only adapter over [`TensorSpanIterator`], yielding [`DimMut`] without positions.
pub struct TensorSpanDims<'a, Scalar: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    inner: TensorSpanIterator<'a, Scalar, MAX_RANK>,
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> Iterator
    for TensorSpanDims<'a, Scalar, MAX_RANK>
{
    type Item = DimMut<'a, Scalar>;

    #[inline]
    fn next(&mut self) -> Option<DimMut<'a, Scalar>> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator
    for TensorSpanDims<'a, Scalar, MAX_RANK>
{
}
impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorSpanDims<'a, Scalar, MAX_RANK>
{
}

// endregion: TensorSpanIterator

// region: Tensor iter() / iter_mut() methods

/// Helper to compute the logical shape and total element count for iteration.
///
/// Shape is already logical (sub-byte types store logical dimensions in shape),
/// so this simply copies the shape and computes the product.
fn logical_shape<const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    ndim: usize,
) -> ([usize; MAX_RANK], usize) {
    let logical = *shape;
    let mut total = 1usize;
    for &dim in &logical[..ndim] {
        total *= dim;
    }
    (logical, total)
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Returns a lazy iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimRef)` pairs. Use `.iter().dims()` for just dimensions.
    /// For sub-byte types, the innermost dimension is expanded.
    pub fn iter(&self) -> TensorViewIterator<'a, Scalar, MAX_RANK> {
        let dims_per_value = Scalar::dimensions_per_value();
        let (logical, total) = logical_shape::<MAX_RANK>(&self.shape, self.ndim);
        TensorViewIterator {
            data: self.data,
            shape: logical,
            strides: self.strides,
            ndim: self.ndim,
            dims_per_value,
            indices: [0; MAX_RANK],
            remaining: total,
            _marker: PhantomData,
        }
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> TensorSpan<'a, Scalar, MAX_RANK> {
    /// Returns a lazy iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimRef)` pairs. Use `.iter().dims()` for just dimensions.
    /// For sub-byte types, the innermost dimension is expanded.
    pub fn iter(&self) -> TensorViewIterator<'_, Scalar, MAX_RANK> {
        let dims_per_value = Scalar::dimensions_per_value();
        let (logical, total) = logical_shape::<MAX_RANK>(&self.shape, self.ndim);
        TensorViewIterator {
            data: self.data,
            shape: logical,
            strides: self.strides,
            ndim: self.ndim,
            dims_per_value,
            indices: [0; MAX_RANK],
            remaining: total,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimMut)` pairs. Use `.iter_mut().dims()` for just dimensions.
    pub fn iter_mut(&mut self) -> TensorSpanIterator<'_, Scalar, MAX_RANK> {
        let dims_per_value = Scalar::dimensions_per_value();
        let (logical, total) = logical_shape::<MAX_RANK>(&self.shape, self.ndim);
        TensorSpanIterator {
            data: self.data,
            shape: logical,
            strides: self.strides,
            ndim: self.ndim,
            dims_per_value,
            indices: [0; MAX_RANK],
            remaining: total,
            _marker: PhantomData,
        }
    }
}

impl<Scalar: FloatConvertible, Alloc: Allocator, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Returns a lazy iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimRef)` pairs. Use `.iter().dims()` for just dimensions.
    pub fn iter(&self) -> TensorViewIterator<'_, Scalar, MAX_RANK> {
        self.view().iter()
    }

    /// Returns a mutable iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimMut)` pairs. Use `.iter_mut().dims()` for just dimensions.
    pub fn iter_mut(&mut self) -> TensorSpanIterator<'_, Scalar, MAX_RANK> {
        let dims_per_value = Scalar::dimensions_per_value();
        let (logical, total) = logical_shape::<MAX_RANK>(&self.shape, self.ndim);
        TensorSpanIterator {
            data: self.data.as_ptr(),
            shape: logical,
            strides: self.strides,
            ndim: self.ndim,
            dims_per_value,
            indices: [0; MAX_RANK],
            remaining: total,
            _marker: PhantomData,
        }
    }
}

// endregion: Tensor iter() / iter_mut() methods

// region: IntoIterator (immutable)

impl<'a, Scalar: FloatConvertible, Alloc: Allocator, const MAX_RANK: usize> IntoIterator
    for &'a Tensor<Scalar, Alloc, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, Scalar>);
    type IntoIter = TensorViewIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> IntoIterator
    for &'a TensorView<'a, Scalar, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, Scalar>);
    type IntoIter = TensorViewIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> IntoIterator
    for &'a TensorSpan<'a, Scalar, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, Scalar>);
    type IntoIter = TensorViewIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// endregion: IntoIterator (immutable)

// region: IntoIterator (mutable)

impl<'a, Scalar: FloatConvertible, Alloc: Allocator, const MAX_RANK: usize> IntoIterator
    for &'a mut Tensor<Scalar, Alloc, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimMut<'a, Scalar>);
    type IntoIter = TensorSpanIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> IntoIterator
    for &'a mut TensorSpan<'a, Scalar, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimMut<'a, Scalar>);
    type IntoIter = TensorSpanIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// endregion: IntoIterator (mutable)

// region: PartialEq

impl<Scalar: StorageElement + PartialEq, Alloc: Allocator, const MAX_RANK: usize> PartialEq
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim
            && self.shape[..self.ndim] == other.shape[..other.ndim]
            && self.as_slice() == other.as_slice()
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> PartialEq
    for TensorView<'a, Scalar, MAX_RANK>
where
    Scalar::DimScalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim
            && self.shape[..self.ndim] == other.shape[..other.ndim]
            && self
                .iter()
                .dims()
                .zip(other.iter().dims())
                .all(|(a, b)| *a == *b)
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> PartialEq
    for TensorSpan<'a, Scalar, MAX_RANK>
where
    Scalar::DimScalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim
            && self.shape[..self.ndim] == other.shape[..other.ndim]
            && self
                .iter()
                .dims()
                .zip(other.iter().dims())
                .all(|(a, b)| *a == *b)
    }
}

// endregion: PartialEq

// endregion: Tolerance Equality (AllCloseOps moved to crate::each)

// region: AsRef

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> AsRef<[Scalar]>
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn as_ref(&self) -> &[Scalar] {
        self.as_slice()
    }
}

// endregion: AsRef

// region: Tensor View and Slice Methods

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Create a view of the entire array.
    pub fn view(&self) -> TensorView<'_, Scalar, MAX_RANK> {
        TensorView {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Create a mutable span of the entire tensor.
    ///
    /// The returned [`TensorSpan`] borrows `self` exclusively for `'_`. Use it
    /// to apply in-place kernels, write to individual elements via
    /// [`TensorSpan::try_flat_mut`] / [`TensorSpan::try_coords_mut`], or pass
    /// it to APIs that need `&mut (impl TensorMut<_, _>)`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use numkong::tensor::Tensor;
    ///
    /// let mut t = Tensor::<f32>::try_full(&[2, 2], 0.0).unwrap();
    /// let mut span = t.span();
    /// *span.try_flat_mut(0_usize).unwrap() = 42.0;
    /// assert_eq!(*span.try_flat(0_usize).unwrap(), 42.0);
    /// ```
    pub fn span(&mut self) -> TensorSpan<'_, Scalar, MAX_RANK> {
        TensorSpan {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Try to get an element by flat logical row-major index.
    pub fn try_flat<AnyIndex: VectorIndex>(&self, index: AnyIndex) -> Result<&Scalar, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data.as_ptr() as *const u8).offset(offset) as *const Scalar) })
    }

    /// Try to get a mutable element by flat logical row-major index.
    pub fn try_flat_mut<AnyIndex: VectorIndex>(
        &mut self,
        index: AnyIndex,
    ) -> Result<&mut Scalar, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &mut *((self.data.as_ptr() as *mut u8).offset(offset) as *mut Scalar) })
    }

    /// Try to get an element by exact coordinates.
    pub fn try_coords<C: TensorCoordinates>(&self, coords: C) -> Result<&Scalar, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &*((self.data.as_ptr() as *const u8).offset(offset) as *const Scalar) })
    }

    /// Try to get a mutable element by exact coordinates.
    pub fn try_coords_mut<C: TensorCoordinates>(
        &mut self,
        coords: C,
    ) -> Result<&mut Scalar, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &mut *((self.data.as_ptr() as *mut u8).offset(offset) as *mut Scalar) })
    }

    /// Try to access the scalar value of a rank-0 tensor.
    pub fn try_scalar(&self) -> Result<&Scalar, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &*self.data.as_ptr() })
    }

    /// Try to access the mutable scalar value of a rank-0 tensor.
    pub fn try_scalar_mut(&mut self) -> Result<&mut Scalar, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &mut *self.data.as_ptr() })
    }

    /// Slice the leading axis by one index, reducing rank by one.
    pub fn slice_leading<AnyIndex: VectorIndex>(
        &self,
        index: AnyIndex,
    ) -> Result<TensorView<'_, Scalar, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data.as_ptr() as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the leading axis mutably by one index, reducing rank by one.
    pub fn slice_leading_mut<AnyIndex: VectorIndex>(
        &mut self,
        index: AnyIndex,
    ) -> Result<TensorSpan<'_, Scalar, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorSpan {
            data: unsafe { (self.data.as_ptr() as *mut u8).offset(offset) as *mut Scalar },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the array along multiple dimensions, returning a zero-copy view.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`. Scalar arguments
    /// reduce the rank, range arguments preserve it, and [`RangeStep`] permits
    /// stepped sampling.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use numkong::tensor::{Tensor, SliceRange};
    ///
    /// let arr = Tensor::<f32>::try_full(&[4, 5], 1.0).unwrap();
    ///
    /// // Tuple syntax — preferred
    /// let block = arr.slice((0..2_usize, ..)).unwrap();     // t[0:2, :]
    /// let row   = arr.slice((1_usize, ..)).unwrap();        // t[1, :]
    /// assert_eq!(block.shape(), &[2, 5]);
    /// assert_eq!(row.shape(),   &[5]);
    ///
    /// // Enum-based syntax still works for programmatic construction
    /// let same_block = arr.slice(&[SliceRange::range(0, 2), SliceRange::full()]).unwrap();
    /// assert_eq!(block.shape(), same_block.shape());
    /// ```
    pub fn slice(
        &self,
        spec: impl SliceSpec,
    ) -> Result<TensorView<'_, Scalar, MAX_RANK>, TensorError> {
        self.view().slice(spec)
    }

    /// Slice the array mutably along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`.
    pub fn slice_mut(
        &mut self,
        spec: impl SliceSpec,
    ) -> Result<TensorSpan<'_, Scalar, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim)?;
        Ok(TensorSpan {
            data: unsafe { (self.data.as_ptr() as *mut u8).offset(offset) as *mut Scalar },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }
}

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize>
    Tensor<Scalar, Alloc, MAX_RANK>
{
    /// Transpose (reverse all dimensions, no data copy).
    pub fn transpose(&self) -> Result<TensorView<'_, Scalar, MAX_RANK>, TensorError> {
        self.view().transpose()
    }

    /// Reshape the array (must have same total elements, contiguous only).
    pub fn reshape(
        &self,
        new_shape: &[usize],
    ) -> Result<TensorView<'_, Scalar, MAX_RANK>, TensorError> {
        self.view().reshape(new_shape)
    }

    /// Flatten to 1D (requires contiguous layout).
    pub fn flatten(&self) -> Result<TensorView<'_, Scalar, MAX_RANK>, TensorError> {
        self.view().flatten()
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorView<'_, Scalar, MAX_RANK> {
        self.view().squeeze()
    }
}

// endregion: Tensor View and Slice Methods

impl<'a, AnyIndex: VectorIndex, Scalar, const MAX_RANK: usize> Index<AnyIndex>
    for TensorView<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: AnyIndex) -> &Self::Output {
        self.try_flat(index).expect("view index out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, Scalar, const MAX_RANK: usize> Index<(I0, I1)>
    for TensorView<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, Scalar, const MAX_RANK: usize>
    Index<(I0, I1, I2)> for TensorView<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3)> for TensorView<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4)> for TensorView<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5)> for TensorView<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6)> for TensorView<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6, I7)> for TensorView<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<'a, AnyIndex: VectorIndex, Scalar, const MAX_RANK: usize> Index<AnyIndex>
    for TensorSpan<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: AnyIndex) -> &Self::Output {
        self.try_flat(index).expect("span index out of bounds")
    }
}

impl<'a, AnyIndex: VectorIndex, Scalar, const MAX_RANK: usize> IndexMut<AnyIndex>
    for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn index_mut(&mut self, index: AnyIndex) -> &mut Self::Output {
        self.try_flat_mut(index).expect("span index out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, Scalar, const MAX_RANK: usize> Index<(I0, I1)>
    for TensorSpan<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, Scalar, const MAX_RANK: usize> IndexMut<(I0, I1)>
    for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, Scalar, const MAX_RANK: usize>
    Index<(I0, I1, I2)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, Scalar, const MAX_RANK: usize>
    IndexMut<(I0, I1, I2)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5, I6)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6, I7)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        Scalar: StorageElement,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5, I6, I7)> for TensorSpan<'a, Scalar, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<AnyIndex: VectorIndex, Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Index<AnyIndex>
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: AnyIndex) -> &Self::Output {
        self.try_flat(index).expect("tensor index out of bounds")
    }
}

impl<AnyIndex: VectorIndex, Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> IndexMut<AnyIndex>
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn index_mut(&mut self, index: AnyIndex) -> &mut Self::Output {
        self.try_flat_mut(index)
            .expect("tensor index out of bounds")
    }
}

impl<I0: VectorIndex, I1: VectorIndex, Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize>
    Index<(I0, I1)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<I0: VectorIndex, I1: VectorIndex, Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize>
    IndexMut<(I0, I1)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5, I6)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6, I7)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    type Output = Scalar;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        Scalar: StorageElement,
        Alloc: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5, I6, I7)> for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

// region: Type Aliases

/// Type alias for a 2D matrix (Tensor with MAX_RANK=2).
pub type Matrix<Scalar, Alloc = Global> = Tensor<Scalar, Alloc, 2>;

/// Type alias for an immutable 2D matrix view.
pub type MatrixView<'a, Scalar> = TensorView<'a, Scalar, 2>;

/// Type alias for a mutable 2D matrix view.
pub type MatrixSpan<'a, Scalar> = TensorSpan<'a, Scalar, 2>;

// endregion: Type Aliases

// region: Tensor Internal Helpers

#[inline]
fn validate_same_shape(left: &[usize], right: &[usize]) -> Result<(), TensorError> {
    if left == right {
        return Ok(());
    }
    if left.len() != right.len() {
        return Err(TensorError::DimensionMismatch {
            expected: left.len(),
            got: right.len(),
        });
    }
    for (i, (&l, &r)) in left.iter().zip(right).enumerate() {
        if l != r {
            return Err(TensorError::ShapeMismatch {
                axis: i,
                expected: l,
                got: r,
            });
        }
    }
    Ok(())
}

#[inline]
fn normalize_axis<AnyIndex: VectorIndex>(
    axis: AnyIndex,
    ndim: usize,
) -> Result<usize, TensorError> {
    if ndim == 0 {
        return Err(TensorError::IndexOutOfBounds {
            index: 0,
            size: ndim,
        });
    }
    axis.resolve(ndim).ok_or(TensorError::IndexOutOfBounds {
        index: 0,
        size: ndim,
    })
}

#[inline]
fn compute_strides_into_<Scalar, const MAX_RANK: usize>(
    shape: &[usize],
    strides: &mut [isize; MAX_RANK],
) {
    let elem_size = core::mem::size_of::<Scalar>();
    if shape.is_empty() {
        return;
    }

    let mut stride = elem_size as isize;
    for dim in (0..shape.len()).rev() {
        strides[dim] = stride;
        stride *= shape[dim] as isize;
    }
}

/// Compute the transposed layout (reverse all dimensions).
fn transpose_layout<const R: usize>(
    shape: &[usize; R],
    strides: &[isize; R],
    ndim: usize,
) -> ([usize; R], [isize; R]) {
    let mut new_shape = [0usize; R];
    let mut new_strides = [0isize; R];
    for i in 0..ndim {
        new_shape[i] = shape[ndim - 1 - i];
        new_strides[i] = strides[ndim - 1 - i];
    }
    (new_shape, new_strides)
}

/// Compute the reshaped layout (same total elements, contiguous only).
fn reshape_layout<Scalar, const R: usize>(
    shape: &[usize; R],
    strides: &[isize; R],
    ndim: usize,
    len: usize,
    new_shape: &[usize],
) -> Result<([usize; R], [isize; R], usize), TensorError> {
    if new_shape.len() > R {
        return Err(TensorError::TooManyRanks {
            got: new_shape.len(),
        });
    }
    let new_len: usize = new_shape.iter().product();
    if new_len != len {
        return Err(TensorError::ShapeMismatch {
            axis: 0,
            expected: new_len,
            got: len,
        });
    }
    // Check contiguous
    let elem_size = core::mem::size_of::<Scalar>() as isize;
    let mut expected = elem_size;
    for i in (0..ndim).rev() {
        if strides[i] != expected {
            return Err(TensorError::NonContiguousRows);
        }
        expected *= shape[i] as isize;
    }
    let mut shape_arr = [0usize; R];
    shape_arr[..new_shape.len()].copy_from_slice(new_shape);
    let mut strides_arr = [0isize; R];
    compute_strides_into_::<Scalar, R>(new_shape, &mut strides_arr);
    Ok((shape_arr, strides_arr, new_shape.len()))
}

/// Compute the squeezed layout (remove dimensions of size 1).
fn squeeze_layout<Scalar, const R: usize>(
    shape: &[usize; R],
    strides: &[isize; R],
    ndim: usize,
) -> ([usize; R], [isize; R], usize) {
    let mut new_shape = [0usize; R];
    let mut new_strides = [0isize; R];
    let mut new_ndim = 0;
    for i in 0..ndim {
        if shape[i] != 1 {
            new_shape[new_ndim] = shape[i];
            new_strides[new_ndim] = strides[i];
            new_ndim += 1;
        }
    }
    if new_ndim == 0 {
        new_ndim = 1;
        new_shape[0] = 1;
        new_strides[0] = core::mem::size_of::<Scalar>() as isize;
    }
    (new_shape, new_strides, new_ndim)
}

#[inline]
fn resolve_index_for_size_<AnyIndex: VectorIndex>(
    index: AnyIndex,
    size: usize,
) -> Result<usize, TensorError> {
    index
        .resolve(size)
        .ok_or(TensorError::IndexOutOfBounds { index: 0, size })
}

#[inline]
fn offset_from_coords_<const MAX_RANK: usize>(
    strides: &[isize; MAX_RANK],
    coords: &[usize; MAX_RANK],
    ndim: usize,
) -> isize {
    let mut offset = 0isize;
    for dim in 0..ndim {
        offset += coords[dim] as isize * strides[dim];
    }
    offset
}

#[inline]
fn offset_from_flat_<const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    strides: &[isize; MAX_RANK],
    ndim: usize,
    mut flat_index: usize,
) -> isize {
    let mut offset = 0isize;
    for dim in (0..ndim).rev() {
        let dim_size = shape[dim];
        let coord = flat_index % dim_size;
        flat_index /= dim_size;
        offset += coord as isize * strides[dim];
    }
    offset
}

fn slice_layout_<const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    strides: &[isize; MAX_RANK],
    ndim: usize,
    ranges: &[SliceRange],
) -> LayoutResult<MAX_RANK> {
    if ranges.len() != ndim {
        return Err(TensorError::DimensionMismatch {
            expected: ndim,
            got: ranges.len(),
        });
    }

    let mut new_shape = [0usize; MAX_RANK];
    let mut new_strides = [0isize; MAX_RANK];
    let mut new_ndim = 0usize;
    let mut offset = 0isize;

    for (dim, range) in ranges.iter().enumerate() {
        let dim_size = shape[dim];
        let dim_stride = strides[dim];

        match *range {
            SliceRange::Full => {
                new_shape[new_ndim] = dim_size;
                new_strides[new_ndim] = dim_stride;
                new_ndim += 1;
            }
            SliceRange::Index(index) => {
                if index >= dim_size {
                    return Err(TensorError::IndexOutOfBounds {
                        index,
                        size: dim_size,
                    });
                }
                offset += index as isize * dim_stride;
            }
            SliceRange::Range { start, end } => {
                if start > end || end > dim_size {
                    return Err(TensorError::IndexOutOfBounds {
                        index: end,
                        size: dim_size,
                    });
                }
                new_shape[new_ndim] = end - start;
                new_strides[new_ndim] = dim_stride;
                new_ndim += 1;
                offset += start as isize * dim_stride;
            }
            SliceRange::RangeStep { start, end, step } => {
                if start >= dim_size || (end > dim_size && step > 0) {
                    return Err(TensorError::IndexOutOfBounds {
                        index: if start >= dim_size { start } else { end },
                        size: dim_size,
                    });
                }
                if step == 0 {
                    return Err(TensorError::InvalidShape {
                        axis: dim,
                        size: 0,
                        reason: "step cannot be zero",
                    });
                }
                let count = if step > 0 {
                    (end.saturating_sub(start) + step as usize - 1) / step as usize
                } else {
                    let abs_step = (-step) as usize;
                    (start.saturating_sub(end) + abs_step - 1) / abs_step
                };
                new_shape[new_ndim] = count;
                new_strides[new_ndim] = dim_stride * step;
                new_ndim += 1;
                offset += start as isize * dim_stride;
            }
        }
    }

    let new_len = if new_ndim == 0 {
        1
    } else {
        new_shape[..new_ndim].iter().product()
    };
    Ok((new_shape, new_strides, new_ndim, offset, new_len))
}

fn slice_leading_layout_<AnyIndex: VectorIndex, const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    strides: &[isize; MAX_RANK],
    ndim: usize,
    index: AnyIndex,
) -> LayoutResult<MAX_RANK> {
    if ndim == 0 {
        return Err(TensorError::IndexOutOfBounds { index: 0, size: 0 });
    }

    let leading = resolve_index_for_size_(index, shape[0])?;
    let mut new_shape = [0usize; MAX_RANK];
    let mut new_strides = [0isize; MAX_RANK];
    new_shape[..ndim - 1].copy_from_slice(&shape[1..ndim]);
    new_strides[..ndim - 1].copy_from_slice(&strides[1..ndim]);
    let new_ndim = ndim - 1;
    let offset = leading as isize * strides[0];
    let new_len = if new_ndim == 0 {
        1
    } else {
        new_shape[..new_ndim].iter().product()
    };
    Ok((new_shape, new_strides, new_ndim, offset, new_len))
}

fn reduced_shape(shape: &[usize], axis: usize, keep_dims: bool) -> Vec<usize> {
    let mut result = Vec::with_capacity(if keep_dims {
        shape.len()
    } else {
        shape.len() - 1
    });
    for (dim_index, &dim_size) in shape.iter().enumerate() {
        if dim_index == axis {
            if keep_dims {
                result.push(1);
            }
        } else {
            result.push(dim_size);
        }
    }
    result
}

fn shared_contiguous_tail_2(
    shape: &[usize],
    first_strides: &[isize],
    first_item_size: isize,
    second_strides: &[isize],
    second_item_size: isize,
) -> usize {
    let mut tail_dims = 0usize;
    let mut expected_first = first_item_size;
    let mut expected_second = second_item_size;
    for dim_index in (0..shape.len()).rev() {
        if first_strides[dim_index] == expected_first
            && second_strides[dim_index] == expected_second
        {
            tail_dims += 1;
            let dim_extent = shape[dim_index] as isize;
            expected_first = expected_first.saturating_mul(dim_extent);
            expected_second = expected_second.saturating_mul(dim_extent);
        } else {
            break;
        }
    }
    tail_dims
}

fn shared_contiguous_tail_3(
    shape: &[usize],
    first_strides: &[isize],
    first_item_size: isize,
    second_strides: &[isize],
    second_item_size: isize,
    third_strides: &[isize],
    third_item_size: isize,
) -> usize {
    let mut tail_dims = 0usize;
    let mut expected_first = first_item_size;
    let mut expected_second = second_item_size;
    let mut expected_third = third_item_size;
    for dim_index in (0..shape.len()).rev() {
        if first_strides[dim_index] == expected_first
            && second_strides[dim_index] == expected_second
            && third_strides[dim_index] == expected_third
        {
            tail_dims += 1;
            let dim_extent = shape[dim_index] as isize;
            expected_first = expected_first.saturating_mul(dim_extent);
            expected_second = expected_second.saturating_mul(dim_extent);
            expected_third = expected_third.saturating_mul(dim_extent);
        } else {
            break;
        }
    }
    tail_dims
}

fn shared_contiguous_tail_4(
    shape: &[usize],
    first_strides: &[isize],
    first_item_size: isize,
    second_strides: &[isize],
    second_item_size: isize,
    third_strides: &[isize],
    third_item_size: isize,
    fourth_strides: &[isize],
    fourth_item_size: isize,
) -> usize {
    let mut tail_dims = 0usize;
    let mut expected_first = first_item_size;
    let mut expected_second = second_item_size;
    let mut expected_third = third_item_size;
    let mut expected_fourth = fourth_item_size;
    for dim_index in (0..shape.len()).rev() {
        if first_strides[dim_index] == expected_first
            && second_strides[dim_index] == expected_second
            && third_strides[dim_index] == expected_third
            && fourth_strides[dim_index] == expected_fourth
        {
            tail_dims += 1;
            let dim_extent = shape[dim_index] as isize;
            expected_first = expected_first.saturating_mul(dim_extent);
            expected_second = expected_second.saturating_mul(dim_extent);
            expected_third = expected_third.saturating_mul(dim_extent);
            expected_fourth = expected_fourth.saturating_mul(dim_extent);
        } else {
            break;
        }
    }
    tail_dims
}

unsafe fn walk_contiguous_blocks_2<TIn, TOut, Kernel>(
    source_ptr: *const TIn,
    source_strides: &[isize],
    target_ptr: *mut TOut,
    target_strides: &[isize],
    shape: &[usize],
    mut kernel: Kernel,
) where
    Kernel: FnMut(*const TIn, *mut TOut, usize),
{
    let tail_dims = shared_contiguous_tail_2(
        shape,
        source_strides,
        core::mem::size_of::<TIn>() as isize,
        target_strides,
        core::mem::size_of::<TOut>() as isize,
    );
    let tail_len = if tail_dims == 0 {
        1
    } else {
        shape[shape.len() - tail_dims..].iter().product()
    };
    let outer_dims = shape.len().saturating_sub(tail_dims);

    unsafe fn recurse<TIn, TOut, Kernel>(
        dim_index: usize,
        outer_dims: usize,
        source_ptr: *const u8,
        source_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut Kernel,
    ) where
        Kernel: FnMut(*const TIn, *mut TOut, usize),
    {
        if dim_index == outer_dims {
            kernel(source_ptr as *const TIn, target_ptr as *mut TOut, tail_len);
            return;
        }
        for offset_index in 0..shape[dim_index] {
            let source_child = source_ptr.offset(offset_index as isize * source_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            recurse::<TIn, TOut, Kernel>(
                dim_index + 1,
                outer_dims,
                source_child,
                source_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            );
        }
    }

    recurse::<TIn, TOut, Kernel>(
        0,
        outer_dims,
        source_ptr as *const u8,
        source_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    );
}

unsafe fn walk_contiguous_blocks_3<TFirst, TSecond, TOut, Kernel>(
    first_ptr: *const TFirst,
    first_strides: &[isize],
    second_ptr: *const TSecond,
    second_strides: &[isize],
    target_ptr: *mut TOut,
    target_strides: &[isize],
    shape: &[usize],
    mut kernel: Kernel,
) where
    Kernel: FnMut(*const TFirst, *const TSecond, *mut TOut, usize),
{
    let tail_dims = shared_contiguous_tail_3(
        shape,
        first_strides,
        core::mem::size_of::<TFirst>() as isize,
        second_strides,
        core::mem::size_of::<TSecond>() as isize,
        target_strides,
        core::mem::size_of::<TOut>() as isize,
    );
    let tail_len = if tail_dims == 0 {
        1
    } else {
        shape[shape.len() - tail_dims..].iter().product()
    };
    let outer_dims = shape.len().saturating_sub(tail_dims);

    unsafe fn recurse<TFirst, TSecond, TOut, Kernel>(
        dim_index: usize,
        outer_dims: usize,
        first_ptr: *const u8,
        first_strides: &[isize],
        second_ptr: *const u8,
        second_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut Kernel,
    ) where
        Kernel: FnMut(*const TFirst, *const TSecond, *mut TOut, usize),
    {
        if dim_index == outer_dims {
            kernel(
                first_ptr as *const TFirst,
                second_ptr as *const TSecond,
                target_ptr as *mut TOut,
                tail_len,
            );
            return;
        }
        for offset_index in 0..shape[dim_index] {
            let first_child = first_ptr.offset(offset_index as isize * first_strides[dim_index]);
            let second_child = second_ptr.offset(offset_index as isize * second_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            recurse::<TFirst, TSecond, TOut, Kernel>(
                dim_index + 1,
                outer_dims,
                first_child,
                first_strides,
                second_child,
                second_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            );
        }
    }

    recurse::<TFirst, TSecond, TOut, Kernel>(
        0,
        outer_dims,
        first_ptr as *const u8,
        first_strides,
        second_ptr as *const u8,
        second_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    );
}

unsafe fn walk_contiguous_blocks_4<TFirst, TSecond, TThird, TOut, Kernel>(
    first_ptr: *const TFirst,
    first_strides: &[isize],
    second_ptr: *const TSecond,
    second_strides: &[isize],
    third_ptr: *const TThird,
    third_strides: &[isize],
    target_ptr: *mut TOut,
    target_strides: &[isize],
    shape: &[usize],
    mut kernel: Kernel,
) where
    Kernel: FnMut(*const TFirst, *const TSecond, *const TThird, *mut TOut, usize),
{
    let tail_dims = shared_contiguous_tail_4(
        shape,
        first_strides,
        core::mem::size_of::<TFirst>() as isize,
        second_strides,
        core::mem::size_of::<TSecond>() as isize,
        third_strides,
        core::mem::size_of::<TThird>() as isize,
        target_strides,
        core::mem::size_of::<TOut>() as isize,
    );
    let tail_len = if tail_dims == 0 {
        1
    } else {
        shape[shape.len() - tail_dims..].iter().product()
    };
    let outer_dims = shape.len().saturating_sub(tail_dims);

    unsafe fn recurse<TFirst, TSecond, TThird, TOut, Kernel>(
        dim_index: usize,
        outer_dims: usize,
        first_ptr: *const u8,
        first_strides: &[isize],
        second_ptr: *const u8,
        second_strides: &[isize],
        third_ptr: *const u8,
        third_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut Kernel,
    ) where
        Kernel: FnMut(*const TFirst, *const TSecond, *const TThird, *mut TOut, usize),
    {
        if dim_index == outer_dims {
            kernel(
                first_ptr as *const TFirst,
                second_ptr as *const TSecond,
                third_ptr as *const TThird,
                target_ptr as *mut TOut,
                tail_len,
            );
            return;
        }
        for offset_index in 0..shape[dim_index] {
            let first_child = first_ptr.offset(offset_index as isize * first_strides[dim_index]);
            let second_child = second_ptr.offset(offset_index as isize * second_strides[dim_index]);
            let third_child = third_ptr.offset(offset_index as isize * third_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            recurse::<TFirst, TSecond, TThird, TOut, Kernel>(
                dim_index + 1,
                outer_dims,
                first_child,
                first_strides,
                second_child,
                second_strides,
                third_child,
                third_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            );
        }
    }

    recurse::<TFirst, TSecond, TThird, TOut, Kernel>(
        0,
        outer_dims,
        first_ptr as *const u8,
        first_strides,
        second_ptr as *const u8,
        second_strides,
        third_ptr as *const u8,
        third_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    );
}

fn for_each_axis_lane<Scalar, const MAX_RANK: usize, Kernel>(
    view: &TensorView<'_, Scalar, MAX_RANK>,
    axis: usize,
    mut callback: Kernel,
) where
    Kernel: FnMut(*const Scalar, usize, isize, usize),
{
    let lane_len = view.shape[axis];
    let lane_stride = view.strides[axis];
    let mut other_dims = [0usize; MAX_RANK];
    let mut other_ndim = 0usize;
    for dim_index in 0..view.ndim {
        if dim_index != axis {
            other_dims[other_ndim] = dim_index;
            other_ndim += 1;
        }
    }

    if other_ndim == 0 {
        callback(view.data, lane_len, lane_stride, 0);
        return;
    }

    let total_lanes: usize = other_dims[..other_ndim]
        .iter()
        .map(|&dim_index| view.shape[dim_index])
        .product();

    // Fast path: when non-axis dims form a uniform-stride progression, iterate with a constant
    // byte increment instead of recomputing offsets from multi-dimensional coordinates.
    let mut other_strides = [0isize; MAX_RANK];
    let mut other_extents = [0usize; MAX_RANK];
    for i in 0..other_ndim {
        other_strides[i] = view.strides[other_dims[i]];
        other_extents[i] = view.shape[other_dims[i]];
    }
    let (other_tail, _other_count, _other_stride) =
        uniform_stride_tail(&other_extents[..other_ndim], &other_strides[..other_ndim]);
    if other_tail == other_ndim {
        let byte_increment = other_strides[other_ndim - 1];
        let mut ptr = view.data as *const u8;
        for lane_index in 0..total_lanes {
            let lane_ptr = ptr as *const Scalar;
            callback(lane_ptr, lane_len, lane_stride, lane_index);
            ptr = unsafe { ptr.offset(byte_increment) };
        }
        return;
    }

    let mut coords = [0usize; MAX_RANK];
    for lane_index in 0..total_lanes {
        let mut lane_offset = 0isize;
        for index in 0..other_ndim {
            let dim_index = other_dims[index];
            lane_offset += coords[index] as isize * view.strides[dim_index];
        }
        let lane_ptr = unsafe { (view.data as *const u8).offset(lane_offset) as *const Scalar };
        callback(lane_ptr, lane_len, lane_stride, lane_index);

        for index in (0..other_ndim).rev() {
            coords[index] += 1;
            if coords[index] < view.shape[other_dims[index]] {
                break;
            }
            coords[index] = 0;
        }
    }
}

/// Detect trailing dimensions where `stride[i] == stride[i+1] * shape[i+1]`.
/// Returns `(tail_dims, element_count, abs_stride_bytes)`.
fn uniform_stride_tail(shape: &[usize], strides: &[isize]) -> (usize, usize, usize) {
    let rank = shape.len();
    if rank == 0 {
        return (0, 1, 0);
    }
    let mut tail: usize = 1;
    let innermost_stride = strides[rank - 1];
    let mut expected_stride = innermost_stride;
    for i in (0..rank - 1).rev() {
        expected_stride *= shape[i + 1] as isize;
        if strides[i] != expected_stride {
            break;
        }
        tail += 1;
    }
    let count: usize = shape[rank - tail..].iter().product();
    let stride_magnitude = innermost_stride.unsigned_abs();
    (tail, count, stride_magnitude)
}

unsafe fn normalize_reduction_lane<Scalar>(
    lane_ptr: *const Scalar,
    lane_len: usize,
    lane_stride: isize,
) -> (*const Scalar, usize, usize, bool) {
    if lane_len == 0 {
        return (lane_ptr, 0, core::mem::size_of::<Scalar>(), false);
    }
    if lane_stride >= 0 {
        return (lane_ptr, lane_len, lane_stride as usize, false);
    }
    let last_ptr =
        (lane_ptr as *const u8).offset((lane_len as isize - 1) * lane_stride) as *const Scalar;
    (last_ptr, lane_len, (-lane_stride) as usize, true)
}

/// Build collapsed shape/strides arrays by merging `tail_dims` trailing dimensions into one.
/// Returns the collapsed rank.
fn build_collapsed_layout(
    shape: &[usize],
    strides: &[isize],
    tail_dims: usize,
    collapsed_count: usize,
    collapsed_shape: &mut [usize],
    collapsed_strides: &mut [isize],
) -> usize {
    let collapsed_rank = shape.len() - tail_dims + 1;
    collapsed_shape[..collapsed_rank - 1].copy_from_slice(&shape[..collapsed_rank - 1]);
    collapsed_strides[..collapsed_rank - 1].copy_from_slice(&strides[..collapsed_rank - 1]);
    collapsed_shape[collapsed_rank - 1] = collapsed_count;
    collapsed_strides[collapsed_rank - 1] = strides[shape.len() - 1];
    collapsed_rank
}

unsafe fn reduce_moments_recursive<Scalar>(
    data: *const Scalar,
    shape: &[usize],
    strides: &[isize],
) -> (Scalar::SumOutput, Scalar::SumSqOutput)
where
    Scalar: ReduceMoments,
    Scalar::SumOutput: Default + core::ops::AddAssign,
    Scalar::SumSqOutput: Default + core::ops::AddAssign,
{
    if shape.is_empty() {
        return Scalar::reduce_moments_raw(data, 1, core::mem::size_of::<Scalar>());
    }
    if shape[0] == 0 {
        return (Scalar::SumOutput::default(), Scalar::SumSqOutput::default());
    }
    // Re-analyze remaining dimensions for collapsibility at each recursive level.
    if shape.len() >= 2 {
        let (tail, count, _stride_magnitude) = uniform_stride_tail(shape, strides);
        if tail == shape.len() {
            let (ptr, count, stride, _) =
                normalize_reduction_lane(data, count, strides[shape.len() - 1]);
            return Scalar::reduce_moments_raw(ptr, count, stride);
        }
        if tail >= 2 {
            let mut collapsed_shape = [0usize; DEFAULT_MAX_RANK];
            let mut collapsed_strides = [0isize; DEFAULT_MAX_RANK];
            let collapsed_rank = build_collapsed_layout(
                shape,
                strides,
                tail,
                count,
                &mut collapsed_shape,
                &mut collapsed_strides,
            );
            return reduce_moments_recursive::<Scalar>(
                data,
                &collapsed_shape[..collapsed_rank],
                &collapsed_strides[..collapsed_rank],
            );
        }
    }
    if shape.len() == 1 {
        let (lane_ptr, lane_len, lane_stride, _) =
            normalize_reduction_lane(data, shape[0], strides[0]);
        return Scalar::reduce_moments_raw(lane_ptr, lane_len, lane_stride);
    }

    let mut sum = Scalar::SumOutput::default();
    let mut sumsq = Scalar::SumSqOutput::default();
    for index in 0..shape[0] {
        let child_ptr = (data as *const u8).offset(index as isize * strides[0]) as *const Scalar;
        let (child_sum, child_sumsq) =
            reduce_moments_recursive::<Scalar>(child_ptr, &shape[1..], &strides[1..]);
        sum += child_sum;
        sumsq += child_sumsq;
    }
    (sum, sumsq)
}

unsafe fn reduce_minmax_recursive<Scalar>(
    data: *const Scalar,
    shape: &[usize],
    strides: &[isize],
    logical_offset: usize,
) -> Option<MinMaxResult<Scalar::Output>>
where
    Scalar: ReduceMinMax,
    Scalar::Output: Clone + PartialOrd,
{
    if shape.is_empty() {
        return Scalar::reduce_minmax_raw(data, 1, core::mem::size_of::<Scalar>()).map(
            |(min_value, _, max_value, _)| MinMaxResult {
                min_value,
                min_index: logical_offset,
                max_value,
                max_index: logical_offset,
            },
        );
    }
    if shape[0] == 0 {
        return None;
    }
    // Re-analyze remaining dimensions for collapsibility at each recursive level.
    if shape.len() >= 2 {
        let (tail, count, _stride_magnitude) = uniform_stride_tail(shape, strides);
        if tail == shape.len() {
            let (lane_ptr, lane_len, lane_stride, reversed) =
                normalize_reduction_lane(data, count, strides[shape.len() - 1]);
            return Scalar::reduce_minmax_raw(lane_ptr, lane_len, lane_stride).map(
                |(min_value, min_index, max_value, max_index)| {
                    let min_index = if reversed {
                        lane_len - 1 - min_index
                    } else {
                        min_index
                    };
                    let max_index = if reversed {
                        lane_len - 1 - max_index
                    } else {
                        max_index
                    };
                    MinMaxResult {
                        min_value,
                        min_index: logical_offset + min_index,
                        max_value,
                        max_index: logical_offset + max_index,
                    }
                },
            );
        }
        if tail >= 2 {
            let mut collapsed_shape = [0usize; DEFAULT_MAX_RANK];
            let mut collapsed_strides = [0isize; DEFAULT_MAX_RANK];
            let collapsed_rank = build_collapsed_layout(
                shape,
                strides,
                tail,
                count,
                &mut collapsed_shape,
                &mut collapsed_strides,
            );
            return reduce_minmax_recursive::<Scalar>(
                data,
                &collapsed_shape[..collapsed_rank],
                &collapsed_strides[..collapsed_rank],
                logical_offset,
            );
        }
    }
    if shape.len() == 1 {
        let (lane_ptr, lane_len, lane_stride, reversed) =
            normalize_reduction_lane(data, shape[0], strides[0]);
        return Scalar::reduce_minmax_raw(lane_ptr, lane_len, lane_stride).map(
            |(min_value, min_index, max_value, max_index)| {
                let min_index = if reversed {
                    lane_len - 1 - min_index
                } else {
                    min_index
                };
                let max_index = if reversed {
                    lane_len - 1 - max_index
                } else {
                    max_index
                };
                MinMaxResult {
                    min_value,
                    min_index: logical_offset + min_index,
                    max_value,
                    max_index: logical_offset + max_index,
                }
            },
        );
    }

    let inner_len: usize = shape[1..].iter().product();
    let mut best_min: Option<(Scalar::Output, usize)> = None;
    let mut best_max: Option<(Scalar::Output, usize)> = None;

    for index in 0..shape[0] {
        let child_ptr = (data as *const u8).offset(index as isize * strides[0]) as *const Scalar;
        let child_offset = logical_offset + index * inner_len;
        if let Some(child) =
            reduce_minmax_recursive::<Scalar>(child_ptr, &shape[1..], &strides[1..], child_offset)
        {
            match &best_min {
                Some((best_value, _))
                    if child.min_value.partial_cmp(best_value)
                        != Some(core::cmp::Ordering::Less) => {}
                _ => best_min = Some((child.min_value, child.min_index)),
            }
            match &best_max {
                Some((best_value, _))
                    if child.max_value.partial_cmp(best_value)
                        != Some(core::cmp::Ordering::Greater) => {}
                _ => best_max = Some((child.max_value, child.max_index)),
            }
        }
    }

    match (best_min, best_max) {
        (Some((min_value, min_index)), Some((max_value, max_index))) => Some(MinMaxResult {
            min_value,
            min_index,
            max_value,
            max_index,
        }),
        _ => None,
    }
}

// SumSqToF64 trait moved to crate::reduce

fn try_alloc_output_like<Destination: Clone + StorageElement, Kernel, const MAX_RANK: usize>(
    shape: &[usize],
    fill: Kernel,
) -> Result<Tensor<Destination, Global, MAX_RANK>, TensorError>
where
    Kernel: FnOnce(&mut TensorSpan<'_, Destination, MAX_RANK>) -> Result<(), TensorError>,
{
    let mut result = unsafe { Tensor::<Destination, Global, MAX_RANK>::try_empty(shape) }?;
    {
        let mut span = result.span();
        fill(&mut span)?;
    }
    Ok(result)
}

pub(crate) fn try_reborrow_tensor_into<
    Scalar: StorageElement + Clone,
    Destination: StorageElement + Clone,
    Kernel,
    const MAX_RANK: usize,
>(
    source: &Tensor<Scalar, Global, MAX_RANK>,
    out: &mut Tensor<Destination, Global, MAX_RANK>,
    apply: Kernel,
) -> Result<(), TensorError>
where
    Kernel: FnOnce(
        &TensorView<'_, Scalar, MAX_RANK>,
        &mut TensorSpan<'_, Destination, MAX_RANK>,
    ) -> Result<(), TensorError>,
{
    let view = source.view();
    let mut span = out.span();
    apply(&view, &mut span)
}

pub(crate) fn try_reborrow_tensor_inplace<
    Scalar: StorageElement + Clone,
    Kernel,
    const MAX_RANK: usize,
>(
    tensor: &mut Tensor<Scalar, Global, MAX_RANK>,
    apply: Kernel,
) -> Result<(), TensorError>
where
    Kernel: FnOnce(
        &TensorView<'_, Scalar, MAX_RANK>,
        &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError>,
{
    let view = TensorView {
        data: tensor.data.as_ptr(),
        shape: tensor.shape,
        strides: tensor.strides,
        ndim: tensor.ndim,
        _marker: PhantomData,
    };
    let mut span = tensor.span();
    apply(&view, &mut span)
}

fn rebind_view_rank<'a, Scalar, const TARGET_MAX_RANK: usize, const SOURCE_MAX_RANK: usize>(
    view: &TensorView<'a, Scalar, SOURCE_MAX_RANK>,
) -> Result<TensorView<'a, Scalar, TARGET_MAX_RANK>, TensorError> {
    if view.ndim > TARGET_MAX_RANK {
        return Err(TensorError::DimensionMismatch {
            expected: TARGET_MAX_RANK,
            got: view.ndim,
        });
    }
    let mut shape = [0usize; TARGET_MAX_RANK];
    let mut strides = [0isize; TARGET_MAX_RANK];
    shape[..view.ndim].copy_from_slice(&view.shape[..view.ndim]);
    strides[..view.ndim].copy_from_slice(&view.strides[..view.ndim]);
    Ok(TensorView {
        data: view.data,
        shape,
        strides,
        ndim: view.ndim,
        _marker: PhantomData,
    })
}

fn try_unary_kernel_into<Source, Destination, Kernel, const MAX_RANK: usize>(
    source: &TensorView<'_, Source, MAX_RANK>,
    out: &mut TensorSpan<'_, Destination, MAX_RANK>,
    mut kernel: Kernel,
) -> Result<(), TensorError>
where
    Kernel: FnMut(&[Source], &mut [Destination]),
{
    validate_same_shape(source.shape(), out.shape())?;
    unsafe {
        walk_contiguous_blocks_2(
            source.data,
            &source.strides[..source.ndim],
            out.data,
            &out.strides[..out.ndim],
            source.shape(),
            |source_ptr, target_ptr, tail_len| {
                let source = core::slice::from_raw_parts(source_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(source, target);
            },
        );
    }
    Ok(())
}

fn try_binary_kernel_into<Alloc, B, Destination, Kernel, const MAX_RANK: usize>(
    first: &TensorView<'_, Alloc, MAX_RANK>,
    second: &TensorView<'_, B, MAX_RANK>,
    out: &mut TensorSpan<'_, Destination, MAX_RANK>,
    mut kernel: Kernel,
) -> Result<(), TensorError>
where
    Kernel: FnMut(&[Alloc], &[B], &mut [Destination]),
{
    validate_same_shape(first.shape(), second.shape())?;
    validate_same_shape(first.shape(), out.shape())?;
    unsafe {
        walk_contiguous_blocks_3(
            first.data,
            &first.strides[..first.ndim],
            second.data,
            &second.strides[..second.ndim],
            out.data,
            &out.strides[..out.ndim],
            first.shape(),
            |first_ptr, second_ptr, target_ptr, tail_len| {
                let first = core::slice::from_raw_parts(first_ptr, tail_len);
                let second = core::slice::from_raw_parts(second_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(first, second, target);
            },
        );
    }
    Ok(())
}

fn try_ternary_kernel_into<Alloc, B, C, Destination, Kernel, const MAX_RANK: usize>(
    first: &TensorView<'_, Alloc, MAX_RANK>,
    second: &TensorView<'_, B, MAX_RANK>,
    third: &TensorView<'_, C, MAX_RANK>,
    out: &mut TensorSpan<'_, Destination, MAX_RANK>,
    mut kernel: Kernel,
) -> Result<(), TensorError>
where
    Kernel: FnMut(&[Alloc], &[B], &[C], &mut [Destination]),
{
    validate_same_shape(first.shape(), second.shape())?;
    validate_same_shape(first.shape(), third.shape())?;
    validate_same_shape(first.shape(), out.shape())?;
    unsafe {
        walk_contiguous_blocks_4(
            first.data,
            &first.strides[..first.ndim],
            second.data,
            &second.strides[..second.ndim],
            third.data,
            &third.strides[..third.ndim],
            out.data,
            &out.strides[..out.ndim],
            first.shape(),
            |first_ptr, second_ptr, third_ptr, target_ptr, tail_len| {
                let first = core::slice::from_raw_parts(first_ptr, tail_len);
                let second = core::slice::from_raw_parts(second_ptr, tail_len);
                let third = core::slice::from_raw_parts(third_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(first, second, third, target);
            },
        );
    }
    Ok(())
}

// endregion: Tensor Internal Helpers

// region: Tensor Elementwise Operations

impl<Scalar: Clone + EachScale, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    /// Apply element-wise scale: result\[i\] = α × self\[i\] + β
    ///
    /// Returns a new array with the scaled values.
    pub fn scale(
        &self,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        self.view().try_scale_tensor(alpha, beta)
    }

    /// Apply element-wise scale in-place: self\[i\] = α × self\[i\] + β
    pub fn scale_inplace(&mut self, alpha: Scalar::Scalar, beta: Scalar::Scalar) {
        let _ = try_reborrow_tensor_inplace(self, |view, span| {
            view.try_scale_tensor_into(alpha, beta, span)
        });
    }
}

impl<Scalar: Clone + EachSum, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    /// Element-wise sum: result\[i\] = self\[i\] + other\[i\]
    ///
    /// Returns a new array with the summed values.
    pub fn add<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<Scalar, Global, OTHER_MAX_RANK>,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<Scalar, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        self.view().try_add_tensor(&other_view)
    }

    /// Element-wise sum in-place: self\[i\] = self\[i\] + other\[i\]
    pub fn add_inplace<const OTHER_MAX_RANK: usize>(
        &mut self,
        other: &Tensor<Scalar, Global, OTHER_MAX_RANK>,
    ) -> Result<(), TensorError> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<Scalar, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_add_tensor_into(&other_view, span)
        })
    }
}

impl<Scalar: Clone + EachBlend, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    /// Blend: result\[i\] = α × self\[i\] + β × other\[i\]
    ///
    /// Returns a new array with the blend.
    pub fn blend<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<Scalar, Global, OTHER_MAX_RANK>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<Scalar, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        self.view().try_blend_tensor(&other_view, alpha, beta)
    }
}

impl<Scalar: Clone + EachFMA, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    /// Fused multiply-add: result\[i\] = α × self\[i\] × b\[i\] + β × c\[i\]
    ///
    /// Returns a new array with the FMA result.
    pub fn fma<const B_MAX_RANK: usize, const C_MAX_RANK: usize>(
        &self,
        b: &Tensor<Scalar, Global, B_MAX_RANK>,
        c: &Tensor<Scalar, Global, C_MAX_RANK>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        validate_same_shape(self.shape(), b.shape())?;
        validate_same_shape(self.shape(), c.shape())?;
        let b_view = rebind_view_rank::<Scalar, MAX_RANK, B_MAX_RANK>(&b.view())?;
        let c_view = rebind_view_rank::<Scalar, MAX_RANK, C_MAX_RANK>(&c.view())?;
        self.view().try_fma_tensors(&b_view, &c_view, alpha, beta)
    }
}

// endregion: Tensor Elementwise Operations

// region: Tensor Explicit Elementwise + Cast

impl<'a, Scalar: Clone + EachScale, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    pub fn try_scale_tensor(
        &self,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| {
            self.try_scale_tensor_into(alpha, beta, span)
        })
    }

    pub fn try_scale_tensor_into(
        &self,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(alpha, beta, out)
    }

    fn try_affine_into(
        &self,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            Scalar::each_scale(source, alpha, beta, target);
        })
    }

    pub fn try_add_scalar(
        &self,
        scalar: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_add_scalar_into(scalar, span))
    }

    pub fn try_sub_scalar(
        &self,
        scalar: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_sub_scalar_into(scalar, span))
    }

    pub fn try_mul_scalar(
        &self,
        scalar: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_mul_scalar_into(scalar, span))
    }

    pub fn try_add_scalar_into(
        &self,
        scalar: Scalar::Scalar,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(Scalar::Scalar::from(1.0f32), scalar, out)
    }

    pub fn try_sub_scalar_into(
        &self,
        scalar: Scalar::Scalar,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(
            Scalar::Scalar::from(1.0f32),
            Scalar::Scalar::from(-1.0f32) * scalar,
            out,
        )
    }

    pub fn try_mul_scalar_into(
        &self,
        scalar: Scalar::Scalar,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(scalar, Scalar::Scalar::from(0.0f32), out)
    }
}

impl<'a, Scalar: Clone + EachSum, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    pub fn try_add_tensor(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK>,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_add_tensor_into(other, span))
    }

    pub fn try_add_tensor_into(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK>,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_binary_kernel_into(self, other, out, |first, second, target| {
            Scalar::each_sum(first, second, target);
        })
    }
}

impl<'a, Scalar: Clone + EachBlend, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    pub fn try_blend_tensor(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| {
            self.try_blend_tensor_into(other, alpha, beta, span)
        })
    }

    pub fn try_blend_tensor_into(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_binary_kernel_into(self, other, out, |first, second, target| {
            Scalar::each_blend(first, second, alpha, beta, target);
        })
    }

    pub fn try_sub_tensor(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK>,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_sub_tensor_into(other, span))
    }

    pub fn try_sub_tensor_into(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK>,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_blend_tensor_into(
            other,
            Scalar::Scalar::from(1.0f32),
            Scalar::Scalar::from(-1.0f32),
            out,
        )
    }
}

impl<'a, Scalar: Clone + EachFMA, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    pub fn try_fma_tensors(
        &self,
        b: &TensorView<'_, Scalar, MAX_RANK>,
        c: &TensorView<'_, Scalar, MAX_RANK>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| {
            self.try_fma_tensors_into(b, c, alpha, beta, span)
        })
    }

    pub fn try_fma_tensors_into(
        &self,
        b: &TensorView<'_, Scalar, MAX_RANK>,
        c: &TensorView<'_, Scalar, MAX_RANK>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_ternary_kernel_into(self, b, c, out, |first, second, third, target| {
            Scalar::each_fma(first, second, third, alpha, beta, target);
        })
    }

    pub fn try_mul_tensor(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK>,
    ) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_mul_tensor_into(other, span))
    }

    pub fn try_mul_tensor_into(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK>,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_fma_tensors_into(
            other,
            self,
            Scalar::Scalar::from(1.0f32),
            Scalar::Scalar::from(0.0f32),
            out,
        )
    }
}

impl<'a, Source: Clone + CastDtype, const MAX_RANK: usize> TensorView<'a, Source, MAX_RANK> {
    pub fn try_cast_dtype<Destination: Clone + CastDtype>(
        &self,
    ) -> Result<Tensor<Destination, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_cast_dtype_into(span))
    }

    pub fn try_cast_dtype_into<Destination: Clone + CastDtype>(
        &self,
        out: &mut TensorSpan<'_, Destination, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            let _ = cast(source, target);
        })
    }
}

// endregion: Tensor Explicit Elementwise + Cast (ScaleOps/SumOps/BlendOps/FmaOps/CastOps moved to crate::each / crate::cast)

// region: Tensor Trigonometry

impl<'a, Scalar: Clone + EachSin, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    pub fn try_sin(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_sin_into(span))
    }

    pub fn try_sin_into(
        &self,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            Scalar::sin(source, target);
        })
    }
}

impl<'a, Scalar: Clone + EachCos, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    pub fn try_cos(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_cos_into(span))
    }

    pub fn try_cos_into(
        &self,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            Scalar::cos(source, target);
        })
    }
}

impl<'a, Scalar: Clone + EachATan, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    pub fn try_atan(&self) -> Result<Tensor<Scalar, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_atan_into(span))
    }

    pub fn try_atan_into(
        &self,
        out: &mut TensorSpan<'_, Scalar, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            Scalar::atan(source, target);
        })
    }
}

// endregion: Tensor Trigonometry (moved to crate::each)

// region: Tensor Reductions

impl<Scalar: Clone + Dot, const MAX_RANK: usize> Tensor<Scalar, Global, MAX_RANK> {
    /// Compute the dot product of this array with another.
    ///
    /// Both arrays must be 1D with the same length.
    pub fn dot_product<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<Scalar, Global, OTHER_MAX_RANK>,
    ) -> Result<Scalar::Output, TensorError> {
        if self.ndim != 1 || other.ndim != 1 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: if self.ndim != 1 {
                    self.ndim
                } else {
                    other.ndim
                },
            });
        }
        if self.numel() != other.numel() {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: self.numel(),
                got: other.numel(),
            });
        }
        // Dot::dot returns Option, unwrap since we verified lengths match
        Ok(Scalar::dot(self.as_slice(), other.as_slice()).expect("dot product failed"))
    }
}

pub(crate) type MomentsAxisResult<Scalar, const MAX_RANK: usize> = Result<
    (
        Tensor<<Scalar as ReduceMoments>::SumOutput, Global, MAX_RANK>,
        Tensor<<Scalar as ReduceMoments>::SumSqOutput, Global, MAX_RANK>,
    ),
    TensorError,
>;

impl<'a, Scalar: Clone + ReduceMoments, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK>
where
    Scalar::SumOutput: Clone + Default + core::ops::AddAssign,
    Scalar::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
    pub fn try_moments_all(&self) -> Result<(Scalar::SumOutput, Scalar::SumSqOutput), TensorError> {
        Ok(unsafe {
            reduce_moments_recursive::<Scalar>(self.data, self.shape(), &self.strides[..self.ndim])
        })
    }

    pub fn try_moments_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> MomentsAxisResult<Scalar, MAX_RANK> {
        let axis = normalize_axis(axis, self.ndim)?;
        let output_shape = reduced_shape(self.shape(), axis, keep_dims);
        let mut sums = Tensor::<Scalar::SumOutput, Global, MAX_RANK>::try_full(
            &output_shape,
            Scalar::SumOutput::default(),
        )?;
        let mut sumsqs = Tensor::<Scalar::SumSqOutput, Global, MAX_RANK>::try_full(
            &output_shape,
            Scalar::SumSqOutput::default(),
        )?;
        self.try_moments_axis_into(axis, keep_dims, &mut sums, &mut sumsqs)?;
        Ok((sums, sumsqs))
    }

    pub fn try_moments_axis_into<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        sum_out: &mut Tensor<Scalar::SumOutput, Global, MAX_RANK>,
        sumsq_out: &mut Tensor<Scalar::SumSqOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, sum_out.shape())?;
        validate_same_shape(&expected_shape, sumsq_out.shape())?;

        for_each_axis_lane(
            self,
            axis,
            |lane_ptr, lane_len, lane_stride, output_index| {
                let (lane_ptr, lane_len, lane_stride, _) =
                    unsafe { normalize_reduction_lane(lane_ptr, lane_len, lane_stride) };
                let (sum, sumsq) =
                    unsafe { Scalar::reduce_moments_raw(lane_ptr, lane_len, lane_stride) };
                sum_out.as_mut_slice()[output_index] = sum;
                sumsq_out.as_mut_slice()[output_index] = sumsq;
            },
        );
        Ok(())
    }

    pub fn try_sum_all(&self) -> Result<Scalar::SumOutput, TensorError> {
        Ok(self.try_moments_all()?.0)
    }

    pub fn try_sum_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<Scalar::SumOutput, Global, MAX_RANK>, TensorError> {
        let (sums, _) = self.try_moments_axis(axis, keep_dims)?;
        Ok(sums)
    }

    pub fn try_sum_axis_into<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        out: &mut Tensor<Scalar::SumOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, out.shape())?;
        let mut scratch = Tensor::<Scalar::SumSqOutput, Global, MAX_RANK>::try_full(
            &expected_shape,
            Scalar::SumSqOutput::default(),
        )?;
        self.try_moments_axis_into(axis, keep_dims, out, &mut scratch)
    }

    pub fn try_norm_all(&self) -> Result<f64, TensorError> {
        let (_, sumsq) = self.try_moments_all()?;
        Ok(Roots::sqrt(sumsq.to_f64()))
    }

    pub fn try_norm_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<f64, Global, MAX_RANK>, TensorError> {
        let (_, sumsqs) = self.try_moments_axis(axis, keep_dims)?;
        let mut norms = Tensor::<f64, Global, MAX_RANK>::try_full(sumsqs.shape(), 0.0)?;
        for (target, value) in norms
            .as_mut_slice()
            .iter_mut()
            .zip(sumsqs.as_slice().iter())
        {
            *target = Roots::sqrt(SumSqToF64::to_f64(*value));
        }
        Ok(norms)
    }

    pub fn try_norm_axis_into<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        out: &mut Tensor<f64, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, out.shape())?;
        let mut scratch_sum = Tensor::<Scalar::SumOutput, Global, MAX_RANK>::try_full(
            &expected_shape,
            Scalar::SumOutput::default(),
        )?;
        let mut scratch_sumsq = Tensor::<Scalar::SumSqOutput, Global, MAX_RANK>::try_full(
            &expected_shape,
            Scalar::SumSqOutput::default(),
        )?;
        self.try_moments_axis_into(axis, keep_dims, &mut scratch_sum, &mut scratch_sumsq)?;
        for (target, value) in out
            .as_mut_slice()
            .iter_mut()
            .zip(scratch_sumsq.as_slice().iter())
        {
            *target = Roots::sqrt(SumSqToF64::to_f64(*value));
        }
        Ok(())
    }
}

pub(crate) type MinMaxAxisResult<Scalar, const MAX_RANK: usize> = Result<
    MinMaxResult<
        Tensor<<Scalar as ReduceMinMax>::Output, Global, MAX_RANK>,
        Tensor<usize, Global, MAX_RANK>,
    >,
    TensorError,
>;

impl<'a, Scalar: Clone + ReduceMinMax, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK>
where
    Scalar::Output: Clone + Default + PartialOrd,
{
    pub fn try_minmax_all(&self) -> Result<MinMaxResult<Scalar::Output>, TensorError> {
        unsafe {
            reduce_minmax_recursive::<Scalar>(
                self.data,
                self.shape(),
                &self.strides[..self.ndim],
                0,
            )
        }
        .ok_or(TensorError::InvalidShape {
            axis: 0,
            size: self.numel(),
            reason: "min/max reduction undefined for empty or NaN-only input",
        })
    }

    pub fn try_minmax_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> MinMaxAxisResult<Scalar, MAX_RANK> {
        let axis = normalize_axis(axis, self.ndim)?;
        let output_shape = reduced_shape(self.shape(), axis, keep_dims);
        let mut min_values = Tensor::<Scalar::Output, Global, MAX_RANK>::try_full(
            &output_shape,
            Scalar::Output::default(),
        )?;
        let mut min_indices = Tensor::<usize, Global, MAX_RANK>::try_full(&output_shape, 0)?;
        let mut max_values = Tensor::<Scalar::Output, Global, MAX_RANK>::try_full(
            &output_shape,
            Scalar::Output::default(),
        )?;
        let mut max_indices = Tensor::<usize, Global, MAX_RANK>::try_full(&output_shape, 0)?;
        self.try_minmax_axis_into(
            axis,
            keep_dims,
            &mut min_values,
            &mut min_indices,
            &mut max_values,
            &mut max_indices,
        )?;
        Ok(MinMaxResult {
            min_value: min_values,
            min_index: min_indices,
            max_value: max_values,
            max_index: max_indices,
        })
    }

    pub fn try_minmax_axis_into<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        min_out: &mut Tensor<Scalar::Output, Global, MAX_RANK>,
        argmin_out: &mut Tensor<usize, Global, MAX_RANK>,
        max_out: &mut Tensor<Scalar::Output, Global, MAX_RANK>,
        argmax_out: &mut Tensor<usize, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, min_out.shape())?;
        validate_same_shape(&expected_shape, argmin_out.shape())?;
        validate_same_shape(&expected_shape, max_out.shape())?;
        validate_same_shape(&expected_shape, argmax_out.shape())?;

        let mut invalid_lane = false;
        for_each_axis_lane(
            self,
            axis,
            |lane_ptr, lane_len, lane_stride, output_index| {
                if invalid_lane {
                    return;
                }
                let (lane_ptr, lane_len, lane_stride, reversed) =
                    unsafe { normalize_reduction_lane(lane_ptr, lane_len, lane_stride) };
                if let Some((min_value, min_index, max_value, max_index)) =
                    unsafe { Scalar::reduce_minmax_raw(lane_ptr, lane_len, lane_stride) }
                {
                    min_out.as_mut_slice()[output_index] = min_value;
                    argmin_out.as_mut_slice()[output_index] = if reversed {
                        lane_len - 1 - min_index
                    } else {
                        min_index
                    };
                    max_out.as_mut_slice()[output_index] = max_value;
                    argmax_out.as_mut_slice()[output_index] = if reversed {
                        lane_len - 1 - max_index
                    } else {
                        max_index
                    };
                } else {
                    invalid_lane = true;
                }
            },
        );

        if invalid_lane {
            return Err(TensorError::InvalidShape {
                axis,
                size: self.shape()[axis],
                reason: "min/max reduction undefined for empty or NaN-only lanes",
            });
        }
        Ok(())
    }

    pub fn try_min_all(&self) -> Result<Scalar::Output, TensorError> {
        Ok(self.try_minmax_all()?.min_value)
    }

    pub fn try_argmin_all(&self) -> Result<usize, TensorError> {
        Ok(self.try_minmax_all()?.min_index)
    }

    pub fn try_max_all(&self) -> Result<Scalar::Output, TensorError> {
        Ok(self.try_minmax_all()?.max_value)
    }

    pub fn try_argmax_all(&self) -> Result<usize, TensorError> {
        Ok(self.try_minmax_all()?.max_index)
    }

    pub fn try_min_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<Scalar::Output, Global, MAX_RANK>, TensorError> {
        Ok(self.try_minmax_axis(axis, keep_dims)?.min_value)
    }

    pub fn try_argmin_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        Ok(self.try_minmax_axis(axis, keep_dims)?.min_index)
    }

    pub fn try_max_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<Scalar::Output, Global, MAX_RANK>, TensorError> {
        Ok(self.try_minmax_axis(axis, keep_dims)?.max_value)
    }

    pub fn try_argmax_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        Ok(self.try_minmax_axis(axis, keep_dims)?.max_index)
    }
}

// MomentsOps and MinMaxOps moved to crate::reduce

impl<const MAX_RANK: usize> Tensor<f32, Global, MAX_RANK> {
    /// Sum all elements of the tensor.
    pub fn sum(&self) -> f32 {
        self.try_sum_all().unwrap_or(0.0) as f32
    }
}

impl<const MAX_RANK: usize> Tensor<f64, Global, MAX_RANK> {
    /// Sum all elements of the tensor.
    pub fn sum(&self) -> f64 {
        self.try_sum_all().unwrap_or(0.0)
    }
}

// endregion: Tensor Reductions

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast::CastOps;
    use crate::each::{AllCloseOps, ScaleOps, SumOps, TrigSinOps};
    use crate::reduce::MomentsOps;
    use crate::types::{bf16c, f16, f16c, f32c};

    #[test]
    fn tensor_creation_from_factories() {
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.numel(), 12);
        assert!(!arr.is_empty());
    }

    #[test]
    fn tensor_from_slice() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.as_slice(), &data[..]);
    }

    #[test]
    fn tensor_clone() {
        let arr = Tensor::<f32>::try_full(&[3, 4], 2.5f32).unwrap();
        let cloned = arr.clone();
        assert_eq!(cloned.shape(), arr.shape());
        assert_eq!(cloned.as_slice(), arr.as_slice());
    }

    #[test]
    fn tensor_error_display() {
        let err = TensorError::AllocationFailed;
        assert_eq!(format!("{}", err), "memory allocation failed");
        let err = TensorError::TooManyRanks { got: 10 };
        assert_eq!(format!("{}", err), "too many ranks: 10");
    }

    #[test]
    fn tensor_row_access() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.row(0), Some(&[0.0, 1.0, 2.0, 3.0][..]));
        assert_eq!(arr.row(1), Some(&[4.0, 5.0, 6.0, 7.0][..]));
        assert_eq!(arr.row(2), Some(&[8.0, 9.0, 10.0, 11.0][..]));
        assert_eq!(arr.row(3), None);
    }

    #[test]
    fn tensor_slicing() {
        let arr = Tensor::<f32>::try_full(&[4, 5], 1.0f32).unwrap();
        let view = arr
            .slice(&[SliceRange::full(), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[4, 5]);
        let view = arr
            .slice(&[SliceRange::range(1, 3), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[2, 5]);
        let view = arr
            .slice(&[SliceRange::index(0), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[5]);
        assert_eq!(view.ndim(), 1);
    }

    #[test]
    fn tensor_transpose_2d() {
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        let transposed = arr.transpose().unwrap();
        assert_eq!(transposed.shape(), &[4, 3]);
    }

    #[test]
    fn tensor_is_contiguous() {
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        let view = arr.view();
        assert!(view.is_contiguous());
        assert!(arr.has_contiguous_rows());
    }

    #[test]
    fn matrix_alias_round_trip() {
        let mat: Matrix<f32> = Matrix::try_full(&[3, 4], 1.0f32).unwrap();
        assert_eq!(mat.shape(), &[3, 4]);
    }

    #[test]
    fn tensor_scalar_lookup_and_views() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut tensor = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();

        assert_eq!(tensor[0_usize], 0.0);
        assert_eq!(tensor[-1_i32], 11.0);
        assert_eq!(tensor[(1_usize, 2_usize)], 6.0);
        assert_eq!(tensor[(2_i32, -1_i32)], 11.0);
        assert_eq!(*tensor.try_flat(5_usize).unwrap(), 5.0);
        assert_eq!(*tensor.try_coords((1_usize, 3_usize)).unwrap(), 7.0);

        tensor[(1_usize, 2_usize)] = 60.0;
        assert_eq!(tensor[(1_usize, 2_usize)], 60.0);
        *tensor.try_coords_mut((2_usize, 0_usize)).unwrap() = 80.0;
        assert_eq!(tensor[(2_usize, 0_usize)], 80.0);

        let view = tensor.view();
        assert_eq!(view[1_usize], 1.0);
        assert_eq!(view[(1_usize, 2_usize)], 60.0);

        let mut span = tensor.span();
        assert_eq!(span[(2_usize, 0_usize)], 80.0);
        span[(0_usize, 1_usize)] = 10.0;
        assert_eq!(span[(0_usize, 1_usize)], 10.0);
    }

    #[test]
    fn tensor_noncontiguous_lookup_and_rank_zero() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let even_columns = tensor
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        assert_eq!(even_columns.shape(), &[3, 2]);
        assert_eq!(even_columns[0_usize], 0.0);
        assert_eq!(even_columns[1_usize], 2.0);
        assert_eq!(even_columns[2_usize], 4.0);
        assert_eq!(even_columns[(1_usize, 1_usize)], 6.0);
        assert_eq!(*even_columns.try_coords((2_usize, 1_usize)).unwrap(), 10.0);

        let row = tensor.slice_leading(1_usize).unwrap();
        assert_eq!(row.shape(), &[4]);
        assert_eq!(row[-1_i32], 7.0);

        let scalar = tensor
            .slice(&[SliceRange::index(2), SliceRange::index(3)])
            .unwrap();
        assert_eq!(scalar.ndim(), 0);
        assert_eq!(*scalar.try_scalar().unwrap(), 11.0);
    }

    #[test]
    fn tensor_ops() {
        // Reshape
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        let reshaped = arr.reshape(&[2, 6]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 6]);
        assert_eq!(reshaped.numel(), 12);

        // Sum f32
        let arr = Tensor::<f32>::try_full(&[100], 1.0f32).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 0.001);

        // Sum f64
        let arr = Tensor::<f64>::try_full(&[100], 1.0f64).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 1e-9);
    }

    #[test]
    fn nd_contraction_global_reductions() {
        // Verify that N-D contiguous tensors produce the same moments/minmax as a flat copy.
        // This tests the uniform-stride-tail collapsing and recursive re-analysis paths.
        let data: Vec<f32> = (0..720).map(|i| (i as f32) * 0.1 - 36.0).collect();

        // 4-D: shape [2, 3, 4, 30] = 720 elements
        let t4d = Tensor::<f32>::try_from_slice(&data, &[2, 3, 4, 30]).unwrap();
        let t1d = Tensor::<f32>::try_from_slice(&data, &[720]).unwrap();
        let (sum_4d, sumsq_4d) = t4d.view().try_moments_all().unwrap();
        let (sum_1d, sumsq_1d) = t1d.view().try_moments_all().unwrap();
        assert!(
            (sum_4d - sum_1d).abs() < 1e-3,
            "4D vs 1D sum: {sum_4d} != {sum_1d}"
        );
        assert!(
            (sumsq_4d - sumsq_1d).abs() < 1e-1,
            "4D vs 1D sumsq: {sumsq_4d} != {sumsq_1d}"
        );
        let mm_4d = t4d.view().try_minmax_all().unwrap();
        let mm_1d = t1d.view().try_minmax_all().unwrap();
        assert_eq!(mm_4d.min_value, mm_1d.min_value);
        assert_eq!(mm_4d.max_value, mm_1d.max_value);
        assert_eq!(mm_4d.min_index, mm_1d.min_index);
        assert_eq!(mm_4d.max_index, mm_1d.max_index);

        // 5-D: shape [2, 3, 4, 5, 6] = 720 elements
        let t5d = Tensor::<f32>::try_from_slice(&data, &[2, 3, 4, 5, 6]).unwrap();
        let (sum_5d, sumsq_5d) = t5d.view().try_moments_all().unwrap();
        assert!((sum_5d - sum_1d).abs() < 1e-3, "5D vs 1D sum");
        assert!((sumsq_5d - sumsq_1d).abs() < 1e-1, "5D vs 1D sumsq");
    }

    #[test]
    fn nd_contraction_strided_views() {
        // Verify reductions on non-contiguous uniform-stride subviews.
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t = Tensor::<f32>::try_from_slice(&data, &[4, 4, 3]).unwrap();

        // Channel subview: [:, :, 1] → shape [4, 4], stride = 3 * sizeof(f32)
        let channel = t
            .slice(&[SliceRange::full(), SliceRange::full(), SliceRange::index(1)])
            .unwrap();
        assert_eq!(channel.shape(), &[4, 4]);
        let (ch_sum, _) = channel.try_moments_all().unwrap();
        let expected: f64 = (0..16).map(|i| (i * 3 + 1) as f64).sum();
        assert!(
            (ch_sum - expected).abs() < 1e-3,
            "channel subview sum: {ch_sum} != {expected}"
        );

        // Row skip: [::2, :, :] → shape [2, 4, 3], outer stride doubled
        let skipped = t
            .slice(&[
                SliceRange::range_step(0, 4, 2),
                SliceRange::full(),
                SliceRange::full(),
            ])
            .unwrap();
        assert_eq!(skipped.shape(), &[2, 4, 3]);
        let (skip_sum, _) = skipped.try_moments_all().unwrap();
        let expected_skip: f64 = data
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < 12 || (*i >= 24 && *i < 36))
            .map(|(_, v)| *v as f64)
            .sum();
        assert!(
            (skip_sum - expected_skip).abs() < 1e-3,
            "row skip sum: {skip_sum} != {expected_skip}"
        );
    }

    #[test]
    fn nd_contraction_axis_reductions() {
        // Verify axis reductions on N-D tensors match manual computation.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::<f32>::try_from_slice(&data, &[2, 3, 4]).unwrap();

        // axis=0: sum over batch → shape [3, 4]
        let (sums_a0, _) = t.view().try_moments_axis(0, false).unwrap();
        assert_eq!(sums_a0.shape(), &[3, 4]);
        // [0..4] + [12..16] = [12, 14, 16, 18]
        assert!((sums_a0.as_slice()[0] - 12.0).abs() < 1e-6);
        assert!((sums_a0.as_slice()[1] - 14.0).abs() < 1e-6);

        // axis=-1: sum over innermost → shape [2, 3]
        let (sums_last, _) = t.view().try_moments_axis(-1_i32, false).unwrap();
        assert_eq!(sums_last.shape(), &[2, 3]);
        // First row: 0+1+2+3 = 6
        assert!((sums_last.as_slice()[0] - 6.0).abs() < 1e-6);

        // axis=1: sum over middle → shape [2, 4]
        let (sums_mid, _) = t.view().try_moments_axis(1, false).unwrap();
        assert_eq!(sums_mid.shape(), &[2, 4]);
        // Column 0: 0+4+8 = 12
        assert!((sums_mid.as_slice()[0] - 12.0).abs() < 1e-6);

        // Axis reduction on strided subview: [::2, :, :] on [4, 3, 4]
        let data48: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t48 = Tensor::<f32>::try_from_slice(&data48, &[4, 3, 4]).unwrap();
        let skipped = t48
            .slice(&[
                SliceRange::range_step(0, 4, 2),
                SliceRange::full(),
                SliceRange::full(),
            ])
            .unwrap();
        assert_eq!(skipped.shape(), &[2, 3, 4]);
        let (sums_skip_last, _) = skipped.try_moments_axis(-1_i32, false).unwrap();
        assert_eq!(sums_skip_last.shape(), &[2, 3]);
        // First lane: 0+1+2+3 = 6
        assert!((sums_skip_last.as_slice()[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn nd_contraction_singleton_dims() {
        // Singleton dimensions (extent=1) should collapse freely.
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let t = Tensor::<f32>::try_from_slice(&data, &[1, 64, 1]).unwrap();
        let t_flat = Tensor::<f32>::try_from_slice(&data, &[64]).unwrap();
        let (sum_nd, sumsq_nd) = t.view().try_moments_all().unwrap();
        let (sum_flat, sumsq_flat) = t_flat.view().try_moments_all().unwrap();
        assert_eq!(sum_nd, sum_flat);
        assert_eq!(sumsq_nd, sumsq_flat);

        let mm_nd = t.view().try_minmax_all().unwrap();
        let mm_flat = t_flat.view().try_minmax_all().unwrap();
        assert_eq!(mm_nd.min_index, mm_flat.min_index);
        assert_eq!(mm_nd.max_index, mm_flat.max_index);
    }

    #[test]
    fn complex_elementwise_view_and_owner_paths() {
        let left_values = [f32c { re: 1.0, im: 2.0 }, f32c { re: 3.0, im: 4.0 }];
        let right_values = [f32c { re: 5.0, im: 6.0 }, f32c { re: 7.0, im: 8.0 }];
        let zeros = Tensor::<f32c>::try_full(&[2], f32c { re: 0.0, im: 0.0 }).unwrap();
        let left = Tensor::<f32c>::try_from_slice(&left_values, &[2]).unwrap();
        let right = Tensor::<f32c>::try_from_slice(&right_values, &[2]).unwrap();

        let added = left.try_add_tensor(&right).unwrap();
        assert_eq!(
            added.as_slice(),
            &[f32c { re: 6.0, im: 8.0 }, f32c { re: 10.0, im: 12.0 }]
        );

        let scaled = left
            .scale(f32c { re: 1.0, im: 0.0 }, f32c { re: 1.0, im: 0.0 })
            .unwrap();
        assert_eq!(
            scaled.as_slice(),
            &[f32c { re: 2.0, im: 2.0 }, f32c { re: 4.0, im: 4.0 }]
        );

        let blended = left
            .view()
            .try_blend_tensor(
                &right.view(),
                f32c { re: 1.0, im: 0.0 },
                f32c { re: -1.0, im: 0.0 },
            )
            .unwrap();
        assert_eq!(
            blended.as_slice(),
            &[f32c { re: -4.0, im: -4.0 }, f32c { re: -4.0, im: -4.0 }]
        );

        let fma = left
            .view()
            .try_fma_tensors(
                &right.view(),
                &zeros.view(),
                f32c { re: 1.0, im: 0.0 },
                f32c { re: 0.0, im: 0.0 },
            )
            .unwrap();
        assert_eq!(
            fma.as_slice(),
            &[
                f32c { re: -7.0, im: 16.0 },
                f32c {
                    re: -11.0,
                    im: 52.0
                }
            ]
        );

        let mut inplace = Tensor::<f32c>::try_from_slice(&left_values, &[2]).unwrap();
        inplace.try_add_tensor_inplace(&right).unwrap();
        assert_eq!(inplace.as_slice(), added.as_slice());

        let widened = left.try_cast_dtype::<bf16c>().unwrap();
        assert_eq!(widened.as_slice()[0].re.to_f32(), 1.0);
        assert_eq!(widened.as_slice()[0].im.to_f32(), 2.0);

        let strided = Tensor::<f16c>::try_from_slice(
            &[
                f16c {
                    re: f16::from_f32(1.0),
                    im: f16::from_f32(2.0),
                },
                f16c {
                    re: f16::from_f32(100.0),
                    im: f16::from_f32(101.0),
                },
                f16c {
                    re: f16::from_f32(3.0),
                    im: f16::from_f32(4.0),
                },
                f16c {
                    re: f16::from_f32(102.0),
                    im: f16::from_f32(103.0),
                },
            ],
            &[2, 2],
        )
        .unwrap();
        let complex_column = strided
            .slice(&[SliceRange::full(), SliceRange::range(0, 1)])
            .unwrap();
        let mut out = Tensor::<f16c>::try_full(
            &[2, 1],
            f16c {
                re: f16::ZERO,
                im: f16::ZERO,
            },
        )
        .unwrap();
        {
            let mut span = out.span();
            complex_column
                .try_scale_tensor_into(
                    f16c {
                        re: f16::ONE,
                        im: f16::ZERO,
                    },
                    f16c {
                        re: f16::ZERO,
                        im: f16::ONE,
                    },
                    &mut span,
                )
                .unwrap();
        }
        assert_eq!(
            out.as_slice(),
            &[
                f16c {
                    re: f16::from_f32(1.0),
                    im: f16::from_f32(3.0)
                },
                f16c {
                    re: f16::from_f32(3.0),
                    im: f16::from_f32(5.0)
                }
            ]
        );
    }
    #[test]
    fn tensor_ref_generic() {
        fn shape_of<Scalar: StorageElement, const R: usize>(
            t: &impl TensorRef<Scalar, R>,
        ) -> Vec<usize> {
            t.shape().to_vec()
        }
        let t = Tensor::<f32>::try_full(&[3, 4], 1.0).unwrap();
        assert_eq!(shape_of(&t), vec![3, 4]);
        assert_eq!(shape_of(&t.view()), vec![3, 4]);

        // TensorRef default methods
        assert_eq!(TensorRef::rank(&t), 2);
        assert!(!TensorRef::is_empty(&t));
        assert!(TensorRef::is_contiguous(&t));
        assert!(TensorRef::has_contiguous_rows(&t));
    }

    #[test]
    fn tensor_ref_extension_traits() {
        crate::capabilities::configure_thread();
        let t = Tensor::<f32>::try_full(&[3, 4], 2.0).unwrap();
        let v = t.view();

        // ScaleOps works on both Tensor and TensorView
        let r1 = ScaleOps::try_add_scalar(&t, 1.0).unwrap();
        let r2 = ScaleOps::try_add_scalar(&v, 1.0).unwrap();
        assert_eq!(r1.as_slice(), r2.as_slice());
        assert!((r1.as_slice()[0] - 3.0).abs() < 0.01);

        // SumOps works on Tensor with TensorView as other
        let other = Tensor::<f32>::try_full(&[3, 4], 1.0).unwrap();
        let r3 = SumOps::try_add_tensor(&t, &other).unwrap();
        assert!((r3.as_slice()[0] - 3.0).abs() < 0.01);

        // TrigSinOps works on both
        let small = Tensor::<f32>::try_full(&[4], 0.0).unwrap();
        let r4 = TrigSinOps::try_sin(&small).unwrap();
        assert!((r4.as_slice()[0] - 0.0).abs() < 0.01);

        // MomentsOps works on both
        let sum_t = MomentsOps::try_sum_all(&t).unwrap();
        let sum_v = MomentsOps::try_sum_all(&v).unwrap();
        assert!((sum_t as f32 - 24.0).abs() < 0.01);
        assert!((sum_v as f32 - 24.0).abs() < 0.01);
    }

    #[test]
    fn tensor_allclose_matching() {
        let a = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::try_full(&[2, 3], 1.0 + 1e-7).unwrap();
        assert!(a.allclose(&b, 1e-6, 0.0));
    }

    #[test]
    fn tensor_allclose_mismatching() {
        let a = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::try_full(&[2, 3], 2.0).unwrap();
        assert!(!a.allclose(&b, 1e-6, 0.0));
    }

    #[test]
    fn tensor_allclose_different_shapes() {
        let a = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::try_full(&[3, 2], 1.0).unwrap();
        assert!(!a.allclose(&b, 1e-6, 1e-6));
    }

    #[test]
    fn tensor_view_allclose() {
        let a = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::try_full(&[2, 3], 1.0 + 1e-7).unwrap();
        assert!(a.view().allclose(&b.view(), 1e-6, 0.0));
    }

    #[test]
    fn tensor_view_iter_logical_scalars() {
        use crate::types::i4x2;
        // Shape [6] logical → 3 i4x2 storage values
        let mut t = Tensor::<i4x2>::try_zeros(&[6]).unwrap();
        // Set nibbles: storage[0] has dims 0,1; storage[1] has dims 2,3; etc.
        let slice = t.as_mut_slice();
        slice[0] = i4x2::pack([1, 2]);
        slice[1] = i4x2::pack([3, 4]);
        slice[2] = i4x2::pack([5, 6]);

        let vals: Vec<i8> = t.iter().map(|(_, v)| *v).collect();
        assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn tensor_span_iter_mut_f32() {
        let mut t = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        for (_, mut value) in &mut t {
            *value += 10.0;
        }
        for (_, v) in t.iter() {
            assert!((*v - 11.0).abs() < 1e-6);
        }
    }

    #[test]
    fn tensor_span_iter_mut_i4x2() {
        use crate::types::i4x2;
        let mut t = Tensor::<i4x2>::try_zeros(&[6]).unwrap();
        for (pos, mut value) in &mut t {
            *value = pos[0] as i8;
        }
        let vals: Vec<i8> = t.iter().map(|(_, v)| *v).collect();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn tensor_iterator_alias_compat() {
        let t = Tensor::<f32>::try_full(&[2], 1.0).unwrap();
        let _it: TensorIterator<'_, f32> = t.iter();
    }

    #[test]
    fn tensor_tuple_slice_syntax() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();

        // Full ranges — tuple of two RangeFull
        let v = t.slice((.., ..)).unwrap();
        assert_eq!(v.shape(), &[3, 4]);

        // Index + full — selects row 0
        let v = t.slice((0_usize, ..)).unwrap();
        assert_eq!(v.shape(), &[4]);
        assert_eq!(v.numel(), 4);

        // Full + index — selects column 1
        let v = t.slice((.., 1_usize)).unwrap();
        assert_eq!(v.shape(), &[3]);

        // Range + full — rows 1..3
        let v = t.slice((1..3_usize, ..)).unwrap();
        assert_eq!(v.shape(), &[2, 4]);

        // Full + RangeTo — columns ..2
        let v = t.slice((.., ..2_usize)).unwrap();
        assert_eq!(v.shape(), &[3, 2]);

        // Full + RangeInclusive — columns 0..=2
        let v = t.slice((.., 0..=2_usize)).unwrap();
        assert_eq!(v.shape(), &[3, 3]);

        // Mixed: tuple with SliceRange pass-through (for RangeStep)
        let v = t.slice((.., RangeStep::new(0, 4, 2))).unwrap();
        assert_eq!(v.shape(), &[3, 2]);

        // Backward compat: &[SliceRange]
        let v = t
            .slice(&[SliceRange::full(), SliceRange::index(0)])
            .unwrap();
        assert_eq!(v.shape(), &[3]);

        // Signed (negative) indices — wrap from end
        let v = t.slice((.., -1_isize)).unwrap(); // last column
        assert_eq!(v.shape(), &[3]);

        let v = t.slice((-2_isize.., ..)).unwrap(); // last 2 rows
        assert_eq!(v.shape(), &[2, 4]);

        let v = t.slice((.., -3..-1_isize)).unwrap(); // columns 1..3
        assert_eq!(v.shape(), &[3, 2]);

        // RangeFrom<usize> — now supported
        let v = t.slice((1_usize.., ..)).unwrap(); // rows 1..end
        assert_eq!(v.shape(), &[2, 4]);

        // Mutable slice with tuple
        let mut t = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let _s = t.slice_mut((.., 0_usize)).unwrap();
    }

    #[test]
    fn tensor_try_from_scalars_and_dims() {
        // Full-byte type: f32
        let scalars: Vec<f32> = (0..6).map(|i| i as f32 + 0.5).collect();
        let tensor = Tensor::<f32>::try_from_scalars(&scalars, &[2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.as_slice(), scalars.as_slice());

        // Length mismatch is reported as a shape error, not a panic.
        let bad = Tensor::<f32>::try_from_scalars(&scalars, &[2, 2]);
        assert!(matches!(bad, Err(TensorError::ShapeMismatch { .. })));

        // try_from_dims on a full-byte type: DimScalar = Scalar for f32.
        let dims: Vec<f32> = scalars.clone();
        let tensor = Tensor::<f32>::try_from_dims(&dims, &[3, 2]).unwrap();
        assert_eq!(tensor.shape(), &[3, 2]);
        assert_eq!(tensor.as_slice(), dims.as_slice());

        // Sub-byte round-trip: i4x2 packs 2 dims/byte.
        let i4_dims: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, 0];
        let tensor = Tensor::<crate::types::i4x2>::try_from_dims(&i4_dims, &[2, 4]).unwrap();
        assert_eq!(tensor.shape(), &[2, 4]);
        assert_eq!(tensor.storage_len(), 4);
    }

    #[test]
    fn fill_zeros_and_fill_on_tensor_and_span() {
        // Tensor: fill with zero, then fill with 7.0, verify storage matches.
        let mut tensor = unsafe { Tensor::<f32>::try_empty(&[3, 4]).unwrap() };
        tensor.fill_zeros();
        assert!(tensor.as_slice().iter().all(|&value| value == 0.0));
        tensor.fill(7.0);
        assert!(tensor.as_slice().iter().all(|&value| value == 7.0));

        // TensorSpan via tensor.span() also exposes Fill methods.
        let mut other = unsafe { Tensor::<f32>::try_empty(&[3, 4]).unwrap() };
        {
            let mut span = other.span();
            span.fill_zeros();
            span.fill(2.5);
        }
        assert!(other.as_slice().iter().all(|&value| value == 2.5));
    }

    #[test]
    fn copy_from_via_trait_round_trips_through_span() {
        let source_tensor = Tensor::<f32>::try_from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mut destination_tensor = unsafe { Tensor::<f32>::try_empty(&[2, 2]).unwrap() };
        // Tensor: CopyFrom<&[Scalar]>.
        destination_tensor.copy_from(source_tensor.as_slice()).unwrap();
        assert_eq!(destination_tensor.as_slice(), source_tensor.as_slice());

        // TensorSpan: CopyFrom<&TensorView>.
        let mut span_destination = unsafe { Tensor::<f32>::try_empty(&[2, 2]).unwrap() };
        {
            let mut span = span_destination.span();
            let view = source_tensor.view();
            span.copy_from(&view).unwrap();
        }
        assert_eq!(span_destination.as_slice(), source_tensor.as_slice());
    }

}

// endregion: Tests
