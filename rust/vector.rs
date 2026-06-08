//! Owning and non-owning vector types with signed indexing and sub-byte support.
//!
//! This module provides:
//!
//! - [`Vector`]: Owning, non-resizable, SIMD-aligned vector
//! - [`VectorView`]: Immutable, strided, non-owning view
//! - [`VectorSpan`]: Mutable, strided, non-owning view
//! - [`VectorIndex`]: Signed indexing trait (negative indices wrap from end)
//!
//! `Vector` and `VectorSpan` implement [`crate::tensor::Fill`] (`fill_zeros` / `fill`) and
//! [`crate::tensor::CopyFrom`] for the natural source type of each container.
//! `Vector<u1x8>`, `VectorView<u1x8>`, and `VectorSpan<u1x8>` expose `popcount` / `any_set` /
//! `none_set` / `all_set` directly so the bit-reduction spelling matches the
//! [`crate::reduce::BitwiseReductions`] trait used on tensor containers.
//!
//! All types use [`StorageElement`] as their element bound, with sub-byte types
//! (i4x2, u4x2, u1x8) supported via `try_get`/`try_set` and iterators.
//!
//! # Signed (Python-style) indexing
//!
//! Negative indices count from the end, mirroring Python's `list[-1]` idiom:
//!
//! ```ignore
//! use numkong::vector::Vector;
//! let v = Vector::<f32>::try_from_dims(&[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
//! // Last element via negative index.
//! assert_eq!(v.try_get(-1_i32).unwrap(), 50.0);
//! // Second-to-last.
//! assert_eq!(v.try_get(-2_i32).unwrap(), 40.0);
//! // Out-of-range negatives return an error rather than panic.
//! assert!(v.try_get(-6_i32).is_err());
//! ```
//!
//! # Sub-byte element iteration
//!
//! For packed types such as `i4x2`, `u4x2`, and `u1x8`, iteration yields one
//! logical dimension per step (not one packed storage value):
//!
//! ```ignore
//! use numkong::vector::Vector;
//! use numkong::types::u1x8;
//! let mut v = Vector::<u1x8>::try_zeros(8).unwrap();
//! for (i, mut bit) in v.iter_mut().enumerate() {
//!     *bit = if i % 2 == 0 { 1 } else { 0 };
//! }
//! assert_eq!(v.try_get(0_usize).unwrap(), 1);
//! assert_eq!(v.try_get(1_usize).unwrap(), 0);
//! ```

extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::tensor::{Allocator, CopyFrom, Fill, Global, Tensor, TensorError, SIMD_ALIGNMENT};
use crate::types::{DimMut, DimRef, FloatConvertible, NumberLike, StorageElement};

// region: VectorIndex — Signed Indexing

mod private {
    pub trait Sealed {}
}

/// Trait for vector index types. Supports signed integers (negative = from end).
pub trait VectorIndex: private::Sealed + Copy {
    /// Resolve this index to a `usize` offset, or `None` if out of bounds.
    fn resolve(self, len: usize) -> Option<usize>;
}

// Unsigned indexing: direct widen-to-usize, then bounds-check.
impl private::Sealed for usize {}
impl VectorIndex for usize {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        if self < len { Some(self) } else { None }
    }
}

impl private::Sealed for u8 {}
impl VectorIndex for u8 {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        let index = self as usize;
        if index < len { Some(index) } else { None }
    }
}

impl private::Sealed for u16 {}
impl VectorIndex for u16 {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        let index = self as usize;
        if index < len { Some(index) } else { None }
    }
}

impl private::Sealed for u32 {}
impl VectorIndex for u32 {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        let index = self as usize;
        if index < len { Some(index) } else { None }
    }
}

impl private::Sealed for u64 {}
impl VectorIndex for u64 {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        let index = self as usize;
        if index < len { Some(index) } else { None }
    }
}

// Signed indexing: negative values wrap from the end (-1 is the last element).
impl private::Sealed for isize {}
impl VectorIndex for isize {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        let index = if self >= 0 {
            self as usize
        } else {
            let negation = (-self) as usize;
            if negation > len { return None; }
            len - negation
        };
        if index < len { Some(index) } else { None }
    }
}

impl private::Sealed for i8 {}
impl VectorIndex for i8 {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        let index = if self >= 0 {
            self as usize
        } else {
            let negation = (-(self as isize)) as usize;
            if negation > len { return None; }
            len - negation
        };
        if index < len { Some(index) } else { None }
    }
}

impl private::Sealed for i16 {}
impl VectorIndex for i16 {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        let index = if self >= 0 {
            self as usize
        } else {
            let negation = (-(self as isize)) as usize;
            if negation > len { return None; }
            len - negation
        };
        if index < len { Some(index) } else { None }
    }
}

impl private::Sealed for i32 {}
impl VectorIndex for i32 {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        let index = if self >= 0 {
            self as usize
        } else {
            let negation = (-(self as isize)) as usize;
            if negation > len { return None; }
            len - negation
        };
        if index < len { Some(index) } else { None }
    }
}

impl private::Sealed for i64 {}
impl VectorIndex for i64 {
    #[inline]
    fn resolve(self, len: usize) -> Option<usize> {
        let index = if self >= 0 {
            self as usize
        } else {
            let negation = (-(self as isize)) as usize;
            if negation > len { return None; }
            len - negation
        };
        if index < len { Some(index) } else { None }
    }
}

// endregion: VectorIndex

// region: Sub-byte Proxy Types

/// Immutable reference to a nibble (4-bit value) within a packed byte.
///
/// Obtained by iterating over or indexing into a `Vector<i4x2>` / `Vector<u4x2>`;
/// the proxy remembers whether it refers to the low or high nibble of the shared
/// byte and decodes accordingly. Prefer [`NibbleRef::get_unsigned`] for `u4x2`
/// storage and [`NibbleRef::get_signed`] for `i4x2` storage (the latter
/// sign-extends bit 3 so the return range is `-8..=7`).
///
/// # Example
///
/// ```ignore
/// use numkong::vector::Vector;
/// use numkong::types::i4x2;
/// let mut v = Vector::<i4x2>::try_zeros(2).unwrap();
/// v.try_set(0_usize, -3).unwrap();
/// // Reading via `try_get` returns the same signed value NibbleRef::get_signed yields.
/// assert_eq!(v.try_get(0_usize).unwrap(), -3);
/// ```
pub struct NibbleRef<'a> {
    byte: *const u8,
    high: bool,
    _marker: PhantomData<&'a u8>,
}

impl<'a> NibbleRef<'a> {
    /// Read the nibble value as u8 (0..16).
    #[inline]
    pub fn get_unsigned(&self) -> u8 {
        // SAFETY: byte pointer is valid for the lifetime 'a
        let byte_value = unsafe { *self.byte };
        if self.high {
            byte_value >> 4
        } else {
            byte_value & 0x0F
        }
    }

    /// Read the nibble value as i8 (-8..7), sign-extending bit 3.
    #[inline]
    pub fn get_signed(&self) -> i8 {
        let nibble = self.get_unsigned();
        if nibble & 0x08 != 0 {
            nibble as i8 | !0x0Fi8
        } else {
            nibble as i8
        }
    }
}

/// Mutable reference to a nibble (4-bit value) within a packed byte.
///
/// Obtained via mutable iteration / indexing of a `Vector<i4x2>` or
/// `Vector<u4x2>`. Writes perform a read-modify-write on the shared byte so the
/// sibling nibble is preserved. Pair `get_unsigned` / `set_unsigned` for
/// unsigned storage and `get_signed` / `set_signed` for signed storage.
///
/// # Example
///
/// ```ignore
/// use numkong::vector::Vector;
/// use numkong::types::u4x2;
/// let mut v = Vector::<u4x2>::try_zeros(4).unwrap();
/// for (i, mut nibble) in v.iter_mut().enumerate() {
///     *nibble = i as u8;
/// }
/// assert_eq!(v.try_get(3_usize).unwrap(), 3);
/// ```
pub struct NibbleRefMut<'a> {
    byte: *mut u8,
    high: bool,
    _marker: PhantomData<&'a mut u8>,
}

impl<'a> NibbleRefMut<'a> {
    /// Read the nibble value as u8.
    #[inline]
    pub fn get_unsigned(&self) -> u8 {
        let byte_value = unsafe { *self.byte };
        if self.high {
            byte_value >> 4
        } else {
            byte_value & 0x0F
        }
    }

    /// Read the nibble value as i8, sign-extending bit 3.
    #[inline]
    pub fn get_signed(&self) -> i8 {
        let nibble = self.get_unsigned();
        if nibble & 0x08 != 0 {
            nibble as i8 | !0x0Fi8
        } else {
            nibble as i8
        }
    }

    /// Set the nibble to an unsigned value (low 4 bits used).
    #[inline]
    pub fn set_unsigned(&self, value: u8) {
        // SAFETY: byte pointer is valid and mutable for the lifetime 'a
        unsafe {
            let byte_value = *self.byte;
            if self.high {
                *self.byte = (byte_value & 0x0F) | ((value & 0x0F) << 4);
            } else {
                *self.byte = (byte_value & 0xF0) | (value & 0x0F);
            }
        }
    }

    /// Set the nibble to a signed value (low 4 bits used).
    #[inline]
    pub fn set_signed(&self, value: i8) {
        self.set_unsigned(value as u8);
    }
}

/// Immutable reference to a single bit within a packed byte.
///
/// Obtained by iterating over or indexing into a `Vector<u1x8>`. The `mask`
/// selects which of the eight bits the proxy represents; the byte itself is
/// shared with the seven sibling proxies for the same byte.
///
/// # Example
///
/// ```ignore
/// use numkong::vector::Vector;
/// use numkong::types::u1x8;
/// let mut v = Vector::<u1x8>::try_zeros(8).unwrap();
/// v.try_set(3_usize, 1).unwrap();
/// // Immutable iteration yields BitRef-like proxies that deref to the bit value.
/// let bits: Vec<u8> = v.iter().map(|b| *b).collect();
/// assert_eq!(bits, vec![0, 0, 0, 1, 0, 0, 0, 0]);
/// ```
pub struct BitRef<'a> {
    byte: *const u8,
    mask: u8,
    _marker: PhantomData<&'a u8>,
}

impl<'a> BitRef<'a> {
    /// Read the bit as bool.
    #[inline]
    pub fn get(&self) -> bool {
        // SAFETY: byte pointer is valid for the lifetime 'a
        (unsafe { *self.byte } & self.mask) != 0
    }
}

/// Mutable reference to a single bit within a packed byte.
///
/// Obtained via mutable iteration / indexing of a `Vector<u1x8>`. Writes OR or
/// AND the stored byte so sibling bits are preserved; the `mask` selects which
/// bit this proxy represents.
///
/// # Example
///
/// ```ignore
/// use numkong::vector::Vector;
/// use numkong::types::u1x8;
/// let mut v = Vector::<u1x8>::try_zeros(8).unwrap();
/// for (i, mut bit) in v.iter_mut().enumerate() {
///     *bit = (i % 2 == 0) as u8;
/// }
/// assert_eq!(v.try_get(0_usize).unwrap(), 1);
/// assert_eq!(v.try_get(1_usize).unwrap(), 0);
/// ```
pub struct BitRefMut<'a> {
    byte: *mut u8,
    mask: u8,
    _marker: PhantomData<&'a mut u8>,
}

impl<'a> BitRefMut<'a> {
    /// Read the bit as bool.
    #[inline]
    pub fn get(&self) -> bool {
        // SAFETY: byte pointer is valid for the lifetime 'a
        (unsafe { *self.byte } & self.mask) != 0
    }

    /// Set the bit.
    #[inline]
    pub fn set(&self, value: bool) {
        // SAFETY: byte pointer is valid and mutable for the lifetime 'a
        unsafe {
            if value {
                *self.byte |= self.mask;
            } else {
                *self.byte &= !self.mask;
            }
        }
    }
}

// endregion: Sub-byte Proxy Types

// region: Vector

/// Owning, non-resizable, SIMD-aligned vector.
///
/// Size is fixed at construction. Uses [`StorageElement`] for element types,
/// including sub-byte packed types via `Scalar::dimensions_per_value()`.
///
/// For normal types (`dimensions_per_value() == 1`), supports `Index`/`IndexMut`.
/// For sub-byte types, use `try_get`/`try_set` or iterators.
pub struct Vector<Scalar: StorageElement, Alloc: Allocator = Global> {
    /// Pointer to the allocated buffer (typed as `Scalar` for alignment).
    data: NonNull<Scalar>,
    /// Number of logical dimensions. Storage size is derived as
    /// `dims_to_values::<Scalar>(self.dims)`.
    dims: usize,
    /// Allocator instance.
    alloc: Alloc,
}

unsafe impl<Scalar: StorageElement + Send, Alloc: Allocator + Send> Send for Vector<Scalar, Alloc> {}
unsafe impl<Scalar: StorageElement + Sync, Alloc: Allocator + Sync> Sync for Vector<Scalar, Alloc> {}

impl<Scalar: StorageElement, Alloc: Allocator> Drop for Vector<Scalar, Alloc> {
    fn drop(&mut self) {
        let storage_count = dims_to_values::<Scalar>(self.dims);
        if storage_count > 0 {
            let layout = alloc::alloc::Layout::from_size_align(
                storage_count * core::mem::size_of::<Scalar>(),
                SIMD_ALIGNMENT,
            )
            .unwrap();
            // SAFETY: data was allocated with this layout in try_zeros_in,
            // and storage_count > 0 guarantees the pointer is non-dangling.
            unsafe {
                self.alloc.deallocate(
                    NonNull::new_unchecked(self.data.as_ptr() as *mut u8),
                    layout,
                );
            }
        }
    }
}

/// Convert dimension count to value count for `Scalar`.
///
/// For sub-byte types where `dimensions_per_value() > 1`, this is a ceiling
/// division via [`usize::div_ceil`].
#[inline]
fn dims_to_values<Scalar: StorageElement>(dims: usize) -> usize {
    dims.div_ceil(Scalar::dimensions_per_value())
}

impl<Scalar: StorageElement, Alloc: Allocator> Vector<Scalar, Alloc> {
    /// Construct a vector from raw parts, taking ownership of the allocation.
    ///
    /// # Safety
    /// - `data` must point to an allocation obtained from `alloc`, aligned to
    ///   [`SIMD_ALIGNMENT`], whose backing buffer is sized for the storage count
    ///   implied by `dims` (`dims_to_values::<Scalar>(dims)` slots of `Scalar`).
    /// - The caller must not free the memory (this vector takes ownership).
    pub unsafe fn from_raw_parts(data: NonNull<Scalar>, dims: usize, alloc: Alloc) -> Self {
        Self { data, dims, alloc }
    }

    /// Try to create a zero-initialized vector with the given number of dimensions.
    pub fn try_zeros_in(dims: usize, alloc: Alloc) -> Result<Self, TensorError> {
        let storage_count = dims_to_values::<Scalar>(dims);
        if storage_count == 0 {
            return Ok(Self {
                data: NonNull::dangling(),
                dims: 0,
                alloc,
            });
        }
        let size = storage_count * core::mem::size_of::<Scalar>();
        let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
            .map_err(|_| TensorError::AllocationFailed)?;
        let ptr = alloc
            .allocate(layout)
            .ok_or(TensorError::AllocationFailed)?;
        unsafe { core::ptr::write_bytes(ptr.as_ptr(), 0, size) };
        Ok(Self {
            data: unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut Scalar) },
            dims,
            alloc,
        })
    }

    /// Try to create a vector filled with `value`.
    pub fn try_full_in(dims: usize, value: Scalar, alloc: Alloc) -> Result<Self, TensorError> {
        let v = Self::try_zeros_in(dims, alloc)?;
        let storage_count = dims_to_values::<Scalar>(v.dims);
        if storage_count > 0 {
            let ptr = v.data.as_ptr();
            for i in 0..storage_count {
                unsafe { ptr.add(i).write(value) };
            }
        }
        Ok(v)
    }

    /// Try to create a vector filled with ones.
    pub fn try_ones_in(dims: usize, alloc: Alloc) -> Result<Self, TensorError>
    where
        Scalar: NumberLike,
    {
        Self::try_full_in(dims, Scalar::one(), alloc)
    }

    /// Try to create an uninitialized vector.
    ///
    /// # Safety
    /// The returned vector's contents are uninitialized. Reading from it before
    /// writing is undefined behavior.
    pub unsafe fn try_empty_in(dims: usize, alloc: Alloc) -> Result<Self, TensorError> {
        let storage_count = dims_to_values::<Scalar>(dims);
        if storage_count == 0 {
            return Ok(Self {
                data: NonNull::dangling(),
                dims: 0,
                alloc,
            });
        }
        let size = storage_count * core::mem::size_of::<Scalar>();
        let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
            .map_err(|_| TensorError::AllocationFailed)?;
        let ptr = alloc
            .allocate(layout)
            .ok_or(TensorError::AllocationFailed)?;
        Ok(Self {
            data: unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut Scalar) },
            dims,
            alloc,
        })
    }

    /// Try to create a vector from a slice of scalars (f32 values).
    ///
    /// Each f32 value is converted through `DimScalar::from_f32()` before storage.
    pub fn try_from_scalars_in(scalars: &[f32], alloc: Alloc) -> Result<Self, TensorError>
    where
        Scalar: FloatConvertible,
    {
        let element_count = scalars.len();
        let mut v = Self::try_zeros_in(element_count, alloc)?;
        for (i, &s) in scalars.iter().enumerate() {
            v.try_set(i, Scalar::DimScalar::from_f32(s))?;
        }
        Ok(v)
    }

    /// Try to create a vector from a slice of per-dimension scalars.
    ///
    /// Each element in `dim_values` corresponds to one logical dimension.
    pub fn try_from_dims_in(
        dim_values: &[Scalar::DimScalar],
        alloc: Alloc,
    ) -> Result<Self, TensorError>
    where
        Scalar: FloatConvertible,
    {
        let element_count = dim_values.len();
        let mut v = Self::try_zeros_in(element_count, alloc)?;
        for (i, &d) in dim_values.iter().enumerate() {
            v.try_set(i, d)?;
        }
        Ok(v)
    }

    /// Number of logical dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of logical dimensions (same as `dims()`).
    #[inline]
    pub fn size(&self) -> usize {
        self.dims
    }

    /// Number of underlying storage values (`Scalar` instances).
    #[inline]
    pub fn size_values(&self) -> usize {
        dims_to_values::<Scalar>(self.dims)
    }

    /// Returns true if the vector has zero dimensions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims == 0
    }

    /// Raw pointer to the underlying data.
    #[inline]
    pub fn as_ptr(&self) -> *const Scalar {
        self.data.as_ptr()
    }

    /// Mutable raw pointer to the underlying data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Scalar {
        self.data.as_ptr()
    }

    /// Size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        dims_to_values::<Scalar>(self.dims) * core::mem::size_of::<Scalar>()
    }

    /// Create an immutable view of this vector.
    #[inline]
    pub fn view(&self) -> VectorView<'_, Scalar> {
        VectorView {
            data: self.data.as_ptr() as *const Scalar,
            dims: self.dims,
            stride_bytes: core::mem::size_of::<Scalar>() as isize,
            _marker: PhantomData,
        }
    }

    /// Create a mutable span of this vector.
    #[inline]
    pub fn span(&mut self) -> VectorSpan<'_, Scalar> {
        VectorSpan {
            data: self.data.as_ptr(),
            dims: self.dims,
            stride_bytes: core::mem::size_of::<Scalar>() as isize,
            _marker: PhantomData,
        }
    }

    /// Try to get the logical dimension at `index` (supports signed indexing).
    ///
    /// Returns the native `DimScalar` type (e.g., `f64` for `Vector<f64>`, `i8` for `Vector<i4x2>`).
    /// For sub-byte types, unpacks the appropriate sub-dimension from the packed storage value.
    ///
    /// # Examples
    ///
    /// Positive and Python-style negative indexing:
    ///
    /// ```ignore
    /// use numkong::vector::Vector;
    /// let v = Vector::<f32>::try_from_dims(&[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    /// assert_eq!(v.try_get(0_usize).unwrap(), 10.0);
    /// assert_eq!(v.try_get(-1_i32).unwrap(), 50.0);
    /// assert_eq!(v.try_get(-5_i32).unwrap(), 10.0);
    /// // Out-of-range indices return an error instead of panicking.
    /// assert!(v.try_get(5_usize).is_err());
    /// assert!(v.try_get(-6_i32).is_err());
    /// ```
    #[inline]
    pub fn try_get<AnyIndex: VectorIndex>(
        &self,
        index: AnyIndex,
    ) -> Result<Scalar::DimScalar, TensorError>
    where
        Scalar: FloatConvertible,
    {
        let i = index
            .resolve(self.dims)
            .ok_or(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.dims,
            })?;
        let dims_per_value = Scalar::dimensions_per_value();
        let value_index = i / dims_per_value;
        let sub_index = i % dims_per_value;
        // SAFETY: value_index < dims_to_values(self.dims), implied by the bounds check above
        let packed = unsafe { *self.data.as_ptr().add(value_index) };
        Ok(packed.unpack().as_ref()[sub_index])
    }

    /// Try to set the logical dimension at `index`.
    ///
    /// Accepts the native `DimScalar` type. For sub-byte types, reads the current packed value,
    /// updates the targeted sub-dimension, and writes back the modified packed value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use numkong::vector::Vector;
    /// let mut v = Vector::<f32>::try_zeros(4).unwrap();
    /// v.try_set(0_usize, 1.0).unwrap();
    /// // Signed indices let you write to the tail without computing the length.
    /// v.try_set(-1_i32, 4.0).unwrap();
    /// assert_eq!(v.try_get(0_usize).unwrap(), 1.0);
    /// assert_eq!(v.try_get(3_usize).unwrap(), 4.0);
    /// ```
    #[inline]
    pub fn try_set<AnyIndex: VectorIndex>(
        &mut self,
        index: AnyIndex,
        value: Scalar::DimScalar,
    ) -> Result<(), TensorError>
    where
        Scalar: FloatConvertible,
    {
        let i = index
            .resolve(self.dims)
            .ok_or(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.dims,
            })?;
        let dims_per_value = Scalar::dimensions_per_value();
        let value_index = i / dims_per_value;
        let sub_index = i % dims_per_value;
        // SAFETY: value_index < dims_to_values(self.dims), implied by the bounds check above
        let ptr = unsafe { self.data.as_ptr().add(value_index) };
        let mut unpacked = unsafe { *ptr }.unpack();
        unpacked.as_mut()[sub_index] = value;
        unsafe { ptr.write(Scalar::pack(unpacked)) };
        Ok(())
    }

    /// Get a slice of the underlying storage values. For sub-byte types this returns
    /// the packed storage values, not the logical dimensions.
    #[inline]
    pub fn as_slice(&self) -> &[Scalar] {
        let storage_count = dims_to_values::<Scalar>(self.dims);
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), storage_count) }
    }

    /// Get a mutable slice of the underlying storage values.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Scalar] {
        let storage_count = dims_to_values::<Scalar>(self.dims);
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), storage_count) }
    }

    /// Returns an iterator over the logical dimension values, yielding [`DimRef`] proxies.
    pub fn iter(&self) -> VectorViewIterator<'_, Scalar>
    where
        Scalar: FloatConvertible,
    {
        self.view().iter()
    }

    /// Returns a mutable iterator over the logical dimension values, yielding [`DimMut`] proxies.
    pub fn iter_mut(&mut self) -> VectorSpanIterator<'_, Scalar>
    where
        Scalar: FloatConvertible,
    {
        VectorSpanIterator {
            data: self.data.as_ptr(),
            stride_bytes: core::mem::size_of::<Scalar>() as isize,
            front: 0,
            back: self.dims,
            _marker: PhantomData,
        }
    }
}

impl<Scalar: StorageElement, Alloc: Allocator> Vector<Scalar, Alloc> {
    /// Convert this vector into a 1D tensor, transferring ownership without copying.
    pub fn try_into_tensor<const MAX_RANK: usize>(
        self,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, TensorError> {
        if MAX_RANK == 0 {
            return Err(TensorError::TooManyRanks { got: 1 });
        }
        let mut shape = [0usize; MAX_RANK];
        shape[0] = self.dims;
        let mut strides = [0isize; MAX_RANK];
        strides[0] = core::mem::size_of::<Scalar>() as isize;
        let data = self.data;
        // SAFETY: we read the allocator out before forget, transferring ownership
        let alloc = unsafe { core::ptr::read(&self.alloc) };
        core::mem::forget(self);
        // SAFETY: storage size derived from `shape[0]` matches the original allocation.
        let tensor = unsafe { Tensor::from_raw_parts(data, shape, strides, 1, alloc) };
        Ok(tensor)
    }
}

impl<Scalar: StorageElement> Vector<Scalar, Global> {
    /// Create a zero-initialized vector with the global allocator.
    pub fn try_zeros(dims: usize) -> Result<Self, TensorError> {
        Self::try_zeros_in(dims, Global)
    }

    /// Create a vector filled with `value`.
    pub fn try_full(dims: usize, value: Scalar) -> Result<Self, TensorError> {
        Self::try_full_in(dims, value, Global)
    }

    /// Create a vector filled with ones.
    pub fn try_ones(dims: usize) -> Result<Self, TensorError>
    where
        Scalar: NumberLike,
    {
        Self::try_full(dims, Scalar::one())
    }

    /// Create an uninitialized vector.
    ///
    /// # Safety
    /// The returned vector's contents are uninitialized. Reading from it before
    /// writing is undefined behavior.
    pub unsafe fn try_empty(dims: usize) -> Result<Self, TensorError> {
        unsafe { Self::try_empty_in(dims, Global) }
    }

    /// Create a vector from scalar f32 values.
    pub fn try_from_scalars(scalars: &[f32]) -> Result<Self, TensorError>
    where
        Scalar: FloatConvertible,
    {
        Self::try_from_scalars_in(scalars, Global)
    }

    /// Create a vector from per-dimension scalars.
    pub fn try_from_dims(dims: &[Scalar::DimScalar]) -> Result<Self, TensorError>
    where
        Scalar: FloatConvertible,
    {
        Self::try_from_dims_in(dims, Global)
    }
}

// Index for normal types (dimensions_per_value == 1)
impl<AnyIndex: VectorIndex, Scalar: StorageElement, Alloc: Allocator> core::ops::Index<AnyIndex>
    for Vector<Scalar, Alloc>
{
    type Output = Scalar;

    #[inline]
    fn index(&self, index: AnyIndex) -> &Scalar {
        let i = index
            .resolve(self.dims)
            .expect("vector index out of bounds");
        debug_assert_eq!(
            Scalar::dimensions_per_value(),
            1,
            "Index trait not supported for sub-byte types"
        );
        unsafe { &*self.data.as_ptr().add(i) }
    }
}

impl<AnyIndex: VectorIndex, Scalar: StorageElement, Alloc: Allocator> core::ops::IndexMut<AnyIndex>
    for Vector<Scalar, Alloc>
{
    #[inline]
    fn index_mut(&mut self, index: AnyIndex) -> &mut Scalar {
        let i = index
            .resolve(self.dims)
            .expect("vector index out of bounds");
        debug_assert_eq!(
            Scalar::dimensions_per_value(),
            1,
            "IndexMut trait not supported for sub-byte types"
        );
        unsafe { &mut *self.data.as_ptr().add(i) }
    }
}

impl<Scalar: StorageElement + Clone, Alloc: Allocator + Clone> Vector<Scalar, Alloc> {
    /// Try to clone this vector, returning an error on allocation failure.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        let storage_count = dims_to_values::<Scalar>(self.dims);
        if storage_count == 0 {
            return Ok(Self {
                data: NonNull::dangling(),
                dims: 0,
                alloc: self.alloc.clone(),
            });
        }
        let size = storage_count * core::mem::size_of::<Scalar>();
        let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
            .map_err(|_| TensorError::AllocationFailed)?;
        let ptr = self
            .alloc
            .allocate(layout)
            .ok_or(TensorError::AllocationFailed)?;
        unsafe {
            core::ptr::copy_nonoverlapping(self.data.as_ptr() as *const u8, ptr.as_ptr(), size);
        }
        Ok(Self {
            data: unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut Scalar) },
            dims: self.dims,
            alloc: self.alloc.clone(),
        })
    }
}

impl<Scalar: StorageElement + Clone, Alloc: Allocator + Clone> Clone for Vector<Scalar, Alloc> {
    fn clone(&self) -> Self {
        self.try_clone().expect("vector clone allocation failed")
    }
}

impl<Scalar: StorageElement> Default for Vector<Scalar, Global> {
    fn default() -> Self {
        Self {
            data: NonNull::dangling(),
            dims: 0,
            alloc: Global,
        }
    }
}

// endregion: Vector

// region: VectorView

/// Immutable, possibly strided, non-owning view into a vector.
///
/// `VectorView` is a zero-copy borrow: it holds a raw pointer, a dimension
/// count, and a byte stride between consecutive logical elements. It never
/// frees its memory and is tied to the lifetime `'a` of whatever owns the
/// storage (typically a [`Vector`] obtained via [`Vector::view`], or a row /
/// column of a tensor).
///
/// Because the stride is stored in bytes and may be negative, views can walk
/// memory in either direction, skip entries (see [`VectorView::try_strided`]),
/// or expose a reversed iteration order (see [`VectorView::rev`]).
///
/// Use [`VectorView`] when only read access is needed. Its mutable counterpart
/// is [`VectorSpan`], which offers the same striding semantics plus element
/// writes. Both types support the [`VectorIndex`] trait, so `view[0_usize]`,
/// `view[-1_i32]`, and friends all resolve via the same Python-style rules.
pub struct VectorView<'a, Scalar: StorageElement> {
    data: *const Scalar,
    dims: usize,
    stride_bytes: isize,
    _marker: PhantomData<&'a Scalar>,
}

unsafe impl<'a, Scalar: StorageElement + Sync> Send for VectorView<'a, Scalar> {}
unsafe impl<'a, Scalar: StorageElement + Sync> Sync for VectorView<'a, Scalar> {}

impl<'a, Scalar: StorageElement> Clone for VectorView<'a, Scalar> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, Scalar: StorageElement> Copy for VectorView<'a, Scalar> {}

impl<'a, Scalar: StorageElement> VectorView<'a, Scalar> {
    /// Create a view from a raw pointer, dimension count, and byte stride.
    ///
    /// # Safety
    /// - `data` must be valid for reads of `dims` elements at the given stride.
    /// - The pointed-to memory must outlive `'a`.
    /// - `stride_bytes` must be non-zero for non-empty views.
    #[inline]
    pub unsafe fn from_raw_parts(data: *const Scalar, dims: usize, stride_bytes: isize) -> Self {
        Self {
            data,
            dims,
            stride_bytes,
            _marker: PhantomData,
        }
    }

    /// Number of logical dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of logical dimensions (alias for dims).
    #[inline]
    pub fn size(&self) -> usize {
        self.dims
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims == 0
    }

    /// Stride in bytes between consecutive elements.
    #[inline]
    pub fn stride_bytes(&self) -> isize {
        self.stride_bytes
    }

    /// Returns true if elements are stored contiguously (stride == sizeof(Scalar)).
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.stride_bytes == core::mem::size_of::<Scalar>() as isize
    }

    /// Get the underlying pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const Scalar {
        self.data
    }

    /// Get a contiguous slice, if this view is contiguous.
    #[inline]
    pub fn as_contiguous_slice(&self) -> Option<&'a [Scalar]> {
        if self.is_contiguous() && Scalar::dimensions_per_value() == 1 {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.dims) })
        } else {
            None
        }
    }

    /// Try to get element at index (supports signed indexing).
    ///
    /// Returns the native `DimScalar` type. For sub-byte types, uses value_index
    /// for stride-based pointer walks to avoid buffer overread.
    #[inline]
    pub fn try_get<AnyIndex: VectorIndex>(
        &self,
        index: AnyIndex,
    ) -> Result<Scalar::DimScalar, TensorError>
    where
        Scalar: FloatConvertible,
    {
        let i = index
            .resolve(self.dims)
            .ok_or(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.dims,
            })?;
        let dims_per_value = Scalar::dimensions_per_value();
        let value_index = i / dims_per_value;
        let sub_index = i % dims_per_value;
        // SAFETY: stride * value_index stays within allocation
        let ptr = unsafe {
            (self.data as *const u8).offset(self.stride_bytes * value_index as isize)
                as *const Scalar
        };
        Ok(unsafe { *ptr }.unpack().as_ref()[sub_index])
    }

    /// Create a reversed view by negating the stride and pointing to the last element.
    ///
    /// The returned view has the same number of dimensions but iterates in the
    /// opposite direction. For an empty view, returns a copy unchanged.
    pub fn rev(&self) -> Self {
        if self.dims == 0 {
            return *self;
        }
        let last_offset = self.stride_bytes * (self.dims as isize - 1);
        Self {
            data: unsafe { (self.data as *const u8).offset(last_offset) as *const Scalar },
            dims: self.dims,
            stride_bytes: -self.stride_bytes,
            _marker: PhantomData,
        }
    }

    /// Create a sub-view with start, end, and step (Python-style slicing).
    ///
    /// Supports negative steps for reverse iteration. `step` must be non-zero.
    /// Returns an error if `start` or `end` exceed `dims()`, or if `step == 0`.
    pub fn try_strided(&self, start: usize, end: usize, step: isize) -> Result<Self, TensorError> {
        if start > self.dims || end > self.dims || step == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: start.max(end),
                size: self.dims,
            });
        }
        let count = if step > 0 {
            if end > start {
                (end - start + step as usize - 1) / step as usize
            } else {
                0
            }
        } else if start > end {
            let abs_step = (-step) as usize;
            (start - end + abs_step - 1) / abs_step
        } else {
            0
        };
        let new_data = unsafe {
            (self.data as *const u8).offset(self.stride_bytes * start as isize) as *const Scalar
        };
        Ok(Self {
            data: new_data,
            dims: count,
            stride_bytes: self.stride_bytes * step,
            _marker: PhantomData,
        })
    }

    /// Returns an iterator over logical dimension values, yielding [`DimRef`] proxies.
    pub fn iter(&self) -> VectorViewIterator<'a, Scalar>
    where
        Scalar: FloatConvertible,
    {
        VectorViewIterator {
            data: self.data,
            stride_bytes: self.stride_bytes,
            front: 0,
            back: self.dims,
            _marker: PhantomData,
        }
    }
}

impl<'a, AnyIndex: VectorIndex, Scalar: StorageElement> core::ops::Index<AnyIndex>
    for VectorView<'a, Scalar>
{
    type Output = Scalar;

    #[inline]
    fn index(&self, index: AnyIndex) -> &Scalar {
        let i = index.resolve(self.dims).expect("view index out of bounds");
        debug_assert_eq!(
            Scalar::dimensions_per_value(),
            1,
            "Index trait not supported for sub-byte types"
        );
        unsafe {
            &*((self.data as *const u8).offset(self.stride_bytes * i as isize) as *const Scalar)
        }
    }
}

// endregion: VectorView

// region: VectorSpan

/// Mutable, possibly strided, non-owning view into a vector.
///
/// `VectorSpan` is the read-write counterpart of [`VectorView`]: it still
/// performs zero-copy borrowing via a raw pointer, dimension count, and byte
/// stride, but additionally allows element writes through `try_set`, `fill`,
/// `iter_mut`, and `IndexMut`. Spans are typically obtained via
/// [`Vector::span`] or a mutable slice of a tensor row.
///
/// Like views, spans support signed ([`VectorIndex`]) indexing, negative
/// strides via [`VectorSpan::as_view`] + [`VectorView::rev`], and Python-style
/// slicing through their view projection. The `'a` lifetime mirrors the owner
/// of the underlying memory, and Rust's borrow checker prevents aliasing a
/// single span with any other reference for the duration of `'a`.
pub struct VectorSpan<'a, Scalar: StorageElement> {
    data: *mut Scalar,
    dims: usize,
    stride_bytes: isize,
    _marker: PhantomData<&'a mut Scalar>,
}

unsafe impl<'a, Scalar: StorageElement + Send> Send for VectorSpan<'a, Scalar> {}
unsafe impl<'a, Scalar: StorageElement + Sync> Sync for VectorSpan<'a, Scalar> {}

impl<'a, Scalar: StorageElement> VectorSpan<'a, Scalar> {
    /// Create a mutable view from a raw pointer, dimension count, and byte stride.
    ///
    /// # Safety
    /// - `data` must be valid for reads and writes of `dims` elements at the given stride.
    /// - The pointed-to memory must outlive `'a`.
    /// - `stride_bytes` must be non-zero for non-empty views.
    /// - No other references to the memory may exist for the duration of `'a`.
    #[inline]
    pub unsafe fn from_raw_parts(data: *mut Scalar, dims: usize, stride_bytes: isize) -> Self {
        Self {
            data,
            dims,
            stride_bytes,
            _marker: PhantomData,
        }
    }

    /// Number of logical dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of logical dimensions (alias for dims).
    #[inline]
    pub fn size(&self) -> usize {
        self.dims
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims == 0
    }

    /// Stride in bytes.
    #[inline]
    pub fn stride_bytes(&self) -> isize {
        self.stride_bytes
    }

    /// Returns true if contiguous.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.stride_bytes == core::mem::size_of::<Scalar>() as isize
    }

    /// Get the underlying pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const Scalar {
        self.data
    }

    /// Get the mutable underlying pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Scalar {
        self.data
    }

    /// Reborrow as an immutable view, sharing the same data pointer and stride.
    pub fn as_view(&self) -> VectorView<'_, Scalar> {
        VectorView {
            data: self.data,
            dims: self.dims,
            stride_bytes: self.stride_bytes,
            _marker: PhantomData,
        }
    }

    /// Try to get element at index.
    #[inline]
    pub fn try_get<AnyIndex: VectorIndex>(
        &self,
        index: AnyIndex,
    ) -> Result<Scalar::DimScalar, TensorError>
    where
        Scalar: FloatConvertible,
    {
        self.as_view().try_get(index)
    }

    /// Try to set the element at `index`.
    ///
    /// Accepts the native `DimScalar` type. For sub-byte types, uses value_index
    /// for stride-based pointer walks to avoid buffer overwrite.
    #[inline]
    pub fn try_set<AnyIndex: VectorIndex>(
        &mut self,
        index: AnyIndex,
        value: Scalar::DimScalar,
    ) -> Result<(), TensorError>
    where
        Scalar: FloatConvertible,
    {
        let i = index
            .resolve(self.dims)
            .ok_or(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.dims,
            })?;
        let dims_per_value = Scalar::dimensions_per_value();
        let value_index = i / dims_per_value;
        let sub_index = i % dims_per_value;
        // SAFETY: stride * value_index stays within allocation
        let ptr = unsafe {
            (self.data as *mut u8).offset(self.stride_bytes * value_index as isize) as *mut Scalar
        };
        let mut unpacked = unsafe { *ptr }.unpack();
        unpacked.as_mut()[sub_index] = value;
        unsafe { ptr.write(Scalar::pack(unpacked)) };
        Ok(())
    }

    /// Get a contiguous slice, if this span is contiguous.
    #[inline]
    pub fn as_contiguous_slice(&self) -> Option<&[Scalar]> {
        if self.is_contiguous() && Scalar::dimensions_per_value() == 1 {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.dims) })
        } else {
            None
        }
    }

    /// Get a mutable contiguous slice, if this span is contiguous.
    #[inline]
    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [Scalar]> {
        if self.is_contiguous() && Scalar::dimensions_per_value() == 1 {
            Some(unsafe { core::slice::from_raw_parts_mut(self.data, self.dims) })
        } else {
            None
        }
    }

    /// Returns an iterator over logical dimension values, yielding [`DimRef`] proxies.
    pub fn iter(&self) -> VectorViewIterator<'_, Scalar>
    where
        Scalar: FloatConvertible,
    {
        VectorViewIterator {
            data: self.data,
            stride_bytes: self.stride_bytes,
            front: 0,
            back: self.dims,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable iterator over logical dimension values, yielding [`DimMut`] proxies.
    pub fn iter_mut(&mut self) -> VectorSpanIterator<'_, Scalar>
    where
        Scalar: FloatConvertible,
    {
        VectorSpanIterator {
            data: self.data,
            stride_bytes: self.stride_bytes,
            front: 0,
            back: self.dims,
            _marker: PhantomData,
        }
    }
}

impl<'a, AnyIndex: VectorIndex, Scalar: StorageElement> core::ops::Index<AnyIndex>
    for VectorSpan<'a, Scalar>
{
    type Output = Scalar;

    #[inline]
    fn index(&self, index: AnyIndex) -> &Scalar {
        let i = index.resolve(self.dims).expect("span index out of bounds");
        debug_assert_eq!(
            Scalar::dimensions_per_value(),
            1,
            "Index trait not supported for sub-byte types"
        );
        unsafe {
            &*((self.data as *const u8).offset(self.stride_bytes * i as isize) as *const Scalar)
        }
    }
}

impl<'a, AnyIndex: VectorIndex, Scalar: StorageElement> core::ops::IndexMut<AnyIndex>
    for VectorSpan<'a, Scalar>
{
    #[inline]
    fn index_mut(&mut self, index: AnyIndex) -> &mut Scalar {
        let i = index.resolve(self.dims).expect("span index out of bounds");
        debug_assert_eq!(
            Scalar::dimensions_per_value(),
            1,
            "IndexMut trait not supported for sub-byte types"
        );
        unsafe {
            &mut *((self.data as *mut u8).offset(self.stride_bytes * i as isize) as *mut Scalar)
        }
    }
}

// endregion: VectorSpan

// region: Bit Reductions on u1x8 vectors

use crate::reduce::ReduceMoments;
use crate::types::u1x8;

/// Population count of a contiguous packed-bit storage slice.
///
/// Shared body for the inherent `popcount` methods on `Vector<u1x8>`,
/// `VectorView<u1x8>`, and `VectorSpan<u1x8>`. Sub-byte vector views always
/// have a one-byte stride (you cannot stride by less than a byte), so
/// contiguous storage is implied — the slice form is always valid.
fn popcount_u1x8_storage(storage: &[u1x8]) -> u64 {
    if storage.is_empty() {
        return 0;
    }
    let (sum, _sum_of_squares) = u1x8::reduce_moments(storage, core::mem::size_of::<u1x8>());
    sum
}

impl<Alloc: Allocator> Vector<u1x8, Alloc> {
    /// Number of set bits across the entire vector.
    pub fn popcount(&self) -> u64 {
        let storage_count = self.dims.div_ceil(u1x8::dimensions_per_value());
        let storage =
            unsafe { core::slice::from_raw_parts(self.data.as_ptr(), storage_count) };
        popcount_u1x8_storage(storage)
    }

    /// `true` if at least one bit in the vector is set.
    pub fn any_set(&self) -> bool {
        self.popcount() != 0
    }

    /// `true` if no bit in the vector is set.
    pub fn none_set(&self) -> bool {
        !self.any_set()
    }

    /// `true` if every bit in the vector is set.
    pub fn all_set(&self) -> bool {
        self.popcount() == self.dims as u64
    }
}

impl<'a> VectorView<'a, u1x8> {
    /// Number of set bits across the entire vector view.
    pub fn popcount(&self) -> u64 {
        let storage_count = self.dims.div_ceil(u1x8::dimensions_per_value());
        let storage = unsafe { core::slice::from_raw_parts(self.data, storage_count) };
        popcount_u1x8_storage(storage)
    }

    /// `true` if at least one bit in the vector view is set.
    pub fn any_set(&self) -> bool {
        self.popcount() != 0
    }

    /// `true` if no bit in the vector view is set.
    pub fn none_set(&self) -> bool {
        !self.any_set()
    }

    /// `true` if every bit in the vector view is set.
    pub fn all_set(&self) -> bool {
        self.popcount() == self.dims as u64
    }
}

impl<'a> VectorSpan<'a, u1x8> {
    /// Number of set bits across the entire vector span.
    pub fn popcount(&self) -> u64 {
        let storage_count = self.dims.div_ceil(u1x8::dimensions_per_value());
        let storage = unsafe { core::slice::from_raw_parts(self.data as *const u1x8, storage_count) };
        popcount_u1x8_storage(storage)
    }

    /// `true` if at least one bit in the vector span is set.
    pub fn any_set(&self) -> bool {
        self.popcount() != 0
    }

    /// `true` if no bit in the vector span is set.
    pub fn none_set(&self) -> bool {
        !self.any_set()
    }

    /// `true` if every bit in the vector span is set.
    pub fn all_set(&self) -> bool {
        self.popcount() == self.dims as u64
    }
}

// endregion: Bit Reductions on u1x8 vectors

// region: Fill / CopyFrom Implementations

impl<Scalar: StorageElement, Alloc: Allocator> Fill<Scalar> for Vector<Scalar, Alloc> {
    fn fill_zeros(&mut self) {
        let storage_count = dims_to_values::<Scalar>(self.dims);
        if storage_count == 0 {
            return;
        }
        unsafe {
            core::ptr::write_bytes(self.data.as_ptr(), 0, storage_count);
        }
    }

    fn fill(&mut self, value: Scalar) {
        self.fill_zeros();
        let storage_count = dims_to_values::<Scalar>(self.dims);
        if storage_count == 0 {
            return;
        }
        // Skip overlay when `value` matches the freshly-zeroed bit pattern.
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
        if value_bytes == default_bytes {
            return;
        }
        if core::mem::size_of::<Scalar>() == 1 {
            unsafe {
                core::ptr::write_bytes(self.data.as_ptr() as *mut u8, value_bytes[0], storage_count);
            }
        } else {
            for storage_index in 0..storage_count {
                unsafe { core::ptr::write(self.data.as_ptr().add(storage_index), value) };
            }
        }
    }
}

impl<Scalar: StorageElement, Alloc: Allocator> CopyFrom<&[Scalar]> for Vector<Scalar, Alloc> {
    fn copy_from(&mut self, source: &[Scalar]) -> Result<(), TensorError> {
        let storage_count = dims_to_values::<Scalar>(self.dims);
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

impl<'a, Scalar: StorageElement> Fill<Scalar> for VectorSpan<'a, Scalar> {
    fn fill_zeros(&mut self) {
        let storage_count = self.dims.div_ceil(Scalar::dimensions_per_value());
        if storage_count == 0 {
            return;
        }
        let element_size = core::mem::size_of::<Scalar>() as isize;
        if self.stride_bytes == element_size {
            // Contiguous fast path.
            unsafe {
                core::ptr::write_bytes(self.data, 0, storage_count);
            }
            return;
        }
        // Strided: zero one element at a time.
        for storage_index in 0..storage_count {
            unsafe {
                let ptr = (self.data as *mut u8)
                    .offset(self.stride_bytes * storage_index as isize)
                    as *mut Scalar;
                core::ptr::write_bytes(ptr, 0, 1);
            }
        }
    }

    fn fill(&mut self, value: Scalar) {
        self.fill_zeros();
        let storage_count = self.dims.div_ceil(Scalar::dimensions_per_value());
        if storage_count == 0 {
            return;
        }
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
        if value_bytes == default_bytes {
            return;
        }
        let element_size = core::mem::size_of::<Scalar>() as isize;
        if self.stride_bytes == element_size {
            // Contiguous fast path.
            if core::mem::size_of::<Scalar>() == 1 {
                unsafe {
                    core::ptr::write_bytes(self.data as *mut u8, value_bytes[0], storage_count);
                }
            } else {
                for storage_index in 0..storage_count {
                    unsafe { core::ptr::write(self.data.add(storage_index), value) };
                }
            }
            return;
        }
        // Strided typed broadcast.
        for storage_index in 0..storage_count {
            unsafe {
                let ptr = (self.data as *mut u8)
                    .offset(self.stride_bytes * storage_index as isize)
                    as *mut Scalar;
                core::ptr::write(ptr, value);
            }
        }
    }
}

impl<'a, 'b, Scalar: StorageElement> CopyFrom<&'b VectorView<'_, Scalar>>
    for VectorSpan<'a, Scalar>
{
    fn copy_from(&mut self, source: &'b VectorView<'_, Scalar>) -> Result<(), TensorError> {
        if source.size() != self.dims {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: self.dims,
                got: source.size(),
            });
        }
        if self.dims == 0 {
            return Ok(());
        }
        let storage_count = self.dims.div_ceil(Scalar::dimensions_per_value());
        let element_size = core::mem::size_of::<Scalar>() as isize;
        if self.stride_bytes == element_size && source.stride_bytes() == element_size {
            unsafe {
                core::ptr::copy_nonoverlapping(source.as_ptr(), self.data, storage_count);
            }
            return Ok(());
        }
        for storage_index in 0..storage_count {
            unsafe {
                let destination_ptr = (self.data as *mut u8)
                    .offset(self.stride_bytes * storage_index as isize)
                    as *mut Scalar;
                let source_ptr = (source.as_ptr() as *const u8)
                    .offset(source.stride_bytes() * storage_index as isize)
                    as *const Scalar;
                core::ptr::write(destination_ptr, *source_ptr);
            }
        }
        Ok(())
    }
}

// endregion: Fill / CopyFrom Implementations

// region: Iterators

/// Stride-aware immutable iterator over logical dimensions, yielding [`DimRef`] proxies.
///
/// For normal types (`dimensions_per_value=1`), yields one proxy per storage value.
/// For sub-byte types, unpacks each storage value and yields sub-dimensions individually.
/// Implements `ExactSizeIterator`, `FusedIterator`, and `DoubleEndedIterator`.
pub struct VectorViewIterator<'a, Scalar: FloatConvertible> {
    data: *const Scalar,
    stride_bytes: isize,
    front: usize,
    back: usize,
    _marker: PhantomData<&'a Scalar>,
}

/// Backward-compatible alias for [`VectorViewIterator`].
pub type VectorIterator<'a, Scalar> = VectorViewIterator<'a, Scalar>;

impl<'a, Scalar: FloatConvertible> Iterator for VectorViewIterator<'a, Scalar> {
    type Item = DimRef<'a, Scalar>;

    #[inline]
    fn next(&mut self) -> Option<DimRef<'a, Scalar>> {
        if self.front >= self.back {
            return None;
        }
        let dims_per_value = Scalar::dimensions_per_value();
        let value_index = self.front / dims_per_value;
        let sub_index = self.front % dims_per_value;
        // SAFETY: value_index < values, stride * value_index within allocation
        let ptr = unsafe {
            (self.data as *const u8).offset(self.stride_bytes * value_index as isize)
                as *const Scalar
        };
        let scalar = unsafe { *ptr }.unpack().as_ref()[sub_index];
        self.front += 1;
        Some(DimRef::new(scalar))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let element_count = self.back - self.front;
        (element_count, Some(element_count))
    }
}

impl<'a, Scalar: FloatConvertible> ExactSizeIterator for VectorViewIterator<'a, Scalar> {}
impl<'a, Scalar: FloatConvertible> core::iter::FusedIterator for VectorViewIterator<'a, Scalar> {}

impl<'a, Scalar: FloatConvertible> DoubleEndedIterator for VectorViewIterator<'a, Scalar> {
    #[inline]
    fn next_back(&mut self) -> Option<DimRef<'a, Scalar>> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let dims_per_value = Scalar::dimensions_per_value();
        let value_index = self.back / dims_per_value;
        let sub_index = self.back % dims_per_value;
        let ptr = unsafe {
            (self.data as *const u8).offset(self.stride_bytes * value_index as isize)
                as *const Scalar
        };
        Some(DimRef::new(unsafe { *ptr }.unpack().as_ref()[sub_index]))
    }
}

/// Stride-aware mutable iterator over logical dimensions, yielding [`DimMut`] proxies.
///
/// For normal types (`dimensions_per_value=1`), yields one proxy per storage value.
/// For sub-byte types, each proxy performs a read-modify-write on drop.
/// Implements `ExactSizeIterator`, `FusedIterator`, and `DoubleEndedIterator`.
pub struct VectorSpanIterator<'a, Scalar: FloatConvertible> {
    data: *mut Scalar,
    stride_bytes: isize,
    front: usize,
    back: usize,
    _marker: PhantomData<&'a mut Scalar>,
}

impl<'a, Scalar: FloatConvertible> Iterator for VectorSpanIterator<'a, Scalar> {
    type Item = DimMut<'a, Scalar>;

    #[inline]
    fn next(&mut self) -> Option<DimMut<'a, Scalar>> {
        if self.front >= self.back {
            return None;
        }
        let dims_per_value = Scalar::dimensions_per_value();
        let value_index = self.front / dims_per_value;
        let sub_index = self.front % dims_per_value;
        let ptr = unsafe {
            (self.data as *mut u8).offset(self.stride_bytes * value_index as isize) as *mut Scalar
        };
        let scalar = unsafe { *ptr }.unpack().as_ref()[sub_index];
        self.front += 1;
        Some(unsafe { DimMut::new(ptr, sub_index, scalar) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let element_count = self.back - self.front;
        (element_count, Some(element_count))
    }
}

impl<'a, Scalar: FloatConvertible> ExactSizeIterator for VectorSpanIterator<'a, Scalar> {}
impl<'a, Scalar: FloatConvertible> core::iter::FusedIterator for VectorSpanIterator<'a, Scalar> {}

impl<'a, Scalar: FloatConvertible> DoubleEndedIterator for VectorSpanIterator<'a, Scalar> {
    #[inline]
    fn next_back(&mut self) -> Option<DimMut<'a, Scalar>> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let dims_per_value = Scalar::dimensions_per_value();
        let value_index = self.back / dims_per_value;
        let sub_index = self.back % dims_per_value;
        let ptr = unsafe {
            (self.data as *mut u8).offset(self.stride_bytes * value_index as isize) as *mut Scalar
        };
        let scalar = unsafe { *ptr }.unpack().as_ref()[sub_index];
        Some(unsafe { DimMut::new(ptr, sub_index, scalar) })
    }
}

// endregion: Iterators

// region: IntoIterator (immutable)

impl<'a, Scalar: FloatConvertible, Alloc: Allocator> IntoIterator for &'a Vector<Scalar, Alloc> {
    type Item = DimRef<'a, Scalar>;
    type IntoIter = VectorViewIterator<'a, Scalar>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, Scalar: FloatConvertible> IntoIterator for &'a VectorView<'a, Scalar> {
    type Item = DimRef<'a, Scalar>;
    type IntoIter = VectorViewIterator<'a, Scalar>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, Scalar: FloatConvertible> IntoIterator for &'a VectorSpan<'a, Scalar> {
    type Item = DimRef<'a, Scalar>;
    type IntoIter = VectorViewIterator<'a, Scalar>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// endregion: IntoIterator (immutable)

// region: IntoIterator (mutable)

impl<'a, Scalar: FloatConvertible, Alloc: Allocator> IntoIterator
    for &'a mut Vector<Scalar, Alloc>
{
    type Item = DimMut<'a, Scalar>;
    type IntoIter = VectorSpanIterator<'a, Scalar>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, Scalar: FloatConvertible> IntoIterator for &'a mut VectorSpan<'a, Scalar> {
    type Item = DimMut<'a, Scalar>;
    type IntoIter = VectorSpanIterator<'a, Scalar>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// endregion: IntoIterator (mutable)

// region: AsRef

impl<Scalar: StorageElement, Alloc: Allocator> AsRef<[Scalar]> for Vector<Scalar, Alloc> {
    fn as_ref(&self) -> &[Scalar] {
        self.as_slice()
    }
}

// endregion: AsRef

// region: PartialEq

impl<Scalar: FloatConvertible, Alloc: Allocator> PartialEq for Vector<Scalar, Alloc>
where
    Scalar::DimScalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<Scalar: FloatConvertible, Alloc: Allocator> PartialEq<[Scalar::DimScalar]>
    for Vector<Scalar, Alloc>
where
    Scalar::DimScalar: PartialEq,
{
    fn eq(&self, other: &[Scalar::DimScalar]) -> bool {
        self.dims == other.len() && self.iter().zip(other.iter()).all(|(a, b)| *a == *b)
    }
}

impl<'a, Scalar: FloatConvertible> PartialEq for VectorView<'a, Scalar>
where
    Scalar::DimScalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<'a, Scalar: FloatConvertible> PartialEq for VectorSpan<'a, Scalar>
where
    Scalar::DimScalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

// endregion: PartialEq

// region: Tolerance Equality

impl<Scalar: FloatConvertible, Alloc: Allocator> Vector<Scalar, Alloc>
where
    Scalar::DimScalar: NumberLike,
{
    /// Check if all elements are within tolerance of `other`.
    ///
    /// Uses the formula `|a - b| <= atol + rtol * |b|` per element.
    /// Returns `false` if dimensions differ.
    pub fn allclose<OtherAlloc: Allocator>(
        &self,
        other: &Vector<Scalar, OtherAlloc>,
        atol: f64,
        rtol: f64,
    ) -> bool {
        self.dims == other.dims
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| crate::types::is_close(a.to_f64(), b.to_f64(), atol, rtol))
    }
}

impl<'a, Scalar: FloatConvertible> VectorView<'a, Scalar>
where
    Scalar::DimScalar: NumberLike,
{
    /// Check if all elements are within tolerance of `other`.
    ///
    /// Uses the formula `|a - b| <= atol + rtol * |b|` per element.
    /// Returns `false` if dimensions differ.
    pub fn allclose(&self, other: &Self, atol: f64, rtol: f64) -> bool {
        self.dims == other.dims
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| crate::types::is_close(a.to_f64(), b.to_f64(), atol, rtol))
    }
}

impl<'a, Scalar: FloatConvertible> VectorSpan<'a, Scalar>
where
    Scalar::DimScalar: NumberLike,
{
    /// Check if all elements are within tolerance of `other`.
    ///
    /// Uses the formula `|a - b| <= atol + rtol * |b|` per element.
    /// Returns `false` if dimensions differ.
    pub fn allclose(&self, other: &Self, atol: f64, rtol: f64) -> bool {
        self.dims == other.dims
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| crate::types::is_close(a.to_f64(), b.to_f64(), atol, rtol))
    }
}

// endregion: Tolerance Equality

// region: Debug and Display

/// Write a truncated, debug-formatted list from an iterator.
fn fmt_debug_list<I: Iterator>(
    f: &mut core::fmt::Formatter<'_>,
    name: &str,
    dims: usize,
    iter: I,
    limit: usize,
) -> core::fmt::Result
where
    I::Item: core::fmt::Debug,
{
    write!(f, "{}(dims={}, [", name, dims)?;
    for (i, value) in iter.enumerate() {
        if i >= limit {
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

/// Write a truncated, display-formatted list from an iterator.
fn fmt_display_list<I: Iterator>(
    f: &mut core::fmt::Formatter<'_>,
    iter: I,
    limit: usize,
) -> core::fmt::Result
where
    I::Item: core::fmt::Display,
{
    let prec = f.precision();
    write!(f, "[")?;
    for (i, value) in iter.enumerate() {
        if i >= limit {
            write!(f, ", ...")?;
            break;
        }
        if i > 0 {
            write!(f, ", ")?;
        }
        if let Some(p) = prec {
            write!(f, "{:.p$}", value)?;
        } else {
            write!(f, "{}", value)?;
        }
    }
    write!(f, "]")
}

impl<Scalar: FloatConvertible, Alloc: Allocator> core::fmt::Debug for Vector<Scalar, Alloc>
where
    Scalar::DimScalar: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt_debug_list(f, "Vector", self.dims, self.iter(), 8)
    }
}

impl<'a, Scalar: FloatConvertible> core::fmt::Debug for VectorView<'a, Scalar>
where
    Scalar::DimScalar: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt_debug_list(f, "VectorView", self.dims, self.iter(), 8)
    }
}

impl<'a, Scalar: FloatConvertible> core::fmt::Debug for VectorSpan<'a, Scalar>
where
    Scalar::DimScalar: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt_debug_list(f, "VectorSpan", self.dims, self.iter(), 8)
    }
}

impl<Scalar: FloatConvertible, Alloc: Allocator> core::fmt::Display for Vector<Scalar, Alloc>
where
    Scalar::DimScalar: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt_display_list(f, self.iter(), 20)
    }
}

// endregion: Debug and Display

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{bf16, f16, i4x2, u1x8, u4x2};

    fn check_vector_roundtrip<Scalar: FloatConvertible>() {
        let dims_per_value = Scalar::dimensions_per_value();
        let test_dims = 16 * dims_per_value;
        let v = Vector::<Scalar>::try_zeros(test_dims).unwrap();
        assert_eq!(v.dims(), test_dims);
        assert_eq!(v.size_values(), test_dims / dims_per_value);
        let count = v.iter().count();
        assert_eq!(count, test_dims);
    }

    fn check_vector_try_get_set<Scalar: FloatConvertible>()
    where
        Scalar::DimScalar: core::fmt::Debug,
    {
        let dims_per_value = Scalar::dimensions_per_value();
        let test_dims = 4 * dims_per_value;
        let mut v = Vector::<Scalar>::try_zeros(test_dims).unwrap();
        let one = Scalar::DimScalar::from_f32(1.0);
        v.try_set(0_usize, one).unwrap();
        v.try_set((test_dims - 1) as i32, one).unwrap();
        let first = v.try_get(0_usize).unwrap();
        let last = v.try_get(-1_i32).unwrap();
        assert!(
            first.to_f32() >= 0.5,
            "first dim should be ~1.0, got {:?}",
            first
        );
        assert!(
            last.to_f32() >= 0.5,
            "last dim should be ~1.0, got {:?}",
            last
        );
    }

    #[test]
    fn vector_roundtrip_all_types() {
        check_vector_roundtrip::<f32>();
        check_vector_roundtrip::<f64>();
        check_vector_roundtrip::<f16>();
        check_vector_roundtrip::<bf16>();
        check_vector_roundtrip::<i4x2>();
        check_vector_roundtrip::<u4x2>();
        check_vector_roundtrip::<u1x8>();
    }

    #[test]
    fn vector_try_get_set_all_types() {
        check_vector_try_get_set::<f32>();
        check_vector_try_get_set::<f64>();
        check_vector_try_get_set::<i4x2>();
        check_vector_try_get_set::<u4x2>();
        check_vector_try_get_set::<u1x8>();
    }

    #[test]
    fn vec_index_signed() {
        let v = Vector::<f32>::try_from_dims(&[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        // Positive indexing
        assert_eq!(v[0], 10.0);
        assert_eq!(v[4], 50.0);
        // Negative indexing (i32 default for integer literals)
        assert_eq!(v[-1_i32], 50.0);
        assert_eq!(v[-2_i32], 40.0);
        assert_eq!(v[-5_i32], 10.0);
    }

    #[test]
    fn vector_view_stride() {
        let v = Vector::<f32>::try_from_scalars(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let view = v.view();
        assert!(view.is_contiguous());
        assert_eq!(view.size(), 5);

        // Reversed view
        let rev = view.rev();
        assert_eq!(rev.try_get(0_usize).unwrap(), 5.0);
        assert_eq!(rev.try_get(4_usize).unwrap(), 1.0);
    }

    #[test]
    fn vector_span_fill() {
        let mut v = Vector::<f32>::try_zeros(4).unwrap();
        {
            let mut span = v.span();
            span.fill(42.0);
        }
        assert_eq!(v[0], 42.0);
        assert_eq!(v[3], 42.0);
    }

    #[test]
    fn vector_iter() {
        let v = Vector::<f32>::try_from_scalars(&[1.0, 2.0, 3.0]).unwrap();
        let values: Vec<f32> = v.iter().map(|x| *x).collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);

        // Double-ended
        let reversed_values: Vec<f32> = v.iter().rev().map(|x| *x).collect();
        assert_eq!(reversed_values, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn view_strided_iteration() {
        let v = Vector::<f32>::try_from_scalars(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let view = v.view();

        // Every other element
        let strided = view.try_strided(0, 5, 2).unwrap();
        assert_eq!(strided.size(), 3);
        let values: Vec<f32> = strided.iter().map(|x| *x).collect();
        assert_eq!(values, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn vector_filled() {
        let v = Vector::<f32>::try_full(3, 7.5).unwrap();
        assert_eq!(v[0], 7.5);
        assert_eq!(v[1], 7.5);
        assert_eq!(v[2], 7.5);
    }

    #[test]
    fn empty_vector() {
        let v = Vector::<f32>::try_zeros(0).unwrap();
        assert!(v.is_empty());
        assert_eq!(v.size(), 0);
    }

    #[test]
    #[should_panic]
    fn index_out_of_bounds() {
        let v = Vector::<f32>::try_zeros(3).unwrap();
        let _ = v[3_usize];
    }

    #[test]
    fn vector_allclose_matching() {
        let a = Vector::<f32>::try_full(4, 1.0).unwrap();
        let b = Vector::<f32>::try_full(4, 1.0 + 1e-7).unwrap();
        assert!(a.allclose(&b, 1e-6, 0.0));
    }

    #[test]
    fn vector_allclose_mismatching() {
        let a = Vector::<f32>::try_full(4, 1.0).unwrap();
        let b = Vector::<f32>::try_full(4, 2.0).unwrap();
        assert!(!a.allclose(&b, 1e-6, 0.0));
    }

    #[test]
    fn vector_allclose_different_dims() {
        let a = Vector::<f32>::try_full(3, 1.0).unwrap();
        let b = Vector::<f32>::try_full(4, 1.0).unwrap();
        assert!(!a.allclose(&b, 1e-6, 1e-6));
    }

    #[test]
    fn display_precision_forwarding() {
        let v = Vector::<f32>::try_full(3, 1.0).unwrap();
        let s = format!("{:.2}", v);
        assert_eq!(s, "[1.00, 1.00, 1.00]");
    }

    #[test]
    fn vector_span_iter_mut_f32() {
        let mut v = Vector::<f32>::try_from_scalars(&[1.0, 2.0, 3.0]).unwrap();
        for mut value in &mut v {
            *value += 10.0;
        }
        let values: Vec<f32> = v.iter().map(|x| *x).collect();
        assert_eq!(values, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn vector_span_iter_mut_i4x2() {
        let mut v = Vector::<i4x2>::try_zeros(4).unwrap();
        {
            let mut span = v.span();
            for (i, mut value) in span.iter_mut().enumerate() {
                *value = (i + 1) as i8;
            }
        }
        assert_eq!(v.try_get(0_usize).unwrap(), 1);
        assert_eq!(v.try_get(1_usize).unwrap(), 2);
        assert_eq!(v.try_get(2_usize).unwrap(), 3);
        assert_eq!(v.try_get(3_usize).unwrap(), 4);
    }

    #[test]
    fn vector_span_iter_mut_u1x8() {
        let mut v = Vector::<u1x8>::try_zeros(8).unwrap();
        for (i, mut value) in v.iter_mut().enumerate() {
            if i % 2 == 0 {
                *value = 1;
            }
        }
        // Even indices should be 1, odd should be 0
        assert_eq!(v.try_get(0_usize).unwrap(), 1);
        assert_eq!(v.try_get(1_usize).unwrap(), 0);
        assert_eq!(v.try_get(2_usize).unwrap(), 1);
        assert_eq!(v.try_get(3_usize).unwrap(), 0);
    }

    #[test]
    fn vector_span_iter_double_ended() {
        let mut v = Vector::<f32>::try_from_scalars(&[1.0, 2.0, 3.0]).unwrap();
        let mut span = v.span();
        let mut it = span.iter_mut();
        // Take from front
        let mut first = it.next().unwrap();
        *first = 10.0;
        drop(first);
        // Take from back
        let mut last = it.next_back().unwrap();
        *last = 30.0;
        drop(last);
        drop(it);
        drop(span);
        assert_eq!(v.try_get(0_usize).unwrap(), 10.0);
        assert_eq!(v.try_get(1_usize).unwrap(), 2.0);
        assert_eq!(v.try_get(2_usize).unwrap(), 30.0);
    }

    #[test]
    fn vector_iterator_alias_compat() {
        // VectorIterator type alias should still work
        let v = Vector::<f32>::try_from_scalars(&[1.0]).unwrap();
        let _it: VectorIterator<'_, f32> = v.iter();
    }

    #[test]
    fn vector_index_signed_negative_resolves_from_end() {
        // After macro removal: i32 negative indices wrap from end.
        assert_eq!(VectorIndex::resolve(-1i32, 5), Some(4));
        assert_eq!(VectorIndex::resolve(-5i32, 5), Some(0));
        assert_eq!(VectorIndex::resolve(-6i32, 5), None); // out of bounds
        // Same logic across all signed widths.
        assert_eq!(VectorIndex::resolve(-1i8, 5), Some(4));
        assert_eq!(VectorIndex::resolve(-1i16, 5), Some(4));
        assert_eq!(VectorIndex::resolve(-1i64, 5), Some(4));
        assert_eq!(VectorIndex::resolve(-1isize, 5), Some(4));
        // Unsigned: bounds check only.
        assert_eq!(VectorIndex::resolve(3u32, 5), Some(3));
        assert_eq!(VectorIndex::resolve(5u32, 5), None);
        assert_eq!(VectorIndex::resolve(0u8, 5), Some(0));
        assert_eq!(VectorIndex::resolve(0usize, 5), Some(0));
    }

    #[test]
    fn fill_zeros_and_fill_on_vector_and_span() {
        // Vector: Fill trait gives fill_zeros + fill methods.
        use crate::tensor::Fill;
        let mut v = Vector::<f32>::try_full(8, 3.5).unwrap();
        v.fill_zeros();
        assert!(v.as_slice().iter().all(|&value| value == 0.0));
        v.fill(2.5);
        assert!(v.as_slice().iter().all(|&value| value == 2.5));

        // VectorSpan: same trait methods.
        let mut data = [0.0f32; 8];
        let mut span = unsafe {
            VectorSpan::<f32>::from_raw_parts(
                data.as_mut_ptr(),
                8,
                core::mem::size_of::<f32>() as isize,
            )
        };
        span.fill(1.25);
        assert!(data.iter().all(|&value| value == 1.25));
    }

    #[test]
    fn copy_from_round_trips_vector_and_span() {
        use crate::tensor::CopyFrom;
        // Vector: CopyFrom<&[Scalar]>.
        let mut destination = Vector::<f32>::try_zeros(4).unwrap();
        let source = [1.0f32, 2.0, 3.0, 4.0];
        destination.copy_from(source.as_slice()).unwrap();
        assert_eq!(destination.as_slice(), &source[..]);

        // VectorSpan: CopyFrom<&VectorView>.
        let source_vec = Vector::<f32>::try_from_scalars(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        let mut destination_buffer = [0.0f32; 4];
        {
            let mut destination_span = unsafe {
                VectorSpan::<f32>::from_raw_parts(
                    destination_buffer.as_mut_ptr(),
                    4,
                    core::mem::size_of::<f32>() as isize,
                )
            };
            let source_view = source_vec.view();
            destination_span.copy_from(&source_view).unwrap();
        }
        assert_eq!(destination_buffer, [5.0f32, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn bit_reductions_on_u1x8_vector_match_method_names() {
        use crate::types::u1x8;
        let mut bits_vector = Vector::<u1x8>::try_zeros(32).unwrap();
        // Set all 32 bits.
        for slot in bits_vector.as_mut_slice().iter_mut() {
            *slot = u1x8(0xFFu8);
        }
        assert_eq!(bits_vector.popcount(), 32);
        assert!(bits_vector.any_set());
        assert!(!bits_vector.none_set());
        assert!(bits_vector.all_set());

        bits_vector.fill_zeros();
        assert_eq!(bits_vector.popcount(), 0);
        assert!(!bits_vector.any_set());
        assert!(bits_vector.none_set());
    }
}

// endregion: Tests
