//! Core N-dimensional tensor types and the `TensorRef` / `TensorMut` structural traits.
//!
//! This module provides:
//!
//! - [`Tensor`]: N-dimensional array with customizable rank and allocator
//! - [`TensorView`]: Immutable view into a tensor
//! - [`TensorSpan`]: Mutable view into a tensor
//! - [`Matrix`]: Type alias for 2D tensors
//! - [`TensorRef`] / [`TensorMut`]: Structural traits implemented by all three above
//! - [`Fill`] / [`CopyFrom`]: Container-level traits for in-place fill and copying a slice or
//!   view into a span
//!
//! Operation extension traits live in their respective domain modules so each `<op>.rs` mirrors the
//! C++ `<op>.hpp` layout:
//! - [`crate::each`]: `ScaleOps` / `SumOps` / `BlendOps` / `FmaOps` / `TrigSinOps` /
//!   `TrigCosOps` / `TrigAtanOps` / `AllCloseOps`
//! - [`crate::reduce`]: `MomentsOps` / `MinMaxOps` / `BitwiseReductionsOps`
//! - [`mod@crate::cast`]: `CastOps` dtype conversions
//!
//! Batch matrix operations live in [`crate::dots`] for GEMM, [`crate::spatials`] for spatial
//! distances, and [`crate::sets`] for binary/set metrics.
//!
//! # Custom allocators
//!
//! [`Tensor`] is generic over [`core::alloc::Allocator`]. Any allocator written against that trait
//! — a bump arena, a pool, a pinned-memory allocator — plugs into the `*_in` constructors with no
//! adapter, and because `core` also implements the trait for `&A`, an arena that is not `Clone`
//! goes in by reference. [`Global`], the default, forwards to the system heap. Views and spans
//! borrow their owner's allocator, and an operation that allocates, like `view.sin()`, allocates
//! its result through a clone of its first operand's allocator; views over foreign memory report
//! [`Global`].
//!
//! ```rust
//! use core::alloc::{AllocError, Allocator, Layout};
//! use core::ptr::NonNull;
//! use numkong::{Global, Tensor};
//!
//! struct MyArena;
//! unsafe impl Allocator for MyArena {
//!     fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
//!         Global.allocate(layout)
//!     }
//!     unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
//!         unsafe { Global.deallocate(ptr, layout) }
//!     }
//! }
//!
//! let arena = MyArena;
//! let owned = Tensor::<f32, _>::full_in(&[1024, 1024], 0.0, &arena).unwrap();
//! ```
//!
//! # Slicing and views
//!
//! Tensors support zero-copy slicing with NumPy-style tuple syntax or the [`SliceRange`] enum.
//! Views and spans share memory with the parent tensor and may have non-contiguous strides:
//!
//! ```rust,ignore
//! let matrix = Tensor::<f32>::full(&[4, 5], 1.0).unwrap();
//! let row = matrix.slice((1_usize, ..)).unwrap();       // t[1, :]
//! let block = matrix.slice((0..2_usize, 1..4_usize)).unwrap();
//! ```
//!
//! # Sub-byte types
//!
//! Sub-byte element types (`i4x2`, `u4x2`, `u1x8`) pack multiple logical elements per storage byte,
//! so sub-byte tensors must be accessed via [`Tensor::flat`], [`Tensor::coords`], and the iterator
//! APIs that yield [`crate::types::DimRef`] / [`crate::types::DimMut`] proxies.
//!
//! File: rust/tensor.rs
//! Author: Ash Vardanian

#[cfg(feature = "alloc")]
extern crate alloc;

pub use core::alloc::{AllocError, Allocator};

use core::{marker::PhantomData, ptr::NonNull};

use crate::{
    capabilities::Status,
    cast::{cast, CastDType},
    dot::Dot,
    each::{EachBlend, EachFma, EachScale, EachSum},
    reduce::{ReduceMinMax, ReduceMoments, SumSqToF64},
    scalar::Roots,
    trigonometry::{TrigAtan, TrigCos, TrigSin},
    types::{DimMut, DimRef, FloatConvertible, StorageElement},
    vector::{Vector, VectorIndex},
};

// region: Constants and Allocator

/// Default maximum rank for tensors.
pub const DEFAULT_MAX_RANK: usize = 8;

/// Alignment for SIMD-friendly allocations — 64 bytes for AVX-512.
pub const SIMD_ALIGNMENT: usize = 64;

/// The [`SIMD_ALIGNMENT`]-aligned layout holding `count` storage values of `Scalar`.
///
/// Every owned allocation in the crate is sized here, so the value-count-to-byte-count multiply is
/// checked in one place. A `count` that overflows describes an allocation no allocator could ever
/// satisfy, so it reports [`Error::AllocationFailed`] rather than wrapping into a small request
/// that the caller would then write past the end of.
pub(crate) fn layout_for<Scalar>(count: usize) -> Result<core::alloc::Layout, Error> {
    let bytes = count
        .checked_mul(core::mem::size_of::<Scalar>())
        .ok_or(Error::AllocationFailed)?;
    layout_for_bytes(bytes)
}

/// The byte-granular twin of [`layout_for`], for the packed buffers whose size the C ABI reports in
/// bytes rather than in typed slots.
pub(crate) fn layout_for_bytes(bytes: usize) -> Result<core::alloc::Layout, Error> {
    core::alloc::Layout::from_size_align(bytes, SIMD_ALIGNMENT).map_err(|_| Error::AllocationFailed)
}

/// The logical element count of `shape`, reporting [`Error::AllocationFailed`] on overflow.
///
/// A wrapped product would leave the stored shape and the derived element count disagreeing, so the
/// per-axis stride arithmetic would index outside the allocation. The empty product of a rank-0
/// shape is 1 — a rank-0 tensor holds one element.
pub(crate) fn shape_product(shape: &[usize]) -> Result<usize, Error> {
    let mut total: usize = 1;
    for &extent in shape {
        total = total.checked_mul(extent).ok_or(Error::AllocationFailed)?;
    }
    Ok(total)
}

/// The default allocator, forwarding to the heap registered with `#[global_allocator]`.
///
/// The [`Allocator`] trait itself is `core`'s, so an arena, a pool, or a pinned-memory allocator
/// written against it — including by reference, through `impl Allocator for &A` — drops into
/// [`Tensor`] and the packed containers with no adapter. This type only names the default, and
/// stays separate from `alloc::alloc::Global` so the heapless build still has one.
///
/// Without the `alloc` feature there is no heap to forward to and every request fails. That is a
/// supported configuration rather than a broken one: the borrowed API — the scalar traits over
/// `&[T]`, the `*_into` verbs writing into caller-provided spans, and the `from_raw_parts` views —
/// never allocates, so only the owning containers become unusable.
#[derive(Debug, Clone, Copy, Default)]
pub struct Global;

/// The allocator every view over foreign memory reports: raw parts, slices and static data.
pub(crate) static GLOBAL: Global = Global;

unsafe impl Allocator for Global {
    #[inline]
    fn allocate(&self, layout: core::alloc::Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0));
        }
        #[cfg(feature = "alloc")]
        {
            // SAFETY: `layout` is non-zero in size, checked directly above.
            let ptr = NonNull::new(unsafe { alloc::alloc::alloc(layout) }).ok_or(AllocError)?;
            Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
        }
        #[cfg(not(feature = "alloc"))]
        {
            Err(AllocError)
        }
    }

    /// Overridden so a zero fill reaches `alloc_zeroed` rather than a `memset` over the block.
    /// The system allocator serves a large request with fresh pages the kernel already zeroed, so
    /// this returns without faulting the whole buffer in.
    #[inline]
    fn allocate_zeroed(&self, layout: core::alloc::Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0));
        }
        #[cfg(feature = "alloc")]
        {
            // SAFETY: `layout` is non-zero in size, checked directly above.
            let ptr = NonNull::new(unsafe { alloc::alloc::alloc_zeroed(layout) }).ok_or(AllocError)?;
            Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
        }
        #[cfg(not(feature = "alloc"))]
        {
            Err(AllocError)
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: core::alloc::Layout) {
        #[cfg(feature = "alloc")]
        if layout.size() > 0 {
            // SAFETY: the caller guarantees `ptr` came from this allocator with a fitting `layout`.
            unsafe { alloc::alloc::dealloc(ptr.as_ptr(), layout) };
        }
        #[cfg(not(feature = "alloc"))]
        let _ = (ptr, layout);
    }
}

/// The repeating byte of `value`, when every one of its bytes is the same.
///
/// A fill whose bytes all agree is a `memset` rather than a loop over typed slots, and one whose
/// repeating byte is zero can skip the write entirely by asking the allocator for zeroed memory.
/// Reading the bytes rather than asking the type keeps the sub-byte and mini-float formats honest:
/// `Ue8m0`'s all-zero encoding is 2^-127, not zero, so a "zero is all-zero-bits" marker on
/// [`StorageElement`] would be wrong for it. `StorageElement: Copy` and every implementor wraps a
/// single primitive, so there is no padding to observe.
fn repeating_byte<Scalar: StorageElement>(value: &Scalar) -> Option<u8> {
    // SAFETY: `Scalar` is `Copy` over a single primitive — no padding, no interior references.
    let bytes =
        unsafe { core::slice::from_raw_parts((value as *const Scalar).cast::<u8>(), core::mem::size_of::<Scalar>()) };
    match bytes.first() {
        Some(&first) if bytes.iter().all(|&byte| byte == first) => Some(first),
        _ => None,
    }
}

/// Allocate room for `count` `Scalar` slots and initialize every one of them to `value`.
///
/// Returns the block and its actual byte size. Picks the cheapest initialization the value allows:
/// zeroed pages from the allocator, a `memset`, or a typed loop.
pub(crate) fn alloc_filled<Scalar: StorageElement, A: Allocator>(
    alloc: &A,
    count: usize,
    value: Scalar,
) -> Result<(NonNull<Scalar>, usize), Error> {
    let layout = layout_for::<Scalar>(count)?;
    match repeating_byte(&value) {
        Some(0) => {
            let block = alloc.allocate_zeroed(layout).map_err(|_| Error::AllocationFailed)?;
            Ok((block.cast(), block.len()))
        }
        Some(byte) => {
            let (block, actual) = alloc_block(alloc, layout)?;
            // SAFETY: the block holds at least `layout.size()` bytes.
            unsafe { block.as_ptr().write_bytes(byte, layout.size()) };
            Ok((block.cast(), actual))
        }
        None => {
            let (block, actual) = alloc_block(alloc, layout)?;
            let start = block.as_ptr().cast::<Scalar>();
            for slot in 0..count {
                // SAFETY: the block holds `count` `Scalar` slots, uninitialized until written here.
                unsafe { core::ptr::write(start.add(slot), value) };
            }
            Ok((block.cast(), actual))
        }
    }
}

/// Allocate `layout` and report the block's _actual_ size, which an arena or a pool may round up
/// past the request.
///
/// Callers record that size as their capacity, so the slack an allocator already handed over is
/// usable rather than discarded. Deallocating with a layout sized anywhere between the request and
/// the reported size is permitted — see the "Memory fitting" contract on [`Allocator`].
pub(crate) fn alloc_block<A: Allocator>(alloc: &A, layout: core::alloc::Layout) -> Result<(NonNull<u8>, usize), Error> {
    let block = alloc.allocate(layout).map_err(|_| Error::AllocationFailed)?;
    Ok((block.cast(), block.len()))
}

// endregion: Constants and Allocator

// region: PackedBuffer

/// A `SIMD_ALIGNMENT`-aligned owning byte buffer shared by the packed-matrix containers.
///
/// The dots, maxsim, and attention packed containers each wrap a single byte blob produced by the C
/// `*_pack` FFI. `PackedBuffer` factors the allocation, reuse, and teardown of that blob into one
/// place — it owns `(data, size, capacity, alloc)` and every container holds one as a private
/// field, exactly as [`alloc::vec::Vec`] holds a `RawVec`. `size` is the live packed length and
/// `capacity` the allocated length (`capacity >= size`); a `capacity == 0` buffer owns no
/// allocation and holds a dangling pointer.
///
/// Two growth policies live here:
/// - [`reset_for_pack`](Self::reset_for_pack) makes room for a fresh pack and discards the old
///   contents — packing overwrites every byte, so nothing is preserved.
/// - [`reserve`](Self::reserve) pre-grows the allocation while preserving the live bytes,
///   so a caller can hoist the allocation out of a decode loop and every later pack reuses it with
///   a stable pointer.
#[derive(Debug)]
pub(crate) struct PackedBuffer<Alloc: Allocator = Global> {
    data: NonNull<u8>,
    size: usize,
    capacity: usize,
    alloc: Alloc,
}

impl<Alloc: Allocator> PackedBuffer<Alloc> {
    /// An empty buffer owning no allocation; fill it with [`reset_for_pack`](Self::reset_for_pack).
    pub(crate) fn empty_in(alloc: Alloc) -> Self {
        Self {
            data: NonNull::dangling(),
            size: 0,
            capacity: 0,
            alloc,
        }
    }

    /// Bytes currently allocated (`>= size`).
    pub(crate) fn capacity(&self) -> usize { self.capacity }

    /// Reference to the stored allocator.
    pub(crate) fn allocator(&self) -> &Alloc { &self.alloc }

    /// Pointer to the packed data.
    pub(crate) fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }

    /// The live packed bytes.
    pub(crate) fn as_bytes(&self) -> &[u8] { unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.size) } }

    /// Reset to logically empty, keeping the allocation for the next pack.
    pub(crate) fn clear(&mut self) { self.size = 0; }

    /// Make room for a fresh `needed`-byte pack, reusing the allocation when it already fits and
    /// reallocating — discarding the old contents, since packing overwrites — only when it must
    /// grow. Marks the live size and returns the writable base pointer.
    pub(crate) fn reset_for_pack(&mut self, needed: usize) -> Result<*mut u8, Error> {
        if needed > self.capacity {
            self.grow_to(needed)?;
        }
        self.size = needed;
        Ok(self.data.as_ptr())
    }

    /// Pre-grow the allocation to at least `needed` bytes, preserving the live bytes, so a later
    /// `reset_for_pack` call stays allocation-free with a stable pointer, doing nothing when the
    /// allocation already fits.
    pub(crate) fn reserve(&mut self, needed: usize) -> Result<(), Error> {
        if needed <= self.capacity {
            return Ok(());
        }
        let new_layout = layout_for_bytes(needed)?;
        // A packed blob is position-independent - every header offset is relative to the base
        // pointer - so the allocator is free to move it. Let it decide: an arena can extend its
        // most recent block in place, where an unconditional allocate-copy-free never could.
        let block = match self.capacity {
            0 => self.alloc.allocate(new_layout),
            current => {
                let old_layout = layout_for_bytes(current)?;
                // SAFETY: `self.data` is currently allocated from `self.alloc` with `old_layout`,
                // and `needed > current` makes the new layout the larger of the two.
                unsafe { self.alloc.grow(self.data, old_layout, new_layout) }
            }
        }
        .map_err(|_| Error::AllocationFailed)?;
        self.data = block.cast();
        self.capacity = block.len();
        Ok(())
    }

    /// Adopt an externally-produced blob by copying `bytes` into a size-exact allocation.
    pub(crate) fn fill_from_bytes(&mut self, bytes: &[u8]) -> Result<(), Error> {
        if bytes.len() > self.capacity {
            self.grow_to(bytes.len())?;
        }
        if !bytes.is_empty() {
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.as_ptr(), bytes.len()) };
        }
        self.size = bytes.len();
        Ok(())
    }

    /// Grow to exactly `needed` bytes — dealloc old, alloc new; contents are discarded.
    fn grow_to(&mut self, needed: usize) -> Result<(), Error> {
        let (new_data, actual) = if needed == 0 {
            (NonNull::dangling(), 0)
        } else {
            alloc_block(&self.alloc, layout_for_bytes(needed)?)?
        };
        if self.capacity > 0 {
            unsafe { self.dealloc_current() };
        }
        self.data = new_data;
        self.capacity = actual;
        Ok(())
    }

    /// Release the buffer's current allocation. Call only while `capacity > 0`.
    ///
    /// # Safety
    /// The buffer must own a live allocation — the `capacity == 0` case holds a dangling pointer.
    unsafe fn dealloc_current(&self) {
        // `capacity` came from a layout that succeeded, so rebuilding it cannot fail.
        if let Ok(layout) = layout_for_bytes(self.capacity) {
            self.alloc.deallocate(self.data, layout);
        }
    }
}

impl<Alloc: Allocator + Clone> PackedBuffer<Alloc> {
    /// Copy the live bytes into a fresh size-exact buffer on the same allocator.
    pub(crate) fn clone(&self) -> Result<Self, Error> {
        let mut cloned = Self::empty_in(self.alloc.clone());
        cloned.fill_from_bytes(self.as_bytes())?;
        Ok(cloned)
    }
}

impl<Alloc: Allocator> Drop for PackedBuffer<Alloc> {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }
        unsafe { self.dealloc_current() };
    }
}

// endregion: PackedBuffer

// region: Error Types

/// Every failure a NumKong call reports, from shape checks in this crate to a kernel's status.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Memory allocation failed.
    AllocationFailed,
    /// Shape mismatch: axis `axis` has size `expected` on one side and `got` on the other.
    ShapeMismatch { axis: usize, expected: usize, got: usize },
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
    /// Too many dimensions — exceeds MAX_RANK.
    TooManyRanks { got: usize },
    /// A resize would exceed the fixed allocated capacity; grow the buffer via `reserve` first.
    CapacityExceeded { requested: usize, capacity: usize },
    /// Operation not supported for sub-byte types: i4x2, u4x2, u1x8.
    SubByteUnsupported,
    /// A kernel refused its operands with this [`Status`], like a matrix packed under other
    /// capabilities than the current [`crate::Capabilities::cpu_enabled`].
    KernelFailed { status: Status },
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl std::error::Error for Error {}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::AllocationFailed => write!(f, "memory allocation failed"),
            Error::ShapeMismatch { axis, expected, got } => {
                write!(f, "shape mismatch on axis {axis}: expected {expected}, got {got}")
            }
            Error::InvalidShape { axis, size, reason } => {
                write!(f, "invalid shape: axis {axis} has size {size}: {reason}")
            }
            Error::NonContiguousRows => {
                write!(f, "operation requires contiguous rows")
            }
            Error::DimensionMismatch { expected, got } => {
                write!(f, "expected {} dimensions, got {}", expected, got)
            }
            Error::IndexOutOfBounds { index, size } => {
                write!(f, "index {} out of bounds for size {}", index, size)
            }
            Error::TooManyRanks { got } => {
                write!(f, "too many ranks: {}", got)
            }
            Error::CapacityExceeded { requested, capacity } => {
                write!(f, "resize needs {requested} storage values but capacity is {capacity}")
            }
            Error::SubByteUnsupported => {
                write!(f, "operation not supported for sub-byte types")
            }
            Error::KernelFailed { status } => write!(f, "kernel failed: {status}"),
        }
    }
}

/// `Ok(())` when a slice kernel's operand of length `got` matches the `expected` one, otherwise a
/// [`Error::ShapeMismatch`] on axis 0.
pub(crate) fn check_len(expected: usize, got: usize) -> Result<(), Error> {
    match expected == got {
        true => Ok(()),
        false => Err(Error::ShapeMismatch { axis: 0, expected, got }),
    }
}

// endregion: Error Types

// region: MinMaxResult

/// Named result from min/max reduction operations.
///
/// `AnyIndex` defaults to `usize` and `Value` is the scalar output type for scalar reductions such
/// as `minmax_all`, while for axis reductions such as `minmax_axis`, `Value` and `AnyIndex` are
/// both tensors.
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
/// Uses raw memory allocation, not std::Vec, for maximum control.
///
/// Supports:
/// - Slicing and subviews, zero-copy
/// - Dot-product multiplication with [`crate::dots::DotsPackedMatrix`]
/// - Reductions — sum, min, max
/// - Elementwise ops — scale, sum, blend, fma
/// - Trigonometry — sin, cos, atan
///
/// # Example
///
/// ```rust,no_run
/// // Requires linking against libnumkong C library
/// use numkong::{Tensor, DotsPackedMatrix};
///
/// let a = Tensor::<f32>::full(&[1024, 512], 1.0).unwrap();
/// let b = Tensor::<f32>::full(&[256, 512], 1.0).unwrap();
///
/// // Pack B once, multiply many times
/// let b_packed = DotsPackedMatrix::new(&b).unwrap();
/// let c = a.dots_packed(&b_packed).unwrap(); // Returns (1024 × 256)
/// ```
pub struct Tensor<Scalar: StorageElement, Alloc: Allocator = Global, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Raw pointer to data buffer.
    data: NonNull<Scalar>,
    /// Shape dimensions, always logical.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Allocated storage-value capacity (`Scalar` slots) — the ceiling `resize` honors and the
    /// count `Drop` frees. Always `>= product(shape) / Scalar::dimensions_per_value()`.
    capacity: usize,
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

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Drop for Tensor<Scalar, Alloc, MAX_RANK> {
    fn drop(&mut self) {
        // Mirror construction's storage sizing exactly: the product of a rank-0 shape is the
        // empty product, 1, so a rank-0 tensor owns one storage element and must be freed here.
        // Special-casing `ndim == 0` to 0 skipped that deallocation and leaked the element.
        if self.capacity == 0 {
            return;
        }
        // `StorageElement: Copy`, so there is nothing to drop in place — only the block to free.
        // `capacity` came from a layout that succeeded, so rebuilding it cannot fail.
        if let Ok(layout) = layout_for::<Scalar>(self.capacity) {
            unsafe { self.alloc.deallocate(self.data.cast(), layout) };
        }
    }
}

impl<Scalar: StorageElement, Alloc: Allocator + Clone, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Clone this tensor, returning an error on allocation failure.
    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Result<Self, Error> {
        Self::from_slice_in(self.as_slice(), self.shape(), self.alloc.clone())
    }
}

// Generic allocator-aware methods
impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Creates a new Tensor filled with a value using a custom allocator.
    ///
    /// The `shape` specifies logical dimensions. For sub-byte types the innermost extent counts
    /// elements, a multiple of the values packed per byte, and storage holds `total` divided by
    /// `dimensions_per_value()` packed values.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn full_in(shape: &[usize], value: Scalar, alloc: Alloc) -> Result<Self, Error> {
        if shape.len() > MAX_RANK {
            return Err(Error::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape_product(shape)?;
        if total == 0 && !shape.is_empty() {
            if let Some(i) = shape.iter().position(|&d| d == 0) {
                return Err(Error::InvalidShape {
                    axis: i,
                    size: 0,
                    reason: "zero-sized dimension",
                });
            }
        }

        let dims_per_value = Scalar::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && !shape[shape.len() - 1].is_multiple_of(dims_per_value) {
            return Err(Error::InvalidShape {
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

        // Allocate a SIMD-aligned buffer and initialize every storage element
        let (data, capacity) = if storage_count == 0 {
            (NonNull::dangling(), 0)
        } else {
            let (block, actual_bytes) = alloc_filled(&alloc, storage_count, value)?;
            (block, actual_bytes / core::mem::size_of::<Scalar>())
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
            capacity,
            alloc,
        })
    }

    /// Creates a zero-initialized Tensor using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn zeros_in(shape: &[usize], alloc: Alloc) -> Result<Self, Error>
    where
        Scalar: Default,
    {
        Self::full_in(shape, Scalar::default(), alloc)
    }

    /// Creates a Tensor filled with ones using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn ones_in(shape: &[usize], alloc: Alloc) -> Result<Self, Error>
    where
        Scalar: crate::types::NumberLike,
    {
        Self::full_in(shape, Scalar::one(), alloc)
    }

    /// Creates an uninitialized Tensor using a custom allocator.
    ///
    /// # Safety
    /// The returned tensor's contents are uninitialized; reading before writing is undefined
    /// behavior.
    pub unsafe fn uninitialized_in(shape: &[usize], alloc: Alloc) -> Result<Self, Error> {
        if shape.len() > MAX_RANK {
            return Err(Error::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape_product(shape)?;
        if total == 0 && !shape.is_empty() {
            if let Some(i) = shape.iter().position(|&d| d == 0) {
                return Err(Error::InvalidShape {
                    axis: i,
                    size: 0,
                    reason: "zero-sized dimension",
                });
            }
        }

        let dims_per_value = Scalar::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && !shape[shape.len() - 1].is_multiple_of(dims_per_value) {
            return Err(Error::InvalidShape {
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

        let (data, capacity) = if storage_count == 0 {
            (NonNull::dangling(), 0)
        } else {
            let layout = layout_for::<Scalar>(storage_count)?;
            let (block, actual_bytes) = alloc_block(&alloc, layout)?;
            let data = unsafe { NonNull::new_unchecked(block.as_ptr() as *mut Scalar) };
            (data, actual_bytes / core::mem::size_of::<Scalar>())
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
            capacity,
            alloc,
        })
    }

    /// Creates a Tensor from existing storage data using a custom allocator.
    ///
    /// The `shape` specifies logical dimensions. For sub-byte types, the `data` slice holds
    /// `shape.product()` divided by `dimensions_per_value()` storage values; for normal types,
    /// `data.len()` equals `shape.product()`.
    pub fn from_slice_in(data: &[Scalar], shape: &[usize], alloc: Alloc) -> Result<Self, Error> {
        if shape.len() > MAX_RANK {
            return Err(Error::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape_product(shape)?;
        let dims_per_value = Scalar::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && !shape[shape.len() - 1].is_multiple_of(dims_per_value) {
            return Err(Error::InvalidShape {
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
            return Err(Error::ShapeMismatch {
                axis: 0,
                expected: expected_storage,
                got: data.len(),
            });
        }

        // Allocate SIMD-aligned buffer and copy using our allocator
        let (ptr, capacity) = if expected_storage == 0 {
            (NonNull::dangling(), 0)
        } else {
            let layout = layout_for::<Scalar>(expected_storage)?;
            let (block, actual_bytes) = alloc_block(&alloc, layout)?;
            // Clone all storage elements
            unsafe {
                let ptr = block.as_ptr() as *mut Scalar;
                for (i, item) in data[..expected_storage].iter().enumerate() {
                    core::ptr::write(ptr.add(i), *item);
                }
                (
                    NonNull::new_unchecked(ptr),
                    actual_bytes / core::mem::size_of::<Scalar>(),
                )
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
            capacity,
            alloc,
        })
    }

    /// Creates a Tensor from per-dimension `f32` values using a custom allocator.
    ///
    /// Each `f32` is converted through `FloatConvertible::DimScalar::from_f32` before storage, so
    /// this works for full-byte types (`f16`, `bf16`, `i8`, …) and sub-byte types (`i4x2`, `u4x2`,
    /// `u1x8`) alike. The length of `scalars` must equal the product of `shape`.
    pub fn from_scalars_in(scalars: &[f32], shape: &[usize], alloc: Alloc) -> Result<Self, Error>
    where
        Scalar: FloatConvertible,
        Alloc: Clone,
    {
        let total: usize = shape_product(shape)?;
        if scalars.len() != total {
            return Err(Error::ShapeMismatch {
                axis: 0,
                expected: total,
                got: scalars.len(),
            });
        }
        // Pack through a Vector so sub-byte types round-trip via `set`.
        let flat = Vector::<Scalar, Alloc>::from_scalars_in(scalars, alloc.clone())?;
        let mut tensor = Self::zeros_in(shape, alloc)?;
        unsafe {
            core::ptr::copy_nonoverlapping(flat.as_ptr(), tensor.as_mut_ptr(), flat.size_values());
        }
        Ok(tensor)
    }

    /// Creates a Tensor from per-dimension `DimScalar` values using a custom allocator.
    ///
    /// Each element of `dim_values` represents one logical dimension; for sub-byte types the values
    /// are packed into their storage representation. The length of `dim_values` must equal the
    /// product of `shape`.
    pub fn from_dims_in(
        dim_values: &[<Scalar as FloatConvertible>::DimScalar],
        shape: &[usize],
        alloc: Alloc,
    ) -> Result<Self, Error>
    where
        Scalar: FloatConvertible,
        Alloc: Clone,
    {
        let total: usize = shape_product(shape)?;
        if dim_values.len() != total {
            return Err(Error::ShapeMismatch {
                axis: 0,
                expected: total,
                got: dim_values.len(),
            });
        }
        let flat = Vector::<Scalar, Alloc>::from_dims_in(dim_values, alloc.clone())?;
        let mut tensor = Self::zeros_in(shape, alloc)?;
        unsafe {
            core::ptr::copy_nonoverlapping(flat.as_ptr(), tensor.as_mut_ptr(), flat.size_values());
        }
        Ok(tensor)
    }

    /// Compute byte strides from a logical shape.
    ///
    /// For sub-byte types (`dims_per_value > 1`), the innermost dimension is divided by
    /// `dims_per_value` before computing strides, so the innermost stride is `size_of::<Scalar>()`
    /// and covers `dims_per_value` logical elements per step.
    fn compute_strides_into(shape: &[usize], dims_per_value: usize, strides: &mut [isize; MAX_RANK]) {
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
    pub fn allocator(&self) -> &Alloc { &self.alloc }

    /// Number of storage values — for sub-byte types, less than numel.
    pub fn storage_len(&self) -> usize { self.numel() / Scalar::dimensions_per_value() }

    /// Convert a 1D contiguous tensor into a [`Vector`], transferring ownership without copying.
    ///
    /// Returns an error if the tensor is not 1D or not contiguous.
    pub fn into_vector(self) -> Result<Vector<Scalar, Alloc>, Error> {
        if self.ndim != 1 {
            return Err(Error::DimensionMismatch {
                expected: 1,
                got: self.ndim,
            });
        }
        let expected_stride = core::mem::size_of::<Scalar>() as isize;
        if self.strides[0] != expected_stride {
            return Err(Error::NonContiguousRows);
        }
        let dims = self.shape[0];
        let data = self.data;
        let alloc = unsafe { core::ptr::read(&self.alloc) };
        core::mem::forget(self);
        // SAFETY: storage size derived from `dims` matches the original allocation.
        Ok(unsafe { Vector::from_raw_parts_in(data, dims, alloc) })
    }
}

// Methods that don't require Clone
impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Construct a tensor from raw parts, taking ownership of the allocation.
    ///
    /// # Safety
    /// - `data` must point to an allocation obtained from `alloc`, aligned to
    ///   [`SIMD_ALIGNMENT`], whose backing buffer is sized for the storage count
    ///   implied by `shape` (`product(shape) / Scalar::dimensions_per_value()`
    ///   slots of `Scalar`).
    /// - `shape`, `strides`, and `ndim` must be consistent with the data layout.
    /// - The caller must not free the memory — this tensor takes ownership.
    pub unsafe fn from_raw_parts_in(
        data: NonNull<Scalar>,
        shape: [usize; MAX_RANK],
        strides: [isize; MAX_RANK],
        ndim: usize,
        alloc: Alloc,
    ) -> Self {
        let capacity = shape[..ndim].iter().product::<usize>() / Scalar::dimensions_per_value();
        Self {
            data,
            shape,
            strides,
            ndim,
            capacity,
            alloc,
        }
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize { self.ndim }

    /// Allocated storage-value capacity (`Scalar` slots) — the ceiling
    /// [`resize`](Self::resize) honors. Always `>= numel() /
    /// Scalar::dimensions_per_value()`.
    pub fn capacity(&self) -> usize { self.capacity }

    /// Validate `shape` and return its packed storage-value count (shared by resize/reserve).
    fn shape_storage_count(shape: &[usize]) -> Result<usize, Error> {
        if shape.len() > MAX_RANK {
            return Err(Error::TooManyRanks { got: shape.len() });
        }
        if let Some(axis) = shape.iter().position(|&d| d == 0) {
            return Err(Error::InvalidShape {
                axis,
                size: 0,
                reason: "zero-sized dimension",
            });
        }
        let dims_per_value = Scalar::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && !shape[shape.len() - 1].is_multiple_of(dims_per_value) {
            return Err(Error::InvalidShape {
                axis: shape.len() - 1,
                size: shape[shape.len() - 1],
                reason: "innermost dimension must be divisible by dimensions_per_value()",
            });
        }
        let total: usize = shape_product(shape)?;
        Ok(if dims_per_value == 1 {
            total
        } else {
            total / dims_per_value
        })
    }

    /// Resize in place to `new_shape` without moving storage.
    ///
    /// Succeeds only when the packed storage fits `capacity()`, so `as_ptr()` stays stable — call
    /// [`reserve`](Self::reserve) first to grow. Returns [`Error::CapacityExceeded`] when it
    /// would overflow or hits a shape error, leaving the tensor unchanged.
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut t = Tensor::<f32>::zeros(&[8, 8])?; // capacity 64
    /// t.resize(&[4, 4])?;                         // shrink within capacity; as_ptr() unchanged
    /// assert!(t.resize(&[9, 8]).is_err());        // beyond capacity
    /// ```
    pub fn resize(&mut self, new_shape: &[usize]) -> Result<(), Error> {
        let storage_count = Self::shape_storage_count(new_shape)?;
        if storage_count > self.capacity {
            return Err(Error::CapacityExceeded {
                requested: storage_count,
                capacity: self.capacity,
            });
        }
        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..new_shape.len()].copy_from_slice(new_shape);
        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(new_shape, Scalar::dimensions_per_value(), &mut strides_arr);
        self.shape = shape_arr;
        self.strides = strides_arr;
        self.ndim = new_shape.len();
        Ok(())
    }

    /// Grow the allocated `capacity()` to hold at least `new_shape`, reallocating and copying the
    /// live elements if needed. A no-op when already large enough. Unlike [`resize`](Self::resize)
    /// it may move storage. Returns [`Error::AllocationFailed`] on failure, leaving it unchanged.
    pub fn reserve(&mut self, new_shape: &[usize]) -> Result<(), Error> {
        let needed = Self::shape_storage_count(new_shape)?;
        if needed <= self.capacity {
            return Ok(());
        }
        let new_layout = layout_for::<Scalar>(needed)?;
        // Hand the move to the allocator rather than always allocate-copy-free: an arena can
        // extend its most recent block in place. `grow` preserves the live elements for us.
        let block = match self.capacity {
            0 => self.alloc.allocate(new_layout),
            current => {
                let old_layout = layout_for::<Scalar>(current)?;
                // SAFETY: `self.data` is currently allocated from `self.alloc` with `old_layout`,
                // and `needed > current` makes the new layout the larger of the two.
                unsafe { self.alloc.grow(self.data.cast(), old_layout, new_layout) }
            }
        }
        .map_err(|_| Error::AllocationFailed)?;
        self.data = block.cast();
        self.capacity = block.len() / core::mem::size_of::<Scalar>();
        Ok(())
    }

    /// Reset to an empty rank-1 shape (`numel() == 0`) while keeping the allocated `capacity()`.
    pub fn clear(&mut self) {
        self.shape = [0usize; MAX_RANK];
        self.strides = [0isize; MAX_RANK];
        self.ndim = 1; // rank-1 extent-0: numel() == 0, distinct from a rank-0 scalar
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }

    /// Returns a pointer to the data.
    pub fn as_ptr(&self) -> *const Scalar { self.data.as_ptr() }

    /// Returns a mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut Scalar { self.data.as_ptr() }

    /// Returns the underlying storage data as a slice.
    ///
    /// For sub-byte types, the slice holds packed storage values, not individual logical elements.
    pub fn as_slice(&self) -> &[Scalar] {
        let count = self.numel() / Scalar::dimensions_per_value();
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), count) }
    }

    /// Returns the underlying storage data as a mutable slice.
    ///
    /// For sub-byte types, the slice holds packed storage values, not individual logical elements.
    pub fn as_mut_slice(&mut self) -> &mut [Scalar] {
        let count = self.numel() / Scalar::dimensions_per_value();
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), count) }
    }

    /// Returns a row of a 2D array.
    pub fn row(&self, i: usize) -> Option<&[Scalar]> {
        if self.ndim != 2 {
            return None;
        }
        let (rows, columns) = (self.shape[0], self.shape[1]);
        if i >= rows {
            return None;
        }
        // `columns` counts dimensions, a multiple of the values per byte, while the slice counts
        // storage values, so rows never straddle a storage value.
        let row_values = Scalar::dimensions_to_values(columns);
        let start = i * row_values;
        Some(&self.as_slice()[start..start + row_values])
    }

    /// Returns a mutable row of a 2D array.
    pub fn row_mut(&mut self, i: usize) -> Option<&mut [Scalar]> {
        if self.ndim != 2 {
            return None;
        }
        let (rows, columns) = (self.shape[0], self.shape[1]);
        if i >= rows {
            return None;
        }
        // `columns` counts dimensions, a multiple of the values per byte, while the slice counts
        // storage values, so rows never straddle a storage value.
        let row_values = Scalar::dimensions_to_values(columns);
        let start = i * row_values;
        Some(&mut self.as_mut_slice()[start..start + row_values])
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
    /// ```rust,no_run
    /// use numkong::tensor::{Tensor, TensorRef};
    ///
    /// let zeros = Tensor::<f32>::full(&[2, 3], 0.0).unwrap();
    /// assert_eq!(zeros.shape(), &[2, 3]);
    /// assert_eq!(zeros.numel(), 6);
    /// assert!(zeros.as_slice().iter().all(|&v| v == 0.0));
    /// ```
    pub fn full(shape: &[usize], value: Scalar) -> Result<Self, Error> { Self::full_in(shape, value, Global) }

    /// Creates a zero-initialized Tensor using the global allocator.
    pub fn zeros(shape: &[usize]) -> Result<Self, Error>
    where
        Scalar: Default,
    {
        Self::zeros_in(shape, Global)
    }

    /// Creates a Tensor filled with ones using the global allocator.
    pub fn ones(shape: &[usize]) -> Result<Self, Error>
    where
        Scalar: crate::types::NumberLike,
    {
        Self::ones_in(shape, Global)
    }

    /// Creates an uninitialized Tensor using the global allocator.
    ///
    /// # Safety
    /// The returned tensor's contents are uninitialized; reading before writing is undefined
    /// behavior.
    pub unsafe fn uninitialized(shape: &[usize]) -> Result<Self, Error> {
        unsafe { Self::uninitialized_in(shape, Global) }
    }

    /// Creates a Tensor from existing slice data using the global allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn from_slice(data: &[Scalar], shape: &[usize]) -> Result<Self, Error> {
        Self::from_slice_in(data, shape, Global)
    }

    /// Creates a Tensor from per-dimension `f32` values using the global allocator.
    ///
    /// Each `f32` is converted through `FloatConvertible::DimScalar::from_f32` before storage. The
    /// length of `scalars` must equal the product of `shape`.
    pub fn from_scalars(scalars: &[f32], shape: &[usize]) -> Result<Self, Error>
    where
        Scalar: FloatConvertible,
    {
        Self::from_scalars_in(scalars, shape, Global)
    }

    /// Creates a Tensor from per-dimension `DimScalar` values using the global allocator.
    ///
    /// Each element of `dim_values` represents one logical dimension. The length of `dim_values`
    /// must equal the product of `shape`.
    pub fn from_dims(dim_values: &[<Scalar as FloatConvertible>::DimScalar], shape: &[usize]) -> Result<Self, Error>
    where
        Scalar: FloatConvertible,
    {
        Self::from_dims_in(dim_values, shape, Global)
    }
}

// endregion: Tensor

// region: SliceRange

/// Represents a range specification for slicing along one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SliceRange {
    /// Full range, equivalent to `..`
    Full,
    /// Single index; reduces the dimension
    Index(usize),
    /// Range from start to end exclusive, equivalent to `start..end`
    Range { start: usize, end: usize },
    /// Range from start to end with step, equivalent to `start..end;step`
    RangeStep { start: usize, end: usize, step: isize },
}

impl SliceRange {
    /// Create a full range.
    pub fn full() -> Self { Self::Full }

    /// Create a single index.
    pub fn index(i: usize) -> Self { Self::Index(i) }

    /// Create a range from start to end.
    pub fn range(start: usize, end: usize) -> Self { Self::Range { start, end } }

    /// Create a range with step.
    pub fn range_step(start: usize, end: usize, step: isize) -> Self { Self::RangeStep { start, end, step } }
}

/// A stepped range for use in tuple-based slicing — compile-time dispatch.
///
/// Rust has no built-in literal for stepped ranges, so this struct fills that gap. Use it inside
/// `.slice()` tuples:
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
    pub fn new(start: usize, end: usize, step: isize) -> Self { Self { start, end, step } }
}

// endregion: SliceRange

// region: SliceArg + SliceSpec

/// Resolve a signed index against a dimension size. Negative values wrap from the end: `-1` →
/// `dim_size - 1`, `-2` → `dim_size - 2`, etc.
#[inline(always)]
fn resolve_signed_(index: isize, dim_size: usize) -> Result<usize, Error> {
    if index >= 0 {
        Ok(index as usize)
    } else {
        dim_size
            .checked_sub(index.unsigned_abs())
            .ok_or(Error::IndexOutOfBounds {
                index: index.unsigned_abs(),
                size: dim_size,
            })
    }
}

/// Processes one axis of a slice operation, directly computing the effect on output shape, strides,
/// and byte offset — no intermediate enum dispatch.
///
/// Each impl is monomorphized and fully inlined, so the compiler sees concrete types at every call
/// site with zero runtime branching overhead.
///
/// Unsigned types: `RangeFull`, `usize`, `Range<usize>`, `RangeTo<usize>`, `RangeFrom<usize>`,
/// `RangeInclusive<usize>`.
///
/// Signed types (negative wraps from end): `isize`, `Range<isize>`, `RangeTo<isize>`,
/// `RangeFrom<isize>`, `RangeInclusive<isize>`.
///
/// Stepped: `RangeStep` — Rust has no built-in stepped range literal.
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
    ) -> Result<(), Error>;
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
    ) -> Result<(), Error> {
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
    ) -> Result<(), Error> {
        if self >= dim_size {
            return Err(Error::IndexOutOfBounds {
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
    ) -> Result<(), Error> {
        resolve_signed_(self, dim_size)?.apply(dim_size, dim_stride, out_shape, out_strides, new_ndim, offset)
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
    ) -> Result<(), Error> {
        if self.start > self.end || self.end > dim_size {
            return Err(Error::IndexOutOfBounds {
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
    ) -> Result<(), Error> {
        let start = resolve_signed_(self.start, dim_size)?;
        let end = resolve_signed_(self.end, dim_size)?;
        (start..end).apply(dim_size, dim_stride, out_shape, out_strides, new_ndim, offset)
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
    ) -> Result<(), Error> {
        (0..self.end).apply(dim_size, dim_stride, out_shape, out_strides, new_ndim, offset)
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
    ) -> Result<(), Error> {
        let end = resolve_signed_(self.end, dim_size)?;
        (0..end).apply(dim_size, dim_stride, out_shape, out_strides, new_ndim, offset)
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
    ) -> Result<(), Error> {
        (self.start..dim_size).apply(dim_size, dim_stride, out_shape, out_strides, new_ndim, offset)
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
    ) -> Result<(), Error> {
        let start = resolve_signed_(self.start, dim_size)?;
        (start..dim_size).apply(dim_size, dim_stride, out_shape, out_strides, new_ndim, offset)
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
    ) -> Result<(), Error> {
        let start = *self.start();
        let end = self.end().saturating_add(1);
        (start..end).apply(dim_size, dim_stride, out_shape, out_strides, new_ndim, offset)
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
    ) -> Result<(), Error> {
        let start = resolve_signed_(*self.start(), dim_size)?;
        let end = resolve_signed_(*self.end(), dim_size)?.saturating_add(1);
        (start..end).apply(dim_size, dim_stride, out_shape, out_strides, new_ndim, offset)
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
    ) -> Result<(), Error> {
        if self.start >= dim_size || (self.end > dim_size && self.step > 0) {
            return Err(Error::IndexOutOfBounds {
                index: if self.start >= dim_size { self.start } else { self.end },
                size: dim_size,
            });
        }
        if self.step == 0 {
            return Err(Error::InvalidShape {
                axis: 0,
                size: 0,
                reason: "step cannot be zero",
            });
        }
        let count = if self.step > 0 {
            self.end.saturating_sub(self.start).div_ceil(self.step as usize)
        } else {
            let abs_step = (-self.step) as usize;
            self.start.saturating_sub(self.end).div_ceil(abs_step)
        };
        out_shape[*new_ndim] = count;
        out_strides[*new_ndim] = dim_stride * self.step;
        *new_ndim += 1;
        *offset += self.start as isize * dim_stride;
        Ok(())
    }
}

type LayoutResult<const MAX_RANK: usize> = Result<([usize; MAX_RANK], [isize; MAX_RANK], usize, isize, usize), Error>;

/// Computes the full slice layout from shape, strides, and ndim.
///
/// Implemented for `&[SliceRange]` / `&[SliceRange; N]` — backward compat, runtime dispatch — and
/// tuples of [`SliceArg`] types from arity 1-8: compile-time dispatch, fully inlined with zero
/// branching overhead.
pub trait SliceSpec {
    ///
    /// `dims_per_value` is the scalar's packing factor: on the innermost axis a range start counts
    /// logical dimensions while the stride spans a whole storage value, so the two must be
    /// reconciled here rather than by the caller.
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK>;
}

impl SliceSpec for &[SliceRange] {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        slice_layout_(shape, strides, ndim, self, dims_per_value)
    }
}

impl<const N: usize> SliceSpec for &[SliceRange; N] {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        slice_layout_(shape, strides, ndim, self.as_slice(), dims_per_value)
    }
}

impl<Axis0: SliceArg> SliceSpec for (Axis0,) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        // Tuple indexing can narrow the innermost axis, whose start counts logical dimensions
        // against a stride spanning a whole storage value. Use the `&[SliceRange]` form, which
        // permits the representable sub-byte case of taking that axis whole.
        if dims_per_value > 1 {
            return Err(Error::SubByteUnsupported);
        }
        if ndim != 1 {
            return Err(Error::DimensionMismatch { expected: 1, got: ndim });
        }
        let mut sliced_shape = [0usize; MAX_RANK];
        let mut sliced_strides = [0isize; MAX_RANK];
        let (mut sliced_ndim, mut byte_offset) = (0usize, 0isize);
        self.0.apply(
            shape[0],
            strides[0],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        let len = if sliced_ndim == 0 {
            1
        } else {
            sliced_shape[..sliced_ndim].iter().product()
        };
        Ok((sliced_shape, sliced_strides, sliced_ndim, byte_offset, len))
    }
}

impl<Axis0: SliceArg, Axis1: SliceArg> SliceSpec for (Axis0, Axis1) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        // Tuple indexing can narrow the innermost axis, whose start counts logical dimensions
        // against a stride spanning a whole storage value. Use the `&[SliceRange]` form, which
        // permits the representable sub-byte case of taking that axis whole.
        if dims_per_value > 1 {
            return Err(Error::SubByteUnsupported);
        }
        if ndim != 2 {
            return Err(Error::DimensionMismatch { expected: 2, got: ndim });
        }
        let mut sliced_shape = [0usize; MAX_RANK];
        let mut sliced_strides = [0isize; MAX_RANK];
        let (mut sliced_ndim, mut byte_offset) = (0usize, 0isize);
        self.0.apply(
            shape[0],
            strides[0],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.1.apply(
            shape[1],
            strides[1],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        let len = if sliced_ndim == 0 {
            1
        } else {
            sliced_shape[..sliced_ndim].iter().product()
        };
        Ok((sliced_shape, sliced_strides, sliced_ndim, byte_offset, len))
    }
}

impl<Axis0: SliceArg, Axis1: SliceArg, Axis2: SliceArg> SliceSpec for (Axis0, Axis1, Axis2) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        // Tuple indexing can narrow the innermost axis, whose start counts logical dimensions
        // against a stride spanning a whole storage value. Use the `&[SliceRange]` form, which
        // permits the representable sub-byte case of taking that axis whole.
        if dims_per_value > 1 {
            return Err(Error::SubByteUnsupported);
        }
        if ndim != 3 {
            return Err(Error::DimensionMismatch { expected: 3, got: ndim });
        }
        let mut sliced_shape = [0usize; MAX_RANK];
        let mut sliced_strides = [0isize; MAX_RANK];
        let (mut sliced_ndim, mut byte_offset) = (0usize, 0isize);
        self.0.apply(
            shape[0],
            strides[0],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.1.apply(
            shape[1],
            strides[1],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.2.apply(
            shape[2],
            strides[2],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        let len = if sliced_ndim == 0 {
            1
        } else {
            sliced_shape[..sliced_ndim].iter().product()
        };
        Ok((sliced_shape, sliced_strides, sliced_ndim, byte_offset, len))
    }
}

impl<Axis0: SliceArg, Axis1: SliceArg, Axis2: SliceArg, Axis3: SliceArg> SliceSpec for (Axis0, Axis1, Axis2, Axis3) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        // Tuple indexing can narrow the innermost axis, whose start counts logical dimensions
        // against a stride spanning a whole storage value. Use the `&[SliceRange]` form, which
        // permits the representable sub-byte case of taking that axis whole.
        if dims_per_value > 1 {
            return Err(Error::SubByteUnsupported);
        }
        if ndim != 4 {
            return Err(Error::DimensionMismatch { expected: 4, got: ndim });
        }
        let mut sliced_shape = [0usize; MAX_RANK];
        let mut sliced_strides = [0isize; MAX_RANK];
        let (mut sliced_ndim, mut byte_offset) = (0usize, 0isize);
        self.0.apply(
            shape[0],
            strides[0],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.1.apply(
            shape[1],
            strides[1],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.2.apply(
            shape[2],
            strides[2],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.3.apply(
            shape[3],
            strides[3],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        let len = if sliced_ndim == 0 {
            1
        } else {
            sliced_shape[..sliced_ndim].iter().product()
        };
        Ok((sliced_shape, sliced_strides, sliced_ndim, byte_offset, len))
    }
}

impl<Axis0: SliceArg, Axis1: SliceArg, Axis2: SliceArg, Axis3: SliceArg, Axis4: SliceArg> SliceSpec
    for (Axis0, Axis1, Axis2, Axis3, Axis4)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        // Tuple indexing can narrow the innermost axis, whose start counts logical dimensions
        // against a stride spanning a whole storage value. Use the `&[SliceRange]` form, which
        // permits the representable sub-byte case of taking that axis whole.
        if dims_per_value > 1 {
            return Err(Error::SubByteUnsupported);
        }
        if ndim != 5 {
            return Err(Error::DimensionMismatch { expected: 5, got: ndim });
        }
        let mut sliced_shape = [0usize; MAX_RANK];
        let mut sliced_strides = [0isize; MAX_RANK];
        let (mut sliced_ndim, mut byte_offset) = (0usize, 0isize);
        self.0.apply(
            shape[0],
            strides[0],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.1.apply(
            shape[1],
            strides[1],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.2.apply(
            shape[2],
            strides[2],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.3.apply(
            shape[3],
            strides[3],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.4.apply(
            shape[4],
            strides[4],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        let len = if sliced_ndim == 0 {
            1
        } else {
            sliced_shape[..sliced_ndim].iter().product()
        };
        Ok((sliced_shape, sliced_strides, sliced_ndim, byte_offset, len))
    }
}

impl<Axis0: SliceArg, Axis1: SliceArg, Axis2: SliceArg, Axis3: SliceArg, Axis4: SliceArg, Axis5: SliceArg> SliceSpec
    for (Axis0, Axis1, Axis2, Axis3, Axis4, Axis5)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        // Tuple indexing can narrow the innermost axis, whose start counts logical dimensions
        // against a stride spanning a whole storage value. Use the `&[SliceRange]` form, which
        // permits the representable sub-byte case of taking that axis whole.
        if dims_per_value > 1 {
            return Err(Error::SubByteUnsupported);
        }
        if ndim != 6 {
            return Err(Error::DimensionMismatch { expected: 6, got: ndim });
        }
        let mut sliced_shape = [0usize; MAX_RANK];
        let mut sliced_strides = [0isize; MAX_RANK];
        let (mut sliced_ndim, mut byte_offset) = (0usize, 0isize);
        self.0.apply(
            shape[0],
            strides[0],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.1.apply(
            shape[1],
            strides[1],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.2.apply(
            shape[2],
            strides[2],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.3.apply(
            shape[3],
            strides[3],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.4.apply(
            shape[4],
            strides[4],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.5.apply(
            shape[5],
            strides[5],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        let len = if sliced_ndim == 0 {
            1
        } else {
            sliced_shape[..sliced_ndim].iter().product()
        };
        Ok((sliced_shape, sliced_strides, sliced_ndim, byte_offset, len))
    }
}

impl<
        Axis0: SliceArg,
        Axis1: SliceArg,
        Axis2: SliceArg,
        Axis3: SliceArg,
        Axis4: SliceArg,
        Axis5: SliceArg,
        Axis6: SliceArg,
    > SliceSpec for (Axis0, Axis1, Axis2, Axis3, Axis4, Axis5, Axis6)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        // Tuple indexing can narrow the innermost axis, whose start counts logical dimensions
        // against a stride spanning a whole storage value. Use the `&[SliceRange]` form, which
        // permits the representable sub-byte case of taking that axis whole.
        if dims_per_value > 1 {
            return Err(Error::SubByteUnsupported);
        }
        if ndim != 7 {
            return Err(Error::DimensionMismatch { expected: 7, got: ndim });
        }
        let mut sliced_shape = [0usize; MAX_RANK];
        let mut sliced_strides = [0isize; MAX_RANK];
        let (mut sliced_ndim, mut byte_offset) = (0usize, 0isize);
        self.0.apply(
            shape[0],
            strides[0],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.1.apply(
            shape[1],
            strides[1],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.2.apply(
            shape[2],
            strides[2],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.3.apply(
            shape[3],
            strides[3],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.4.apply(
            shape[4],
            strides[4],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.5.apply(
            shape[5],
            strides[5],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.6.apply(
            shape[6],
            strides[6],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        let len = if sliced_ndim == 0 {
            1
        } else {
            sliced_shape[..sliced_ndim].iter().product()
        };
        Ok((sliced_shape, sliced_strides, sliced_ndim, byte_offset, len))
    }
}

impl<
        Axis0: SliceArg,
        Axis1: SliceArg,
        Axis2: SliceArg,
        Axis3: SliceArg,
        Axis4: SliceArg,
        Axis5: SliceArg,
        Axis6: SliceArg,
        Axis7: SliceArg,
    > SliceSpec for (Axis0, Axis1, Axis2, Axis3, Axis4, Axis5, Axis6, Axis7)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
        dims_per_value: usize,
    ) -> LayoutResult<MAX_RANK> {
        // Tuple indexing can narrow the innermost axis, whose start counts logical dimensions
        // against a stride spanning a whole storage value. Use the `&[SliceRange]` form, which
        // permits the representable sub-byte case of taking that axis whole.
        if dims_per_value > 1 {
            return Err(Error::SubByteUnsupported);
        }
        if ndim != 8 {
            return Err(Error::DimensionMismatch { expected: 8, got: ndim });
        }
        let mut sliced_shape = [0usize; MAX_RANK];
        let mut sliced_strides = [0isize; MAX_RANK];
        let (mut sliced_ndim, mut byte_offset) = (0usize, 0isize);
        self.0.apply(
            shape[0],
            strides[0],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.1.apply(
            shape[1],
            strides[1],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.2.apply(
            shape[2],
            strides[2],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.3.apply(
            shape[3],
            strides[3],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.4.apply(
            shape[4],
            strides[4],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.5.apply(
            shape[5],
            strides[5],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.6.apply(
            shape[6],
            strides[6],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        self.7.apply(
            shape[7],
            strides[7],
            &mut sliced_shape,
            &mut sliced_strides,
            &mut sliced_ndim,
            &mut byte_offset,
        )?;
        let len = if sliced_ndim == 0 {
            1
        } else {
            sliced_shape[..sliced_ndim].iter().product()
        };
        Ok((sliced_shape, sliced_strides, sliced_ndim, byte_offset, len))
    }
}

// endregion: SliceArg + SliceSpec

#[doc(hidden)]
pub trait TensorCoordinates {
    const ARITY: usize;

    fn resolve<const MAX_RANK: usize>(self, shape: &[usize; MAX_RANK], ndim: usize)
        -> Result<[usize; MAX_RANK], Error>;
}

impl<I0: VectorIndex, I1: VectorIndex> TensorCoordinates for (I0, I1) {
    const ARITY: usize = 2;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], Error> {
        if ndim != Self::ARITY {
            return Err(Error::DimensionMismatch {
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
    ) -> Result<[usize; MAX_RANK], Error> {
        if ndim != Self::ARITY {
            return Err(Error::DimensionMismatch {
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

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, I3: VectorIndex> TensorCoordinates for (I0, I1, I2, I3) {
    const ARITY: usize = 4;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], Error> {
        if ndim != Self::ARITY {
            return Err(Error::DimensionMismatch {
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

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, I3: VectorIndex, I4: VectorIndex> TensorCoordinates
    for (I0, I1, I2, I3, I4)
{
    const ARITY: usize = 5;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], Error> {
        if ndim != Self::ARITY {
            return Err(Error::DimensionMismatch {
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

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, I3: VectorIndex, I4: VectorIndex, I5: VectorIndex>
    TensorCoordinates for (I0, I1, I2, I3, I4, I5)
{
    const ARITY: usize = 6;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], Error> {
        if ndim != Self::ARITY {
            return Err(Error::DimensionMismatch {
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
    ) -> Result<[usize; MAX_RANK], Error> {
        if ndim != Self::ARITY {
            return Err(Error::DimensionMismatch {
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
    ) -> Result<[usize; MAX_RANK], Error> {
        if ndim != Self::ARITY {
            return Err(Error::DimensionMismatch {
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
/// `TensorView` borrows the parent tensor's memory for the duration of `'a` without owning or
/// copying any data. Views may have arbitrary byte strides, so they transparently represent
/// slicing, transposition, or step-sampling of the underlying storage — iteration and indexing
/// honour those strides.
///
/// A view is normally obtained by calling [`Tensor::view`] or by slicing:
/// `tensor.slice((0..4_usize, ..))`. For lower-level construction from a raw pointer plus
/// shape/stride arrays, see [`TensorView::from_raw_parts`].
///
/// `TensorView` is the immutable counterpart of [`TensorSpan`]. Both share the same layout fields,
/// but a view cannot be used to mutate the backing storage. Multiple views into the same tensor may
/// coexist, subject to Rust's borrow rules; a mutable span excludes all other references.
///
/// The `'a` lifetime ties the view to the source tensor or outer view, ensuring the referenced
/// memory outlives the view itself.
pub struct TensorView<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK, Alloc = Global> {
    /// Pointer to first element of view.
    data: *const Scalar,
    /// Shape of the view, always logical.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// The allocator of the tensor this view borrows, which allocating operations clone.
    allocator: &'a Alloc,
    /// Lifetime marker.
    _marker: PhantomData<&'a Scalar>,
}

impl<Scalar, const MAX_RANK: usize, Alloc> Clone for TensorView<'_, Scalar, MAX_RANK, Alloc> {
    fn clone(&self) -> Self { *self }
}

impl<Scalar, const MAX_RANK: usize, Alloc> Copy for TensorView<'_, Scalar, MAX_RANK, Alloc> {}

impl<Scalar, const MAX_RANK: usize, Alloc> core::fmt::Debug for TensorView<'_, Scalar, MAX_RANK, Alloc> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorView")
            .field("data", &self.data)
            .field("shape", &&self.shape[..self.ndim])
            .field("strides", &&self.strides[..self.ndim])
            .finish()
    }
}

// Safety: a view is a shared borrow with a layout attached — no ownership, no interior mutability,
// no `Drop`. It misses the auto-traits only because the data pointer is raw, so it takes exactly
// the bounds `&[Scalar]` would: sending or sharing a shared borrow needs `Scalar: Sync`.
unsafe impl<Scalar: Sync, const MAX_RANK: usize, Alloc: Sync> Send for TensorView<'_, Scalar, MAX_RANK, Alloc> {}
unsafe impl<Scalar: Sync, const MAX_RANK: usize, Alloc: Sync> Sync for TensorView<'_, Scalar, MAX_RANK, Alloc> {}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize> TensorView<'a, Scalar, MAX_RANK> {
    /// Create a view from a raw pointer, shape, and byte strides, reporting [`Global`] as its
    /// allocator.
    ///
    /// The `shape` specifies logical dimensions. For sub-byte types, the storage count is
    /// `shape.product()` divided by `dimensions_per_value()`; for normal types the two are equal.
    ///
    /// # Safety
    /// - `data` must be valid for reads over the region described by `shape` and `strides_bytes`.
    /// - The pointed-to memory must outlive `'a`.
    ///
    /// Returns `Err` if `shape.len() > MAX_RANK` or `shape.len() != strides_bytes.len()`.
    pub unsafe fn from_raw_parts(data: *const Scalar, shape: &[usize], strides_bytes: &[isize]) -> Result<Self, Error> {
        Self::from_raw_parts_in(data, shape, strides_bytes, &GLOBAL)
    }
}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc> {
    /// Like [`TensorView::from_raw_parts`], reporting `allocator` to the operations that allocate.
    ///
    /// # Safety
    /// The same as for [`TensorView::from_raw_parts`].
    pub unsafe fn from_raw_parts_in(
        data: *const Scalar,
        shape: &[usize],
        strides_bytes: &[isize],
        allocator: &'a Alloc,
    ) -> Result<Self, Error> {
        validate_raw_parts::<MAX_RANK>(shape, strides_bytes)?;
        let ndim = shape.len();
        let mut shape_storage = [0usize; MAX_RANK];
        let mut stride_storage = [0isize; MAX_RANK];
        shape_storage[..ndim].copy_from_slice(shape);
        stride_storage[..ndim].copy_from_slice(strides_bytes);
        Ok(Self {
            data,
            shape: shape_storage,
            strides: stride_storage,
            ndim,
            allocator,
            _marker: PhantomData,
        })
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize { self.ndim }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const Scalar { self.data }

    /// Get element at flat index; only valid for contiguous views.
    ///
    /// # Safety
    /// Caller must ensure the view is contiguous and index is in bounds.
    pub unsafe fn get_unchecked(&self, index: usize) -> &Scalar { &*self.data.add(index) }

    /// Get an element by flat logical row-major index.
    ///
    /// Negative `isize` indices wrap from the end (`-1` is the last element).
    /// Returns [`Error::DimensionMismatch`] on a rank-0 view and
    /// [`Error::IndexOutOfBounds`] when the index is outside the logical element count.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use numkong::tensor::Tensor;
    ///
    /// let t = Tensor::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let view = t.view();
    /// assert_eq!(*view.flat(0_usize).unwrap(), 1.0);
    /// assert_eq!(*view.flat(-1_i32).unwrap(), 4.0);
    /// ```
    pub fn flat<AnyIndex: VectorIndex>(&self, index: AnyIndex) -> Result<&Scalar, Error> {
        if self.ndim == 0 {
            return Err(Error::DimensionMismatch { expected: 1, got: 0 });
        }
        let logical_index = resolve_index_for_size_(index, self.shape[..self.ndim].iter().product::<usize>())?;
        let offset = offset_from_flat_::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const Scalar) })
    }

    /// Get an element by exact coordinates.
    pub fn coords<C: TensorCoordinates>(&self, coords: C) -> Result<&Scalar, Error> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_::<Scalar, MAX_RANK>(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const Scalar) })
    }

    /// Access the scalar value of a rank-0 tensor view.
    pub fn scalar(&self) -> Result<&Scalar, Error> {
        if self.ndim != 0 {
            return Err(Error::DimensionMismatch {
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
    ) -> Result<TensorView<'a, Scalar, MAX_RANK, Alloc>, Error> {
        let (shape, strides, ndim, offset, _) = slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    /// Slice the view along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`. Indexing with a scalar reduces the
    /// rank, while range arguments preserve it.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use numkong::tensor::{Tensor, SliceRange};
    ///
    /// let t = Tensor::<f32>::full(&[4, 5], 1.0).unwrap();
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
    pub fn slice(&self, spec: impl SliceSpec) -> Result<TensorView<'a, Scalar, MAX_RANK, Alloc>, Error> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim, Scalar::dimensions_per_value())?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }
}

impl<'a, Scalar: Clone + StorageElement, const MAX_RANK: usize, Alloc: Allocator>
    TensorView<'a, Scalar, MAX_RANK, Alloc>
{
    /// Copy the view contents to a new owned Tensor.
    pub fn to_owned(&self) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        if self.is_contiguous() {
            let slice = unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) };
            Tensor::from_slice_in(slice, self.shape(), self.allocator.clone())
        } else {
            // For non-contiguous views, we need to copy element by element
            let mut result = Tensor::full_in(self.shape(), unsafe { *self.data }, self.allocator.clone())?;
            self.copy_to_contiguous(result.as_mut_slice());
            Ok(result)
        }
    }

    /// Number of storage values — for sub-byte types, less than numel.
    pub fn storage_len(&self) -> usize { self.numel() / Scalar::dimensions_per_value() }

    /// Convert to slice; only valid for contiguous views.
    pub fn as_contiguous_slice(&self) -> Option<&[Scalar]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) })
        } else {
            None
        }
    }

    /// True when the underlying storage _bytes_ are densely packed in memory.
    ///
    /// Unlike [`is_contiguous`](Self::is_contiguous), this accounts for sub-byte packing: the
    /// innermost-axis extent is measured in storage values (`shape / dimensions_per_value`), so a
    /// freshly-allocated `e2m1x2` tensor, whose row stride is `columns / 2` bytes, is recognized as
    /// packed. Used by the block-scaled casts, which hand raw byte buffers to the C kernel.
    pub fn is_packed_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<Scalar>() as isize;
        let dims_per_value = Scalar::dimensions_per_value();
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            let extent = if i == self.ndim - 1 && dims_per_value > 1 {
                self.shape[i] / dims_per_value
            } else {
                self.shape[i]
            };
            expected_stride *= extent as isize;
        }
        true
    }

    /// Storage values as a slice when the bytes are packed, sub-byte aware.
    ///
    /// Returns `None` for strided/transposed views. This is the sub-byte-aware companion to
    /// [`as_contiguous_slice`](Self::as_contiguous_slice).
    pub fn as_packed_slice(&self) -> Option<&[Scalar]> {
        if self.is_packed_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) })
        } else {
            None
        }
    }

    fn copy_to_contiguous(&self, dest: &mut [Scalar]) {
        // For 2D case, optimize the copy
        if self.ndim == 2 {
            let rows = self.shape[0];
            let columns = self.shape[1];
            let row_stride = self.strides[0];
            let col_stride = self.strides[1];
            let mut dest_idx = 0;
            for r in 0..rows {
                let row_ptr = unsafe { (self.data as *const u8).offset(r as isize * row_stride) as *const Scalar };
                for c in 0..columns {
                    let elem_ptr = unsafe { (row_ptr as *const u8).offset(c as isize * col_stride) as *const Scalar };
                    dest[dest_idx] = unsafe { *elem_ptr };
                    dest_idx += 1;
                }
            }
        } else {
            // General N-dimensional case: iterate in row-major order
            let mut indices = [0usize; MAX_RANK];
            for dest_slot in dest[..self.numel()].iter_mut() {
                let mut offset = 0isize;
                for (index_val, stride_val) in indices[..self.ndim].iter().zip(self.strides[..self.ndim].iter()) {
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
/// `TensorSpan` borrows the parent tensor's memory exclusively for the duration of `'a`, giving
/// write access without taking ownership. As with [`TensorView`], a span may carry non-contiguous
/// byte strides so slicing, transposition, and stepped sub-views all remain free of data copies.
///
/// Spans are typically produced by [`Tensor::span`] or by a mutable slicing method such as
/// `tensor.slice_mut((..,0_usize))`. Lower-level construction from a raw pointer is available
/// through [`TensorSpan::from_raw_parts`].
///
/// A span is the mutable counterpart of [`TensorView`]. At most one span to a given region may
/// exist at a time; reborrow via [`TensorSpan::as_view`] to hand out immutable sub-views without
/// surrendering the span.
///
/// The `'a` lifetime ties the span to the owning tensor, so the referenced memory stays valid for
/// as long as the span exists.
pub struct TensorSpan<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK, Alloc = Global> {
    /// Pointer to first element of view.
    data: *mut Scalar,
    /// Shape of the view, always logical.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// The allocator of the tensor this span borrows, which allocating operations clone.
    allocator: &'a Alloc,
    /// Lifetime marker.
    _marker: PhantomData<&'a mut Scalar>,
}

impl<Scalar, const MAX_RANK: usize, Alloc> core::fmt::Debug for TensorSpan<'_, Scalar, MAX_RANK, Alloc> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorSpan")
            .field("data", &self.data)
            .field("shape", &&self.shape[..self.ndim])
            .field("strides", &&self.strides[..self.ndim])
            .finish()
    }
}

// Safety: a span is a unique borrow with a layout attached, so it takes the bounds `&mut [Scalar]`
// would — moving one to another thread hands over the data itself, which needs `Scalar: Send`,
// while sharing it only exposes reads, which needs `Scalar: Sync`.
unsafe impl<Scalar: Send, const MAX_RANK: usize, Alloc: Sync> Send for TensorSpan<'_, Scalar, MAX_RANK, Alloc> {}
unsafe impl<Scalar: Sync, const MAX_RANK: usize, Alloc: Sync> Sync for TensorSpan<'_, Scalar, MAX_RANK, Alloc> {}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize> TensorSpan<'a, Scalar, MAX_RANK> {
    /// Create a mutable view from a raw pointer, shape, and byte strides, reporting [`Global`] as
    /// its allocator.
    ///
    /// # Safety
    /// - `data` must be valid for reads and writes over the region described by `shape` and
    ///   `strides_bytes`.
    /// - The pointed-to memory must outlive `'a`.
    /// - No other references to the memory may exist for the duration of `'a`.
    ///
    /// Returns `Err` if `shape.len() > MAX_RANK` or `shape.len() != strides_bytes.len()`.
    pub unsafe fn from_raw_parts(data: *mut Scalar, shape: &[usize], strides_bytes: &[isize]) -> Result<Self, Error> {
        Self::from_raw_parts_in(data, shape, strides_bytes, &GLOBAL)
    }
}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
    /// Like [`TensorSpan::from_raw_parts`], reporting `allocator` to the operations that allocate.
    ///
    /// # Safety
    /// The same as for [`TensorSpan::from_raw_parts`].
    pub unsafe fn from_raw_parts_in(
        data: *mut Scalar,
        shape: &[usize],
        strides_bytes: &[isize],
        allocator: &'a Alloc,
    ) -> Result<Self, Error> {
        validate_raw_parts::<MAX_RANK>(shape, strides_bytes)?;
        let ndim = shape.len();
        let mut shape_storage = [0usize; MAX_RANK];
        let mut stride_storage = [0isize; MAX_RANK];
        shape_storage[..ndim].copy_from_slice(shape);
        stride_storage[..ndim].copy_from_slice(strides_bytes);
        Ok(Self {
            data,
            shape: shape_storage,
            strides: stride_storage,
            ndim,
            allocator,
            _marker: PhantomData,
        })
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize { self.ndim }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const Scalar { self.data }

    /// Returns a mutable pointer to the first element.
    pub fn as_mut_ptr(&mut self) -> *mut Scalar { self.data }

    /// Reborrow as immutable view.
    pub fn as_view(&self) -> TensorView<'_, Scalar, MAX_RANK, Alloc> {
        TensorView {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        }
    }

    /// Get an element by flat logical row-major index.
    pub fn flat<AnyIndex: VectorIndex>(&self, index: AnyIndex) -> Result<&Scalar, Error> {
        if self.ndim == 0 {
            return Err(Error::DimensionMismatch { expected: 1, got: 0 });
        }
        let logical_index = resolve_index_for_size_(index, self.shape[..self.ndim].iter().product::<usize>())?;
        let offset = offset_from_flat_::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const Scalar) })
    }

    /// Get a mutable element by flat logical row-major index.
    ///
    /// Negative `isize` indices wrap from the end. Returns [`Error::DimensionMismatch`] on a
    /// rank-0 span and [`Error::IndexOutOfBounds`] when the index is out of range.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use numkong::tensor::Tensor;
    ///
    /// let mut t = Tensor::<f32>::full(&[3], 0.0).unwrap();
    /// let mut span = t.span();
    /// *span.flat_mut(-1_i32).unwrap() = 9.0;
    /// assert_eq!(*span.flat(2_usize).unwrap(), 9.0);
    /// ```
    pub fn flat_mut<AnyIndex: VectorIndex>(&mut self, index: AnyIndex) -> Result<&mut Scalar, Error> {
        if self.ndim == 0 {
            return Err(Error::DimensionMismatch { expected: 1, got: 0 });
        }
        let logical_index = resolve_index_for_size_(index, self.shape[..self.ndim].iter().product::<usize>())?;
        let offset = offset_from_flat_::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &mut *((self.data as *mut u8).offset(offset) as *mut Scalar) })
    }

    /// Get an element by exact coordinates.
    pub fn coords<C: TensorCoordinates>(&self, coords: C) -> Result<&Scalar, Error> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_::<Scalar, MAX_RANK>(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const Scalar) })
    }

    /// Get a mutable element by exact coordinates.
    pub fn coords_mut<C: TensorCoordinates>(&mut self, coords: C) -> Result<&mut Scalar, Error> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_::<Scalar, MAX_RANK>(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &mut *((self.data as *mut u8).offset(offset) as *mut Scalar) })
    }

    /// Access the scalar value of a rank-0 tensor span.
    pub fn scalar(&self) -> Result<&Scalar, Error> {
        if self.ndim != 0 {
            return Err(Error::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &*self.data })
    }

    /// Access the mutable scalar value of a rank-0 tensor span.
    pub fn scalar_mut(&mut self) -> Result<&mut Scalar, Error> {
        if self.ndim != 0 {
            return Err(Error::DimensionMismatch {
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
    ) -> Result<TensorView<'_, Scalar, MAX_RANK, Alloc>, Error> {
        let (shape, strides, ndim, offset, _) = slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    /// Slice the leading axis mutably by one index, reducing rank by one.
    pub fn slice_leading_mut<AnyIndex: VectorIndex>(
        &mut self,
        index: AnyIndex,
    ) -> Result<TensorSpan<'_, Scalar, MAX_RANK, Alloc>, Error> {
        let (shape, strides, ndim, offset, _) = slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorSpan {
            data: unsafe { (self.data as *mut u8).offset(offset) as *mut Scalar },
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    /// Slice the span along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`.
    pub fn slice(&self, spec: impl SliceSpec) -> Result<TensorView<'_, Scalar, MAX_RANK, Alloc>, Error> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim, Scalar::dimensions_per_value())?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    /// Slice the span mutably along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`.
    pub fn slice_mut(&mut self, spec: impl SliceSpec) -> Result<TensorSpan<'_, Scalar, MAX_RANK, Alloc>, Error> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim, Scalar::dimensions_per_value())?;
        Ok(TensorSpan {
            data: unsafe { (self.data as *mut u8).offset(offset) as *mut Scalar },
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }
}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
    /// Number of storage values — for sub-byte types, less than numel.
    pub fn storage_len(&self) -> usize { self.numel() / Scalar::dimensions_per_value() }

    /// Convert to slice; only valid for contiguous views.
    pub fn as_contiguous_slice(&self) -> Option<&[Scalar]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) })
        } else {
            None
        }
    }

    /// Convert to mutable slice; only valid for contiguous views.
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
/// Implemented by [`Tensor`], [`TensorView`], and [`TensorSpan`], enabling generic code over any
/// tensor-like container. Most operations in this crate accept `&(impl TensorRef<Scalar, MAX_RANK>
/// + ?Sized)` so you can mix owned tensors, views, and spans freely without copying.
///
/// The trait only exposes _structural_ accessors — shape, strides, pointer, and a borrow-as-view
/// constructor. Element reads happen through the resulting [`TensorView`]. See [`TensorMut`] for
/// the mutable counterpart.
pub trait TensorRef<Scalar: StorageElement, const MAX_RANK: usize> {
    /// The allocator of the owning tensor, which allocating operations clone for their result.
    ///
    /// An arena that is not `Clone` works by reference: `&Arena` is itself a `Clone` allocator.
    type Alloc: Allocator;

    /// The owning tensor's allocator, or [`Global`] for a view over foreign memory.
    fn allocator(&self) -> &Self::Alloc;

    /// Logical shape as a slice of length `ndim()`.
    ///
    /// For sub-byte types this is the number of individual elements, not the packed storage count.
    fn shape(&self) -> &[usize];

    /// Number of dimensions currently in use.
    ///
    /// Always satisfies `ndim() <= MAX_RANK`. A rank-0 tensor reports `0`.
    fn ndim(&self) -> usize;

    /// Byte stride along dimension `dim`.
    ///
    /// Negative values indicate reverse traversal, such as from a stepped slice. Only indices in
    /// `0..ndim()` are valid.
    ///
    /// # Example
    /// ```rust,ignore
    /// let t = Tensor::<f32>::full(&[3, 4], 0.0).unwrap();
    /// assert_eq!(t.stride_bytes(1), 4);  // innermost f32 stride
    /// ```
    fn stride_bytes(&self, dim: usize) -> isize;

    /// Raw pointer to the first storage element.
    ///
    /// The pointer is valid for `shape().iter().product()` _logical_ reads, divided by
    /// `Scalar::dimensions_per_value()` for sub-byte types.
    fn as_ptr(&self) -> *const Scalar;

    /// Borrow as an immutable [`TensorView`] with the same shape and strides.
    ///
    /// This is a zero-cost reborrow and is the normal entry point for extension-trait methods that
    /// want to delegate to view-level code.
    fn view(&self) -> TensorView<'_, Scalar, MAX_RANK, Self::Alloc>;

    /// Total number of logical elements, the product of the shape dimensions.
    ///
    /// # Example
    /// ```rust,ignore
    /// let t = Tensor::<f32>::full(&[2, 3, 4], 0.0).unwrap();
    /// assert_eq!(t.numel(), 24);
    /// ```
    fn numel(&self) -> usize { self.shape().iter().product() }

    /// Alias for [`ndim`](TensorRef::ndim) — number of dimensions.
    fn rank(&self) -> usize { self.ndim() }

    /// Returns `true` if the tensor contains zero logical elements.
    ///
    /// A rank-0 tensor is a scalar and is _not_ empty — it has one element.
    fn is_empty(&self) -> bool { self.numel() == 0 }

    /// Returns `true` for rank-2 tensors whose innermost stride equals one element — the layout
    /// required by GEMM's left-hand matrix.
    fn has_contiguous_rows(&self) -> bool {
        self.ndim() == 2 && self.stride_bytes(1) == core::mem::size_of::<Scalar>() as isize
    }

    /// Returns `true` if the entire tensor is stored in row-major contiguous order with no gaps.
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
/// `TensorMut` is a supertrait of [`TensorRef`] that adds a single method — a mutable raw pointer
/// to the first element. It is implemented by [`Tensor`], which owns its memory, and by
/// [`TensorSpan`], which borrows a mutable sub-region. Immutable views ([`TensorView`])
/// deliberately do not implement `TensorMut`.
///
/// Generic write-access helpers accept `&mut (impl TensorMut<Scalar, R>)` so they work uniformly
/// against owned tensors and span reborrows.
pub trait TensorMut<Scalar: StorageElement, const MAX_RANK: usize>: TensorRef<Scalar, MAX_RANK> {
    /// Raw mutable pointer to the first storage element.
    ///
    /// The pointer is valid for `numel()` divided by `dimensions_per_value()` writes and preserves
    /// whatever stride layout [`TensorRef::stride_bytes`] reports — callers must honour the
    /// per-axis strides when striding through non-contiguous memory.
    fn as_mut_ptr(&mut self) -> *mut Scalar;
}

/// In-place fill operations shared by every mutable container of `Scalar`.
///
/// Implemented for [`Tensor`], [`TensorSpan`], [`crate::vector::Vector`], and
/// [`crate::vector::VectorSpan`]. Mirrors `tensor_span::fill_zeros / fill` and
/// `vector_span::fill_zeros / fill` from the C++ side. Both methods are infallible and operate on
/// every storage byte the container owns or borrows.
pub trait Fill<Scalar: StorageElement> {
    /// Set every storage byte to zero. Works for every `StorageElement` since the binary-zero of
    /// all NumKong scalar dtypes is the value-zero.
    fn fill_zeros(&mut self);

    /// Set every storage element to `value`. For 1-byte storage, including sub-byte packings, this
    /// becomes a single byte-pattern memset; for multi-byte storage it falls through to a typed
    /// broadcast loop on the pre-zeroed buffer, so float NaN/inf patterns from uninitialised memory
    /// cannot leak through.
    fn fill(&mut self, value: Scalar);
}

/// Copy storage from a `Source` into this container in place.
///
/// `Source` is the natural read-side counterpart for the implementor:
/// `&[Scalar]`, the raw storage slice, for an owning `Tensor` or `Vector`, or the matching `View`
/// type for span implementors, mirroring `tensor_span::copy_from` and `vector_span::copy_from` on
/// the C++ side.
pub trait CopyFrom<Source> {
    /// Copy from `source` into self. Returns an error on a shape or storage mismatch and does not
    /// modify the destination on error.
    fn copy_from(&mut self, source: Source) -> Result<(), Error>;
}

impl<Scalar: StorageElement, Alloc: Allocator, const R: usize> TensorRef<Scalar, R> for Tensor<Scalar, Alloc, R> {
    type Alloc = Alloc;
    fn allocator(&self) -> &Alloc { &self.alloc }
    fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }
    fn ndim(&self) -> usize { self.ndim }
    fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }
    fn as_ptr(&self) -> *const Scalar { self.data.as_ptr() }
    fn view(&self) -> TensorView<'_, Scalar, R, Alloc> {
        TensorView {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            allocator: &self.alloc,
            _marker: PhantomData,
        }
    }
}

impl<Scalar: StorageElement, Alloc: Allocator, const R: usize> TensorMut<Scalar, R> for Tensor<Scalar, Alloc, R> {
    fn as_mut_ptr(&mut self) -> *mut Scalar { self.data.as_ptr() }
}

impl<'a, Scalar: StorageElement, const R: usize, Alloc: Allocator> TensorRef<Scalar, R>
    for TensorView<'a, Scalar, R, Alloc>
{
    type Alloc = Alloc;
    fn allocator(&self) -> &Alloc { self.allocator }
    fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }
    fn ndim(&self) -> usize { self.ndim }
    fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }
    fn as_ptr(&self) -> *const Scalar { self.data }
    fn view(&self) -> TensorView<'_, Scalar, R, Alloc> { *self }
}

impl<'a, Scalar: StorageElement, const R: usize, Alloc: Allocator> TensorRef<Scalar, R>
    for TensorSpan<'a, Scalar, R, Alloc>
{
    type Alloc = Alloc;
    fn allocator(&self) -> &Alloc { self.allocator }
    fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }
    fn ndim(&self) -> usize { self.ndim }
    fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }
    fn as_ptr(&self) -> *const Scalar { self.data }
    fn view(&self) -> TensorView<'_, Scalar, R, Alloc> { self.as_view() }
}

impl<'a, Scalar: StorageElement, const R: usize, Alloc: Allocator> TensorMut<Scalar, R>
    for TensorSpan<'a, Scalar, R, Alloc>
{
    fn as_mut_ptr(&mut self) -> *mut Scalar { self.data }
}

// endregion: TensorRef / TensorMut Traits

// region: Fill / CopyFrom Implementations

/// Compares the bit pattern of a fresh `Scalar::default()` to `value` to decide whether the
/// post-`fill_zeros` overlay can be skipped. Treats `+0.0` and the default `Scalar::default()`
/// representation as identical so a `fill(0.0)` on a freshly-defaulted buffer is a single memset.
#[inline]
fn value_is_default_bit_pattern<Scalar: StorageElement>(value: Scalar) -> bool {
    let default_value = Scalar::default();
    let value_bytes =
        unsafe { core::slice::from_raw_parts((&value as *const Scalar) as *const u8, core::mem::size_of::<Scalar>()) };
    let default_bytes = unsafe {
        core::slice::from_raw_parts(
            (&default_value as *const Scalar) as *const u8,
            core::mem::size_of::<Scalar>(),
        )
    };
    value_bytes == default_bytes
}

/// Body shared by every `Fill::fill` impl: zero the buffer, then a byte-pattern memset for 1-byte
/// storage or a typed broadcast for multi-byte storage.
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

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Fill<Scalar> for Tensor<Scalar, Alloc, MAX_RANK> {
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
    fn copy_from(&mut self, source: &[Scalar]) -> Result<(), Error> {
        let storage_count = self.storage_len();
        if source.len() != storage_count {
            return Err(Error::ShapeMismatch {
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

impl<'a, Scalar: StorageElement, const MAX_RANK: usize, Alloc: Allocator> Fill<Scalar>
    for TensorSpan<'a, Scalar, MAX_RANK, Alloc>
{
    fn fill_zeros(&mut self) {
        // Contiguous fast path: one write_bytes for the whole span.
        if self.is_contiguous() {
            let storage_count = self.numel() / Scalar::dimensions_per_value();
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
            let storage_count = self.numel() / Scalar::dimensions_per_value();
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

impl<'a, 'b, Scalar: StorageElement, const MAX_RANK: usize, Alloc: Allocator, SourceAlloc: Allocator>
    CopyFrom<&'b TensorView<'_, Scalar, MAX_RANK, SourceAlloc>> for TensorSpan<'a, Scalar, MAX_RANK, Alloc>
{
    fn copy_from(&mut self, source: &'b TensorView<'_, Scalar, MAX_RANK, SourceAlloc>) -> Result<(), Error> {
        if self.shape() != source.shape() {
            return Err(Error::ShapeMismatch {
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
            let storage_count = self.numel() / Scalar::dimensions_per_value();
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
/// Each item is a `TensorView` with the iterated dimension removed, leaving rank minus one. For a
/// rank-2 matrix, `axis_views(0)` yields row views.
pub struct AxisIterator<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK, Alloc = Global> {
    data: *const Scalar,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    axis: usize,
    axis_size: usize,
    axis_stride: isize,
    current: usize,
    allocator: &'a Alloc,
    _marker: PhantomData<&'a Scalar>,
}

impl<'a, Scalar, const MAX_RANK: usize, Alloc> Iterator for AxisIterator<'a, Scalar, MAX_RANK, Alloc> {
    type Item = TensorView<'a, Scalar, MAX_RANK, Alloc>;

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
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.axis_size - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, Scalar, const MAX_RANK: usize, Alloc> ExactSizeIterator for AxisIterator<'a, Scalar, MAX_RANK, Alloc> {}
impl<'a, Scalar, const MAX_RANK: usize, Alloc> core::iter::FusedIterator for AxisIterator<'a, Scalar, MAX_RANK, Alloc> {}

/// Mutable iterator over sub-tensor spans along a given axis.
///
/// Each item is a `TensorSpan` with the iterated dimension removed, leaving rank minus one. For a
/// rank-2 matrix, `axis_spans(0)` yields mutable row spans.
pub struct AxisIteratorMut<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK, Alloc = Global> {
    data: *mut Scalar,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    axis: usize,
    axis_size: usize,
    axis_stride: isize,
    current: usize,
    allocator: &'a Alloc,
    _marker: PhantomData<&'a mut Scalar>,
}

impl<'a, Scalar, const MAX_RANK: usize, Alloc> Iterator for AxisIteratorMut<'a, Scalar, MAX_RANK, Alloc> {
    type Item = TensorSpan<'a, Scalar, MAX_RANK, Alloc>;

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
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.axis_size - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, Scalar, const MAX_RANK: usize, Alloc> ExactSizeIterator for AxisIteratorMut<'a, Scalar, MAX_RANK, Alloc> {}
impl<'a, Scalar, const MAX_RANK: usize, Alloc> core::iter::FusedIterator
    for AxisIteratorMut<'a, Scalar, MAX_RANK, Alloc>
{
}

impl<'a, Scalar, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc> {
    /// Iterate along the given axis, yielding sub-tensor views with rank-1.
    pub fn axis_views<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
    ) -> Result<AxisIterator<'a, Scalar, MAX_RANK, Alloc>, Error> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(Error::IndexOutOfBounds {
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
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }
}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc> {
    /// Transpose — reverse all dimensions, no data copy.
    ///
    /// Returns an error for sub-byte types with ndim >= 2, since transposing would produce
    /// non-contiguous strides that break packed element addressing.
    pub fn transpose(&self) -> Result<TensorView<'a, Scalar, MAX_RANK, Alloc>, Error> {
        if self.ndim < 2 {
            return Ok(TensorView {
                data: self.data,
                shape: self.shape,
                strides: self.strides,
                ndim: self.ndim,
                allocator: self.allocator,
                _marker: PhantomData,
            });
        }
        if Scalar::dimensions_per_value() > 1 {
            return Err(Error::SubByteUnsupported);
        }
        let (shape, strides) = transpose_layout(&self.shape, &self.strides, self.ndim);
        Ok(TensorView {
            data: self.data,
            shape,
            strides,
            ndim: self.ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    /// Reshape the view — must have same total elements, contiguous only.
    ///
    /// For sub-byte types this returns an error, since a reshape would invalidate the packed
    /// element layout.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<'a, Scalar, MAX_RANK, Alloc>, Error> {
        if Scalar::dimensions_per_value() > 1 {
            return Err(Error::SubByteUnsupported);
        }
        let (shape, strides, ndim) =
            reshape_layout::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim, self.numel(), new_shape)?;
        Ok(TensorView {
            data: self.data,
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    /// Flatten to 1D; requires contiguous layout.
    pub fn flatten(&self) -> Result<TensorView<'a, Scalar, MAX_RANK, Alloc>, Error> { self.reshape(&[self.numel()]) }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorView<'a, Scalar, MAX_RANK, Alloc> {
        let (shape, strides, ndim) = squeeze_layout::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim);
        TensorView {
            data: self.data,
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        }
    }
}

impl<'a, Scalar: StorageElement, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
    /// Transpose — reverse all dimensions, no data copy.
    ///
    /// Returns an error for sub-byte types with ndim >= 2.
    pub fn transpose(&self) -> Result<TensorSpan<'a, Scalar, MAX_RANK, Alloc>, Error> {
        if self.ndim < 2 {
            return Ok(TensorSpan {
                data: self.data,
                shape: self.shape,
                strides: self.strides,
                ndim: self.ndim,
                allocator: self.allocator,
                _marker: PhantomData,
            });
        }
        if Scalar::dimensions_per_value() > 1 {
            return Err(Error::SubByteUnsupported);
        }
        let (shape, strides) = transpose_layout(&self.shape, &self.strides, self.ndim);
        Ok(TensorSpan {
            data: self.data,
            shape,
            strides,
            ndim: self.ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    /// Reshape the span — must have same total elements, contiguous only.
    ///
    /// Returns an error for sub-byte types.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorSpan<'a, Scalar, MAX_RANK, Alloc>, Error> {
        if Scalar::dimensions_per_value() > 1 {
            return Err(Error::SubByteUnsupported);
        }
        let (shape, strides, ndim) =
            reshape_layout::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim, self.numel(), new_shape)?;
        Ok(TensorSpan {
            data: self.data,
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }

    /// Flatten to 1D; requires contiguous layout.
    pub fn flatten(&self) -> Result<TensorSpan<'a, Scalar, MAX_RANK, Alloc>, Error> { self.reshape(&[self.numel()]) }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
        let (shape, strides, ndim) = squeeze_layout::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim);
        TensorSpan {
            data: self.data,
            shape,
            strides,
            ndim,
            allocator: self.allocator,
            _marker: PhantomData,
        }
    }
}

impl<'a, Scalar: Clone + StorageElement, const MAX_RANK: usize, Alloc: Allocator>
    TensorSpan<'a, Scalar, MAX_RANK, Alloc>
{
    /// Copy the span contents to a new owned Tensor.
    pub fn to_owned(&self) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        self.as_view().to_owned()
    }
}

impl<'a, Scalar, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
    /// Iterate mutably along the given axis, yielding sub-tensor spans with rank-1.
    pub fn axis_spans<AnyIndex: VectorIndex>(
        &mut self,
        axis: AnyIndex,
    ) -> Result<AxisIteratorMut<'a, Scalar, MAX_RANK, Alloc>, Error> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(Error::IndexOutOfBounds {
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
            allocator: self.allocator,
            _marker: PhantomData,
        })
    }
}

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Iterate along the given axis, yielding sub-tensor views with rank-1.
    pub fn axis_views<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
    ) -> Result<AxisIterator<'_, Scalar, MAX_RANK, Alloc>, Error> {
        self.view().axis_views(axis)
    }

    /// Iterate mutably along the given axis, yielding sub-tensor spans with rank-1.
    pub fn axis_spans<AnyIndex: VectorIndex>(
        &mut self,
        axis: AnyIndex,
    ) -> Result<AxisIteratorMut<'_, Scalar, MAX_RANK, Alloc>, Error> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(Error::IndexOutOfBounds {
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
            allocator: &self.alloc,
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
/// Yields `(position, DimRef<'a, Scalar>)` pairs in row-major order at the logical-scalar level.
/// For sub-byte types, the innermost dimension is expanded by `dimensions_per_value`; use
/// [`.dims()`](TensorViewIterator::dims) when only the dimension proxies are needed.
///
/// Cost per `next()` is O(1) amortized (O(ndim) only on carry propagation).
pub struct TensorViewIterator<'a, Scalar: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
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

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> Iterator for TensorViewIterator<'a, Scalar, MAX_RANK> {
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

    fn size_hint(&self) -> (usize, Option<usize>) { (self.remaining, Some(self.remaining)) }
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
    pub fn dims(self) -> TensorViewDims<'a, Scalar, MAX_RANK> { TensorViewDims { inner: self } }
}

/// Dimension-only adapter over [`TensorViewIterator`], yielding [`DimRef`] without positions.
///
/// Created by [`TensorViewIterator::dims()`].
pub struct TensorViewDims<'a, Scalar: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    inner: TensorViewIterator<'a, Scalar, MAX_RANK>,
}

/// Backward-compatible alias for [`TensorViewDims`].
pub type TensorDims<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK> = TensorViewDims<'a, Scalar, MAX_RANK>;
/// Backward-compatible alias for [`TensorViewDims`].
pub type TensorValues<'a, Scalar, const MAX_RANK: usize = DEFAULT_MAX_RANK> = TensorViewDims<'a, Scalar, MAX_RANK>;

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> Iterator for TensorViewDims<'a, Scalar, MAX_RANK> {
    type Item = DimRef<'a, Scalar>;

    #[inline]
    fn next(&mut self) -> Option<DimRef<'a, Scalar>> { self.inner.next().map(|(_, v)| v) }

    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator for TensorViewDims<'a, Scalar, MAX_RANK> {}
impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorViewDims<'a, Scalar, MAX_RANK>
{
}

// endregion: TensorViewIterator

// region: TensorSpanIterator

/// Mutable element iterator over possibly non-contiguous tensor data.
///
/// Yields `(position, DimMut<'a, Scalar>)` pairs in row-major order at the logical-scalar level.
/// For sub-byte types, each proxy performs a read-modify-write on drop.
pub struct TensorSpanIterator<'a, Scalar: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    data: *mut Scalar,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    dims_per_value: usize,
    indices: [usize; MAX_RANK],
    remaining: usize,
    _marker: PhantomData<&'a mut Scalar>,
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> Iterator for TensorSpanIterator<'a, Scalar, MAX_RANK> {
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

    fn size_hint(&self) -> (usize, Option<usize>) { (self.remaining, Some(self.remaining)) }
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
    pub fn dims(self) -> TensorSpanDims<'a, Scalar, MAX_RANK> { TensorSpanDims { inner: self } }
}

/// A mutable dimension-only adapter over [`TensorSpanIterator`], yielding [`DimMut`] items and
/// discarding positions.
pub struct TensorSpanDims<'a, Scalar: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    inner: TensorSpanIterator<'a, Scalar, MAX_RANK>,
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> Iterator for TensorSpanDims<'a, Scalar, MAX_RANK> {
    type Item = DimMut<'a, Scalar>;

    #[inline]
    fn next(&mut self) -> Option<DimMut<'a, Scalar>> { self.inner.next().map(|(_, v)| v) }

    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator for TensorSpanDims<'a, Scalar, MAX_RANK> {}
impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorSpanDims<'a, Scalar, MAX_RANK>
{
}

// endregion: TensorSpanIterator

// region: Tensor iter() / iter_mut() methods

/// Helper to compute the logical shape and total element count for iteration.
///
/// Shape is already logical — sub-byte types store logical dimensions in shape — so this simply
/// copies the shape and computes the product.
fn logical_shape<const MAX_RANK: usize>(shape: &[usize; MAX_RANK], ndim: usize) -> ([usize; MAX_RANK], usize) {
    let logical = *shape;
    let mut total = 1usize;
    for &dim in &logical[..ndim] {
        total *= dim;
    }
    (logical, total)
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc> {
    /// Returns a lazy iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimRef)` pairs. Use `.iter().dims()` for just dimensions. For sub-byte
    /// types, the innermost dimension is expanded.
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

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
    /// Returns a lazy iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimRef)` pairs. Use `.iter().dims()` for just dimensions. For sub-byte
    /// types, the innermost dimension is expanded.
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

impl<Scalar: FloatConvertible, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Returns a lazy iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimRef)` pairs. Use `.iter().dims()` for just dimensions.
    pub fn iter(&self) -> TensorViewIterator<'_, Scalar, MAX_RANK> { self.view().iter() }

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

// region: IntoIterator, immutable

impl<'a, Scalar: FloatConvertible, Alloc: Allocator, const MAX_RANK: usize> IntoIterator
    for &'a Tensor<Scalar, Alloc, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, Scalar>);
    type IntoIter = TensorViewIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize, Alloc: Allocator> IntoIterator
    for &'a TensorView<'a, Scalar, MAX_RANK, Alloc>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, Scalar>);
    type IntoIter = TensorViewIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize, Alloc: Allocator> IntoIterator
    for &'a TensorSpan<'a, Scalar, MAX_RANK, Alloc>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, Scalar>);
    type IntoIter = TensorViewIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

// endregion: IntoIterator, immutable

// region: IntoIterator, mutable

impl<'a, Scalar: FloatConvertible, Alloc: Allocator, const MAX_RANK: usize> IntoIterator
    for &'a mut Tensor<Scalar, Alloc, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimMut<'a, Scalar>);
    type IntoIter = TensorSpanIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize, Alloc: Allocator> IntoIterator
    for &'a mut TensorSpan<'a, Scalar, MAX_RANK, Alloc>
{
    type Item = ([usize; MAX_RANK], DimMut<'a, Scalar>);
    type IntoIter = TensorSpanIterator<'a, Scalar, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

// endregion: IntoIterator, mutable

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

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize, Alloc: Allocator> PartialEq
    for TensorView<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::DimScalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim
            && self.shape[..self.ndim] == other.shape[..other.ndim]
            && self.iter().dims().zip(other.iter().dims()).all(|(a, b)| *a == *b)
    }
}

impl<'a, Scalar: FloatConvertible, const MAX_RANK: usize, Alloc: Allocator> PartialEq
    for TensorSpan<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::DimScalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim
            && self.shape[..self.ndim] == other.shape[..other.ndim]
            && self.iter().dims().zip(other.iter().dims()).all(|(a, b)| *a == *b)
    }
}

// endregion: PartialEq

// region: AsRef

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> AsRef<[Scalar]>
    for Tensor<Scalar, Alloc, MAX_RANK>
{
    fn as_ref(&self) -> &[Scalar] { self.as_slice() }
}

// endregion: AsRef

// region: Tensor View and Slice Methods

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Create a view of the entire array.
    pub fn view(&self) -> TensorView<'_, Scalar, MAX_RANK, Alloc> {
        TensorView {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            allocator: &self.alloc,
            _marker: PhantomData,
        }
    }

    /// Create a mutable span of the entire tensor.
    ///
    /// The returned [`TensorSpan`] borrows `self` exclusively for `'_`. Use it to apply in-place
    /// kernels, write to individual elements via [`TensorSpan::flat_mut`] /
    /// [`TensorSpan::coords_mut`], or pass it to APIs that need `&mut (impl TensorMut<_, _>)`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use numkong::tensor::Tensor;
    ///
    /// let mut t = Tensor::<f32>::full(&[2, 2], 0.0).unwrap();
    /// let mut span = t.span();
    /// *span.flat_mut(0_usize).unwrap() = 42.0;
    /// assert_eq!(*span.flat(0_usize).unwrap(), 42.0);
    /// ```
    pub fn span(&mut self) -> TensorSpan<'_, Scalar, MAX_RANK, Alloc> {
        TensorSpan {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            allocator: &self.alloc,
            _marker: PhantomData,
        }
    }

    /// Get an element by flat logical row-major index.
    pub fn flat<AnyIndex: VectorIndex>(&self, index: AnyIndex) -> Result<&Scalar, Error> {
        if self.ndim == 0 {
            return Err(Error::DimensionMismatch { expected: 1, got: 0 });
        }
        let logical_index = resolve_index_for_size_(index, self.shape[..self.ndim].iter().product::<usize>())?;
        let offset = offset_from_flat_::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data.as_ptr() as *const u8).offset(offset) as *const Scalar) })
    }

    /// Get a mutable element by flat logical row-major index.
    pub fn flat_mut<AnyIndex: VectorIndex>(&mut self, index: AnyIndex) -> Result<&mut Scalar, Error> {
        if self.ndim == 0 {
            return Err(Error::DimensionMismatch { expected: 1, got: 0 });
        }
        let logical_index = resolve_index_for_size_(index, self.shape[..self.ndim].iter().product::<usize>())?;
        let offset = offset_from_flat_::<Scalar, MAX_RANK>(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &mut *((self.data.as_ptr() as *mut u8).offset(offset) as *mut Scalar) })
    }

    /// Get an element by exact coordinates.
    pub fn coords<C: TensorCoordinates>(&self, coords: C) -> Result<&Scalar, Error> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_::<Scalar, MAX_RANK>(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &*((self.data.as_ptr() as *const u8).offset(offset) as *const Scalar) })
    }

    /// Get a mutable element by exact coordinates.
    pub fn coords_mut<C: TensorCoordinates>(&mut self, coords: C) -> Result<&mut Scalar, Error> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_::<Scalar, MAX_RANK>(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &mut *((self.data.as_ptr() as *mut u8).offset(offset) as *mut Scalar) })
    }

    /// Access the scalar value of a rank-0 tensor.
    pub fn scalar(&self) -> Result<&Scalar, Error> {
        if self.ndim != 0 {
            return Err(Error::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &*self.data.as_ptr() })
    }

    /// Access the mutable scalar value of a rank-0 tensor.
    pub fn scalar_mut(&mut self) -> Result<&mut Scalar, Error> {
        if self.ndim != 0 {
            return Err(Error::DimensionMismatch {
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
    ) -> Result<TensorView<'_, Scalar, MAX_RANK, Alloc>, Error> {
        let (shape, strides, ndim, offset, _) = slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data.as_ptr() as *const u8).offset(offset) as *const Scalar },
            shape,
            strides,
            ndim,
            allocator: &self.alloc,
            _marker: PhantomData,
        })
    }

    /// Slice the leading axis mutably by one index, reducing rank by one.
    pub fn slice_leading_mut<AnyIndex: VectorIndex>(
        &mut self,
        index: AnyIndex,
    ) -> Result<TensorSpan<'_, Scalar, MAX_RANK, Alloc>, Error> {
        let (shape, strides, ndim, offset, _) = slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorSpan {
            data: unsafe { (self.data.as_ptr() as *mut u8).offset(offset) as *mut Scalar },
            shape,
            strides,
            ndim,
            allocator: &self.alloc,
            _marker: PhantomData,
        })
    }

    /// Slice the array along multiple dimensions, returning a zero-copy view.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`. Scalar arguments reduce the rank,
    /// range arguments preserve it, and [`RangeStep`] permits stepped sampling.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use numkong::tensor::{Tensor, SliceRange};
    ///
    /// let arr = Tensor::<f32>::full(&[4, 5], 1.0).unwrap();
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
    pub fn slice(&self, spec: impl SliceSpec) -> Result<TensorView<'_, Scalar, MAX_RANK, Alloc>, Error> {
        self.view().slice(spec)
    }

    /// Slice the array mutably along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`.
    pub fn slice_mut(&mut self, spec: impl SliceSpec) -> Result<TensorSpan<'_, Scalar, MAX_RANK, Alloc>, Error> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim, Scalar::dimensions_per_value())?;
        Ok(TensorSpan {
            data: unsafe { (self.data.as_ptr() as *mut u8).offset(offset) as *mut Scalar },
            shape,
            strides,
            ndim,
            allocator: &self.alloc,
            _marker: PhantomData,
        })
    }
}

impl<Scalar: StorageElement, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Transpose — reverse all dimensions, no data copy.
    pub fn transpose(&self) -> Result<TensorView<'_, Scalar, MAX_RANK, Alloc>, Error> { self.view().transpose() }

    /// Reshape the array — must have same total elements, contiguous only.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<'_, Scalar, MAX_RANK, Alloc>, Error> {
        self.view().reshape(new_shape)
    }

    /// Flatten to 1D; requires contiguous layout.
    pub fn flatten(&self) -> Result<TensorView<'_, Scalar, MAX_RANK, Alloc>, Error> { self.view().flatten() }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorView<'_, Scalar, MAX_RANK, Alloc> { self.view().squeeze() }
}

// endregion: Tensor View and Slice Methods

// region: Type Aliases

/// Type alias for a 2D matrix — a Tensor with MAX_RANK=2.
pub type Matrix<Scalar, Alloc = Global> = Tensor<Scalar, Alloc, 2>;

/// Type alias for an immutable 2D matrix view.
pub type MatrixView<'a, Scalar, Alloc = Global> = TensorView<'a, Scalar, 2, Alloc>;

/// Type alias for a mutable 2D matrix view.
pub type MatrixSpan<'a, Scalar, Alloc = Global> = TensorSpan<'a, Scalar, 2, Alloc>;

// endregion: Type Aliases

// region: Tensor Internal Helpers

fn validate_raw_parts<const MAX_RANK: usize>(shape: &[usize], strides_bytes: &[isize]) -> Result<(), Error> {
    if shape.len() > MAX_RANK {
        return Err(Error::TooManyRanks { got: shape.len() });
    }
    if shape.len() != strides_bytes.len() {
        return Err(Error::DimensionMismatch {
            expected: shape.len(),
            got: strides_bytes.len(),
        });
    }
    Ok(())
}

#[inline]
fn validate_same_shape(left: &[usize], right: &[usize]) -> Result<(), Error> {
    if left == right {
        return Ok(());
    }
    if left.len() != right.len() {
        return Err(Error::DimensionMismatch {
            expected: left.len(),
            got: right.len(),
        });
    }
    for (i, (&l, &r)) in left.iter().zip(right).enumerate() {
        if l != r {
            return Err(Error::ShapeMismatch {
                axis: i,
                expected: l,
                got: r,
            });
        }
    }
    Ok(())
}

#[inline]
fn normalize_axis<AnyIndex: VectorIndex>(axis: AnyIndex, ndim: usize) -> Result<usize, Error> {
    if ndim == 0 {
        return Err(Error::IndexOutOfBounds { index: 0, size: ndim });
    }
    axis.resolve(ndim)
        .ok_or(Error::IndexOutOfBounds { index: 0, size: ndim })
}

#[inline]
fn compute_strides_into_<Scalar, const MAX_RANK: usize>(shape: &[usize], strides: &mut [isize; MAX_RANK]) {
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

/// Compute the transposed layout; reverse all dimensions.
fn transpose_layout<const R: usize>(shape: &[usize; R], strides: &[isize; R], ndim: usize) -> ([usize; R], [isize; R]) {
    let mut new_shape = [0usize; R];
    let mut new_strides = [0isize; R];
    for i in 0..ndim {
        new_shape[i] = shape[ndim - 1 - i];
        new_strides[i] = strides[ndim - 1 - i];
    }
    (new_shape, new_strides)
}

/// Compute the reshaped layout — same total elements, contiguous only.
fn reshape_layout<Scalar, const R: usize>(
    shape: &[usize; R],
    strides: &[isize; R],
    ndim: usize,
    len: usize,
    new_shape: &[usize],
) -> Result<([usize; R], [isize; R], usize), Error> {
    if new_shape.len() > R {
        return Err(Error::TooManyRanks { got: new_shape.len() });
    }
    let new_len: usize = shape_product(new_shape)?;
    if new_len != len {
        return Err(Error::ShapeMismatch {
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
            return Err(Error::NonContiguousRows);
        }
        expected *= shape[i] as isize;
    }
    let mut shape_arr = [0usize; R];
    shape_arr[..new_shape.len()].copy_from_slice(new_shape);
    let mut strides_arr = [0isize; R];
    compute_strides_into_::<Scalar, R>(new_shape, &mut strides_arr);
    Ok((shape_arr, strides_arr, new_shape.len()))
}

/// Compute the squeezed layout; remove dimensions of size 1.
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
fn resolve_index_for_size_<AnyIndex: VectorIndex>(index: AnyIndex, size: usize) -> Result<usize, Error> {
    index.resolve(size).ok_or(Error::IndexOutOfBounds { index: 0, size })
}

#[inline]
/// Byte offset of the storage value holding the element at `coords`.
///
/// The innermost coordinate is a logical dimension, but the innermost stride spans a whole storage
/// value — `dimensions_per_value()` logical dimensions for the sub-byte types. Locating the
/// dimension first is what keeps the two in the same units; without it a `u1x8` coordinate produced
/// an offset eight times too large, which the logical-count bounds check could not catch.
fn offset_from_coords_<Scalar: StorageElement, const MAX_RANK: usize>(
    strides: &[isize; MAX_RANK],
    coords: &[usize; MAX_RANK],
    ndim: usize,
) -> isize {
    let mut offset = 0isize;
    for dim in 0..ndim - 1 {
        offset += coords[dim] as isize * strides[dim];
    }
    let innermost = Scalar::locate_dim(coords[ndim - 1]).value_index;
    offset + innermost as isize * strides[ndim - 1]
}

#[inline]
/// Byte offset of the storage value holding the element at row-major `flat_index`.
///
/// Decomposes into per-axis coordinates and then defers to [`offset_from_coords_`], so the sub-byte
/// units are handled in exactly one place.
fn offset_from_flat_<Scalar: StorageElement, const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    strides: &[isize; MAX_RANK],
    ndim: usize,
    mut flat_index: usize,
) -> isize {
    let mut coords = [0usize; MAX_RANK];
    for dim in (0..ndim).rev() {
        coords[dim] = flat_index % shape[dim];
        flat_index /= shape[dim];
    }
    offset_from_coords_::<Scalar, MAX_RANK>(strides, &coords, ndim)
}

fn slice_layout_<const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    strides: &[isize; MAX_RANK],
    ndim: usize,
    ranges: &[SliceRange],
    dims_per_value: usize,
) -> LayoutResult<MAX_RANK> {
    // On the innermost axis a start counts logical dimensions but the stride spans a whole storage
    // value, so for a sub-byte scalar a nonzero start would offset the pointer by `dims_per_value`
    // times too much and hand back a view outside the allocation. Taking that axis whole — which is
    // what a leading-axis row slice does — is the case that stays representable.
    if dims_per_value > 1 && ndim > 0 && !matches!(ranges[ndim - 1], SliceRange::Full) {
        return Err(Error::SubByteUnsupported);
    }
    if ranges.len() != ndim {
        return Err(Error::DimensionMismatch {
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
                    return Err(Error::IndexOutOfBounds { index, size: dim_size });
                }
                offset += index as isize * dim_stride;
            }
            SliceRange::Range { start, end } => {
                if start > end || end > dim_size {
                    return Err(Error::IndexOutOfBounds {
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
                    return Err(Error::IndexOutOfBounds {
                        index: if start >= dim_size { start } else { end },
                        size: dim_size,
                    });
                }
                if step == 0 {
                    return Err(Error::InvalidShape {
                        axis: dim,
                        size: 0,
                        reason: "step cannot be zero",
                    });
                }
                let count = if step > 0 {
                    end.saturating_sub(start).div_ceil(step as usize)
                } else {
                    let abs_step = (-step) as usize;
                    start.saturating_sub(end).div_ceil(abs_step)
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
        return Err(Error::IndexOutOfBounds { index: 0, size: 0 });
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

/// Write the axis-reduced shape into `out`, a stack buffer whose rank is bounded by `MAX_RANK`, and
/// return its length. Heap-free, so it needs no `alloc`.
fn reduced_shape_into(shape: &[usize], axis: usize, keep_dims: bool, out: &mut [usize]) -> usize {
    let mut ndim = 0;
    for (dim_index, &dim_size) in shape.iter().enumerate() {
        if dim_index == axis {
            if keep_dims {
                out[ndim] = 1;
                ndim += 1;
            }
        } else {
            out[ndim] = dim_size;
            ndim += 1;
        }
    }
    ndim
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
        if first_strides[dim_index] == expected_first && second_strides[dim_index] == expected_second {
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
) -> Result<(), Error>
where
    Kernel: FnMut(*const TIn, *mut TOut, usize) -> Result<(), Error>,
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

    unsafe fn walk_outer_dimension<TIn, TOut, Kernel>(
        dim_index: usize,
        outer_dims: usize,
        source_ptr: *const u8,
        source_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut Kernel,
    ) -> Result<(), Error>
    where
        Kernel: FnMut(*const TIn, *mut TOut, usize) -> Result<(), Error>,
    {
        if dim_index == outer_dims {
            return kernel(source_ptr as *const TIn, target_ptr as *mut TOut, tail_len);
        }
        for offset_index in 0..shape[dim_index] {
            let source_child = source_ptr.offset(offset_index as isize * source_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            walk_outer_dimension::<TIn, TOut, Kernel>(
                dim_index + 1,
                outer_dims,
                source_child,
                source_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            )?;
        }
        Ok(())
    }

    walk_outer_dimension::<TIn, TOut, Kernel>(
        0,
        outer_dims,
        source_ptr as *const u8,
        source_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    )
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
) -> Result<(), Error>
where
    Kernel: FnMut(*const TFirst, *const TSecond, *mut TOut, usize) -> Result<(), Error>,
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

    unsafe fn walk_outer_dimension<TFirst, TSecond, TOut, Kernel>(
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
    ) -> Result<(), Error>
    where
        Kernel: FnMut(*const TFirst, *const TSecond, *mut TOut, usize) -> Result<(), Error>,
    {
        if dim_index == outer_dims {
            return kernel(
                first_ptr as *const TFirst,
                second_ptr as *const TSecond,
                target_ptr as *mut TOut,
                tail_len,
            );
        }
        for offset_index in 0..shape[dim_index] {
            let first_child = first_ptr.offset(offset_index as isize * first_strides[dim_index]);
            let second_child = second_ptr.offset(offset_index as isize * second_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            walk_outer_dimension::<TFirst, TSecond, TOut, Kernel>(
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
            )?;
        }
        Ok(())
    }

    walk_outer_dimension::<TFirst, TSecond, TOut, Kernel>(
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
    )
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
) -> Result<(), Error>
where
    Kernel: FnMut(*const TFirst, *const TSecond, *const TThird, *mut TOut, usize) -> Result<(), Error>,
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

    unsafe fn walk_outer_dimension<TFirst, TSecond, TThird, TOut, Kernel>(
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
    ) -> Result<(), Error>
    where
        Kernel: FnMut(*const TFirst, *const TSecond, *const TThird, *mut TOut, usize) -> Result<(), Error>,
    {
        if dim_index == outer_dims {
            return kernel(
                first_ptr as *const TFirst,
                second_ptr as *const TSecond,
                third_ptr as *const TThird,
                target_ptr as *mut TOut,
                tail_len,
            );
        }
        for offset_index in 0..shape[dim_index] {
            let first_child = first_ptr.offset(offset_index as isize * first_strides[dim_index]);
            let second_child = second_ptr.offset(offset_index as isize * second_strides[dim_index]);
            let third_child = third_ptr.offset(offset_index as isize * third_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            walk_outer_dimension::<TFirst, TSecond, TThird, TOut, Kernel>(
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
            )?;
        }
        Ok(())
    }

    walk_outer_dimension::<TFirst, TSecond, TThird, TOut, Kernel>(
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
    )
}

fn for_each_axis_lane<Scalar, const MAX_RANK: usize, Alloc, Kernel>(
    view: &TensorView<'_, Scalar, MAX_RANK, Alloc>,
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

/// Byte offset of the `flat_index`-th logical element, row-major over `shape`, inside a container
/// whose per-axis byte `strides` are given.
///
/// Reduction `_into` kernels enumerate output lanes with a flat row-major counter; when the
/// destination is a strided sub-span rather than a dense owned tensor, that counter must be
/// re-expanded into multi-dimensional coordinates and re-projected through the destination's
/// per-axis strides.
#[inline]
fn logical_index_byte_offset(flat_index: usize, shape: &[usize], strides: &[isize]) -> isize {
    let mut remaining = flat_index;
    let mut offset = 0isize;
    for dim in (0..shape.len()).rev() {
        let extent = shape[dim];
        if extent == 0 {
            continue;
        }
        let coord = remaining % extent;
        remaining /= extent;
        offset += coord as isize * strides[dim];
    }
    offset
}

/// Detect trailing dimensions where `stride[i] == stride[i+1] * shape[i+1]`.
/// Returns `(tail_dims, element_count, abs_stride)`.
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
    let last_ptr = (lane_ptr as *const u8).offset((lane_len as isize - 1) * lane_stride) as *const Scalar;
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
) -> Result<(Scalar::SumOutput, Scalar::SumSqOutput), Error>
where
    Scalar: ReduceMoments,
    Scalar::SumOutput: Default + core::ops::AddAssign,
    Scalar::SumSqOutput: Default + core::ops::AddAssign,
{
    if shape.is_empty() {
        return Scalar::reduce_moments(core::slice::from_raw_parts(data, 1), core::mem::size_of::<Scalar>());
    }
    if shape[0] == 0 {
        return Ok((Scalar::SumOutput::default(), Scalar::SumSqOutput::default()));
    }
    // Re-analyze remaining dimensions for collapsibility at each recursive level.
    if shape.len() >= 2 {
        let (tail, count, _stride_magnitude) = uniform_stride_tail(shape, strides);
        if tail == shape.len() {
            let (ptr, count, stride, _) = normalize_reduction_lane(data, count, strides[shape.len() - 1]);
            return Scalar::reduce_moments(core::slice::from_raw_parts(ptr, count), stride);
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
        let (lane_ptr, lane_len, lane_stride, _) = normalize_reduction_lane(data, shape[0], strides[0]);
        return Scalar::reduce_moments(core::slice::from_raw_parts(lane_ptr, lane_len), lane_stride);
    }

    let mut sum = Scalar::SumOutput::default();
    let mut sumsq = Scalar::SumSqOutput::default();
    for index in 0..shape[0] {
        let child_ptr = (data as *const u8).offset(index as isize * strides[0]) as *const Scalar;
        let (child_sum, child_sumsq) = reduce_moments_recursive::<Scalar>(child_ptr, &shape[1..], &strides[1..])?;
        sum += child_sum;
        sumsq += child_sumsq;
    }
    Ok((sum, sumsq))
}

unsafe fn reduce_minmax_recursive<Scalar>(
    data: *const Scalar,
    shape: &[usize],
    strides: &[isize],
    logical_offset: usize,
) -> Result<Option<MinMaxResult<Scalar::Output>>, Error>
where
    Scalar: ReduceMinMax,
    Scalar::Output: Clone + PartialOrd,
{
    if shape.is_empty() {
        return Ok(
            Scalar::reduce_minmax(core::slice::from_raw_parts(data, 1), core::mem::size_of::<Scalar>())?.map(|lane| {
                MinMaxResult {
                    min_index: logical_offset,
                    max_index: logical_offset,
                    ..lane
                }
            }),
        );
    }
    if shape[0] == 0 {
        return Ok(None);
    }
    // Re-analyze remaining dimensions for collapsibility at each recursive level.
    if shape.len() >= 2 {
        let (tail, count, _stride_magnitude) = uniform_stride_tail(shape, strides);
        if tail == shape.len() {
            let (lane_ptr, lane_len, lane_stride, reversed) =
                normalize_reduction_lane(data, count, strides[shape.len() - 1]);
            return Ok(
                Scalar::reduce_minmax(core::slice::from_raw_parts(lane_ptr, lane_len), lane_stride)?.map(|lane| {
                    let min_index = if reversed {
                        lane_len - 1 - lane.min_index
                    } else {
                        lane.min_index
                    };
                    let max_index = if reversed {
                        lane_len - 1 - lane.max_index
                    } else {
                        lane.max_index
                    };
                    MinMaxResult {
                        min_index: logical_offset + min_index,
                        max_index: logical_offset + max_index,
                        ..lane
                    }
                }),
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
        let (lane_ptr, lane_len, lane_stride, reversed) = normalize_reduction_lane(data, shape[0], strides[0]);
        return Ok(
            Scalar::reduce_minmax(core::slice::from_raw_parts(lane_ptr, lane_len), lane_stride)?.map(|lane| {
                let min_index = if reversed {
                    lane_len - 1 - lane.min_index
                } else {
                    lane.min_index
                };
                let max_index = if reversed {
                    lane_len - 1 - lane.max_index
                } else {
                    lane.max_index
                };
                MinMaxResult {
                    min_index: logical_offset + min_index,
                    max_index: logical_offset + max_index,
                    ..lane
                }
            }),
        );
    }

    let inner_len: usize = shape[1..].iter().product();
    let mut best_min: Option<(Scalar::Output, usize)> = None;
    let mut best_max: Option<(Scalar::Output, usize)> = None;

    for index in 0..shape[0] {
        let child_ptr = (data as *const u8).offset(index as isize * strides[0]) as *const Scalar;
        let child_offset = logical_offset + index * inner_len;
        if let Some(child) = reduce_minmax_recursive::<Scalar>(child_ptr, &shape[1..], &strides[1..], child_offset)? {
            match &best_min {
                Some((best_value, _)) if child.min_value.partial_cmp(best_value) != Some(core::cmp::Ordering::Less) => {
                }
                _ => best_min = Some((child.min_value, child.min_index)),
            }
            match &best_max {
                Some((best_value, _))
                    if child.max_value.partial_cmp(best_value) != Some(core::cmp::Ordering::Greater) => {}
                _ => best_max = Some((child.max_value, child.max_index)),
            }
        }
    }

    Ok(match (best_min, best_max) {
        (Some((min_value, min_index)), Some((max_value, max_index))) => Some(MinMaxResult {
            min_value,
            min_index,
            max_value,
            max_index,
        }),
        _ => None,
    })
}

// SumSqToF64 trait moved to crate::reduce

/// Allocates the result of an operation through `alloc`, a clone of its first operand's allocator,
/// and lets `fill` write it.
fn alloc_output_like<Destination, Alloc, Kernel, const MAX_RANK: usize>(
    shape: &[usize],
    alloc: Alloc,
    fill: Kernel,
) -> Result<Tensor<Destination, Alloc, MAX_RANK>, Error>
where
    Destination: Clone + StorageElement,
    Alloc: Allocator,
    Kernel: FnOnce(&mut TensorSpan<'_, Destination, MAX_RANK, Alloc>) -> Result<(), Error>,
{
    let mut result = unsafe { Tensor::<Destination, Alloc, MAX_RANK>::uninitialized_in(shape, alloc) }?;
    {
        let mut span = result.span();
        fill(&mut span)?;
    }
    Ok(result)
}

fn rebind_view_rank<'a, Scalar, Alloc, const TARGET_MAX_RANK: usize, const SOURCE_MAX_RANK: usize>(
    view: &TensorView<'a, Scalar, SOURCE_MAX_RANK, Alloc>,
) -> Result<TensorView<'a, Scalar, TARGET_MAX_RANK, Alloc>, Error> {
    if view.ndim > TARGET_MAX_RANK {
        return Err(Error::DimensionMismatch {
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
        allocator: view.allocator,
        _marker: PhantomData,
    })
}

fn unary_kernel_into<Source, SourceAlloc, Destination, OutputTensor, Kernel, const MAX_RANK: usize>(
    source: &TensorView<'_, Source, MAX_RANK, SourceAlloc>,
    out: &mut OutputTensor,
    mut kernel: Kernel,
) -> Result<(), Error>
where
    Source: StorageElement,
    SourceAlloc: Allocator,
    Destination: StorageElement,
    OutputTensor: TensorMut<Destination, MAX_RANK> + ?Sized,
    Kernel: FnMut(&[Source], &mut [Destination]) -> Result<(), Error>,
{
    validate_same_shape(source.shape(), out.shape())?;
    let mut target_strides = [0isize; MAX_RANK];
    let out_ndim = out.ndim();
    for (dim, target_stride) in target_strides.iter_mut().enumerate().take(out_ndim) {
        *target_stride = out.stride_bytes(dim);
    }
    let target_ptr = out.as_mut_ptr();
    unsafe {
        walk_contiguous_blocks_2(
            source.data,
            &source.strides[..source.ndim],
            target_ptr,
            &target_strides[..out_ndim],
            source.shape(),
            |source_ptr, target_ptr, tail_len| {
                let source = core::slice::from_raw_parts(source_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(source, target)
            },
        )
    }
}

fn binary_kernel_into<
    First,
    FirstAlloc,
    Second,
    SecondAlloc,
    Destination,
    OutputTensor,
    Kernel,
    const MAX_RANK: usize,
>(
    first: &TensorView<'_, First, MAX_RANK, FirstAlloc>,
    second: &TensorView<'_, Second, MAX_RANK, SecondAlloc>,
    out: &mut OutputTensor,
    mut kernel: Kernel,
) -> Result<(), Error>
where
    First: StorageElement,
    FirstAlloc: Allocator,
    Second: StorageElement,
    SecondAlloc: Allocator,
    Destination: StorageElement,
    OutputTensor: TensorMut<Destination, MAX_RANK> + ?Sized,
    Kernel: FnMut(&[First], &[Second], &mut [Destination]) -> Result<(), Error>,
{
    validate_same_shape(first.shape(), second.shape())?;
    validate_same_shape(first.shape(), out.shape())?;
    let mut target_strides = [0isize; MAX_RANK];
    let out_ndim = out.ndim();
    for (dim, target_stride) in target_strides.iter_mut().enumerate().take(out_ndim) {
        *target_stride = out.stride_bytes(dim);
    }
    let target_ptr = out.as_mut_ptr();
    unsafe {
        walk_contiguous_blocks_3(
            first.data,
            &first.strides[..first.ndim],
            second.data,
            &second.strides[..second.ndim],
            target_ptr,
            &target_strides[..out_ndim],
            first.shape(),
            |first_ptr, second_ptr, target_ptr, tail_len| {
                let first = core::slice::from_raw_parts(first_ptr, tail_len);
                let second = core::slice::from_raw_parts(second_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(first, second, target)
            },
        )
    }
}

fn ternary_kernel_into<
    First,
    Second,
    Third,
    FirstAlloc,
    SecondAlloc,
    ThirdAlloc,
    Destination,
    OutputTensor,
    Kernel,
    const MAX_RANK: usize,
>(
    first: &TensorView<'_, First, MAX_RANK, FirstAlloc>,
    second: &TensorView<'_, Second, MAX_RANK, SecondAlloc>,
    third: &TensorView<'_, Third, MAX_RANK, ThirdAlloc>,
    out: &mut OutputTensor,
    mut kernel: Kernel,
) -> Result<(), Error>
where
    First: StorageElement,
    Second: StorageElement,
    Third: StorageElement,
    FirstAlloc: Allocator,
    SecondAlloc: Allocator,
    ThirdAlloc: Allocator,
    Destination: StorageElement,
    OutputTensor: TensorMut<Destination, MAX_RANK> + ?Sized,
    Kernel: FnMut(&[First], &[Second], &[Third], &mut [Destination]) -> Result<(), Error>,
{
    validate_same_shape(first.shape(), second.shape())?;
    validate_same_shape(first.shape(), third.shape())?;
    validate_same_shape(first.shape(), out.shape())?;
    let mut target_strides = [0isize; MAX_RANK];
    let out_ndim = out.ndim();
    for (dim, target_stride) in target_strides.iter_mut().enumerate().take(out_ndim) {
        *target_stride = out.stride_bytes(dim);
    }
    let target_ptr = out.as_mut_ptr();
    unsafe {
        walk_contiguous_blocks_4(
            first.data,
            &first.strides[..first.ndim],
            second.data,
            &second.strides[..second.ndim],
            third.data,
            &third.strides[..third.ndim],
            target_ptr,
            &target_strides[..out_ndim],
            first.shape(),
            |first_ptr, second_ptr, third_ptr, target_ptr, tail_len| {
                let first = core::slice::from_raw_parts(first_ptr, tail_len);
                let second = core::slice::from_raw_parts(second_ptr, tail_len);
                let third = core::slice::from_raw_parts(third_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(first, second, third, target)
            },
        )
    }
}

// endregion: Tensor Internal Helpers

// region: TensorSpan In-Place Elementwise Operations
//
// In-place mutation operates _through the mutable span_ on its own storage. Each closure forms at
// most a single `&mut [T]` over the target, derived from the span's own pointer, and, for binary
// ops, a disjoint `&[T]` over `other`. No fabricated read-view aliases the span's storage, so no
// overlapping `&[T]` + `&mut [T]` is ever constructed — sound under Stacked/Tree Borrows.

impl<'a, Scalar: Clone + EachScale, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    /// In-place affine: selfᵢ = α × selfᵢ + β.
    pub fn scale_inplace(&mut self, alpha: Scalar::Scalar, beta: Scalar::Scalar) -> Result<(), Error> {
        let ptr = self.data;
        let ndim = self.ndim;
        unsafe {
            walk_contiguous_blocks_2(
                ptr as *const Scalar,
                &self.strides[..ndim],
                ptr,
                &self.strides[..ndim],
                &self.shape[..ndim],
                |_src, dst, len| Scalar::each_scale_inplace(core::slice::from_raw_parts_mut(dst, len), alpha, beta),
            )
        }
    }

    /// In-place add scalar: selfᵢ = selfᵢ + scalar.
    pub fn add_scalar_inplace(&mut self, scalar: Scalar::Scalar) -> Result<(), Error> {
        self.scale_inplace(Scalar::Scalar::from(1.0f32), scalar)
    }

    /// In-place subtract scalar: selfᵢ = selfᵢ − scalar.
    pub fn sub_scalar_inplace(&mut self, scalar: Scalar::Scalar) -> Result<(), Error> {
        self.scale_inplace(Scalar::Scalar::from(1.0f32), Scalar::Scalar::from(-1.0f32) * scalar)
    }

    /// In-place multiply scalar: selfᵢ = selfᵢ × scalar.
    pub fn mul_scalar_inplace(&mut self, scalar: Scalar::Scalar) -> Result<(), Error> {
        self.scale_inplace(scalar, Scalar::Scalar::from(0.0f32))
    }
}

impl<'a, Scalar: Clone + EachSum, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
    /// In-place sum: selfᵢ = selfᵢ + otherᵢ.
    pub fn add_inplace<OtherAlloc: Allocator>(
        &mut self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
    ) -> Result<(), Error> {
        validate_same_shape(self.shape(), other.shape())?;
        let ptr = self.data;
        let ndim = self.ndim;
        unsafe {
            walk_contiguous_blocks_2(
                other.data,
                &other.strides[..other.ndim],
                ptr,
                &self.strides[..ndim],
                &self.shape[..ndim],
                |op, sp, len| {
                    Scalar::each_sum_inplace(
                        core::slice::from_raw_parts_mut(sp, len),
                        core::slice::from_raw_parts(op, len),
                    )
                },
            )
        }
    }
}

impl<'a, Scalar: Clone + EachBlend, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::Scalar: From<f32> + Copy,
{
    /// In-place subtract tensor: selfᵢ = selfᵢ − otherᵢ.
    pub fn sub_inplace<OtherAlloc: Allocator>(
        &mut self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
    ) -> Result<(), Error> {
        validate_same_shape(self.shape(), other.shape())?;
        let ptr = self.data;
        let ndim = self.ndim;
        let alpha = Scalar::Scalar::from(1.0f32);
        let beta = Scalar::Scalar::from(-1.0f32);
        unsafe {
            walk_contiguous_blocks_2(
                other.data,
                &other.strides[..other.ndim],
                ptr,
                &self.strides[..ndim],
                &self.shape[..ndim],
                |op, sp, len| {
                    Scalar::each_blend_inplace(
                        core::slice::from_raw_parts_mut(sp, len),
                        core::slice::from_raw_parts(op, len),
                        alpha,
                        beta,
                    )
                },
            )
        }
    }
}

impl<'a, Scalar: Clone + EachFma, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::Scalar: From<f32> + Copy,
{
    /// In-place multiply tensor: selfᵢ = selfᵢ × otherᵢ.
    ///
    /// Mirrors out-of-place `mul_tensor`, which is an FMA with `c = self` and `alpha = 1`, `beta =
    /// 0`. The `c` operand is bound to `self` inside the kernel via the single span pointer, so
    /// `other` remains the only foreign slice.
    pub fn mul_inplace<OtherAlloc: Allocator>(
        &mut self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
    ) -> Result<(), Error> {
        validate_same_shape(self.shape(), other.shape())?;
        let ptr = self.data;
        let ndim = self.ndim;
        let alpha = Scalar::Scalar::from(1.0f32);
        let beta = Scalar::Scalar::from(0.0f32);
        unsafe {
            walk_contiguous_blocks_2(
                other.data,
                &other.strides[..other.ndim],
                ptr,
                &self.strides[..ndim],
                &self.shape[..ndim],
                |op, sp, len| {
                    Scalar::each_fma_inplace(
                        core::slice::from_raw_parts_mut(sp, len),
                        core::slice::from_raw_parts(op, len),
                        alpha,
                        beta,
                    )
                },
            )
        }
    }
}

impl<'a, Scalar: Clone + TrigSin, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
    /// In-place sine: selfᵢ = sin(selfᵢ).
    pub fn sin_inplace(&mut self) -> Result<(), Error> {
        let ptr = self.data;
        let ndim = self.ndim;
        unsafe {
            walk_contiguous_blocks_2(
                ptr as *const Scalar,
                &self.strides[..ndim],
                ptr,
                &self.strides[..ndim],
                &self.shape[..ndim],
                |_src, dst, len| Scalar::sin_inplace(core::slice::from_raw_parts_mut(dst, len)),
            )
        }
    }
}

impl<'a, Scalar: Clone + TrigCos, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
    /// In-place cosine: selfᵢ = cos(selfᵢ).
    pub fn cos_inplace(&mut self) -> Result<(), Error> {
        let ptr = self.data;
        let ndim = self.ndim;
        unsafe {
            walk_contiguous_blocks_2(
                ptr as *const Scalar,
                &self.strides[..ndim],
                ptr,
                &self.strides[..ndim],
                &self.shape[..ndim],
                |_src, dst, len| Scalar::cos_inplace(core::slice::from_raw_parts_mut(dst, len)),
            )
        }
    }
}

impl<'a, Scalar: Clone + TrigAtan, const MAX_RANK: usize, Alloc: Allocator> TensorSpan<'a, Scalar, MAX_RANK, Alloc> {
    /// In-place arctangent: selfᵢ = atan(selfᵢ).
    pub fn atan_inplace(&mut self) -> Result<(), Error> {
        let ptr = self.data;
        let ndim = self.ndim;
        unsafe {
            walk_contiguous_blocks_2(
                ptr as *const Scalar,
                &self.strides[..ndim],
                ptr,
                &self.strides[..ndim],
                &self.shape[..ndim],
                |_src, dst, len| Scalar::atan_inplace(core::slice::from_raw_parts_mut(dst, len)),
            )
        }
    }
}

// endregion: TensorSpan In-Place Elementwise Operations

// region: Tensor Elementwise Operations

impl<Scalar: Clone + EachScale, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    /// Apply element-wise scale: result\[i\] = α × self\[i\] + β
    ///
    /// Returns a new array with the scaled values.
    pub fn scale(&self, alpha: Scalar::Scalar, beta: Scalar::Scalar) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        self.view().scale_tensor(alpha, beta)
    }

    /// Apply element-wise scale in-place: self\[i\] = α × self\[i\] + β
    pub fn scale_inplace(&mut self, alpha: Scalar::Scalar, beta: Scalar::Scalar) -> Result<(), Error> {
        self.span().scale_inplace(alpha, beta)
    }
}

impl<Scalar: Clone + EachSum, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Element-wise sum: result\[i\] = self\[i\] + other\[i\]
    ///
    /// Returns a new array with the summed values.
    pub fn add<OtherAlloc: Allocator, const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<Scalar, OtherAlloc, OTHER_MAX_RANK>,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<Scalar, OtherAlloc, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        self.view().add_tensor(&other_view)
    }

    /// Element-wise sum in-place: self\[i\] = self\[i\] + other\[i\]
    pub fn add_inplace<OtherAlloc: Allocator, const OTHER_MAX_RANK: usize>(
        &mut self,
        other: &Tensor<Scalar, OtherAlloc, OTHER_MAX_RANK>,
    ) -> Result<(), Error> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<Scalar, OtherAlloc, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        self.span().add_inplace(&other_view)
    }
}

impl<Scalar: Clone + EachBlend, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    /// Blend: result\[i\] = α × self\[i\] + β × other\[i\]
    ///
    /// Returns a new array with the blend.
    pub fn blend<OtherAlloc: Allocator, const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<Scalar, OtherAlloc, OTHER_MAX_RANK>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<Scalar, OtherAlloc, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        self.view().blend_tensor(&other_view, alpha, beta)
    }
}

impl<Scalar: Clone + EachFma, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK>
where
    Scalar::Scalar: From<f32> + Copy,
{
    /// Fused multiply-add: result\[i\] = α × self\[i\] × b\[i\] + β × c\[i\]
    ///
    /// Returns a new array with the FMA result.
    pub fn fma<BAlloc: Allocator, CAlloc: Allocator, const B_MAX_RANK: usize, const C_MAX_RANK: usize>(
        &self,
        multiplier: &Tensor<Scalar, BAlloc, B_MAX_RANK>,
        addend: &Tensor<Scalar, CAlloc, C_MAX_RANK>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        validate_same_shape(self.shape(), multiplier.shape())?;
        validate_same_shape(self.shape(), addend.shape())?;
        let multiplier_view = rebind_view_rank::<Scalar, BAlloc, MAX_RANK, B_MAX_RANK>(&multiplier.view())?;
        let addend_view = rebind_view_rank::<Scalar, CAlloc, MAX_RANK, C_MAX_RANK>(&addend.view())?;
        self.view().fma_tensors(&multiplier_view, &addend_view, alpha, beta)
    }
}

// endregion: Tensor Elementwise Operations

// region: Tensor Explicit Elementwise + Cast

impl<'a, Scalar: Clone + EachScale, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::Scalar: From<f32> + core::ops::Mul<Output = Scalar::Scalar> + Copy,
{
    pub fn scale_tensor(
        &self,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| {
            self.scale_tensor_into(alpha, beta, span)
        })
    }

    pub fn scale_tensor_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        self.affine_into(alpha, beta, out)
    }

    fn affine_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        unary_kernel_into(self, out, |source, target| {
            Scalar::each_scale(source, alpha, beta, target)
        })
    }

    pub fn add_scalar(&self, scalar: Scalar::Scalar) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| {
            self.add_scalar_into(scalar, span)
        })
    }

    pub fn sub_scalar(&self, scalar: Scalar::Scalar) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| {
            self.sub_scalar_into(scalar, span)
        })
    }

    pub fn mul_scalar(&self, scalar: Scalar::Scalar) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| {
            self.mul_scalar_into(scalar, span)
        })
    }

    pub fn add_scalar_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        scalar: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        self.affine_into(Scalar::Scalar::from(1.0f32), scalar, out)
    }

    pub fn sub_scalar_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        scalar: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        self.affine_into(
            Scalar::Scalar::from(1.0f32),
            Scalar::Scalar::from(-1.0f32) * scalar,
            out,
        )
    }

    pub fn mul_scalar_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        scalar: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        self.affine_into(scalar, Scalar::Scalar::from(0.0f32), out)
    }
}

impl<'a, Scalar: Clone + EachSum, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc> {
    pub fn add_tensor<OtherAlloc: Allocator>(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| {
            self.add_tensor_into(other, span)
        })
    }

    pub fn add_tensor_into<OtherAlloc: Allocator, OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        binary_kernel_into(self, other, out, |first, second, target| {
            Scalar::each_sum(first, second, target)
        })
    }
}

impl<'a, Scalar: Clone + EachBlend, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::Scalar: From<f32> + Copy,
{
    pub fn blend_tensor<OtherAlloc: Allocator>(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| {
            self.blend_tensor_into(other, alpha, beta, span)
        })
    }

    pub fn blend_tensor_into<OtherAlloc: Allocator, OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        binary_kernel_into(self, other, out, |first, second, target| {
            Scalar::each_blend(first, second, alpha, beta, target)
        })
    }

    pub fn sub_tensor<OtherAlloc: Allocator>(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| {
            self.sub_tensor_into(other, span)
        })
    }

    pub fn sub_tensor_into<OtherAlloc: Allocator, OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        self.blend_tensor_into(other, Scalar::Scalar::from(1.0f32), Scalar::Scalar::from(-1.0f32), out)
    }
}

impl<'a, Scalar: Clone + EachFma, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::Scalar: From<f32> + Copy,
{
    pub fn fma_tensors<BAlloc: Allocator, CAlloc: Allocator>(
        &self,
        b: &TensorView<'_, Scalar, MAX_RANK, BAlloc>,
        c: &TensorView<'_, Scalar, MAX_RANK, CAlloc>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| {
            self.fma_tensors_into(b, c, alpha, beta, span)
        })
    }

    pub fn fma_tensors_into<
        BAlloc: Allocator,
        CAlloc: Allocator,
        OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized,
    >(
        &self,
        b: &TensorView<'_, Scalar, MAX_RANK, BAlloc>,
        c: &TensorView<'_, Scalar, MAX_RANK, CAlloc>,
        alpha: Scalar::Scalar,
        beta: Scalar::Scalar,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        ternary_kernel_into(self, b, c, out, |first, second, third, target| {
            Scalar::each_fma(first, second, third, alpha, beta, target)
        })
    }

    pub fn mul_tensor<OtherAlloc: Allocator>(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
    ) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| {
            self.mul_tensor_into(other, span)
        })
    }

    pub fn mul_tensor_into<OtherAlloc: Allocator, OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        other: &TensorView<'_, Scalar, MAX_RANK, OtherAlloc>,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        self.fma_tensors_into(
            other,
            self,
            Scalar::Scalar::from(1.0f32),
            Scalar::Scalar::from(0.0f32),
            out,
        )
    }
}

impl<'a, Source: Clone + CastDType, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Source, MAX_RANK, Alloc> {
    pub fn cast<Destination: Clone + CastDType>(&self) -> Result<Tensor<Destination, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| self.cast_into(span))
    }

    pub fn cast_into<Destination, OutputTensor>(&self, out: &mut OutputTensor) -> Result<(), Error>
    where
        Destination: Clone + CastDType,
        OutputTensor: TensorMut<Destination, MAX_RANK> + ?Sized,
    {
        unary_kernel_into(self, out, |source, target: &mut [Destination]| cast(source, target))
    }
}

// endregion: Tensor Explicit Elementwise + Cast

// region: Tensor Trigonometry

impl<'a, Scalar: Clone + TrigSin, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc> {
    pub fn sin(&self) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| self.sin_into(span))
    }

    pub fn sin_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        unary_kernel_into(self, out, |source, target| Scalar::sin(source, target))
    }
}

impl<'a, Scalar: Clone + TrigCos, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc> {
    pub fn cos(&self) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| self.cos_into(span))
    }

    pub fn cos_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        unary_kernel_into(self, out, |source, target| Scalar::cos(source, target))
    }
}

impl<'a, Scalar: Clone + TrigAtan, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc> {
    pub fn atan(&self) -> Result<Tensor<Scalar, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        alloc_output_like(self.shape(), self.allocator.clone(), |span| self.atan_into(span))
    }

    pub fn atan_into<OutputTensor: TensorMut<Scalar, MAX_RANK> + ?Sized>(
        &self,
        out: &mut OutputTensor,
    ) -> Result<(), Error> {
        unary_kernel_into(self, out, |source, target| Scalar::atan(source, target))
    }
}

// endregion: Tensor Trigonometry

// region: Tensor Reductions

impl<Scalar: Clone + Dot, Alloc: Allocator, const MAX_RANK: usize> Tensor<Scalar, Alloc, MAX_RANK> {
    /// Compute the dot product of this array with another.
    ///
    /// Both arrays must be 1D with the same length.
    pub fn dot_product<OtherAlloc: Allocator, const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<Scalar, OtherAlloc, OTHER_MAX_RANK>,
    ) -> Result<Scalar::Output, Error> {
        if self.ndim != 1 || other.ndim != 1 {
            return Err(Error::DimensionMismatch {
                expected: 1,
                got: if self.ndim != 1 { self.ndim } else { other.ndim },
            });
        }
        if self.numel() != other.numel() {
            return Err(Error::ShapeMismatch {
                axis: 0,
                expected: self.numel(),
                got: other.numel(),
            });
        }
        Scalar::dot(self.as_slice(), other.as_slice())
    }
}

pub(crate) type MomentsAxisResult<Scalar, const MAX_RANK: usize, Alloc> = Result<
    (
        Tensor<<Scalar as ReduceMoments>::SumOutput, Alloc, MAX_RANK>,
        Tensor<<Scalar as ReduceMoments>::SumSqOutput, Alloc, MAX_RANK>,
    ),
    Error,
>;

impl<'a, Scalar: Clone + ReduceMoments, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::SumOutput: Clone + Default + core::ops::AddAssign,
    Scalar::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
    pub fn moments_all(&self) -> Result<(Scalar::SumOutput, Scalar::SumSqOutput), Error> {
        unsafe { reduce_moments_recursive::<Scalar>(self.data, self.shape(), &self.strides[..self.ndim]) }
    }

    pub fn moments_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> MomentsAxisResult<Scalar, MAX_RANK, Alloc>
    where
        Alloc: Clone,
    {
        let axis = normalize_axis(axis, self.ndim)?;
        let mut shape_buf = [0usize; MAX_RANK];
        let reduced_ndim = reduced_shape_into(self.shape(), axis, keep_dims, &mut shape_buf);
        let output_shape = &shape_buf[..reduced_ndim];
        let mut sums = Tensor::full_in(output_shape, Scalar::SumOutput::default(), self.allocator.clone())?;
        let mut sumsqs = Tensor::full_in(output_shape, Scalar::SumSqOutput::default(), self.allocator.clone())?;
        self.moments_axis_into(axis, keep_dims, &mut sums, &mut sumsqs)?;
        Ok((sums, sumsqs))
    }

    pub fn moments_axis_into<AnyIndex, SumTensor, SumSqTensor>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        sum_out: &mut SumTensor,
        sumsq_out: &mut SumSqTensor,
    ) -> Result<(), Error>
    where
        AnyIndex: VectorIndex,
        SumTensor: TensorMut<Scalar::SumOutput, MAX_RANK> + ?Sized,
        SumSqTensor: TensorMut<Scalar::SumSqOutput, MAX_RANK> + ?Sized,
    {
        let axis = normalize_axis(axis, self.ndim)?;
        let mut shape_buf = [0usize; MAX_RANK];
        let reduced_ndim = reduced_shape_into(self.shape(), axis, keep_dims, &mut shape_buf);
        let expected_shape = &shape_buf[..reduced_ndim];
        validate_same_shape(expected_shape, sum_out.shape())?;
        validate_same_shape(expected_shape, sumsq_out.shape())?;

        let out_ndim = sum_out.ndim();
        let mut sum_strides = [0isize; MAX_RANK];
        let mut sumsq_strides = [0isize; MAX_RANK];
        for dim in 0..out_ndim {
            sum_strides[dim] = sum_out.stride_bytes(dim);
            sumsq_strides[dim] = sumsq_out.stride_bytes(dim);
        }
        let sum_base = sum_out.as_mut_ptr() as *mut u8;
        let sumsq_base = sumsq_out.as_mut_ptr() as *mut u8;

        let mut outcome = Ok(());
        for_each_axis_lane(self, axis, |lane_ptr, lane_len, lane_stride, output_index| {
            let (lane_ptr, lane_len, lane_stride, _) =
                unsafe { normalize_reduction_lane(lane_ptr, lane_len, lane_stride) };
            let moments =
                unsafe { Scalar::reduce_moments(core::slice::from_raw_parts(lane_ptr, lane_len), lane_stride) };
            let (sum, sumsq) = match moments {
                Ok(moments) => moments,
                Err(error) => {
                    outcome = Err(error);
                    return;
                }
            };
            let sum_offset = logical_index_byte_offset(output_index, expected_shape, &sum_strides[..out_ndim]);
            let sumsq_offset = logical_index_byte_offset(output_index, expected_shape, &sumsq_strides[..out_ndim]);
            unsafe {
                *(sum_base.offset(sum_offset) as *mut Scalar::SumOutput) = sum;
                *(sumsq_base.offset(sumsq_offset) as *mut Scalar::SumSqOutput) = sumsq;
            }
        });
        outcome
    }

    pub fn sum_all(&self) -> Result<Scalar::SumOutput, Error> { Ok(self.moments_all()?.0) }

    pub fn sum_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<Scalar::SumOutput, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        let (sums, _) = self.moments_axis(axis, keep_dims)?;
        Ok(sums)
    }

    pub fn sum_axis_into<AnyIndex, SumTensor>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        out: &mut SumTensor,
    ) -> Result<(), Error>
    where
        AnyIndex: VectorIndex,
        SumTensor: TensorMut<Scalar::SumOutput, MAX_RANK> + ?Sized,
    {
        let axis = normalize_axis(axis, self.ndim)?;
        let mut shape_buf = [0usize; MAX_RANK];
        let reduced_ndim = reduced_shape_into(self.shape(), axis, keep_dims, &mut shape_buf);
        let expected_shape = &shape_buf[..reduced_ndim];
        validate_same_shape(expected_shape, out.shape())?;
        let mut scratch: Tensor<_, &Alloc, MAX_RANK> =
            Tensor::full_in(expected_shape, Scalar::SumSqOutput::default(), self.allocator)?;
        self.moments_axis_into(axis, keep_dims, out, &mut scratch)
    }

    pub fn norm_all(&self) -> Result<f64, Error> {
        let (_, sumsq) = self.moments_all()?;
        Ok(Roots::sqrt(sumsq.to_f64()))
    }

    pub fn norm_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<f64, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        let (_, sumsqs) = self.moments_axis(axis, keep_dims)?;
        let mut norms = Tensor::full_in(sumsqs.shape(), 0.0, self.allocator.clone())?;
        for (target, value) in norms.as_mut_slice().iter_mut().zip(sumsqs.as_slice().iter()) {
            *target = Roots::sqrt(SumSqToF64::to_f64(*value));
        }
        Ok(norms)
    }

    pub fn norm_axis_into<AnyIndex, NormTensor>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        out: &mut NormTensor,
    ) -> Result<(), Error>
    where
        AnyIndex: VectorIndex,
        NormTensor: TensorMut<f64, MAX_RANK> + ?Sized,
    {
        let axis = normalize_axis(axis, self.ndim)?;
        let mut shape_buf = [0usize; MAX_RANK];
        let reduced_ndim = reduced_shape_into(self.shape(), axis, keep_dims, &mut shape_buf);
        let expected_shape = &shape_buf[..reduced_ndim];
        validate_same_shape(expected_shape, out.shape())?;
        let mut scratch_sum: Tensor<_, &Alloc, MAX_RANK> =
            Tensor::full_in(expected_shape, Scalar::SumOutput::default(), self.allocator)?;
        let mut scratch_sumsq: Tensor<_, &Alloc, MAX_RANK> =
            Tensor::full_in(expected_shape, Scalar::SumSqOutput::default(), self.allocator)?;
        self.moments_axis_into(axis, keep_dims, &mut scratch_sum, &mut scratch_sumsq)?;

        let out_ndim = out.ndim();
        let mut out_strides = [0isize; MAX_RANK];
        for (dim, out_stride) in out_strides.iter_mut().enumerate().take(out_ndim) {
            *out_stride = out.stride_bytes(dim);
        }
        let out_base = out.as_mut_ptr() as *mut u8;
        for (flat_index, value) in scratch_sumsq.as_slice().iter().enumerate() {
            let offset = logical_index_byte_offset(flat_index, expected_shape, &out_strides[..out_ndim]);
            unsafe {
                *(out_base.offset(offset) as *mut f64) = Roots::sqrt(SumSqToF64::to_f64(*value));
            }
        }
        Ok(())
    }
}

pub(crate) type MinMaxAxisResult<Scalar, const MAX_RANK: usize, Alloc> = Result<
    MinMaxResult<Tensor<<Scalar as ReduceMinMax>::Output, Alloc, MAX_RANK>, Tensor<usize, Alloc, MAX_RANK>>,
    Error,
>;

impl<'a, Scalar: Clone + ReduceMinMax, const MAX_RANK: usize, Alloc: Allocator> TensorView<'a, Scalar, MAX_RANK, Alloc>
where
    Scalar::Output: Clone + Default + PartialOrd,
{
    pub fn minmax_all(&self) -> Result<MinMaxResult<Scalar::Output>, Error> {
        unsafe { reduce_minmax_recursive::<Scalar>(self.data, self.shape(), &self.strides[..self.ndim], 0) }?.ok_or(
            Error::InvalidShape {
                axis: 0,
                size: self.numel(),
                reason: "min/max reduction undefined for empty or NaN-only input",
            },
        )
    }

    pub fn minmax_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> MinMaxAxisResult<Scalar, MAX_RANK, Alloc>
    where
        Alloc: Clone,
    {
        let axis = normalize_axis(axis, self.ndim)?;
        let mut shape_buf = [0usize; MAX_RANK];
        let reduced_ndim = reduced_shape_into(self.shape(), axis, keep_dims, &mut shape_buf);
        let output_shape = &shape_buf[..reduced_ndim];
        let mut min_values = Tensor::full_in(output_shape, Scalar::Output::default(), self.allocator.clone())?;
        let mut min_indices = Tensor::full_in(output_shape, 0, self.allocator.clone())?;
        let mut max_values = Tensor::full_in(output_shape, Scalar::Output::default(), self.allocator.clone())?;
        let mut max_indices = Tensor::full_in(output_shape, 0, self.allocator.clone())?;
        self.minmax_axis_into(
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

    pub fn minmax_axis_into<AnyIndex, ValueTensor, IndexTensor>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
        min_out: &mut ValueTensor,
        argmin_out: &mut IndexTensor,
        max_out: &mut ValueTensor,
        argmax_out: &mut IndexTensor,
    ) -> Result<(), Error>
    where
        AnyIndex: VectorIndex,
        ValueTensor: TensorMut<Scalar::Output, MAX_RANK> + ?Sized,
        IndexTensor: TensorMut<usize, MAX_RANK> + ?Sized,
    {
        let axis = normalize_axis(axis, self.ndim)?;
        let mut shape_buf = [0usize; MAX_RANK];
        let reduced_ndim = reduced_shape_into(self.shape(), axis, keep_dims, &mut shape_buf);
        let expected_shape = &shape_buf[..reduced_ndim];
        validate_same_shape(expected_shape, min_out.shape())?;
        validate_same_shape(expected_shape, argmin_out.shape())?;
        validate_same_shape(expected_shape, max_out.shape())?;
        validate_same_shape(expected_shape, argmax_out.shape())?;

        let out_ndim = min_out.ndim();
        let mut min_strides = [0isize; MAX_RANK];
        let mut argmin_strides = [0isize; MAX_RANK];
        let mut max_strides = [0isize; MAX_RANK];
        let mut argmax_strides = [0isize; MAX_RANK];
        for dim in 0..out_ndim {
            min_strides[dim] = min_out.stride_bytes(dim);
            argmin_strides[dim] = argmin_out.stride_bytes(dim);
            max_strides[dim] = max_out.stride_bytes(dim);
            argmax_strides[dim] = argmax_out.stride_bytes(dim);
        }
        let min_base = min_out.as_mut_ptr() as *mut u8;
        let argmin_base = argmin_out.as_mut_ptr() as *mut u8;
        let max_base = max_out.as_mut_ptr() as *mut u8;
        let argmax_base = argmax_out.as_mut_ptr() as *mut u8;

        let mut outcome = Ok(());
        for_each_axis_lane(self, axis, |lane_ptr, lane_len, lane_stride, output_index| {
            if outcome.is_err() {
                return;
            }
            let (lane_ptr, lane_len, lane_stride, reversed) =
                unsafe { normalize_reduction_lane(lane_ptr, lane_len, lane_stride) };
            let lane = unsafe { Scalar::reduce_minmax(core::slice::from_raw_parts(lane_ptr, lane_len), lane_stride) };
            let MinMaxResult {
                min_value,
                min_index,
                max_value,
                max_index,
            } = match lane {
                Ok(Some(extremes)) => extremes,
                Ok(None) => {
                    outcome = Err(Error::InvalidShape {
                        axis,
                        size: lane_len,
                        reason: "min/max reduction undefined for empty or NaN-only lanes",
                    });
                    return;
                }
                Err(error) => {
                    outcome = Err(error);
                    return;
                }
            };
            let min_offset = logical_index_byte_offset(output_index, expected_shape, &min_strides[..out_ndim]);
            let argmin_offset = logical_index_byte_offset(output_index, expected_shape, &argmin_strides[..out_ndim]);
            let max_offset = logical_index_byte_offset(output_index, expected_shape, &max_strides[..out_ndim]);
            let argmax_offset = logical_index_byte_offset(output_index, expected_shape, &argmax_strides[..out_ndim]);
            unsafe {
                *(min_base.offset(min_offset) as *mut Scalar::Output) = min_value;
                *(argmin_base.offset(argmin_offset) as *mut usize) =
                    if reversed { lane_len - 1 - min_index } else { min_index };
                *(max_base.offset(max_offset) as *mut Scalar::Output) = max_value;
                *(argmax_base.offset(argmax_offset) as *mut usize) =
                    if reversed { lane_len - 1 - max_index } else { max_index };
            }
        });
        outcome
    }

    pub fn min_all(&self) -> Result<Scalar::Output, Error> { Ok(self.minmax_all()?.min_value) }

    pub fn argmin_all(&self) -> Result<usize, Error> { Ok(self.minmax_all()?.min_index) }

    pub fn max_all(&self) -> Result<Scalar::Output, Error> { Ok(self.minmax_all()?.max_value) }

    pub fn argmax_all(&self) -> Result<usize, Error> { Ok(self.minmax_all()?.max_index) }

    pub fn min_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<Scalar::Output, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        Ok(self.minmax_axis(axis, keep_dims)?.min_value)
    }

    pub fn argmin_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        Ok(self.minmax_axis(axis, keep_dims)?.min_index)
    }

    pub fn max_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<Scalar::Output, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        Ok(self.minmax_axis(axis, keep_dims)?.max_value)
    }

    pub fn argmax_axis<AnyIndex: VectorIndex>(
        &self,
        axis: AnyIndex,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Alloc, MAX_RANK>, Error>
    where
        Alloc: Clone,
    {
        Ok(self.minmax_axis(axis, keep_dims)?.max_index)
    }
}

// MomentsOps and MinMaxOps moved to crate::reduce

// endregion: Tensor Reductions

// region: Block-Scaled Tensor

use crate::cast::BlockScaledFormat;

/// An owning block-scaled tensor: two composed [`Tensor`]s plus an optional per-tensor scale.
///
/// A block-scaled tensor stores quantized values in blocks along its last axis. It is exactly:
///
/// - `elements`: a `Tensor<F::Element>` of logical shape `(rows, columns)`, sub-byte packed
///   for FP4 formats, and
/// - `block_scales`: a `Tensor<F::Scale>` of shape `(rows, columns / F::BLOCK_SIZE)` — one
///   scale byte per block, and
/// - `tensor_scale`: an `Option<f32>` per-tensor multiplier (`Some` for NVFP4, `None` for MX).
///
/// This composes the existing tensor family rather than introducing a parallel hierarchy; the scale
/// newtypes ([`crate::Ue4m3`] / [`crate::Ue8m0`]) and the packed element scalars are plain
/// [`StorageElement`]s. Construct one with `dense.view().cast_to_scaled::<F>()` and decode it back
/// with `scaled.view().cast::<f32>()`.
#[derive(Debug)]
pub struct ScaledTensor<F: BlockScaledFormat, A: Allocator = Global> {
    elements: Tensor<F::Element, A>,
    block_scales: Tensor<F::Scale, A>,
    tensor_scale: Option<f32>,
}

impl<F: BlockScaledFormat, A: Allocator> ScaledTensor<F, A> {
    /// Assemble a block-scaled tensor from its two component tensors and an optional scale.
    ///
    /// The cast verbs in [`mod@crate::cast`] are the usual way to build these; this constructor is
    /// public so callers holding pre-quantized buffers can wrap them without a re-encode.
    ///
    /// Both parts must agree, because nothing downstream re-checks them: decoding derives the scale
    /// count from the _elements_ shape alone and hands the scales pointer to the kernel, so a
    /// `block_scales` shorter than that shape is read past its end. Whether a per-tensor multiplier
    /// exists at all is fixed by the format rather than the caller, so a mismatch there is rejected
    /// too — its value, for the formats that have one, is genuine data.
    pub fn from_parts(
        elements: Tensor<F::Element, A>,
        block_scales: Tensor<F::Scale, A>,
        tensor_scale: Option<f32>,
    ) -> Result<Self, Error> {
        let mut expected = [0usize; DEFAULT_MAX_RANK];
        let expected_ndim = Self::scales_shape_into(elements.shape(), &mut expected)?;
        if block_scales.shape() != &expected[..expected_ndim] {
            return Err(Error::ShapeMismatch {
                axis: expected_ndim - 1,
                expected: expected[expected_ndim - 1],
                got: block_scales.shape().last().copied().unwrap_or(0),
            });
        }
        if tensor_scale.is_some() != F::HAS_TENSOR_SCALE {
            return Err(Error::InvalidShape {
                axis: 0,
                size: 0,
                reason: "per-tensor scale must be present exactly when the format defines one",
            });
        }
        Ok(ScaledTensor {
            elements,
            block_scales,
            tensor_scale,
        })
    }

    /// Borrow the packed element values.
    pub fn elements(&self) -> TensorView<'_, F::Element, DEFAULT_MAX_RANK, A> { self.elements.view() }

    /// Borrow the per-block scale bytes.
    pub fn block_scales(&self) -> TensorView<'_, F::Scale, DEFAULT_MAX_RANK, A> { self.block_scales.view() }

    /// The per-tensor multiplier (`Some` for NVFP4, `None` for the MX family).
    pub fn tensor_scale(&self) -> Option<f32> { self.tensor_scale }

    /// Logical `(rows, columns)` shape of the elements tensor.
    pub fn shape(&self) -> &[usize] { self.elements.shape() }

    /// Borrow the whole tensor as a [`ScaledTensorView`].
    pub fn view(&self) -> ScaledTensorView<'_, F, A> {
        ScaledTensorView {
            elements: self.elements.view(),
            block_scales: self.block_scales.view(),
            tensor_scale: self.tensor_scale,
        }
    }

    /// Borrow the whole tensor mutably as a [`ScaledTensorSpan`].
    pub fn span(&mut self) -> ScaledTensorSpan<'_, F, A> {
        ScaledTensorSpan {
            elements: self.elements.span(),
            block_scales: self.block_scales.span(),
            tensor_scale: self.tensor_scale,
        }
    }

    /// Allocated element-storage capacity of the packed `elements` buffer (`F::Element` slots) —
    /// the ceiling a coordinated [`resize`](Self::resize) honors.
    pub fn capacity(&self) -> usize { self.elements.capacity() }

    /// Derive the paired scales shape for an element `shape`, with the last axis counted in blocks.
    fn scales_shape_into(shape: &[usize], out: &mut [usize]) -> Result<usize, Error> {
        let (&last, leading) = shape
            .split_last()
            .ok_or(Error::DimensionMismatch { expected: 1, got: 0 })?;
        if last % F::BLOCK_SIZE != 0 {
            return Err(Error::InvalidShape {
                axis: leading.len(),
                size: last,
                reason: "last axis must be divisible by the format block size",
            });
        }
        if shape.len() > out.len() {
            return Err(Error::TooManyRanks { got: shape.len() });
        }
        out[..shape.len()].copy_from_slice(shape);
        out[shape.len() - 1] = last / F::BLOCK_SIZE;
        Ok(shape.len())
    }

    /// Resize the packed elements and the per-block scales in lockstep, moving neither buffer.
    ///
    /// Atomic: fails leaving both children unchanged if either would exceed its capacity — call
    /// [`reserve`](Self::reserve) first to grow.
    pub fn resize(&mut self, new_shape: &[usize]) -> Result<(), Error> {
        let mut scales_buf = [0usize; DEFAULT_MAX_RANK];
        let scales_ndim = Self::scales_shape_into(new_shape, &mut scales_buf)?;
        let scales_shape = &scales_buf[..scales_ndim];
        // Pre-validate both capacities so neither child is mutated on failure.
        let elem_storage = Tensor::<F::Element, A>::shape_storage_count(new_shape)?;
        if elem_storage > self.elements.capacity() {
            return Err(Error::CapacityExceeded {
                requested: elem_storage,
                capacity: self.elements.capacity(),
            });
        }
        let scale_storage = Tensor::<F::Scale, A>::shape_storage_count(scales_shape)?;
        if scale_storage > self.block_scales.capacity() {
            return Err(Error::CapacityExceeded {
                requested: scale_storage,
                capacity: self.block_scales.capacity(),
            });
        }
        self.elements.resize(new_shape)?;
        self.block_scales.resize(scales_shape)?;
        Ok(())
    }

    /// Grow both children's capacity to hold `new_shape` and its paired scales, reallocating if
    /// needed. May move storage. A no-op when already large enough.
    pub fn reserve(&mut self, new_shape: &[usize]) -> Result<(), Error> {
        let mut scales_buf = [0usize; DEFAULT_MAX_RANK];
        let scales_ndim = Self::scales_shape_into(new_shape, &mut scales_buf)?;
        let scales_shape = &scales_buf[..scales_ndim];
        self.elements.reserve(new_shape)?;
        self.block_scales.reserve(scales_shape)?;
        Ok(())
    }

    /// Reset both children to empty while keeping their capacities.
    pub fn clear(&mut self) {
        self.elements.clear();
        self.block_scales.clear();
    }
}

/// A borrowed, read-only view into a [`ScaledTensor`] — composed [`TensorView`]s plus the scale.
#[derive(Debug)]
pub struct ScaledTensorView<'a, F: BlockScaledFormat, A = Global> {
    elements: TensorView<'a, F::Element, DEFAULT_MAX_RANK, A>,
    block_scales: TensorView<'a, F::Scale, DEFAULT_MAX_RANK, A>,
    tensor_scale: Option<f32>,
}

impl<F: BlockScaledFormat, A> Clone for ScaledTensorView<'_, F, A> {
    fn clone(&self) -> Self { *self }
}

impl<F: BlockScaledFormat, A> Copy for ScaledTensorView<'_, F, A> {}

impl<'a, F: BlockScaledFormat, A: Allocator> ScaledTensorView<'a, F, A> {
    /// Borrow the packed element values.
    pub fn elements(&self) -> TensorView<'_, F::Element, DEFAULT_MAX_RANK, A> { self.elements }

    /// Borrow the per-block scale bytes.
    pub fn block_scales(&self) -> TensorView<'_, F::Scale, DEFAULT_MAX_RANK, A> { self.block_scales }

    /// The per-tensor multiplier (`Some` for NVFP4, `None` for the MX family).
    pub fn tensor_scale(&self) -> Option<f32> { self.tensor_scale }

    /// Logical `(rows, columns)` shape.
    pub fn shape(&self) -> &[usize] { self.elements.shape() }

    /// Borrow leading-axis index `i` as a new [`ScaledTensorView`], slicing both sub-tensors in
    /// lockstep and keeping the rank, with the leading extent reduced to 1 rather than dropped.
    pub fn row(&self, i: usize) -> Result<ScaledTensorView<'a, F, A>, Error> { self.rows(i, i + 1) }

    /// Slice a contiguous leading-axis range `start..end`, slicing both sub-tensors in lockstep.
    pub fn rows(&self, start: usize, end: usize) -> Result<ScaledTensorView<'a, F, A>, Error> {
        let leading = self.elements.shape().first().copied().unwrap_or(0);
        if end > leading || start > end {
            return Err(Error::IndexOutOfBounds {
                index: end,
                size: leading,
            });
        }
        // Slice the leading axis; keep every trailing axis, including the quantized last axis, intact.
        let ndim = self.elements.ndim();
        let mut spec = [SliceRange::Full; DEFAULT_MAX_RANK];
        spec[0] = SliceRange::range(start, end);
        // Trailing axes stay `SliceRange::Full` from the initializer.
        let elements = self.elements.slice(&spec[..ndim])?;
        let block_scales = self.block_scales.slice(&spec[..ndim])?;
        Ok(ScaledTensorView {
            elements,
            block_scales,
            tensor_scale: self.tensor_scale,
        })
    }
}

/// A borrowed, mutable view into a [`ScaledTensor`] — composed [`TensorSpan`]s plus the scale.
#[derive(Debug)]
pub struct ScaledTensorSpan<'a, F: BlockScaledFormat, A = Global> {
    elements: TensorSpan<'a, F::Element, DEFAULT_MAX_RANK, A>,
    block_scales: TensorSpan<'a, F::Scale, DEFAULT_MAX_RANK, A>,
    tensor_scale: Option<f32>,
}

impl<'a, F: BlockScaledFormat, A: Allocator> ScaledTensorSpan<'a, F, A> {
    /// Mutably borrow the packed element values.
    pub fn elements(&mut self) -> &mut TensorSpan<'a, F::Element, DEFAULT_MAX_RANK, A> { &mut self.elements }

    /// Mutably borrow the per-block scale bytes.
    pub fn block_scales(&mut self) -> &mut TensorSpan<'a, F::Scale, DEFAULT_MAX_RANK, A> { &mut self.block_scales }

    /// The per-tensor multiplier (`Some` for NVFP4, `None` for the MX family).
    pub fn tensor_scale(&self) -> Option<f32> { self.tensor_scale }

    /// Logical shape of the packed elements.
    pub fn shape(&self) -> &[usize] { self.elements.shape() }

    /// Reborrow as a read-only [`ScaledTensorView`].
    pub fn as_view(&self) -> ScaledTensorView<'_, F, A> {
        ScaledTensorView {
            elements: self.elements.as_view(),
            block_scales: self.block_scales.as_view(),
            tensor_scale: self.tensor_scale,
        }
    }
}

// endregion: Block-Scaled Tensor

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cast::{CastOps, DenseToScaledOps},
        each::{AllCloseOps, ScaleOps, SumOps},
        reduce::MomentsOps,
        trigonometry::TrigSinOps,
        types::{bf16c, f16, f16c, f32c},
    };

    /// Property test: a materialized random slice of a random iota tensor holds exactly the source
    /// elements its strides address, exercising stride, offset, and slice math on random shapes.
    #[test]
    fn prop_slice_materializes_correct_elements() {
        // Self-contained xorshift64 PRNG — reproducible, no external dependency.
        let mut state: u64 = 0x00C0_FFEE_1234_5678;
        let mut below = |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as usize) % n
        };
        for _ in 0..256 {
            let rows = 1 + below(7);
            let columns = 1 + below(7);
            let data: Vec<f32> = (0..(rows * columns) as u32).map(|i| i as f32).collect();
            let t = Tensor::<f32>::from_slice(&data, &[rows, columns]).unwrap();

            let r0 = below(rows);
            let r1 = r0 + 1 + below(rows - r0);
            let c0 = below(columns);
            let c1 = c0 + 1 + below(columns - c0);

            let view = t
                .slice(&[SliceRange::range(r0, r1), SliceRange::range(c0, c1)])
                .unwrap();
            assert_eq!(view.shape(), [r1 - r0, c1 - c0]);

            let mut expected: Vec<f32> = Vec::new();
            for r in r0..r1 {
                for c in c0..c1 {
                    expected.push(data[r * columns + c]);
                }
            }
            assert_eq!(view.to_owned().unwrap().as_slice(), expected.as_slice());
        }
    }

    #[test]
    fn tensor_creation_from_factories() {
        let arr = Tensor::<f32>::full(&[3, 4], 1.0f32).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.numel(), 12);
        assert!(!arr.is_empty());
    }

    #[test]
    fn tensor_resize_capacity() {
        let mut t = Tensor::<f32>::zeros(&[8, 8]).unwrap();
        assert_eq!(t.capacity(), 64);
        let ptr = t.as_ptr();
        // Shrink within capacity: storage does not move.
        t.resize(&[4, 4]).unwrap();
        assert_eq!(t.shape(), &[4, 4]);
        assert_eq!(t.numel(), 16);
        assert_eq!(t.capacity(), 64);
        assert_eq!(t.as_ptr(), ptr, "resize must not move storage");
        // Beyond capacity fails, leaving the tensor unchanged.
        assert!(matches!(t.resize(&[9, 8]), Err(Error::CapacityExceeded { .. })));
        assert_eq!(t.shape(), &[4, 4]);
        // Reserve grows capacity and preserves the live contents.
        let mut u = Tensor::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        u.reserve(&[64]).unwrap();
        assert!(u.capacity() >= 64);
        assert_eq!(u.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        u.resize(&[64]).unwrap();
        assert_eq!(u.numel(), 64);
        // Clear keeps capacity.
        u.clear();
        assert!(u.is_empty());
        assert!(u.capacity() >= 64);
    }

    #[test]
    fn scaled_tensor_resize() {
        let data = vec![0.5f32; 2 * 32];
        let dense = Tensor::<f32>::from_slice(&data, &[2, 32]).unwrap();
        let mut scaled = dense.cast_to_scaled::<crate::cast::Nvfp4>().unwrap();
        assert_eq!(scaled.shape(), &[2, 32]);
        let cap = scaled.capacity();
        // Shrink within capacity — elements and scales move in lockstep, neither buffer relocates.
        scaled.resize(&[2, 16]).unwrap();
        assert_eq!(scaled.shape(), &[2, 16]);
        assert_eq!(scaled.block_scales().shape(), &[2, 1]);
        assert_eq!(scaled.capacity(), cap);
        // Beyond capacity fails atomically — both children unchanged.
        assert!(matches!(scaled.resize(&[4, 32]), Err(Error::CapacityExceeded { .. })));
        assert_eq!(scaled.shape(), &[2, 16]);
        // Reserve both children, then resize into the grown envelope.
        scaled.reserve(&[4, 32]).unwrap();
        scaled.resize(&[4, 32]).unwrap();
        assert_eq!(scaled.shape(), &[4, 32]);
        assert_eq!(scaled.block_scales().shape(), &[4, 2]);
        // Clear resets both children.
        scaled.clear();
        assert_eq!(scaled.shape(), &[0]);
    }

    #[test]
    fn tensor_from_slice() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.as_slice(), &data[..]);
    }

    #[test]
    fn transposed_view_to_owned_roundtrips() {
        // Materializing a non-contiguous (transposed) view must honor every stride, not assume the
        // inner axis is packed.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::from_slice(&data, &[3, 4]).unwrap();
        let transposed = arr.view().transpose().unwrap();
        assert_eq!(transposed.shape(), &[4, 3]);

        let owned = transposed.to_owned().unwrap();
        assert_eq!(owned.shape(), &[4, 3]);
        let got = owned.as_slice();
        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(
                    got[i * 3 + j],
                    data[j * 4 + i],
                    "transpose materialization at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn tensor_clone() {
        let arr = Tensor::<f32>::full(&[3, 4], 2.5f32).unwrap();
        let cloned = arr.clone().unwrap();
        assert_eq!(cloned.shape(), arr.shape());
        assert_eq!(cloned.as_slice(), arr.as_slice());
    }

    #[test]
    fn allocators_are_honored_by_reference_and_across_growth() {
        use core::{
            alloc::AllocError,
            sync::atomic::{AtomicUsize, Ordering},
        };

        // Deliberately not `Clone`, so it can only reach a container by reference — the case the
        // in-house trait could not express, because a blanket bridge forecloses `impl for &A`.
        struct Counting {
            live_bytes: AtomicUsize,
        }
        unsafe impl Allocator for Counting {
            fn allocate(&self, layout: core::alloc::Layout) -> Result<NonNull<[u8]>, AllocError> {
                self.live_bytes.fetch_add(layout.size(), Ordering::Relaxed);
                Global.allocate(layout)
            }
            unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: core::alloc::Layout) {
                self.live_bytes.fetch_sub(layout.size(), Ordering::Relaxed);
                unsafe { Global.deallocate(ptr, layout) };
            }
        }

        let arena = Counting {
            live_bytes: AtomicUsize::new(0),
        };
        {
            let mut tensor = Tensor::<f32, _>::from_slice_in(&[1.0, 2.0, 3.0, 4.0], &[4], &arena).unwrap();
            // Growth goes through `Allocator::grow`, which may move the block. The live elements
            // must survive it, and the AMX kernels cast packed bytes straight to a 64-byte-aligned
            // tile type, so the alignment has to survive it too.
            for extent in [64_usize, 4096, 100_000] {
                tensor.reserve(&[extent]).unwrap();
                assert!(tensor.capacity() >= extent);
                assert_eq!(tensor.as_ptr() as usize % SIMD_ALIGNMENT, 0);
                assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
            }
            assert!(arena.live_bytes.load(Ordering::Relaxed) >= 100_000 * 4);
        }
        // Every byte the container took came from this arena and went back to it.
        assert_eq!(arena.live_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn operations_on_views_allocate_through_the_owners_allocator() {
        use core::{
            alloc::AllocError,
            sync::atomic::{AtomicUsize, Ordering},
        };

        struct Counting {
            allocations: AtomicUsize,
        }
        unsafe impl Allocator for Counting {
            fn allocate(&self, layout: core::alloc::Layout) -> Result<NonNull<[u8]>, AllocError> {
                self.allocations.fetch_add(1, Ordering::Relaxed);
                Global.allocate(layout)
            }
            unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: core::alloc::Layout) {
                unsafe { Global.deallocate(ptr, layout) };
            }
        }

        let arena = Counting {
            allocations: AtomicUsize::new(0),
        };
        let owner = Tensor::<f32, _>::from_slice_in(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], &arena).unwrap();
        let row = owner.slice((1_usize, ..)).unwrap();
        let before = arena.allocations.load(Ordering::Relaxed);
        let sines = row.sin().unwrap();
        let widened = row.cast::<f64>().unwrap();
        let sums = owner.view().sum_axis(1_usize, false).unwrap();
        let gram = owner.view().dots_symmetric().unwrap();
        assert!(core::ptr::eq(*sines.allocator(), &arena));
        assert!(core::ptr::eq(*widened.allocator(), &arena));
        assert!(core::ptr::eq(*sums.allocator(), &arena));
        assert!(core::ptr::eq(*gram.allocator(), &arena));
        // `sum_axis` also allocates the sums of squares it computes alongside the sums.
        assert_eq!(arena.allocations.load(Ordering::Relaxed), before + 5);

        // A view over static data borrows no tensor, so its results come from `Global`.
        static DATA: [f32; 3] = [0.0, 1.0, 2.0];
        let foreign = unsafe { TensorView::<f32, 1>::from_raw_parts(DATA.as_ptr(), &[3], &[4]) }.unwrap();
        let _: &Global = foreign.allocator();
        let _: Tensor<f32, Global, 1> = foreign.sin().unwrap();
        assert_eq!(arena.allocations.load(Ordering::Relaxed), before + 5);
    }

    #[test]
    fn uniform_fills_pick_the_cheap_path() {
        // An all-zero fill is served by `allocate_zeroed`, a repeating-byte fill by one `memset`,
        // and anything else by the typed loop.
        assert_eq!(repeating_byte(&0.0_f32), Some(0));
        assert_eq!(repeating_byte(&(-1_i8)), Some(0xFF));
        assert_eq!(repeating_byte(&1.5_f32), None);
        assert!(Tensor::<i8>::full(&[3, 4], -1)
            .unwrap()
            .as_slice()
            .iter()
            .all(|&v| v == -1));
    }

    #[test]
    fn borrowed_views_are_shareable_across_threads() {
        // A view is a shared borrow and a span a unique one, so each is exactly as shareable as the
        // borrow it stands for. `ScaledTensor`'s views hold only these plus an `Option<f32>`, so
        // they pick both up by auto-derivation.
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<TensorView<'_, f32>>();
        assert_sync::<TensorView<'_, f32>>();
        assert_send::<TensorSpan<'_, f32>>();
        assert_sync::<TensorSpan<'_, f32>>();
        assert_send::<Matrix<f32>>();
        assert_send::<MatrixView<'_, f32>>();
        assert_sync::<MatrixView<'_, f32>>();
        assert_send::<MatrixSpan<'_, f32>>();
        assert_sync::<ScaledTensorView<'_, crate::cast::Mxfp4>>();
        assert_send::<ScaledTensorSpan<'_, crate::cast::Mxfp4>>();
    }

    #[test]
    fn sub_byte_indexing_stays_inside_storage() {
        use crate::types::{i4x2, u1x8, u4x2, DimLocation};

        // 64 logical bits live in 8 bytes. The bounds check counts dimensions, the offset counts
        // storage values, and locating the dimension is what keeps the two in the same units —
        // without it index 63 addressed byte 63 of an 8-byte allocation.
        let packed = Tensor::<u1x8>::zeros(&[64]).unwrap();
        let base = packed.as_ptr() as usize;
        for index in 0..64_usize {
            let slot = packed.flat(index).unwrap() as *const u1x8 as usize;
            assert!(slot - base < 8, "index {index} addressed byte {} of 8", slot - base);
        }
        assert!(packed.flat(64_usize).is_err());

        // Eight consecutive dimensions share one byte, and the nibble types pair up.
        assert_eq!(
            u1x8::locate_dim(63),
            DimLocation {
                value_index: 7,
                sub_index: 7
            }
        );
        assert_eq!(
            i4x2::locate_dim(5),
            DimLocation {
                value_index: 2,
                sub_index: 1
            }
        );
        assert_eq!(
            f32::locate_dim(5),
            DimLocation {
                value_index: 5,
                sub_index: 0
            }
        );
        assert_eq!(u1x8::dimensions_to_values(16), 2);
        assert_eq!(f32::dimensions_to_values(9), 9);

        // `row` returns storage values, so its range must be in those units too.
        let nibbles = Tensor::<u4x2>::zeros(&[2, 4]).unwrap();
        assert_eq!(nibbles.row(0).unwrap().len(), 2);
        assert_eq!(nibbles.row(1).unwrap().len(), 2);

        // Taking the innermost axis whole is representable; narrowing it is not, and used to hand
        // back a view whose data pointer already sat outside the allocation.
        let leading_only = [SliceRange::range(0, 1), SliceRange::Full];
        assert!(nibbles.view().slice(&leading_only[..]).is_ok());
        let narrows_innermost = [SliceRange::Full, SliceRange::range(0, 2)];
        assert_eq!(
            nibbles.view().slice(&narrows_innermost[..]).unwrap_err(),
            Error::SubByteUnsupported
        );
    }

    #[test]
    fn mismatched_parts_and_ranks_are_errors() {
        use crate::cast::Mxfp4;

        // Decoding derives the scale count from the elements shape alone, so a short scale tensor
        // would be read past its end by the kernel.
        let dense = Tensor::<f32>::zeros(&[1, 64]).unwrap();
        let scaled = dense.view().cast_to_scaled::<Mxfp4>().unwrap();
        let (elements, scales) = (scaled.elements(), scaled.block_scales());
        assert_eq!(elements.shape(), &[1, 64]);
        assert_eq!(scales.shape(), &[1, 2]);
        let too_few = Tensor::<crate::types::Ue8m0>::zeros(&[1, 1]).unwrap();
        let elements_owned = Tensor::<crate::types::e2m1x2>::zeros(&[1, 64]).unwrap();
        assert!(matches!(
            ScaledTensor::<Mxfp4>::from_parts(elements_owned, too_few, None).unwrap_err(),
            Error::ShapeMismatch { .. }
        ));

        // A rank past `DEFAULT_MAX_RANK` overran the fixed scratch buffer instead of reporting.
        let deep = [1usize, 1, 1, 1, 1, 1, 1, 1, 32];
        assert_eq!(
            ScaledTensor::<Mxfp4>::scales_shape_into(&deep, &mut [0usize; DEFAULT_MAX_RANK]).unwrap_err(),
            Error::TooManyRanks { got: 9 }
        );
    }

    #[test]
    fn allocation_size_overflow_is_an_error() {
        // A quarter of the address space in `f32` slots is its whole size in bytes, which wraps to
        // zero, so an unchecked multiply produced a zero-sized layout, a dangling pointer, and a
        // constructor that then wrote that many elements through it.
        assert_eq!(
            Tensor::<f32>::full(&[1_usize << (usize::BITS - 2), 1], 1.0).unwrap_err(),
            Error::AllocationFailed
        );
        assert_eq!(Vector::<f32>::zeros(usize::MAX).unwrap_err(), Error::AllocationFailed);
        // The shape product wraps before any byte count is involved, which would leave the stored
        // shape and the derived element count disagreeing.
        assert_eq!(
            Tensor::<u8>::zeros(&[1_usize << (usize::BITS / 2 + 1), 1_usize << (usize::BITS / 2 + 1), 4]).unwrap_err(),
            Error::AllocationFailed
        );
    }

    #[test]
    fn tensor_error_display() {
        let err = Error::AllocationFailed;
        assert_eq!(format!("{}", err), "memory allocation failed");
        let err = Error::TooManyRanks { got: 10 };
        assert_eq!(format!("{}", err), "too many ranks: 10");
    }

    #[test]
    fn tensor_row_access() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.row(0), Some(&[0.0, 1.0, 2.0, 3.0][..]));
        assert_eq!(arr.row(1), Some(&[4.0, 5.0, 6.0, 7.0][..]));
        assert_eq!(arr.row(2), Some(&[8.0, 9.0, 10.0, 11.0][..]));
        assert_eq!(arr.row(3), None);
    }

    #[test]
    fn tensor_slicing() {
        let arr = Tensor::<f32>::full(&[4, 5], 1.0f32).unwrap();
        let view = arr.slice(&[SliceRange::full(), SliceRange::full()]).unwrap();
        assert_eq!(view.shape(), &[4, 5]);
        let view = arr.slice(&[SliceRange::range(1, 3), SliceRange::full()]).unwrap();
        assert_eq!(view.shape(), &[2, 5]);
        let view = arr.slice(&[SliceRange::index(0), SliceRange::full()]).unwrap();
        assert_eq!(view.shape(), &[5]);
        assert_eq!(view.ndim(), 1);
    }

    #[test]
    fn tensor_transpose_2d() {
        let arr = Tensor::<f32>::full(&[3, 4], 1.0f32).unwrap();
        let transposed = arr.transpose().unwrap();
        assert_eq!(transposed.shape(), &[4, 3]);
    }

    #[test]
    fn tensor_is_contiguous() {
        let arr = Tensor::<f32>::full(&[3, 4], 1.0f32).unwrap();
        let view = arr.view();
        assert!(view.is_contiguous());
        assert!(arr.has_contiguous_rows());
    }

    #[test]
    fn matrix_alias_round_trip() {
        let mat: Matrix<f32> = Matrix::full(&[3, 4], 1.0f32).unwrap();
        assert_eq!(mat.shape(), &[3, 4]);
    }

    #[test]
    fn tensor_scalar_lookup_and_views() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut tensor = Tensor::<f32>::from_slice(&data, &[3, 4]).unwrap();

        assert_eq!(*tensor.flat(0_usize).unwrap(), 0.0);
        assert_eq!(*tensor.flat(-1_i32).unwrap(), 11.0);
        assert_eq!(*tensor.coords((1_usize, 2_usize)).unwrap(), 6.0);
        assert_eq!(*tensor.coords((2_i32, -1_i32)).unwrap(), 11.0);
        assert_eq!(*tensor.flat(5_usize).unwrap(), 5.0);
        assert_eq!(*tensor.coords((1_usize, 3_usize)).unwrap(), 7.0);
        assert!(tensor.flat(12_usize).is_err());
        assert!(tensor.coords((3_usize, 0_usize)).is_err());

        *tensor.coords_mut((1_usize, 2_usize)).unwrap() = 60.0;
        assert_eq!(*tensor.coords((1_usize, 2_usize)).unwrap(), 60.0);
        *tensor.coords_mut((2_usize, 0_usize)).unwrap() = 80.0;
        assert_eq!(*tensor.coords((2_usize, 0_usize)).unwrap(), 80.0);

        let view = tensor.view();
        assert_eq!(*view.flat(1_usize).unwrap(), 1.0);
        assert_eq!(*view.coords((1_usize, 2_usize)).unwrap(), 60.0);

        let mut span = tensor.span();
        assert_eq!(*span.coords((2_usize, 0_usize)).unwrap(), 80.0);
        *span.coords_mut((0_usize, 1_usize)).unwrap() = 10.0;
        assert_eq!(*span.coords((0_usize, 1_usize)).unwrap(), 10.0);
    }

    #[test]
    fn tensor_noncontiguous_lookup_and_rank_zero() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = Tensor::<f32>::from_slice(&data, &[3, 4]).unwrap();
        let even_columns = tensor
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        assert_eq!(even_columns.shape(), &[3, 2]);
        assert_eq!(*even_columns.flat(0_usize).unwrap(), 0.0);
        assert_eq!(*even_columns.flat(1_usize).unwrap(), 2.0);
        assert_eq!(*even_columns.flat(2_usize).unwrap(), 4.0);
        assert_eq!(*even_columns.coords((1_usize, 1_usize)).unwrap(), 6.0);
        assert_eq!(*even_columns.coords((2_usize, 1_usize)).unwrap(), 10.0);

        let row = tensor.slice_leading(1_usize).unwrap();
        assert_eq!(row.shape(), &[4]);
        assert_eq!(*row.flat(-1_i32).unwrap(), 7.0);

        let scalar = tensor.slice(&[SliceRange::index(2), SliceRange::index(3)]).unwrap();
        assert_eq!(scalar.ndim(), 0);
        assert_eq!(*scalar.scalar().unwrap(), 11.0);
    }

    #[test]
    fn tensor_ops() {
        // Reshape
        let arr = Tensor::<f32>::full(&[3, 4], 1.0f32).unwrap();
        let reshaped = arr.reshape(&[2, 6]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 6]);
        assert_eq!(reshaped.numel(), 12);

        // Sum f32
        let arr = Tensor::<f32>::full(&[100], 1.0f32).unwrap();
        let sum = arr.sum_all().unwrap();
        assert!((sum - 100.0).abs() < 0.001);

        // Sum f64
        let arr = Tensor::<f64>::full(&[100], 1.0f64).unwrap();
        let sum = arr.sum_all().unwrap();
        assert!((sum - 100.0).abs() < 1e-9);

        // Rank-0 (scalar) sum returns the single element; the reduce path must feed the kernel a
        // real element stride, never zero — a zero stride hangs the SIMD strided moments kernel.
        let scalar = Tensor::<f32>::full(&[], 3.5f32).unwrap();
        assert_eq!(scalar.numel(), 1);
        assert!((scalar.sum_all().unwrap() - 3.5).abs() < 1e-6);

        // A zero-sized dimension is rejected at construction: NumKong's Rust tensors are non-empty
        // by contract, unlike the Python/C++ bindings, which allow zero-element shapes.
        assert!(Tensor::<f32>::zeros(&[0]).is_err());
    }

    #[test]
    fn nd_contraction_global_reductions() {
        // Verify that N-D contiguous tensors produce the same moments/minmax as a flat copy.
        // This tests the uniform-stride-tail collapsing and recursive re-analysis paths.
        let data: Vec<f32> = (0..720).map(|i| (i as f32) * 0.1 - 36.0).collect();

        // 4-D: shape [2, 3, 4, 30] = 720 elements
        let t4d = Tensor::<f32>::from_slice(&data, &[2, 3, 4, 30]).unwrap();
        let t1d = Tensor::<f32>::from_slice(&data, &[720]).unwrap();
        let (sum_4d, sumsq_4d) = t4d.view().moments_all().unwrap();
        let (sum_1d, sumsq_1d) = t1d.view().moments_all().unwrap();
        assert!((sum_4d - sum_1d).abs() < 1e-3, "4D vs 1D sum: {sum_4d} != {sum_1d}");
        assert!(
            (sumsq_4d - sumsq_1d).abs() < 1e-1,
            "4D vs 1D sumsq: {sumsq_4d} != {sumsq_1d}"
        );
        let mm_4d = t4d.view().minmax_all().unwrap();
        let mm_1d = t1d.view().minmax_all().unwrap();
        assert_eq!(mm_4d.min_value, mm_1d.min_value);
        assert_eq!(mm_4d.max_value, mm_1d.max_value);
        assert_eq!(mm_4d.min_index, mm_1d.min_index);
        assert_eq!(mm_4d.max_index, mm_1d.max_index);

        // 5-D: shape [2, 3, 4, 5, 6] = 720 elements
        let t5d = Tensor::<f32>::from_slice(&data, &[2, 3, 4, 5, 6]).unwrap();
        let (sum_5d, sumsq_5d) = t5d.view().moments_all().unwrap();
        assert!((sum_5d - sum_1d).abs() < 1e-3, "5D vs 1D sum");
        assert!((sumsq_5d - sumsq_1d).abs() < 1e-1, "5D vs 1D sumsq");
    }

    #[test]
    fn nd_contraction_strided_views() {
        // Verify reductions on non-contiguous uniform-stride subviews.
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t = Tensor::<f32>::from_slice(&data, &[4, 4, 3]).unwrap();

        // Channel subview: [:, :, 1] → shape [4, 4], stride = 3 * sizeof(f32)
        let channel = t
            .slice(&[SliceRange::full(), SliceRange::full(), SliceRange::index(1)])
            .unwrap();
        assert_eq!(channel.shape(), &[4, 4]);
        let (ch_sum, _) = channel.moments_all().unwrap();
        let expected: f64 = (0..16).map(|i| (i * 3 + 1) as f64).sum();
        assert!(
            (ch_sum - expected).abs() < 1e-3,
            "channel subview sum: {ch_sum} != {expected}"
        );

        // Row skip: [::2, :, :] → shape [2, 4, 3], outer stride doubled
        let skipped = t
            .slice(&[SliceRange::range_step(0, 4, 2), SliceRange::full(), SliceRange::full()])
            .unwrap();
        assert_eq!(skipped.shape(), &[2, 4, 3]);
        let (skip_sum, _) = skipped.moments_all().unwrap();
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
        let t = Tensor::<f32>::from_slice(&data, &[2, 3, 4]).unwrap();

        // axis=0: sum over batch → shape [3, 4]
        let (sums_a0, _) = t.view().moments_axis(0, false).unwrap();
        assert_eq!(sums_a0.shape(), &[3, 4]);
        // [0..4] + [12..16] = [12, 14, 16, 18]
        assert!((sums_a0.as_slice()[0] - 12.0).abs() < 1e-6);
        assert!((sums_a0.as_slice()[1] - 14.0).abs() < 1e-6);

        // axis=-1: sum over innermost → shape [2, 3]
        let (sums_last, _) = t.view().moments_axis(-1_i32, false).unwrap();
        assert_eq!(sums_last.shape(), &[2, 3]);
        // First row: 0+1+2+3 = 6
        assert!((sums_last.as_slice()[0] - 6.0).abs() < 1e-6);

        // axis=1: sum over middle → shape [2, 4]
        let (sums_mid, _) = t.view().moments_axis(1, false).unwrap();
        assert_eq!(sums_mid.shape(), &[2, 4]);
        // Column 0: 0+4+8 = 12
        assert!((sums_mid.as_slice()[0] - 12.0).abs() < 1e-6);

        // Axis reduction on strided subview: [::2, :, :] on [4, 3, 4]
        let data48: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t48 = Tensor::<f32>::from_slice(&data48, &[4, 3, 4]).unwrap();
        let skipped = t48
            .slice(&[SliceRange::range_step(0, 4, 2), SliceRange::full(), SliceRange::full()])
            .unwrap();
        assert_eq!(skipped.shape(), &[2, 3, 4]);
        let (sums_skip_last, _) = skipped.moments_axis(-1_i32, false).unwrap();
        assert_eq!(sums_skip_last.shape(), &[2, 3]);
        // First lane: 0+1+2+3 = 6
        assert!((sums_skip_last.as_slice()[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn nd_contraction_singleton_dims() {
        // Singleton dimensions (extent=1) should collapse freely.
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let t = Tensor::<f32>::from_slice(&data, &[1, 64, 1]).unwrap();
        let t_flat = Tensor::<f32>::from_slice(&data, &[64]).unwrap();
        let (sum_nd, sumsq_nd) = t.view().moments_all().unwrap();
        let (sum_flat, sumsq_flat) = t_flat.view().moments_all().unwrap();
        assert_eq!(sum_nd, sum_flat);
        assert_eq!(sumsq_nd, sumsq_flat);

        let mm_nd = t.view().minmax_all().unwrap();
        let mm_flat = t_flat.view().minmax_all().unwrap();
        assert_eq!(mm_nd.min_index, mm_flat.min_index);
        assert_eq!(mm_nd.max_index, mm_flat.max_index);
    }

    #[test]
    fn complex_elementwise_view_and_owner_paths() {
        let left_values = [f32c { re: 1.0, im: 2.0 }, f32c { re: 3.0, im: 4.0 }];
        let right_values = [f32c { re: 5.0, im: 6.0 }, f32c { re: 7.0, im: 8.0 }];
        let zeros = Tensor::<f32c>::full(&[2], f32c { re: 0.0, im: 0.0 }).unwrap();
        let left = Tensor::<f32c>::from_slice(&left_values, &[2]).unwrap();
        let right = Tensor::<f32c>::from_slice(&right_values, &[2]).unwrap();

        let added = left.add_tensor(&right).unwrap();
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
            .blend_tensor(&right.view(), f32c { re: 1.0, im: 0.0 }, f32c { re: -1.0, im: 0.0 })
            .unwrap();
        assert_eq!(
            blended.as_slice(),
            &[f32c { re: -4.0, im: -4.0 }, f32c { re: -4.0, im: -4.0 }]
        );

        let fma = left
            .view()
            .fma_tensors(
                &right.view(),
                &zeros.view(),
                f32c { re: 1.0, im: 0.0 },
                f32c { re: 0.0, im: 0.0 },
            )
            .unwrap();
        assert_eq!(
            fma.as_slice(),
            &[f32c { re: -7.0, im: 16.0 }, f32c { re: -11.0, im: 52.0 }]
        );

        let mut inplace = Tensor::<f32c>::from_slice(&left_values, &[2]).unwrap();
        inplace.add_tensor_inplace(&right).unwrap();
        assert_eq!(inplace.as_slice(), added.as_slice());

        let widened = left.cast::<bf16c>().unwrap();
        assert_eq!(widened.as_slice()[0].re.to_f32(), 1.0);
        assert_eq!(widened.as_slice()[0].im.to_f32(), 2.0);

        let strided = Tensor::<f16c>::from_slice(
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
        let complex_column = strided.slice(&[SliceRange::full(), SliceRange::range(0, 1)]).unwrap();
        let mut out = Tensor::<f16c>::full(
            &[2, 1],
            f16c {
                re: f16::ZERO,
                im: f16::ZERO,
            },
        )
        .unwrap();
        complex_column
            .scale_tensor_into(
                f16c {
                    re: f16::ONE,
                    im: f16::ZERO,
                },
                f16c {
                    re: f16::ZERO,
                    im: f16::ONE,
                },
                &mut out,
            )
            .unwrap();
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
        fn shape_of<Scalar: StorageElement, const R: usize>(t: &impl TensorRef<Scalar, R>) -> Vec<usize> {
            t.shape().to_vec()
        }
        let t = Tensor::<f32>::full(&[3, 4], 1.0).unwrap();
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
        crate::capabilities::configure_cpu_thread().unwrap();
        let t = Tensor::<f32>::full(&[3, 4], 2.0).unwrap();
        let v = t.view();

        // ScaleOps works on both Tensor and TensorView
        let r1 = ScaleOps::add_scalar(&t, 1.0).unwrap();
        let r2 = ScaleOps::add_scalar(&v, 1.0).unwrap();
        assert_eq!(r1.as_slice(), r2.as_slice());
        assert!((r1.as_slice()[0] - 3.0).abs() < 0.01);

        // SumOps works on Tensor with TensorView as other
        let other = Tensor::<f32>::full(&[3, 4], 1.0).unwrap();
        let r3 = SumOps::add_tensor(&t, &other).unwrap();
        assert!((r3.as_slice()[0] - 3.0).abs() < 0.01);

        // TrigSinOps works on both
        let small = Tensor::<f32>::full(&[4], 0.0).unwrap();
        let r4 = TrigSinOps::sin(&small).unwrap();
        assert!((r4.as_slice()[0] - 0.0).abs() < 0.01);

        // MomentsOps works on both
        let sum_t = MomentsOps::sum_all(&t).unwrap();
        let sum_v = MomentsOps::sum_all(&v).unwrap();
        assert!((sum_t as f32 - 24.0).abs() < 0.01);
        assert!((sum_v as f32 - 24.0).abs() < 0.01);
    }

    #[test]
    fn tensor_allclose_matching() {
        let a = Tensor::<f32>::full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::full(&[2, 3], 1.0 + 1e-7).unwrap();
        assert!(a.allclose(&b, 1e-6, 0.0));
    }

    #[test]
    fn tensor_allclose_mismatching() {
        let a = Tensor::<f32>::full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::full(&[2, 3], 2.0).unwrap();
        assert!(!a.allclose(&b, 1e-6, 0.0));
    }

    #[test]
    fn tensor_allclose_different_shapes() {
        let a = Tensor::<f32>::full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::full(&[3, 2], 1.0).unwrap();
        assert!(!a.allclose(&b, 1e-6, 1e-6));
    }

    #[test]
    fn tensor_view_allclose() {
        let a = Tensor::<f32>::full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::full(&[2, 3], 1.0 + 1e-7).unwrap();
        assert!(a.view().allclose(&b.view(), 1e-6, 0.0));
    }

    #[test]
    fn tensor_view_iter_logical_scalars() {
        use crate::types::i4x2;

        // Shape [6] logical → 3 i4x2 storage values
        let mut t = Tensor::<i4x2>::zeros(&[6]).unwrap();
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
        let mut t = Tensor::<f32>::full(&[2, 3], 1.0).unwrap();
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

        let mut t = Tensor::<i4x2>::zeros(&[6]).unwrap();
        for (pos, mut value) in &mut t {
            *value = pos[0] as i8;
        }
        let vals: Vec<i8> = t.iter().map(|(_, v)| *v).collect();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn tensor_iterator_alias_compat() {
        let t = Tensor::<f32>::full(&[2], 1.0).unwrap();
        let _it: TensorIterator<'_, f32> = t.iter();
    }

    #[test]
    fn tensor_tuple_slice_syntax() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::<f32>::from_slice(&data, &[3, 4]).unwrap();

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
        let v = t.slice(&[SliceRange::full(), SliceRange::index(0)]).unwrap();
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
        let mut t = Tensor::<f32>::from_slice(&data, &[3, 4]).unwrap();
        let _s = t.slice_mut((.., 0_usize)).unwrap();
    }

    #[test]
    fn tensor_try_from_scalars_and_dims() {
        // Full-byte type: f32
        let scalars: Vec<f32> = (0..6).map(|i| i as f32 + 0.5).collect();
        let tensor = Tensor::<f32>::from_scalars(&scalars, &[2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.as_slice(), scalars.as_slice());

        // Length mismatch is reported as a shape error, not a panic.
        let bad = Tensor::<f32>::from_scalars(&scalars, &[2, 2]);
        assert!(matches!(bad, Err(Error::ShapeMismatch { .. })));

        // from_dims on a full-byte type: DimScalar = Scalar for f32.
        let dims: Vec<f32> = scalars.clone();
        let tensor = Tensor::<f32>::from_dims(&dims, &[3, 2]).unwrap();
        assert_eq!(tensor.shape(), &[3, 2]);
        assert_eq!(tensor.as_slice(), dims.as_slice());

        // Sub-byte round-trip: i4x2 packs 2 dims/byte.
        let i4_dims: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, 0];
        let tensor = Tensor::<crate::types::i4x2>::from_dims(&i4_dims, &[2, 4]).unwrap();
        assert_eq!(tensor.shape(), &[2, 4]);
        assert_eq!(tensor.storage_len(), 4);
    }

    #[test]
    fn fill_zeros_and_fill_on_tensor_and_span() {
        // Tensor: fill with zero, then fill with 7.0, verify storage matches.
        let mut tensor = unsafe { Tensor::<f32>::uninitialized(&[3, 4]).unwrap() };
        tensor.fill_zeros();
        assert!(tensor.as_slice().iter().all(|&value| value == 0.0));
        tensor.fill(7.0);
        assert!(tensor.as_slice().iter().all(|&value| value == 7.0));

        // TensorSpan via tensor.span() also exposes Fill methods.
        let mut other = unsafe { Tensor::<f32>::uninitialized(&[3, 4]).unwrap() };
        {
            let mut span = other.span();
            span.fill_zeros();
            span.fill(2.5);
        }
        assert!(other.as_slice().iter().all(|&value| value == 2.5));
    }

    #[test]
    fn copy_from_via_trait_round_trips_through_span() {
        let source_tensor = Tensor::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mut destination_tensor = unsafe { Tensor::<f32>::uninitialized(&[2, 2]).unwrap() };
        // Tensor: CopyFrom<&[Scalar]>.
        destination_tensor.copy_from(source_tensor.as_slice()).unwrap();
        assert_eq!(destination_tensor.as_slice(), source_tensor.as_slice());

        // TensorSpan: CopyFrom<&TensorView>.
        let mut span_destination = unsafe { Tensor::<f32>::uninitialized(&[2, 2]).unwrap() };
        {
            let mut span = span_destination.span();
            let view = source_tensor.view();
            span.copy_from(&view).unwrap();
        }
        assert_eq!(span_destination.as_slice(), source_tensor.as_slice());
    }
}

// endregion: Tests
