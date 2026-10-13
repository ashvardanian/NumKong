//! Many-to-many shape distances between 3D point clouds of any size.
//!
//! [`ShapesPackedMatrix`] packs every cloud on its own: centered on its centroid and rotated onto
//! its principal axes, so a pack is the same on every machine. Its methods then score every packed
//! cloud of one set, the rows, against every packed cloud of another, the columns:
//!
//! - [`ShapesPackedMatrix::chamfers`]: the mean nearest-partner distance both ways, halved
//! - [`ShapesPackedMatrix::hausdorffs`]: the worst nearest-partner distance either way
//! - [`ShapesPackedMatrix::wassersteinprocrustes`]: the exact transport RMSD after the best
//!   rotation or reflection, on equal sizes at most the Kabsch RMSD of a known correspondence, and
//!   equal to it when that correspondence is an optimal plan
//! - [`ShapesPackedMatrix::gromovwassersteins`]: the transport distance between the clouds' own
//!   distance matrices
//!
//! None of them needs a correspondence between points, and the two sides may differ in size. Every
//! kind is invariant to rotation, reflection and translation, and between packs made by
//! [`ShapesPackedMatrix::new_normalized`], which divide each cloud by its RMS distance from the
//! centroid, to scale as well. Distances are `f32` for `f16` clouds and `f64` for `f32` ones, and
//! the optional rotations are row-major `[f32; 9]` matrices taking each column cloud onto each row
//! cloud through the centroids and scales: `aᵢ ≈ (scale_a / scale_b) × R × (bⱼ − b̄) + ā`.
//!
//! The pack layout is private: the accessors report each cloud's pose and bounds, and
//! [`ShapesPackedMatrix::unpack`] writes a cloud back in its original pose and scale.
//!
//! # Example
//!
//! ```rust,no_run
//! use numkong::ShapesPackedMatrix;
//!
//! let tetrahedron = [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
//! let triangle = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
//!
//! let rows = ShapesPackedMatrix::new(&[&tetrahedron[..]]).unwrap();
//! let columns = ShapesPackedMatrix::new(&[&tetrahedron[..], &triangle[..]]).unwrap();
//! let mut distances = [0.0_f64; 2];
//! rows.wassersteinprocrustes(&columns, &mut distances, None).unwrap();
//! assert!(distances[0] < 1e-6);
//! ```
//!
//! File: rust/shapes.rs
//! Author: Ash Vardanian

use core::{alloc::Layout, ffi::c_void, marker::PhantomData, ptr::null_mut};

use crate::{
    capabilities::{nk_capability_t, nk_size_t, nk_status_t, Capabilities, StatusCode},
    tensor::{check_len, Allocator, Error, Global, PackedBuffer},
    types::{f16, StorageElement},
};

// region: FFI

#[link(name = "numkong")]
extern "C" {
    fn nk_shapes_pack_size_f16_best(
        points: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_shapes_pack_size_f32_best(
        points: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_shapes_pack_f16_best(
        points: *const f16,
        count: nk_size_t,
        packed: *mut u8,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_shapes_pack_f32_best(
        points: *const f32,
        count: nk_size_t,
        packed: *mut u8,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_shapes_pack_normalized_f16_best(
        points: *const f16,
        count: nk_size_t,
        packed: *mut u8,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_shapes_pack_normalized_f32_best(
        points: *const f32,
        count: nk_size_t,
        packed: *mut u8,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_shapes_packed_shape_f16_best(
        packed: *const u8,
        points: *mut nk_size_t,
        centroid: *mut f32,
        frame: *mut f32,
        scale: *mut f32,
        spreads: *mut f32,
        max_radius: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_shapes_packed_shape_f32_best(
        packed: *const u8,
        points: *mut nk_size_t,
        centroid: *mut f64,
        frame: *mut f64,
        scale: *mut f64,
        spreads: *mut f64,
        max_radius: *mut f64,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_shapes_unpack_f16_best(
        packed: *const u8,
        points: *mut f16,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_shapes_unpack_f32_best(
        packed: *const u8,
        points: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_chamfers_packed_f16_best(
        a_packed: *const *const u8,
        rows: nk_size_t,
        b_packed: *const *const u8,
        columns: nk_size_t,
        results: *mut f32,
        results_stride: nk_size_t,
        rotations: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_chamfers_packed_f32_best(
        a_packed: *const *const u8,
        rows: nk_size_t,
        b_packed: *const *const u8,
        columns: nk_size_t,
        results: *mut f64,
        results_stride: nk_size_t,
        rotations: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_hausdorffs_packed_f16_best(
        a_packed: *const *const u8,
        rows: nk_size_t,
        b_packed: *const *const u8,
        columns: nk_size_t,
        results: *mut f32,
        results_stride: nk_size_t,
        rotations: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_hausdorffs_packed_f32_best(
        a_packed: *const *const u8,
        rows: nk_size_t,
        b_packed: *const *const u8,
        columns: nk_size_t,
        results: *mut f64,
        results_stride: nk_size_t,
        rotations: *mut f32,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_wassersteinprocrustes_workspace_size_f16_best(
        row_points: nk_size_t,
        column_points: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_wassersteinprocrustes_workspace_size_f32_best(
        row_points: nk_size_t,
        column_points: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_gromovwassersteins_workspace_size_f16_best(
        row_points: nk_size_t,
        column_points: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_gromovwassersteins_workspace_size_f32_best(
        row_points: nk_size_t,
        column_points: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;

    fn nk_wassersteinprocrustes_packed_f16_best(
        a_packed: *const *const u8,
        rows: nk_size_t,
        b_packed: *const *const u8,
        columns: nk_size_t,
        results: *mut f32,
        results_stride: nk_size_t,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_wassersteinprocrustes_packed_f32_best(
        a_packed: *const *const u8,
        rows: nk_size_t,
        b_packed: *const *const u8,
        columns: nk_size_t,
        results: *mut f64,
        results_stride: nk_size_t,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_gromovwassersteins_packed_f16_best(
        a_packed: *const *const u8,
        rows: nk_size_t,
        b_packed: *const *const u8,
        columns: nk_size_t,
        results: *mut f32,
        results_stride: nk_size_t,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_gromovwassersteins_packed_f32_best(
        a_packed: *const *const u8,
        rows: nk_size_t,
        b_packed: *const *const u8,
        columns: nk_size_t,
        results: *mut f64,
        results_stride: nk_size_t,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
}

// endregion: FFI

// region: Shapes trait

/// Byte boundary every pack and every transport workspace starts on, the C library's
/// `NUMKONG_DEFAULT_ALIGNMENT` for the same target.
#[cfg(target_arch = "s390x")]
const DEFAULT_ALIGNMENT: usize = 256;
#[cfg(target_family = "wasm")]
const DEFAULT_ALIGNMENT: usize = 64;
#[cfg(not(any(target_arch = "s390x", target_family = "wasm")))]
const DEFAULT_ALIGNMENT: usize = 128;

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for super::f16 {}
}

/// Cloud packing and the many-to-many shape kernels, for `f16` and `f32` clouds.
///
/// # Errors
///
/// [`Error::KernelFailed`] when the kernel refuses to run, like on a misaligned workspace.
pub trait Shapes: private::Sealed + StorageElement {
    /// Distance type and pose type: `f32` for `f16` clouds, `f64` for `f32` ones.
    type Distance: Copy + Default;

    /// Returns the bytes a pack of `points` points takes.
    fn shapes_pack_size(points: usize) -> Result<usize, Error>;

    /// Packs one cloud: centers it and rotates it onto its principal axes.
    ///
    /// # Safety
    /// `packed` must have `shapes_pack_size(points.len())` writable bytes on an
    /// `nk_default_alignment_k` boundary.
    unsafe fn shapes_pack(points: &[[Self; 3]], packed: *mut u8) -> Result<(), Error>;

    /// Packs one cloud like [`Shapes::shapes_pack`], also dividing it by its RMS distance from the
    /// centroid, which the pack keeps as its scale, or by 1 when the cloud has no spread.
    ///
    /// # Safety
    /// The contract of [`Shapes::shapes_pack`].
    unsafe fn shapes_pack_normalized(points: &[[Self; 3]], packed: *mut u8) -> Result<(), Error>;

    /// Reads the pose and bounds of a pack into every non-null output: its number of points, its
    /// mean input point, the unit quaternion (x, y, z, w) rotating the centered cloud onto its
    /// principal axes, what the rotated cloud was divided by, the descending eigenvalues of the
    /// packed points' second moments, and the largest packed point norm.
    ///
    /// # Safety
    /// `packed` must point to a pack of `Self`, and the non-null outputs must hold 1, 3, 4, 1, 3
    /// and 1 writable values in that order.
    unsafe fn shapes_packed_shape(
        packed: *const u8,
        points: *mut usize,
        centroid: *mut Self::Distance,
        frame: *mut Self::Distance,
        scale: *mut Self::Distance,
        spreads: *mut Self::Distance,
        max_radius: *mut Self::Distance,
    ) -> Result<(), Error>;

    /// Writes the cloud of a pack back interleaved, in its original pose and scale, within one
    /// rounding to `Self`.
    ///
    /// # Safety
    /// `packed` must point to a pack of `Self`, and `points` must hold as many writable points as
    /// it packed.
    unsafe fn shapes_unpack(packed: *const u8, points: *mut [Self; 3]) -> Result<(), Error>;

    /// Chamfer distances between every row cloud and every column cloud.
    ///
    /// # Safety
    /// - `a_packed` and `b_packed` must hold `rows` and `columns` pointers to packs of `Self`
    /// - `results` must hold `rows` rows of `columns` distances, `results_stride` bytes apart
    /// - `rotations` must be null or hold `rows × columns × 9` writable values
    unsafe fn chamfers_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut Self::Distance,
        results_stride: usize,
        rotations: *mut f32,
    ) -> Result<(), Error>;

    /// Hausdorff distances between every row cloud and every column cloud.
    ///
    /// # Safety
    /// The contract of [`Shapes::chamfers_packed`].
    unsafe fn hausdorffs_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut Self::Distance,
        results_stride: usize,
        rotations: *mut f32,
    ) -> Result<(), Error>;

    /// Returns the workspace bytes Wasserstein-Procrustes needs between clouds of up to
    /// `row_points` and `column_points` points.
    fn wassersteinprocrustes_workspace_size(row_points: usize, column_points: usize) -> Result<usize, Error>;

    /// Wasserstein-Procrustes distances between every row cloud and every column cloud.
    ///
    /// # Safety
    /// The contract of [`Shapes::chamfers_packed`], plus `workspace` must have `workspace_bytes`
    /// writable bytes on an `nk_default_alignment_k` boundary.
    unsafe fn wassersteinprocrustes_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut Self::Distance,
        results_stride: usize,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> Result<(), Error>;

    /// Returns the workspace bytes Gromov-Wasserstein needs between clouds of up to `row_points`
    /// and `column_points` points.
    fn gromovwassersteins_workspace_size(row_points: usize, column_points: usize) -> Result<usize, Error>;

    /// Gromov-Wasserstein distances between every row cloud and every column cloud.
    ///
    /// # Safety
    /// The contract of [`Shapes::wassersteinprocrustes_packed`].
    unsafe fn gromovwassersteins_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut Self::Distance,
        results_stride: usize,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> Result<(), Error>;
}

// endregion: Shapes trait

// region: Shapes impls

impl Shapes for f16 {
    type Distance = f32;

    fn shapes_pack_size(points: usize) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe { nk_shapes_pack_size_f16_best(points, Capabilities::CPUS.bits(), &mut bytes) }.check()?;
        Ok(bytes)
    }

    unsafe fn shapes_pack(points: &[[Self; 3]], packed: *mut u8) -> Result<(), Error> {
        nk_shapes_pack_f16_best(
            points.as_ptr().cast(),
            points.len(),
            packed,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    unsafe fn shapes_pack_normalized(points: &[[Self; 3]], packed: *mut u8) -> Result<(), Error> {
        nk_shapes_pack_normalized_f16_best(
            points.as_ptr().cast(),
            points.len(),
            packed,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    unsafe fn shapes_packed_shape(
        packed: *const u8,
        points: *mut usize,
        centroid: *mut f32,
        frame: *mut f32,
        scale: *mut f32,
        spreads: *mut f32,
        max_radius: *mut f32,
    ) -> Result<(), Error> {
        nk_shapes_packed_shape_f16_best(
            packed,
            points,
            centroid,
            frame,
            scale,
            spreads,
            max_radius,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    unsafe fn shapes_unpack(packed: *const u8, points: *mut [Self; 3]) -> Result<(), Error> {
        nk_shapes_unpack_f16_best(packed, points.cast(), Capabilities::CPUS.bits(), null_mut()).check()
    }

    unsafe fn chamfers_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut f32,
        results_stride: usize,
        rotations: *mut f32,
    ) -> Result<(), Error> {
        nk_chamfers_packed_f16_best(
            a_packed,
            rows,
            b_packed,
            columns,
            results,
            results_stride,
            rotations,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    unsafe fn hausdorffs_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut f32,
        results_stride: usize,
        rotations: *mut f32,
    ) -> Result<(), Error> {
        nk_hausdorffs_packed_f16_best(
            a_packed,
            rows,
            b_packed,
            columns,
            results,
            results_stride,
            rotations,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    fn wassersteinprocrustes_workspace_size(row_points: usize, column_points: usize) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe {
            nk_wassersteinprocrustes_workspace_size_f16_best(
                row_points,
                column_points,
                Capabilities::CPUS.bits(),
                &mut bytes,
            )
        }
        .check()?;
        Ok(bytes)
    }

    unsafe fn wassersteinprocrustes_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut f32,
        results_stride: usize,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> Result<(), Error> {
        nk_wassersteinprocrustes_packed_f16_best(
            a_packed,
            rows,
            b_packed,
            columns,
            results,
            results_stride,
            rotations,
            workspace,
            workspace_bytes,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    fn gromovwassersteins_workspace_size(row_points: usize, column_points: usize) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe {
            nk_gromovwassersteins_workspace_size_f16_best(
                row_points,
                column_points,
                Capabilities::CPUS.bits(),
                &mut bytes,
            )
        }
        .check()?;
        Ok(bytes)
    }

    unsafe fn gromovwassersteins_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut f32,
        results_stride: usize,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> Result<(), Error> {
        nk_gromovwassersteins_packed_f16_best(
            a_packed,
            rows,
            b_packed,
            columns,
            results,
            results_stride,
            rotations,
            workspace,
            workspace_bytes,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }
}

impl Shapes for f32 {
    type Distance = f64;

    fn shapes_pack_size(points: usize) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe { nk_shapes_pack_size_f32_best(points, Capabilities::CPUS.bits(), &mut bytes) }.check()?;
        Ok(bytes)
    }

    unsafe fn shapes_pack(points: &[[Self; 3]], packed: *mut u8) -> Result<(), Error> {
        nk_shapes_pack_f32_best(
            points.as_ptr().cast(),
            points.len(),
            packed,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    unsafe fn shapes_pack_normalized(points: &[[Self; 3]], packed: *mut u8) -> Result<(), Error> {
        nk_shapes_pack_normalized_f32_best(
            points.as_ptr().cast(),
            points.len(),
            packed,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    unsafe fn shapes_packed_shape(
        packed: *const u8,
        points: *mut usize,
        centroid: *mut f64,
        frame: *mut f64,
        scale: *mut f64,
        spreads: *mut f64,
        max_radius: *mut f64,
    ) -> Result<(), Error> {
        nk_shapes_packed_shape_f32_best(
            packed,
            points,
            centroid,
            frame,
            scale,
            spreads,
            max_radius,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    unsafe fn shapes_unpack(packed: *const u8, points: *mut [Self; 3]) -> Result<(), Error> {
        nk_shapes_unpack_f32_best(packed, points.cast(), Capabilities::CPUS.bits(), null_mut()).check()
    }

    unsafe fn chamfers_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut f64,
        results_stride: usize,
        rotations: *mut f32,
    ) -> Result<(), Error> {
        nk_chamfers_packed_f32_best(
            a_packed,
            rows,
            b_packed,
            columns,
            results,
            results_stride,
            rotations,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    unsafe fn hausdorffs_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut f64,
        results_stride: usize,
        rotations: *mut f32,
    ) -> Result<(), Error> {
        nk_hausdorffs_packed_f32_best(
            a_packed,
            rows,
            b_packed,
            columns,
            results,
            results_stride,
            rotations,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    fn wassersteinprocrustes_workspace_size(row_points: usize, column_points: usize) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe {
            nk_wassersteinprocrustes_workspace_size_f32_best(
                row_points,
                column_points,
                Capabilities::CPUS.bits(),
                &mut bytes,
            )
        }
        .check()?;
        Ok(bytes)
    }

    unsafe fn wassersteinprocrustes_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut f64,
        results_stride: usize,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> Result<(), Error> {
        nk_wassersteinprocrustes_packed_f32_best(
            a_packed,
            rows,
            b_packed,
            columns,
            results,
            results_stride,
            rotations,
            workspace,
            workspace_bytes,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }

    fn gromovwassersteins_workspace_size(row_points: usize, column_points: usize) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe {
            nk_gromovwassersteins_workspace_size_f32_best(
                row_points,
                column_points,
                Capabilities::CPUS.bits(),
                &mut bytes,
            )
        }
        .check()?;
        Ok(bytes)
    }

    unsafe fn gromovwassersteins_packed(
        a_packed: *const *const u8,
        rows: usize,
        b_packed: *const *const u8,
        columns: usize,
        results: *mut f64,
        results_stride: usize,
        rotations: *mut f32,
        workspace: *mut u8,
        workspace_bytes: usize,
    ) -> Result<(), Error> {
        nk_gromovwassersteins_packed_f32_best(
            a_packed,
            rows,
            b_packed,
            columns,
            results,
            results_stride,
            rotations,
            workspace,
            workspace_bytes,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
        .check()
    }
}

// endregion: Shapes impls

// region: ShapesPackedMatrix

/// Point clouds packed one by one onto their principal axes, scored as rows against the columns
/// of another set.
///
/// One allocation holds the pointer to every pack, as the kernels take them, then the packs.
#[derive(Debug)]
pub struct ShapesPackedMatrix<Scalar: Shapes, Alloc: Allocator = Global> {
    buffer: PackedBuffer<Alloc>,
    clouds: usize,
    _marker: PhantomData<Scalar>,
}

// Safety: the pointers only reach into the buffer this value owns
unsafe impl<Scalar: Shapes + Send, Alloc: Allocator + Send> Send for ShapesPackedMatrix<Scalar, Alloc> {}
unsafe impl<Scalar: Shapes + Sync, Alloc: Allocator + Sync> Sync for ShapesPackedMatrix<Scalar, Alloc> {}

impl<Scalar: Shapes, Alloc: Allocator> ShapesPackedMatrix<Scalar, Alloc> {
    /// Packs every cloud of `clouds`, of any sizes, using a custom allocator.
    pub fn new_in(clouds: &[&[[Scalar; 3]]], alloc: Alloc) -> Result<Self, Error> {
        Self::pack_in(clouds, alloc, Scalar::shapes_pack)
    }

    /// Packs every cloud of `clouds` like [`new_in`](Self::new_in), each divided by its RMS
    /// distance from the centroid, so every kind compares shapes regardless of size.
    pub fn new_normalized_in(clouds: &[&[[Scalar; 3]]], alloc: Alloc) -> Result<Self, Error> {
        Self::pack_in(clouds, alloc, Scalar::shapes_pack_normalized)
    }

    fn pack_in(
        clouds: &[&[[Scalar; 3]]],
        alloc: Alloc,
        pack_cloud: unsafe fn(&[[Scalar; 3]], *mut u8) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        let pointers_bytes = clouds.len() * core::mem::size_of::<*const u8>();
        // The slack moves the packs from `PackedBuffer`'s `SIMD_ALIGNMENT` to `DEFAULT_ALIGNMENT`
        let mut total_bytes = pointers_bytes + DEFAULT_ALIGNMENT - 1;
        for cloud in clouds {
            total_bytes = total_bytes
                .checked_add(Scalar::shapes_pack_size(cloud.len())?.next_multiple_of(DEFAULT_ALIGNMENT))
                .ok_or(Error::AllocationFailed)?;
        }
        let mut buffer = PackedBuffer::empty_in(alloc);
        let base = buffer.reset_for_pack(total_bytes)?;
        let mut offset = (base.addr() + pointers_bytes).next_multiple_of(DEFAULT_ALIGNMENT) - base.addr();
        for (index, cloud) in clouds.iter().enumerate() {
            let pack = unsafe { base.add(offset) };
            unsafe { pack_cloud(cloud, pack) }?;
            unsafe { base.cast::<*const u8>().add(index).write(pack) };
            offset += Scalar::shapes_pack_size(cloud.len())?.next_multiple_of(DEFAULT_ALIGNMENT);
        }
        Ok(Self {
            buffer,
            clouds: clouds.len(),
            _marker: PhantomData,
        })
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { self.buffer.allocator() }

    /// Number of packed clouds.
    pub fn clouds(&self) -> usize { self.clouds }

    /// Total points across the packed clouds.
    pub fn points(&self) -> usize {
        (0..self.clouds)
            .map(|index| self.cloud_points(index).unwrap_or(0))
            .sum()
    }

    /// Returns the shape __[clouds,points]__, the clouds and their total points.
    pub fn shape(&self) -> (usize, usize) { (self.clouds, self.points()) }

    /// Bytes one packed cloud of `points` points takes, to size a buffer without packing.
    pub fn pack_size(points: usize) -> Result<usize, Error> { Scalar::shapes_pack_size(points) }

    /// The pointer to every pack, as the `nk_*_packed_*` shape kernels take them.
    pub fn packs(&self) -> &[*const u8] {
        match self.clouds {
            0 => &[],
            clouds => unsafe { core::slice::from_raw_parts(self.buffer.as_ptr().cast::<*const u8>(), clouds) },
        }
    }

    fn pack(&self, index: usize) -> Result<*const u8, Error> {
        self.packs().get(index).copied().ok_or(Error::IndexOutOfBounds {
            index,
            size: self.clouds,
        })
    }

    /// Number of points in cloud `index`.
    pub fn cloud_points(&self, index: usize) -> Result<usize, Error> {
        let (pack, mut points, none) = (self.pack(index)?, 0, null_mut());
        unsafe { Scalar::shapes_packed_shape(pack, &mut points, none, none, none, none, none) }?;
        Ok(points)
    }

    /// Mean input point of cloud `index`.
    pub fn centroid(&self, index: usize) -> Result<[Scalar::Distance; 3], Error> {
        let (pack, mut centroid, none) = (self.pack(index)?, [Scalar::Distance::default(); 3], null_mut());
        unsafe { Scalar::shapes_packed_shape(pack, null_mut(), centroid.as_mut_ptr(), none, none, none, none) }?;
        Ok(centroid)
    }

    /// Unit quaternion (x, y, z, w) rotating centered cloud `index` onto its principal axes.
    pub fn frame(&self, index: usize) -> Result<[Scalar::Distance; 4], Error> {
        let (pack, mut frame, none) = (self.pack(index)?, [Scalar::Distance::default(); 4], null_mut());
        unsafe { Scalar::shapes_packed_shape(pack, null_mut(), none, frame.as_mut_ptr(), none, none, none) }?;
        Ok(frame)
    }

    /// What rotated cloud `index` was divided by: its RMS distance from the centroid for normalized
    /// packs, and 1 otherwise.
    pub fn scale(&self, index: usize) -> Result<Scalar::Distance, Error> {
        let (pack, mut scale, none) = (self.pack(index)?, Scalar::Distance::default(), null_mut());
        unsafe { Scalar::shapes_packed_shape(pack, null_mut(), none, none, &mut scale, none, none) }?;
        Ok(scale)
    }

    /// Descending eigenvalues of the second moments of packed cloud `index`.
    pub fn spreads(&self, index: usize) -> Result<[Scalar::Distance; 3], Error> {
        let (pack, mut spreads, none) = (self.pack(index)?, [Scalar::Distance::default(); 3], null_mut());
        unsafe { Scalar::shapes_packed_shape(pack, null_mut(), none, none, none, spreads.as_mut_ptr(), none) }?;
        Ok(spreads)
    }

    /// Largest packed point norm of cloud `index`.
    pub fn max_radius(&self, index: usize) -> Result<Scalar::Distance, Error> {
        let (pack, mut max_radius, none) = (self.pack(index)?, Scalar::Distance::default(), null_mut());
        unsafe { Scalar::shapes_packed_shape(pack, null_mut(), none, none, none, none, &mut max_radius) }?;
        Ok(max_radius)
    }

    /// Writes cloud `index` into `out` as it was packed: interleaved, in its original pose and
    /// scale, within one rounding to `Scalar`.
    ///
    /// Returns `Err` if `index` is out of range or `out` does not hold
    /// [`cloud_points`](Self::cloud_points) points.
    pub fn unpack(&self, index: usize, out: &mut [[Scalar; 3]]) -> Result<(), Error> {
        check_len(self.cloud_points(index)?, out.len())?;
        unsafe { Scalar::shapes_unpack(self.pack(index)?, out.as_mut_ptr()) }
    }

    fn largest(&self) -> usize {
        (0..self.clouds)
            .map(|index| self.cloud_points(index).unwrap_or(0))
            .max()
            .unwrap_or(0)
    }

    /// Chamfer distances from every cloud of `self` to every cloud of `columns`, the best of the 8
    /// sign flips of the principal axes, zero between two empty clouds and infinite between an
    /// empty and a non-empty one.
    ///
    /// Fills `results` with __[rows,columns]__ distances and, when given, `rotations` with as many
    /// matrices taking each column cloud onto each row cloud.
    ///
    /// Returns `Err` if `results` or `rotations` does not hold __rows × columns__ entries.
    pub fn chamfers<OtherAlloc: Allocator>(
        &self,
        columns: &ShapesPackedMatrix<Scalar, OtherAlloc>,
        results: &mut [Scalar::Distance],
        rotations: Option<&mut [[f32; 9]]>,
    ) -> Result<(), Error> {
        let (results_stride, rotations) = check_outputs(self.clouds, columns.clouds, results, rotations)?;
        let (rows, columns) = (self.packs(), columns.packs());
        unsafe {
            Scalar::chamfers_packed(
                rows.as_ptr(),
                rows.len(),
                columns.as_ptr(),
                columns.len(),
                results.as_mut_ptr(),
                results_stride,
                rotations,
            )
        }
    }

    /// Hausdorff distances from every cloud of `self` to every cloud of `columns`, the best of the
    /// 8 sign flips of the principal axes, with the outputs and conventions of
    /// [`chamfers`](Self::chamfers).
    pub fn hausdorffs<OtherAlloc: Allocator>(
        &self,
        columns: &ShapesPackedMatrix<Scalar, OtherAlloc>,
        results: &mut [Scalar::Distance],
        rotations: Option<&mut [[f32; 9]]>,
    ) -> Result<(), Error> {
        let (results_stride, rotations) = check_outputs(self.clouds, columns.clouds, results, rotations)?;
        let (rows, columns) = (self.packs(), columns.packs());
        unsafe {
            Scalar::hausdorffs_packed(
                rows.as_ptr(),
                rows.len(),
                columns.as_ptr(),
                columns.len(),
                results.as_mut_ptr(),
                results_stride,
                rotations,
            )
        }
    }

    /// Wasserstein-Procrustes distances from every cloud of `self` to every cloud of `columns`:
    /// the RMSD of the exact optimal transport plan between uniform weights, after the rotation or
    /// reflection minimizing it.
    ///
    /// The workspace comes from `self`'s allocator for the call. Outputs and conventions match
    /// [`chamfers`](Self::chamfers).
    pub fn wassersteinprocrustes<OtherAlloc: Allocator>(
        &self,
        columns: &ShapesPackedMatrix<Scalar, OtherAlloc>,
        results: &mut [Scalar::Distance],
        rotations: Option<&mut [[f32; 9]]>,
    ) -> Result<(), Error> {
        let (results_stride, rotations) = check_outputs(self.clouds, columns.clouds, results, rotations)?;
        let workspace_bytes = Scalar::wassersteinprocrustes_workspace_size(self.largest(), columns.largest())?;
        let (rows, columns) = (self.packs(), columns.packs());
        with_workspace(self.allocator(), workspace_bytes, |workspace| unsafe {
            Scalar::wassersteinprocrustes_packed(
                rows.as_ptr(),
                rows.len(),
                columns.as_ptr(),
                columns.len(),
                results.as_mut_ptr(),
                results_stride,
                rotations,
                workspace,
                workspace_bytes,
            )
        })
    }

    /// Gromov-Wasserstein distances from every cloud of `self` to every cloud of `columns`: the
    /// transport distance between the clouds' own distance matrices, under the best plan found from
    /// the Wasserstein-Procrustes plans and the product plan.
    ///
    /// The workspace comes from `self`'s allocator for the call. Outputs and conventions match
    /// [`chamfers`](Self::chamfers).
    pub fn gromovwassersteins<OtherAlloc: Allocator>(
        &self,
        columns: &ShapesPackedMatrix<Scalar, OtherAlloc>,
        results: &mut [Scalar::Distance],
        rotations: Option<&mut [[f32; 9]]>,
    ) -> Result<(), Error> {
        let (results_stride, rotations) = check_outputs(self.clouds, columns.clouds, results, rotations)?;
        let workspace_bytes = Scalar::gromovwassersteins_workspace_size(self.largest(), columns.largest())?;
        let (rows, columns) = (self.packs(), columns.packs());
        with_workspace(self.allocator(), workspace_bytes, |workspace| unsafe {
            Scalar::gromovwassersteins_packed(
                rows.as_ptr(),
                rows.len(),
                columns.as_ptr(),
                columns.len(),
                results.as_mut_ptr(),
                results_stride,
                rotations,
                workspace,
                workspace_bytes,
            )
        })
    }
}

impl<Scalar: Shapes> ShapesPackedMatrix<Scalar, Global> {
    /// Packs every cloud of `clouds`, of any sizes, using the global allocator.
    pub fn new(clouds: &[&[[Scalar; 3]]]) -> Result<Self, Error> { Self::new_in(clouds, Global) }

    /// Packs every cloud of `clouds` normalized, using the global allocator.
    pub fn new_normalized(clouds: &[&[[Scalar; 3]]]) -> Result<Self, Error> { Self::new_normalized_in(clouds, Global) }
}

/// Checks that `results` and `rotations` hold `rows × columns` entries, returning the bytes between
/// rows of `results` and the rotations pointer, null without them.
fn check_outputs<Distance>(
    rows: usize,
    columns: usize,
    results: &[Distance],
    rotations: Option<&mut [[f32; 9]]>,
) -> Result<(usize, *mut f32), Error> {
    let pairs = rows.saturating_mul(columns);
    check_len(pairs, results.len())?;
    let rotations = match rotations {
        Some(rotations) => {
            check_len(pairs, rotations.len())?;
            rotations.as_mut_ptr().cast::<f32>()
        }
        None => null_mut(),
    };
    Ok((columns * core::mem::size_of::<Distance>(), rotations))
}

/// Runs `kernel` on a workspace of `bytes` from `alloc`, released when it returns.
fn with_workspace<Alloc: Allocator>(
    alloc: &Alloc,
    bytes: usize,
    kernel: impl FnOnce(*mut u8) -> Result<(), Error>,
) -> Result<(), Error> {
    let layout = Layout::from_size_align(bytes, DEFAULT_ALIGNMENT).map_err(|_| Error::AllocationFailed)?;
    let workspace = alloc
        .allocate(layout)
        .map_err(|_| Error::AllocationFailed)?
        .cast::<u8>();
    let result = kernel(workspace.as_ptr());
    unsafe { alloc.deallocate(workspace, layout) };
    result
}

// endregion: ShapesPackedMatrix

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, FloatLike, NumberLike};

    /// An irregular cloud, so its principal axes and their order are unambiguous.
    const CLOUD: [[f32; 3]; 7] = [
        [0.0, 0.0, 0.0],
        [3.0, 0.2, 0.1],
        [0.4, 2.0, -0.3],
        [-0.2, 0.3, 1.0],
        [1.5, -1.1, 0.4],
        [-1.7, 0.6, -0.5],
        [0.9, 1.4, 0.8],
    ];

    /// A proper rotation, 40° about (1, 2, 2) / 3.
    fn rotation() -> [[f32; 3]; 3] {
        let (axis, angle) = ([1.0_f32 / 3.0, 2.0 / 3.0, 2.0 / 3.0], 40.0_f32.to_radians());
        let (sine, cosine) = angle.sin_cos();
        let mut matrix = [[0.0_f32; 3]; 3];
        for row in 0..3 {
            for column in 0..3 {
                let identity = if row == column { cosine } else { 0.0 };
                let cross = match (row, column) {
                    (0, 1) => -axis[2],
                    (0, 2) => axis[1],
                    (1, 0) => axis[2],
                    (1, 2) => -axis[0],
                    (2, 0) => -axis[1],
                    (2, 1) => axis[0],
                    _ => 0.0,
                };
                matrix[row][column] = identity + sine * cross + (1.0 - cosine) * axis[row] * axis[column];
            }
        }
        matrix
    }

    /// `CLOUD` rotated by `rotation()` and moved by `shift`, in reverse point order.
    fn moved_cloud(shift: [f32; 3]) -> Vec<[f32; 3]> {
        let rotation = rotation();
        let moved = CLOUD.iter().rev().map(|point| {
            core::array::from_fn(|row| (0..3).map(|k| rotation[row][k] * point[k]).sum::<f32>() + shift[row])
        });
        moved.collect()
    }

    fn convert_cloud<Scalar: FloatLike>(cloud: &[[f32; 3]]) -> Vec<[Scalar; 3]> {
        let convert = |point: &[f32; 3]| point.map(Scalar::from_f32);
        cloud.iter().map(convert).collect()
    }

    /// The accessors report every cloud's pose and bounds, `max_radius` within `tolerance` of the
    /// farthest input point from the centroid.
    fn check_poses<Scalar>(tolerance: f64)
    where
        Scalar: Shapes + FloatLike,
        Scalar::Distance: NumberLike,
    {
        let shifted = convert_cloud::<Scalar>(&moved_cloud([10.0, -4.0, 2.5]));
        let packed = ShapesPackedMatrix::new(&[&shifted[..], &[]]).unwrap();
        assert_eq!(packed.shape(), (2, 7));
        assert_eq!(
            (packed.cloud_points(0).unwrap(), packed.cloud_points(1).unwrap()),
            (7, 0)
        );
        let centroid = packed.centroid(0).unwrap().map(NumberLike::to_f64);
        for axis in 0..3 {
            let mean = shifted.iter().map(|point| point[axis].to_f64()).sum::<f64>() / 7.0;
            assert_close(centroid[axis], mean, 1e-5, 1e-5, "centroid");
        }
        let spreads = packed.spreads(0).unwrap().map(NumberLike::to_f64);
        assert!(spreads[0] >= spreads[1] && spreads[1] >= spreads[2] && spreads[2] >= 0.0);
        let frame = packed.frame(0).unwrap().map(NumberLike::to_f64);
        assert_close(
            frame.iter().map(|value| value * value).sum(),
            1.0,
            1e-5,
            0.0,
            "frame norm",
        );
        let farthest = shifted
            .iter()
            .map(|point| {
                (0..3)
                    .map(|axis| (point[axis].to_f64() - centroid[axis]).powi(2))
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
            .sqrt();
        assert_close(
            packed.max_radius(0).unwrap().to_f64(),
            farthest,
            tolerance,
            0.0,
            "max radius",
        );
        assert_eq!(packed.max_radius(1).unwrap().to_f64(), 0.0);
        assert!(matches!(
            packed.cloud_points(2),
            Err(Error::IndexOutOfBounds { index: 2, size: 2 })
        ));
    }

    #[test]
    fn shapes_pack_reports_poses() {
        check_poses::<f32>(1e-5);
        check_poses::<f16>(1e-2);
    }

    /// Unpacking plain and normalized packs returns every input point within `tolerance`.
    fn check_unpack<Scalar>(tolerance: f64)
    where
        Scalar: Shapes + FloatLike,
    {
        let shifted = convert_cloud::<Scalar>(&moved_cloud([10.0, -4.0, 2.5]));
        let plain = ShapesPackedMatrix::new(&[&shifted[..], &[]]).unwrap();
        let normalized = ShapesPackedMatrix::new_normalized(&[&shifted[..]]).unwrap();
        let mut unpacked = vec![[Scalar::from_f32(0.0); 3]; 7];
        for packed in [&plain, &normalized] {
            packed.unpack(0, &mut unpacked).unwrap();
            for (point, input) in unpacked.iter().zip(&shifted) {
                for axis in 0..3 {
                    let name = core::any::type_name::<Scalar>();
                    assert_close(point[axis].to_f64(), input[axis].to_f64(), tolerance, 0.0, name);
                }
            }
        }
        plain.unpack(1, &mut []).unwrap();
        let mismatch = Err(Error::ShapeMismatch {
            axis: 0,
            expected: 7,
            got: 6,
        });
        assert_eq!(plain.unpack(0, &mut unpacked[..6]), mismatch);
    }

    #[test]
    fn shapes_unpack_round_trips() {
        check_unpack::<f32>(1e-5);
        check_unpack::<f16>(1e-2);
    }

    /// Every kind vanishes within `tolerance` between a cloud and its rigid motion, and the
    /// rotations superpose them.
    fn check_rigid_copies<Scalar>(tolerance: f64)
    where
        Scalar: Shapes + FloatLike,
        Scalar::Distance: NumberLike,
    {
        let shift = [1.0_f32, -2.0, 0.5];
        let cloud = convert_cloud::<Scalar>(&CLOUD);
        let moved = convert_cloud::<Scalar>(&moved_cloud(shift));
        let rows = ShapesPackedMatrix::new(&[&cloud[..]]).unwrap();
        let columns = ShapesPackedMatrix::new(&[&moved[..]]).unwrap();
        let name = core::any::type_name::<Scalar>();
        let mut distances = [Scalar::Distance::default(); 1];
        let mut rotations = [[0.0_f32; 9]; 1];

        rows.chamfers(&columns, &mut distances, None).unwrap();
        assert_close(distances[0].to_f64(), 0.0, tolerance, 0.0, &format!("chamfers<{name}>"));
        rows.hausdorffs(&columns, &mut distances, None).unwrap();
        assert_close(
            distances[0].to_f64(),
            0.0,
            tolerance,
            0.0,
            &format!("hausdorffs<{name}>"),
        );
        rows.gromovwassersteins(&columns, &mut distances, None).unwrap();
        assert_close(
            distances[0].to_f64(),
            0.0,
            tolerance,
            0.0,
            &format!("gromovwassersteins<{name}>"),
        );
        rows.wassersteinprocrustes(&columns, &mut distances, Some(&mut rotations))
            .unwrap();
        assert_close(
            distances[0].to_f64(),
            0.0,
            tolerance,
            0.0,
            &format!("wassersteinprocrustes<{name}>"),
        );

        // `CLOUD` was reversed, so row point `i` pairs with column point `6 - i`
        let (row_centroid, column_centroid) = (rows.centroid(0).unwrap(), columns.centroid(0).unwrap());
        let rotation = rotations[0];
        for (index, point) in cloud.iter().enumerate() {
            let partner = moved[6 - index];
            for axis in 0..3 {
                let superposed = (0..3)
                    .map(|k| rotation[axis * 3 + k] as f64 * (partner[k].to_f64() - column_centroid[k].to_f64()))
                    .sum::<f64>()
                    + row_centroid[axis].to_f64();
                assert_close(
                    superposed,
                    point[axis].to_f64(),
                    tolerance,
                    0.0,
                    &format!("superposed<{name}>"),
                );
            }
        }
    }

    #[test]
    fn shapes_rigid_copies_vanish() {
        // Rounding the moved inputs costs at most 3e-7 for `f32` and 1.4e-3 for `f16`
        check_rigid_copies::<f32>(1e-5);
        check_rigid_copies::<f16>(1e-2);
    }

    #[test]
    fn shapes_normalized_packs_ignore_size() {
        let grown: Vec<[f32; 3]> = moved_cloud([1.0, -2.0, 0.5])
            .iter()
            .map(|point| point.map(|value| value * 3.7))
            .collect();
        let radius = |cloud: &[[f32; 3]]| {
            let centroid: [f64; 3] =
                core::array::from_fn(|axis| cloud.iter().map(|point| point[axis] as f64).sum::<f64>() / 7.0);
            let squares = cloud
                .iter()
                .flat_map(|point| (0..3).map(move |axis| point[axis] as f64 - centroid[axis]));
            (squares.map(|delta| delta * delta).sum::<f64>() / 7.0).sqrt()
        };
        let rows = ShapesPackedMatrix::new_normalized(&[&CLOUD[..]]).unwrap();
        let columns = ShapesPackedMatrix::new_normalized(&[&grown[..]]).unwrap();
        assert_close(rows.scale(0).unwrap(), radius(&CLOUD), 0.0, 1e-6, "scale");
        assert_close(columns.scale(0).unwrap(), radius(&grown), 0.0, 1e-6, "grown scale");
        let plain_rows = ShapesPackedMatrix::new(&[&CLOUD[..]]).unwrap();
        let plain_columns = ShapesPackedMatrix::new(&[&grown[..]]).unwrap();
        assert_eq!(plain_rows.scale(0).unwrap(), 1.0);
        type Kind = fn(&ShapesPackedMatrix<f32>, &ShapesPackedMatrix<f32>, &mut [f64]) -> Result<(), Error>;
        let kinds: [Kind; 4] = [
            |a, b, results| a.chamfers(b, results, None),
            |a, b, results| a.hausdorffs(b, results, None),
            |a, b, results| a.wassersteinprocrustes(b, results, None),
            |a, b, results| a.gromovwassersteins(b, results, None),
        ];
        let tolerance = 1e-5;
        for kind in kinds {
            let (mut normalized, mut plain) = ([0.0_f64; 1], [0.0_f64; 1]);
            kind(&rows, &columns, &mut normalized).unwrap();
            kind(&plain_rows, &plain_columns, &mut plain).unwrap();
            assert!(
                normalized[0] <= tolerance,
                "a grown copy scores {} between normalized packs",
                normalized[0]
            );
            assert!(
                plain[0] > tolerance * columns.scale(0).unwrap(),
                "a grown copy scores {} between plain packs",
                plain[0]
            );
        }
    }

    #[test]
    fn shapes_swapped_sides_match() {
        let small = &CLOUD[..5];
        let large = moved_cloud([0.3, 0.0, -1.0]);
        let empty: &[[f32; 3]] = &[];
        let rows = ShapesPackedMatrix::new(&[small, &large[..], empty]).unwrap();
        let columns = ShapesPackedMatrix::new(&[&large[..], small, empty]).unwrap();
        type Kind = fn(&ShapesPackedMatrix<f32>, &ShapesPackedMatrix<f32>, &mut [f64]) -> Result<(), Error>;
        let kinds: [Kind; 4] = [
            |a, b, results| a.chamfers(b, results, None),
            |a, b, results| a.hausdorffs(b, results, None),
            |a, b, results| a.wassersteinprocrustes(b, results, None),
            |a, b, results| a.gromovwassersteins(b, results, None),
        ];
        for kind in kinds {
            let (mut forward, mut backward) = ([0.0_f64; 9], [0.0_f64; 9]);
            kind(&rows, &columns, &mut forward).unwrap();
            kind(&columns, &rows, &mut backward).unwrap();
            for (row, column) in (0..3).flat_map(|row| (0..3).map(move |column| (row, column))) {
                assert_eq!(
                    forward[row * 3 + column].to_bits(),
                    backward[column * 3 + row].to_bits()
                );
            }
            assert_eq!(forward[8], 0.0, "two empty clouds");
            assert!(forward[2].is_infinite() && forward[6].is_infinite(), "one empty cloud");
        }
    }

    #[test]
    fn shapes_reject_short_outputs() {
        let packed = ShapesPackedMatrix::new(&[&CLOUD[..], &CLOUD[..3]]).unwrap();
        let mut results = [0.0_f64; 3];
        let mismatch = Err(Error::ShapeMismatch {
            axis: 0,
            expected: 4,
            got: 3,
        });
        assert_eq!(packed.chamfers(&packed, &mut results, None), mismatch);
        let mut results = [0.0_f64; 4];
        let mut rotations = [[0.0_f32; 9]; 3];
        assert_eq!(packed.hausdorffs(&packed, &mut results, Some(&mut rotations)), mismatch);
    }
}
