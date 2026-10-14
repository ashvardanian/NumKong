//! Many-to-many geospatial distances: squared Euclidean in 3D, Haversine and Vincenty.
//!
//! This module provides:
//!
//! - [`SqEuclideans3D`]: Squared Euclidean distances between `[x, y, z]` points, exact for `i32`
//! - [`HaversinesPackedMatrix`]: Great-circle distances on the mean sphere
//! - [`VincentysPackedMatrix`]: Geodesic distances on the WGS-84 ellipsoid
//!
//! Each kernel scores every point of one set, the rows, against every point of another, the
//! columns, into a caller's row-major __[rows,columns]__ slice.
//!
//! # Units
//!
//! Both geodesics use a fixed input / output unit convention. Callers holding degree values must
//! convert with `.to_radians()` first:
//!
//! - __Input latitudes and longitudes__: radians, not degrees
//! - __Output distance__: meters
//! - __Earth radius__ for Haversine: `6 371 008.8 m`, the IUGG mean radius
//! - __Ellipsoid__ for Vincenty: WGS-84 (`a = 6 378 137 m`, `f = 1/298.257223563`)
//!
//! A packed matrix converts its points once: a Haversine pack holds unit vectors, a Vincenty pack
//! the longitudes and the sines and cosines of the reduced latitudes, so the kernels evaluate no
//! trigonometry before their per-pair work.
//!
//! # Example
//!
//! ```rust
//! use numkong::{HaversinesPackedMatrix, SqEuclideans3D};
//!
//! // London and Sydney as rows, New York as the only column
//! let rows = HaversinesPackedMatrix::new(&[0.8990, -0.5911], &[-0.0022, 2.6391]).unwrap();
//! let columns = HaversinesPackedMatrix::new(&[0.7106], &[-1.2917]).unwrap();
//! let mut meters = [0.0_f64; 2];
//! rows.haversines(&columns, &mut meters).unwrap();
//! assert!((meters[0] - 5_570_000.0).abs() < 10_000.0);
//!
//! let points = [[0_i32, 0, 0], [3, 4, 12]];
//! let mut squares = [0_u64; 4];
//! i32::sqeuclideans3d(&points, &points, &mut squares).unwrap();
//! assert_eq!(squares, [0, 169, 169, 0]);
//! ```
//!
//! File: rust/geospatial.rs
//! Author: Ash Vardanian

use core::{ffi::c_void, ptr::null_mut};

use crate::{
    capabilities::{nk_capability_t, nk_size_t, nk_status_t, Capabilities, StatusCode},
    tensor::{check_len, Allocator, Error, Global, PackedBuffer},
};

// region: FFI

#[link(name = "numkong")]
extern "C" {
    fn nk_sqeuclideans3d_i32_best(
        a: *const i32,
        rows: nk_size_t,
        b: *const i32,
        columns: nk_size_t,
        results: *mut u64,
        results_stride: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_sqeuclideans3d_f64_best(
        a: *const f64,
        rows: nk_size_t,
        b: *const f64,
        columns: nk_size_t,
        results: *mut f64,
        results_stride: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_haversines_pack_size_f64_best(
        points: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_vincentys_pack_size_f64_best(
        points: nk_size_t,
        capabilities: nk_capability_t,
        bytes: *mut nk_size_t,
    ) -> nk_status_t;
    fn nk_haversines_pack_f64_best(
        latitudes: *const f64,
        longitudes: *const f64,
        count: nk_size_t,
        packed: *mut u8,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_vincentys_pack_f64_best(
        latitudes: *const f64,
        longitudes: *const f64,
        count: nk_size_t,
        packed: *mut u8,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;

    fn nk_haversines_packed_f64_best(
        a_packed: *const u8,
        b_packed: *const u8,
        results: *mut f64,
        results_stride: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_vincentys_packed_f64_best(
        a_packed: *const u8,
        b_packed: *const u8,
        results: *mut f64,
        results_stride: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
}

// endregion: FFI

// region: SqEuclideans3D

/// Computes __squared Euclidean distances__ between every row point and every column point in 3D.
///
/// `i32` coordinates yield exact `u64` squares whenever every axis difference fits 32 bits, which
/// Earth-fixed centimeters below 4,359 km of altitude always satisfy. `f64` coordinates sum their
/// three squares in Dot2.
pub trait SqEuclideans3D: Sized {
    /// Squared distance type: `u64` for `i32` points, `f64` for `f64` ones.
    type Distance: Copy + Default;

    /// Fills `results` with the __[rows,columns]__ squared distances from every point of `a` to
    /// every point of `b`.
    ///
    /// Returns `Err` if `results` does not hold __rows × columns__ entries.
    fn sqeuclideans3d(a: &[[Self; 3]], b: &[[Self; 3]], results: &mut [Self::Distance]) -> Result<(), Error>;
}

impl SqEuclideans3D for i32 {
    type Distance = u64;

    fn sqeuclideans3d(a: &[[i32; 3]], b: &[[i32; 3]], results: &mut [u64]) -> Result<(), Error> {
        let results_stride = check_results(a.len(), b.len(), results)?;
        unsafe {
            nk_sqeuclideans3d_i32_best(
                a.as_ptr().cast(),
                a.len(),
                b.as_ptr().cast(),
                b.len(),
                results.as_mut_ptr(),
                results_stride,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
        }
        .check()
    }
}

impl SqEuclideans3D for f64 {
    type Distance = f64;

    fn sqeuclideans3d(a: &[[f64; 3]], b: &[[f64; 3]], results: &mut [f64]) -> Result<(), Error> {
        let results_stride = check_results(a.len(), b.len(), results)?;
        unsafe {
            nk_sqeuclideans3d_f64_best(
                a.as_ptr().cast(),
                a.len(),
                b.as_ptr().cast(),
                b.len(),
                results.as_mut_ptr(),
                results_stride,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
        }
        .check()
    }
}

// endregion: SqEuclideans3D

// region: HaversinesPackedMatrix

/// Points packed once as unit vectors on the mean sphere, scored with __great-circle distances__ as
/// rows against the columns of another pack.
///
/// The distance is `2R × atan2(‖a − b‖, ‖a + b‖)` over unit vectors `a` and `b`, which stays
/// conditioned at every separation, from a millimeter to the antipode, with `R` the mean radius.
#[derive(Debug)]
pub struct HaversinesPackedMatrix<Alloc: Allocator = Global> {
    buffer: PackedBuffer<Alloc>,
    points: usize,
}

// Safety: the pack is plain bytes this value owns
unsafe impl<Alloc: Allocator + Send> Send for HaversinesPackedMatrix<Alloc> {}
unsafe impl<Alloc: Allocator + Sync> Sync for HaversinesPackedMatrix<Alloc> {}

impl<Alloc: Allocator> HaversinesPackedMatrix<Alloc> {
    /// Packs the points at `latitudes` and `longitudes`, in radians, using a custom allocator.
    ///
    /// Returns `Err` if the two slices differ in length or allocation fails.
    pub fn new_in(latitudes: &[f64], longitudes: &[f64], alloc: Alloc) -> Result<Self, Error> {
        let bytes = Self::pack_size(latitudes.len())?;
        let buffer = pack_in(latitudes, longitudes, bytes, alloc, nk_haversines_pack_f64_best)?;
        Ok(Self {
            buffer,
            points: latitudes.len(),
        })
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { self.buffer.allocator() }

    /// Number of packed points.
    pub fn points(&self) -> usize { self.points }

    /// Bytes one pack of `points` points takes, to size a buffer without packing.
    pub fn pack_size(points: usize) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe { nk_haversines_pack_size_f64_best(points, Capabilities::CPUS.bits(), &mut bytes) }.check()?;
        Ok(bytes)
    }

    /// Great-circle distances in meters from every point of `self` to every point of `columns`,
    /// filling `results` with __[rows,columns]__ entries.
    ///
    /// Returns `Err` if `results` does not hold __rows × columns__ entries.
    pub fn haversines<OtherAlloc: Allocator>(
        &self,
        columns: &HaversinesPackedMatrix<OtherAlloc>,
        results: &mut [f64],
    ) -> Result<(), Error> {
        let results_stride = check_results(self.points, columns.points, results)?;
        unsafe {
            nk_haversines_packed_f64_best(
                self.buffer.as_ptr(),
                columns.buffer.as_ptr(),
                results.as_mut_ptr(),
                results_stride,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
        }
        .check()
    }
}

impl HaversinesPackedMatrix<Global> {
    /// Packs the points at `latitudes` and `longitudes`, in radians, using the global allocator.
    pub fn new(latitudes: &[f64], longitudes: &[f64]) -> Result<Self, Error> {
        Self::new_in(latitudes, longitudes, Global)
    }
}

// endregion: HaversinesPackedMatrix

// region: VincentysPackedMatrix

/// Points packed once as longitudes and the sines and cosines of their reduced latitudes, scored
/// with __Vincenty geodesic distances__ on the WGS-84 ellipsoid as rows against the columns of
/// another pack.
///
/// Uses Vincenty's iterative formula for oblate spheroid geodesics:
///
/// 1. Reduced latitudes: `tan(U) = (1−f) × tan(φ)`
/// 2. Iterate until convergence: `λ → L + (1−C) × f × sin(α) × [σ + C × sin(σ) × ...]`
/// 3. Compute: `u² = cos²(α) × (a² − b²)/b²`
/// 4. Series coefficients A, B from u²
/// 5. Distance: `s = b × A × (σ − Δσ)`
///
/// Where a = equatorial radius, b = polar radius, f = flattening. ~20× more accurate than Haversine
/// for long distances. Inputs are in radians, outputs in meters. A pair still unsettled after the
/// iteration limit, which happens only near the antipode, reports its last iterate.
#[derive(Debug)]
pub struct VincentysPackedMatrix<Alloc: Allocator = Global> {
    buffer: PackedBuffer<Alloc>,
    points: usize,
}

// Safety: the pack is plain bytes this value owns
unsafe impl<Alloc: Allocator + Send> Send for VincentysPackedMatrix<Alloc> {}
unsafe impl<Alloc: Allocator + Sync> Sync for VincentysPackedMatrix<Alloc> {}

impl<Alloc: Allocator> VincentysPackedMatrix<Alloc> {
    /// Packs the points at `latitudes` and `longitudes`, in radians, using a custom allocator.
    ///
    /// Returns `Err` if the two slices differ in length or allocation fails.
    pub fn new_in(latitudes: &[f64], longitudes: &[f64], alloc: Alloc) -> Result<Self, Error> {
        let bytes = Self::pack_size(latitudes.len())?;
        let buffer = pack_in(latitudes, longitudes, bytes, alloc, nk_vincentys_pack_f64_best)?;
        Ok(Self {
            buffer,
            points: latitudes.len(),
        })
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &Alloc { self.buffer.allocator() }

    /// Number of packed points.
    pub fn points(&self) -> usize { self.points }

    /// Bytes one pack of `points` points takes, to size a buffer without packing.
    pub fn pack_size(points: usize) -> Result<usize, Error> {
        let mut bytes = 0;
        unsafe { nk_vincentys_pack_size_f64_best(points, Capabilities::CPUS.bits(), &mut bytes) }.check()?;
        Ok(bytes)
    }

    /// Geodesic distances in meters from every point of `self` to every point of `columns`,
    /// filling `results` with __[rows,columns]__ entries.
    ///
    /// Returns `Err` if `results` does not hold __rows × columns__ entries.
    pub fn vincentys<OtherAlloc: Allocator>(
        &self,
        columns: &VincentysPackedMatrix<OtherAlloc>,
        results: &mut [f64],
    ) -> Result<(), Error> {
        let results_stride = check_results(self.points, columns.points, results)?;
        unsafe {
            nk_vincentys_packed_f64_best(
                self.buffer.as_ptr(),
                columns.buffer.as_ptr(),
                results.as_mut_ptr(),
                results_stride,
                Capabilities::CPUS.bits(),
                null_mut(),
            )
        }
        .check()
    }
}

impl VincentysPackedMatrix<Global> {
    /// Packs the points at `latitudes` and `longitudes`, in radians, using the global allocator.
    pub fn new(latitudes: &[f64], longitudes: &[f64]) -> Result<Self, Error> {
        Self::new_in(latitudes, longitudes, Global)
    }
}

// endregion: VincentysPackedMatrix

/// Packs `latitudes` and `longitudes` with `pack`, a Haversine or Vincenty packer, into a buffer
/// from `alloc` holding a `bytes`-long pack on a [`crate::DEFAULT_ALIGNMENT`] boundary.
fn pack_in<Alloc: Allocator>(
    latitudes: &[f64],
    longitudes: &[f64],
    bytes: usize,
    alloc: Alloc,
    pack: unsafe extern "C" fn(*const f64, *const f64, nk_size_t, *mut u8, nk_capability_t, *mut c_void) -> nk_status_t,
) -> Result<PackedBuffer<Alloc>, Error> {
    check_len(latitudes.len(), longitudes.len())?;
    let mut buffer = PackedBuffer::empty_in(alloc);
    let packed = buffer.reset_for_pack(bytes)?;
    unsafe {
        pack(
            latitudes.as_ptr(),
            longitudes.as_ptr(),
            latitudes.len(),
            packed,
            Capabilities::CPUS.bits(),
            null_mut(),
        )
    }
    .check()?;
    Ok(buffer)
}

/// Checks that `results` holds `rows × columns` entries, returning the bytes between its rows.
fn check_results<Distance>(rows: usize, columns: usize, results: &[Distance]) -> Result<usize, Error> {
    check_len(rows.saturating_mul(columns), results.len())?;
    Ok(columns * core::mem::size_of::<Distance>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::assert_close;

    /// Rows in degrees: London, Sydney and the North Pole.
    const ROW_LATITUDES: [f64; 3] = [51.5074, -33.8688, 90.0];
    const ROW_LONGITUDES: [f64; 3] = [-0.1278, 151.2093, 0.0];

    /// Columns in degrees: New York, Sydney's antipode, London's near-antipode across the
    /// antimeridian, a point a millimeter north of London, and one beside the South Pole.
    const COLUMN_LATITUDES: [f64; 5] = [40.7128, 33.8688, -51.5074, 51.50740001, -89.9];
    const COLUMN_LONGITUDES: [f64; 5] = [-74.0060, -28.7907, 179.8722, -0.1278, 180.0];

    fn radians(degrees: &[f64]) -> Vec<f64> { degrees.iter().map(|value| value.to_radians()).collect() }

    /// Great-circle meters between two points in radians, from their unit vectors on the sphere.
    fn haversine_reference(first: [f64; 2], second: [f64; 2]) -> f64 {
        let unit = |[latitude, longitude]: [f64; 2]| {
            [
                latitude.cos() * longitude.cos(),
                latitude.cos() * longitude.sin(),
                latitude.sin(),
            ]
        };
        let (first, second) = (unit(first), unit(second));
        let norm = |sign: f64| {
            (0..3)
                .map(|axis| (first[axis] + sign * second[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        2.0 * 6_371_008.8 * norm(-1.0).atan2(norm(1.0))
    }

    #[test]
    fn geospatial_haversines_match_unit_vectors() {
        let (row_latitudes, row_longitudes) = (radians(&ROW_LATITUDES), radians(&ROW_LONGITUDES));
        let (column_latitudes, column_longitudes) = (radians(&COLUMN_LATITUDES), radians(&COLUMN_LONGITUDES));
        let rows = HaversinesPackedMatrix::new(&row_latitudes, &row_longitudes).unwrap();
        let columns = HaversinesPackedMatrix::new(&column_latitudes, &column_longitudes).unwrap();
        assert_eq!((rows.points(), columns.points()), (3, 5));
        let mut results = [0.0_f64; 15];
        rows.haversines(&columns, &mut results).unwrap();
        for row in 0..3 {
            for column in 0..5 {
                let expected = haversine_reference(
                    [row_latitudes[row], row_longitudes[row]],
                    [column_latitudes[column], column_longitudes[column]],
                );
                assert_close(results[row * 5 + column], expected, 1e-6, 0.0, "haversines");
            }
        }
    }

    #[test]
    fn geospatial_vincentys_flinders_peak_to_buninyong() {
        let radians =
            |degrees: f64, minutes: f64, seconds: f64| (degrees + minutes / 60.0 + seconds / 3600.0).to_radians();
        let flinders_peak = [-radians(37.0, 57.0, 3.72030), radians(144.0, 25.0, 29.52440)];
        let buninyong = [-radians(37.0, 39.0, 10.15610), radians(143.0, 55.0, 35.38390)];
        let rows = VincentysPackedMatrix::new(&[flinders_peak[0]], &[flinders_peak[1]]).unwrap();
        let columns =
            VincentysPackedMatrix::new(&[buninyong[0], flinders_peak[0]], &[buninyong[1], flinders_peak[1]]).unwrap();
        let mut results = [f64::NAN; 2];
        rows.vincentys(&columns, &mut results).unwrap();
        assert_close(results[0], 54_972.271_1, 1e-3, 0.0, "Flinders Peak to Buninyong");
        assert_eq!(results[1], 0.0, "a point to itself");
    }

    #[test]
    fn geospatial_sqeuclideans3d_exact() {
        // Earth-fixed centimeters, whose squares sum past 2⁵³
        let a: [[i32; 3]; 2] = [[637_813_700, -637_813_700, 0], [-1, 2, -3]];
        let b: [[i32; 3]; 3] = [[-637_813_700, 637_813_700, 1], [0, 0, 0], [5, -7, 11]];
        let mut results = [0_u64; 6];
        i32::sqeuclideans3d(&a, &b, &mut results).unwrap();
        for row in 0..2 {
            for column in 0..3 {
                let expected = (0..3)
                    .map(|axis| (a[row][axis] as i64 - b[column][axis] as i64).unsigned_abs().pow(2))
                    .sum::<u64>();
                assert_eq!(results[row * 3 + column], expected);
            }
        }

        let mut results = [f64::NAN; 2];
        f64::sqeuclideans3d(&[[1.0, 2.0, 3.0]], &[[4.0, 6.0, 3.0], [1.0, 2.0, 3.0]], &mut results).unwrap();
        assert_eq!(results, [25.0, 0.0]);
    }

    #[test]
    fn geospatial_reject_mismatched_slices() {
        let mismatch = |expected, got| Err(Error::ShapeMismatch { axis: 0, expected, got });
        let unequal = HaversinesPackedMatrix::new(&[0.0, 1.0], &[0.0]);
        assert_eq!(unequal.map(|_| ()), mismatch(2, 1));
        let packed = VincentysPackedMatrix::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
        assert_eq!(packed.vincentys(&packed, &mut [0.0; 3]), mismatch(4, 3));
        let short = i32::sqeuclideans3d(&[[0; 3]], &[[0; 3]; 2], &mut [0; 1]);
        assert_eq!(short, mismatch(2, 1));
    }
}
