//! Geospatial distance functions: Haversine and Vincenty.
//!
//! This module provides:
//!
//! - [`Haversine`]: Great-circle distance on a sphere
//! - [`Vincenty`]: Geodesic distance on an ellipsoid
//! - [`Geospatial`]: Blanket trait combining `Haversine + Vincenty`
//!
//! # Units
//!
//! Both kernels use a fixed input / output unit convention. Callers holding degree
//! values must convert with `.to_radians()` first:
//!
//! - **Input latitudes and longitudes**: radians (not degrees)
//! - **Output distance**: meters
//! - **Earth radius** for Haversine: `6 335 439 m` (WGS-84 mean meridional radius)
//! - **Ellipsoid** for Vincenty: WGS-84 (`a = 6 378 137 m`, `f = 1/298.257223563`)
//!
//! The four latitude/longitude slices must be equal length; the output slice
//! must have the same length and is filled with per-pair distances in meters.

#[link(name = "numkong")]
extern "C" {
    fn nk_haversine_f32(
        a_lats: *const f32,
        a_lons: *const f32,
        b_lats: *const f32,
        b_lons: *const f32,
        n: usize,
        results: *mut f32,
    );
    fn nk_haversine_f64(
        a_lats: *const f64,
        a_lons: *const f64,
        b_lats: *const f64,
        b_lons: *const f64,
        n: usize,
        results: *mut f64,
    );
    fn nk_vincenty_f32(
        a_lats: *const f32,
        a_lons: *const f32,
        b_lats: *const f32,
        b_lons: *const f32,
        n: usize,
        results: *mut f32,
    );
    fn nk_vincenty_f64(
        a_lats: *const f64,
        a_lons: *const f64,
        b_lats: *const f64,
        b_lons: *const f64,
        n: usize,
        results: *mut f64,
    );
}

/// Computes **great-circle distances** between geographic coordinates on Earth.
///
/// Uses the Haversine formula for spherical Earth approximation:
///
/// - `a = sin²(Δφ/2) + cos(φ₁) × cos(φ₂) × sin²(Δλ/2)`
/// - `c = 2 × atan2(√a, √(1−a))`
/// - `d = R × c`
///
/// Where φ = latitude, λ = longitude, R = Earth's radius (6335 km).
/// Inputs are in radians, outputs in meters.
pub trait Haversine: Sized {
    /// Compute the great-circle distance for paired coordinates.
    ///
    /// All four coordinate slices must be the same length, matching the output
    /// slice. Returns `None` on length mismatch. Inputs are in **radians** and
    /// results are written in **meters**.
    ///
    /// # Examples
    ///
    /// ```
    /// use numkong::Haversine;
    /// // New York (40.7128°, -74.0060°) → Los Angeles (34.0522°, -118.2437°)
    /// let a_lat = [40.7128_f64.to_radians()];
    /// let a_lon = [(-74.0060_f64).to_radians()];
    /// let b_lat = [34.0522_f64.to_radians()];
    /// let b_lon = [(-118.2437_f64).to_radians()];
    /// let mut distance_meters = [0.0_f64];
    /// f64::haversine(&a_lat, &a_lon, &b_lat, &b_lon, &mut distance_meters).unwrap();
    /// // Approximate NY→LA great-circle distance ≈ 3 914 km.
    /// assert!((distance_meters[0] - 3_914_000.0).abs() < 30_000.0);
    /// ```
    fn haversine(a_lat: &[Self], a_lon: &[Self], b_lat: &[Self], b_lon: &[Self], result: &mut [Self]) -> Option<()>;
}

/// Computes **Vincenty geodesic distances** on the WGS84 ellipsoid.
///
/// Uses Vincenty's iterative formula for oblate spheroid geodesics:
///
/// 1. Reduced latitudes: `tan(U) = (1−f) × tan(φ)`
/// 2. Iterate until convergence: `λ → L + (1−C) × f × sin(α) × [σ + C × sin(σ) × ...]`
/// 3. Compute: `u² = cos²(α) × (a² − b²)/b²`
/// 4. Series coefficients A, B from u²
/// 5. Distance: `s = b × A × (σ − Δσ)`
///
/// Where a = equatorial radius, b = polar radius, f = flattening.
/// ~20× more accurate than Haversine for long distances.
/// Inputs are in radians, outputs in meters.
pub trait Vincenty: Sized {
    fn vincenty(a_lat: &[Self], a_lon: &[Self], b_lat: &[Self], b_lon: &[Self], result: &mut [Self]) -> Option<()>;
}

/// Combined trait for all geospatial distance computations.
pub trait Geospatial: Haversine + Vincenty {}

impl Haversine for f64 {
    fn haversine(a_lat: &[Self], a_lon: &[Self], b_lat: &[Self], b_lon: &[Self], result: &mut [Self]) -> Option<()> {
        let coordinate_count = a_lat.len();
        if a_lon.len() != coordinate_count
            || b_lat.len() != coordinate_count
            || b_lon.len() != coordinate_count
            || result.len() != coordinate_count
        {
            return None;
        }
        unsafe {
            nk_haversine_f64(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                coordinate_count,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Vincenty for f64 {
    fn vincenty(a_lat: &[Self], a_lon: &[Self], b_lat: &[Self], b_lon: &[Self], result: &mut [Self]) -> Option<()> {
        let coordinate_count = a_lat.len();
        if a_lon.len() != coordinate_count
            || b_lat.len() != coordinate_count
            || b_lon.len() != coordinate_count
            || result.len() != coordinate_count
        {
            return None;
        }
        unsafe {
            nk_vincenty_f64(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                coordinate_count,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Geospatial for f64 {}

impl Haversine for f32 {
    fn haversine(a_lat: &[Self], a_lon: &[Self], b_lat: &[Self], b_lon: &[Self], result: &mut [Self]) -> Option<()> {
        let coordinate_count = a_lat.len();
        if a_lon.len() != coordinate_count
            || b_lat.len() != coordinate_count
            || b_lon.len() != coordinate_count
            || result.len() != coordinate_count
        {
            return None;
        }
        unsafe {
            nk_haversine_f32(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                coordinate_count,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Vincenty for f32 {
    fn vincenty(a_lat: &[Self], a_lon: &[Self], b_lat: &[Self], b_lon: &[Self], result: &mut [Self]) -> Option<()> {
        let coordinate_count = a_lat.len();
        if a_lon.len() != coordinate_count
            || b_lat.len() != coordinate_count
            || b_lon.len() != coordinate_count
            || result.len() != coordinate_count
        {
            return None;
        }
        unsafe {
            nk_vincenty_f32(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                coordinate_count,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Geospatial for f32 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, FloatLike, TestableType};

    fn check_haversine<Scalar>(
        a_lat_deg: f64,
        a_lon_deg: f64,
        b_lat_deg: f64,
        b_lon_deg: f64,
        expected_meters: f64,
        tolerance: f64,
    ) where
        Scalar: FloatLike + TestableType + Haversine,
    {
        let a_lat = [Scalar::from_f32(a_lat_deg.to_radians() as f32)];
        let a_lon = [Scalar::from_f32(a_lon_deg.to_radians() as f32)];
        let b_lat = [Scalar::from_f32(b_lat_deg.to_radians() as f32)];
        let b_lon = [Scalar::from_f32(b_lon_deg.to_radians() as f32)];
        let mut result = [Scalar::zero()];
        Scalar::haversine(&a_lat, &a_lon, &b_lat, &b_lon, &mut result).unwrap();
        assert_close(
            result[0].to_f64(),
            expected_meters,
            tolerance,
            0.0,
            &format!("haversine<{}>", core::any::type_name::<Scalar>()),
        );
    }

    fn check_vincenty<Scalar>(
        a_lat_deg: f64,
        a_lon_deg: f64,
        b_lat_deg: f64,
        b_lon_deg: f64,
        expected_meters: f64,
        tolerance: f64,
    ) where
        Scalar: FloatLike + TestableType + Vincenty,
    {
        let a_lat = [Scalar::from_f32(a_lat_deg.to_radians() as f32)];
        let a_lon = [Scalar::from_f32(a_lon_deg.to_radians() as f32)];
        let b_lat = [Scalar::from_f32(b_lat_deg.to_radians() as f32)];
        let b_lon = [Scalar::from_f32(b_lon_deg.to_radians() as f32)];
        let mut result = [Scalar::zero()];
        Scalar::vincenty(&a_lat, &a_lon, &b_lat, &b_lon, &mut result).unwrap();
        assert_close(
            result[0].to_f64(),
            expected_meters,
            tolerance,
            0.0,
            &format!("vincenty<{}>", core::any::type_name::<Scalar>()),
        );
    }

    #[test]
    fn geospatial() {
        // New York → Los Angeles
        // Haversine uses NK_EARTH_MEDIATORIAL_RADIUS (6,335,439m) → ~3,913,778m
        // Vincenty uses the WGS-84 ellipsoid → ~3,944,422m
        let hav_expected = 3_914_000.0;
        let vin_expected = 3_944_000.0;
        check_haversine::<f64>(40.7128, -74.0060, 34.0522, -118.2437, hav_expected, 20_000.0);
        check_haversine::<f32>(40.7128, -74.0060, 34.0522, -118.2437, hav_expected, 50_000.0);
        check_vincenty::<f64>(40.7128, -74.0060, 34.0522, -118.2437, vin_expected, 20_000.0);
        check_vincenty::<f32>(40.7128, -74.0060, 34.0522, -118.2437, vin_expected, 50_000.0);
    }
}
