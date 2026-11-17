//! Test-only helpers shared by every module's tests: the run's seed and streams, per-type
//! tolerances, closeness checks, and the problem sizes the batched suites sweep.
//!
//! File: rust/testing.rs
//! Author: Ash Vardanian
//! Date: November 16, 2026

#![allow(dead_code, reason = "the Shared Testing region is the same in every sibling library, used or not")]

use crate::{
    tensor::Error,
    types::{FloatLike, StorageElement, bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2},
};

/// The variable [`seed`] reads, the one the C and C++ suites read.
const SEED_VARIABLE: &str = "NUMKONG_SEED";

// region Shared Testing
/// Announces a skipped test on stderr, so a run shows what it left out rather than passing in
/// silence.
pub(crate) fn skipped(reason: &str) {
    eprintln!("skipped: {reason}");
}

/// The run's seed, read once from the variable [`SEED_VARIABLE`] names: 42 when unset, drawn afresh
/// for `random` and then printed, so the run can be repeated.
pub(crate) fn seed() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SEED.get_or_init(read_seed)
}

/// Parses [`SEED_VARIABLE`] once for [`seed`].
fn read_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};

    const EXPECTED: &str = "an unsigned integer or random";
    let text = std::env::var(SEED_VARIABLE).unwrap_or_default();
    if text.is_empty() {
        return 42;
    }
    if text != "random" {
        let Ok(parsed) = text.parse::<u64>() else {
            panic!("{SEED_VARIABLE}=\"{text}\" does not parse, expected {EXPECTED}");
        };
        return parsed;
    }
    let hasher = std::collections::hash_map::RandomState::new().build_hasher();
    let drawn = hasher.finish();
    eprintln!("{SEED_VARIABLE}={drawn}");
    drawn
}

/// SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits.
pub(crate) const fn split_mix_bits(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The `index`-th stream of the family `name` under `seed`, identical in C, C++, Python and
/// Mojo.
pub(crate) fn stream_key(seed: u64, name: &str, index: u64) -> u64 {
    let mut hashed = 0xCBF2_9CE4_8422_2325_u64;
    for byte in name.bytes() {
        hashed = (hashed ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3);
    }
    split_mix_bits(split_mix_bits(seed ^ hashed).wrapping_add(index))
}

/// The `index`-th stream of the running test, keyed by the name libtest gives its thread.
pub(crate) fn case_stream_key(index: u64) -> u64 {
    let thread = std::thread::current();
    let name = thread.name().unwrap_or("main");
    stream_key(seed(), name, index)
}

/// A directory of its own under the system's temporary one, removed when dropped.
pub(crate) struct ScratchDirectory {
    path: std::path::PathBuf,
}

impl ScratchDirectory {
    /// Creates `<temp>/<name>-<pid>-<nanos>`, so parallel tests never share one.
    pub(crate) fn new(name: &str) -> Self {
        let since = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
        let nanos = since.map_or(0, |elapsed| elapsed.as_nanos());
        let leaf = format!("{name}-{}-{nanos}", std::process::id());
        let path = std::env::temp_dir().join(leaf);
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    /// Where the directory is.
    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Polls `future` to completion on this thread with a no-op waker.
pub(crate) fn block_on<F: core::future::Future>(future: F) -> F::Output {
    let mut future = core::pin::pin!(future);
    let waker = core::task::Waker::noop();
    let mut context = core::task::Context::from_waker(waker);
    loop {
        if let core::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        std::thread::yield_now();
    }
}
// endregion Shared Testing

/// The tolerances a type's results are held to, as `atol + rtol * |expected|`.
pub(crate) trait TestableType: FloatLike {
    /// Base absolute tolerance for this type.
    fn atol() -> f64;
    /// Base relative tolerance for this type.
    fn rtol() -> f64;
}

/// Asserts `actual` lies within `atol + rtol * |expected|` of `expected`, naming `msg` on failure.
#[track_caller]
pub(crate) fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, msg: &str) {
    let tol = atol + rtol * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "{}: expected {} but got {} (atol={}, rtol={}, tol={})",
        msg,
        expected,
        actual,
        atol,
        rtol,
        tol
    );
}

/// Test a two-input metric: convert f32 inputs to Scalar, call `op`, compare to `expected`.
pub(crate) fn check_binary<Scalar, R, F>(a_vals: &[f32], b_vals: &[f32], op: F, expected: f64, label: &str)
where
    Scalar: FloatLike + TestableType,
    R: FloatLike,
    F: FnOnce(&[Scalar], &[Scalar]) -> Result<R, Error>,
{
    let first: Vec<Scalar> = a_vals.iter().map(|&v| Scalar::from_f32(v)).collect();
    let second: Vec<Scalar> = b_vals.iter().map(|&v| Scalar::from_f32(v)).collect();
    let result = op(&first, &second).unwrap().to_f64();
    assert_close(
        result,
        expected,
        Scalar::atol(),
        Scalar::rtol(),
        &format!("{}<{}>", label, core::any::type_name::<Scalar>()),
    );
}

/// One-time SIMD thread setup shared by the batched dots/spatials/sets tests.
static PACKED_TEST_INIT: std::sync::Once = std::sync::Once::new();

/// Configure the current thread's SIMD features once per test process.
pub(crate) fn init_thread() {
    PACKED_TEST_INIT.call_once(|| {
        crate::capabilities::configure_cpu_thread().unwrap();
    });
}

/// Standard `(rows, columns, depth)` problem sizes swept by the batched tests.
pub(crate) const DIMS: &[(usize, usize, usize)] = &[(1, 1, 1), (1, 8, 3), (3, 1, 7), (7, 5, 3), (33, 17, 65)];

/// Lifts a test `depth` to the next valid dimension count, a multiple of the values per byte.
pub(crate) fn align_depth<Scalar: StorageElement>(depth: usize) -> usize {
    let dims_per_value = Scalar::dimensions_per_value();
    depth.div_ceil(dims_per_value) * dims_per_value
}

/// Compare only the upper-triangle elements of two __\[n,n\]__ buffers; symmetric kernels do not
/// write the lower triangle.
pub(crate) fn assert_upper_triangle_eq<X: Copy + PartialEq + core::fmt::Debug>(
    left: &[X],
    right: &[X],
    n: usize,
    tag: &str,
) {
    for i in 0..n {
        for j in i..n {
            let index = i * n + j;
            assert_eq!(left[index], right[index], "{tag}[{i},{j}]");
        }
    }
}

impl TestableType for f32 {
    fn atol() -> f64 {
        1e-4
    }
    fn rtol() -> f64 {
        1e-4
    }
}
impl TestableType for f64 {
    fn atol() -> f64 {
        1e-9
    }
    fn rtol() -> f64 {
        1e-9
    }
}
impl TestableType for f16 {
    fn atol() -> f64 {
        0.05
    }
    fn rtol() -> f64 {
        0.05
    }
}
impl TestableType for bf16 {
    fn atol() -> f64 {
        0.1
    }
    fn rtol() -> f64 {
        0.1
    }
}
impl TestableType for e4m3 {
    fn atol() -> f64 {
        0.5
    }
    fn rtol() -> f64 {
        0.1
    }
}
impl TestableType for e5m2 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.1
    }
}
impl TestableType for e2m3 {
    fn atol() -> f64 {
        0.5
    }
    fn rtol() -> f64 {
        0.1
    }
}
impl TestableType for e3m2 {
    fn atol() -> f64 {
        0.5
    }
    fn rtol() -> f64 {
        0.1
    }
}
impl TestableType for i8 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for u8 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for i32 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for u32 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for i16 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for u16 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for i64 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for u64 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for i4x2 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for u4x2 {
    fn atol() -> f64 {
        1.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
impl TestableType for u1x8 {
    fn atol() -> f64 {
        0.0
    }
    fn rtol() -> f64 {
        0.0
    }
}
