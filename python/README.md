# NumKong for Python

NumKong for Python is the main high-level SDK in the project.
It targets the gap between `numpy` and low-level native kernels: you keep buffer-protocol interoperability and shape-aware outputs, but you stop giving up mixed precision, widened accumulators, packed reuse, and backend-specific optimizations every time you leave `float64`.
It combines NumPy-friendly buffers with native mixed-precision kernels, zero-copy tensor views, packed and symmetric matrix operations, sparse helpers, geometric mesh alignment, and MaxSim.
The API feels NumPy-shaped with familiar scalar, batched, and all-pairs entrypoints, while `Tensor` keeps shape, dtype, and strides visible through a memoryview-backed container.
Low-precision dtypes (BFloat16, Float8, Float6, packed bits) flow through the same API, and dense, packed, and symmetric kernels release the GIL around native work.

## Ecosystem Comparison

| Feature                         | NumKong                                                                                                                      | [NumPy](https://github.com/numpy/numpy)/[SciPy](https://github.com/scipy/scipy) | [PyTorch](https://github.com/pytorch/pytorch)                               |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Operation families              | dots, distances, binary, probability, geospatial, curved, mesh, sparse, MaxSim, elementwise, reductions, cast, trig          | dots, distances, elementwise, reductions, some probability via `cdist`          | dots, distances, elementwise, reductions                                    |
| Precision                       | BFloat16 through sub-byte — Float8, Float6, Int4, packed bits; automatic widening; Kahan summation; 0 ULP in Float32/Float64 | Float16, partial BFloat16; no auto-widening; standard accuracy                  | Float16, BFloat16, partial Float8; explicit AMP required; standard accuracy |
| Runtime SIMD dispatch           | auto-selects best ISA per-thread at runtime on x86, ARM, RISC-V                                                              | compile-time only                                                               | CPU: compile-time; CUDA: runtime                                            |
| Packed matrix, GEMM-like        | pack once, reuse across query batches                                                                                        | `np.dot`/`@` — no persistent packing                                            | `torch.mm` — no persistent distance-oriented packing                        |
| Symmetric kernels, SYRK-like    | skip duplicate pairs, up to 2x speedup for self-distance                                                                     | `pdist` computes one triangle; `cdist` recomputes both                          | `X @ X.T` recomputes both triangles                                         |
| Output parameter `out=`         | Yes — all major entrypoints                                                                                                  | Yes — most `ufunc`s and functions; SciPy: some functions only                   | Yes for `torch.mm`, `torch.matmul`; No for `torch.cdist`                    |
| Fast CPython calling convention | Yes — direct `METH_FASTCALL`                                                                                                 | Yes — `vectorcall` in 2.0+                                                      | No — tensor dispatch overhead                                               |
| GIL release                     | batched, packed, and symmetric kernels                                                                                       | some ops only                                                                   | most ops                                                                    |



## Quickstart

```python
import numpy as np
import numkong as nk

a, b = np.random.randn(1536).astype(np.float32), np.random.randn(1536).astype(np.float32)
dot = nk.dot(a, b)  # widened accumulation, not same-dtype
print(dot)
```

## Installation

From PyPI:

```sh
python -m pip install numkong
```

From a local checkout:

```sh
python -m pip install .
```

Quick runtime check:

```sh
python -c "import numkong as nk; print(nk.get_capabilities())"
```

## Wheel Compatibility and Building from Source

Pre-built wheels are available on PyPI for Linux (x86_64, aarch64, riscv64, plus i686, ppc64le, s390x), macOS (x86_64, arm64), and Windows (AMD64, ARM64).
Python 3.9 through 3.14 is supported, including free-threading variants (3.13t, 3.14t).
Every wheel is built with `NK_DYNAMIC_DISPATCH=1`, so a single wheel covers all CPU generations on a given architecture.

When building from source, the compiler requirements depend on the platform.
On macOS x86 only AVX2 is available; on macOS ARM NEON is always present, but SME requires Apple M4+ with Xcode 16+ (AppleClang 16+).
RISC-V builds require Clang and LLD because GCC lacks `zvfh`, `zvfbfwma`, and `zvbb` support.
On Windows, MSVC 19.44+ (Visual Studio 2022 17.14+) is recommended for full AVX-512 with FP16/BF16/VNNI.
Build parallelism is controlled by `NK_BUILD_PARALLEL`, which defaults to `min(cpu_count, 4)` and should be lowered in memory-constrained containers.
There is no OpenMP dependency.
Python-side parallelism uses `concurrent.futures` with GIL-free kernels.

```sh
NK_BUILD_PARALLEL=2 pip install . --no-build-isolation
```

## Dot Products

Dot products are their own family because storage type, conjugation rules, and output widening matter.

```python
import numpy as np
import numkong as nk

a = (np.random.randn(256) + 1j * np.random.randn(256)).astype(np.complex64)
b = (np.random.randn(256) + 1j * np.random.randn(256)).astype(np.complex64)

dot = nk.dot(a, b)   # numpy.dot(a, b)
vdot = nk.vdot(a, b) # numpy.vdot(a, b)

print(dot, vdot)
```

Real low-precision inputs can also be routed through explicit dtype tags when the storage buffer itself is raw bytes.

## Dense Distances

The dense distance entrypoints cover `sqeuclidean`, `euclidean`, and `angular`.
The first important difference from NumPy or SciPy is that the accumulator policy is not forced to match the storage dtype.

```python
import numpy as np
import numkong as nk

a = np.random.randn(768).astype(np.float16)
b = np.random.randn(768).astype(np.float16)

sqeuclidean = nk.sqeuclidean(a, b)
euclidean = nk.euclidean(a, b)
angular = nk.angular(a, b)
```

For `float16`, a naive same-dtype implementation is exactly the kind of path that loses precision or widens too late.
NumKong's API makes the widening policy part of the kernel contract.

### Output Control: `out=`, `dtype=`, and `out_dtype=`

Most distance and dot-product entrypoints accept `out=`, `dtype=`, and `out_dtype=` keyword arguments.
Passing them avoids dynamic memory allocations for temporary objects.

```python
import numpy as np
import numkong as nk

queries = np.random.randn(100, 768).astype(np.float32)
database = np.random.randn(100, 768).astype(np.float32)

# Pre-allocated output with out=
out = nk.zeros((100,), dtype="float32")
nk.sqeuclidean(queries, database[:100], out=out)  # writes in-place, returns None

# Explicit input dtype for raw byte buffers
raw = np.frombuffer(some_bytes, dtype=np.uint16)
nk.dot(raw, raw, dtype=nk.bfloat16)  # reinterpret uint16 as bf16

# Output dtype override
nk.euclidean(queries[0], database[0], out_dtype="float32")  # accumulate in f64, downcast result
```

When `out=` is provided, the function writes results in-place and returns `None`.
The `out` array must be pre-allocated with the correct shape and a supported dtype.
For custom float types (`bfloat16`, `float16`, `float8_e4m3`, `float8_e5m2`, `float6_e2m3`, `float6_e3m2`), type objects are preferred over strings — they are faster to dispatch and provide IDE autocomplete:

```python
nk.dot(a, b, dtype=nk.bfloat16) # works faster
nk.dot(a, b, dtype="bfloat16")  # works a bit slower
```

## Set Similarity

Packed-binary metrics operate on packed bits.
That is why the right NumPy equivalent uses `np.packbits`, not `bool` arrays fed to scalar Python code.

```python
import numpy as np
import numkong as nk

a_bits = np.random.randint(0, 2, size=256, dtype=np.uint8)
b_bits = np.random.randint(0, 2, size=256, dtype=np.uint8)
a, b = np.packbits(a_bits), np.packbits(b_bits)

hamming = nk.hamming(a, b, dtype="uint1")
jaccard = nk.jaccard(a, b, dtype="uint1")
```

Integer set Jaccard works on sorted ascending arrays of integer identifiers.
Both inputs must be sorted in ascending order for correct results.

```python
set_a = np.array([1, 3, 5, 7, 9], dtype=np.uint32)  # must be sorted ascending
set_b = np.array([3, 5, 8, 9, 10], dtype=np.uint32)  # must be sorted ascending
jaccard_sets = nk.jaccard(set_a, set_b) # |A ∩ B| / |A ∪ B|
assert 0.0 < jaccard_sets < 1.0, "|A ∩ B| / |A ∪ B| should be in (0, 1)"
```

## Probability Metrics

Probability divergences deserve their own section because they are not just "one more distance".

```python
import numpy as np
import numkong as nk

p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
q = np.array([0.1, 0.3, 0.6], dtype=np.float32)

kl_forward, kl_reverse = nk.kullbackleibler(p, q), nk.kullbackleibler(q, p)
assert kl_forward != kl_reverse, "KLD is asymmetric"

js_forward, js_reverse = nk.jensenshannon(p, q), nk.jensenshannon(q, p)
np.testing.assert_allclose(js_forward, js_reverse, atol=1e-6)  # JSD is symmetric
```

## Geospatial Metrics

Geospatial kernels take four coordinate arrays.
Inputs are in radians.
Outputs are in meters.

```python
import numpy as np
import numkong as nk

# Statue of Liberty (40.6892°N, 74.0445°W) → Big Ben (51.5007°N, 0.1246°W)
liberty_lat, liberty_lon = np.array([0.7101605100], dtype=np.float64), np.array([-1.2923203180], dtype=np.float64)
big_ben_lat, big_ben_lon = np.array([0.8988567821], dtype=np.float64), np.array([-0.0021746802], dtype=np.float64)

vincenty = nk.vincenty(liberty_lat, liberty_lon, big_ben_lat, big_ben_lon)    # ≈ 5,589,857 m (ellipsoidal, baseline)
haversine = nk.haversine(liberty_lat, liberty_lon, big_ben_lat, big_ben_lon)  # ≈ 5,543,723 m (spherical, ~46 km less)

# Vincenty in f32 — drifts ~2 m from f64
liberty_lat32 = liberty_lat.astype(np.float32)
liberty_lon32 = liberty_lon.astype(np.float32)
big_ben_lat32 = big_ben_lat.astype(np.float32)
big_ben_lon32 = big_ben_lon.astype(np.float32)
vincenty_f32 = nk.vincenty(liberty_lat32, liberty_lon32, big_ben_lat32, big_ben_lon32)  # ≈ 5,589,859 m (+2 m drift)
```

## Curved Metrics

Curved-space kernels use an extra metric tensor or inverse covariance and should not be mixed into the Euclidean section.

```python
import numpy as np
import numkong as nk

# Complex bilinear form: aᴴ M b
a = (np.ones(16) + 1j * np.zeros(16)).astype(np.complex64)
b = (np.zeros(16) + 1j * np.ones(16)).astype(np.complex64)
m = np.eye(16, dtype=np.complex64)
bilinear = nk.bilinear(a, b, m)

# Real Mahalanobis distance: √((a−b)ᵀ M⁻¹ (a−b))
x = np.ones(32, dtype=np.float32)
y = np.full(32, 2.0, dtype=np.float32)
inv_cov = np.eye(32, dtype=np.float32)
mahalanobis = nk.mahalanobis(x, y, inv_cov)
```

## Scalar Types and Low-Precision Formats

NumKong exposes two different low-precision stories in Python.
It exposes Python scalar objects for a few formats.
And it exposes tensor dtypes for the broader buffer-oriented path.

The six scalar types have stable payload sizes even though Python object headers are not:

| Type             | Bits   | Bytes | Range     | Inf | NaN |
| ---------------- | ------ | ----- | --------- | --- | --- |
| `nk.float16`     | 1+5+10 | 2     | ±65504    | yes | yes |
| `nk.bfloat16`    | 1+8+7  | 2     | ±3.4×10³⁸ | yes | yes |
| `nk.float8_e4m3` | 1+4+3  | 1     | ±448      | no  | yes |
| `nk.float8_e5m2` | 1+5+2  | 1     | ±57344    | yes | yes |
| `nk.float6_e2m3` | 1+2+3  | 1     | ±7.5      | no  | no  |
| `nk.float6_e3m2` | 1+3+2  | 1     | ±28       | no  | no  |

The Bits column shows sign + exponent + mantissa bit counts.
The Bytes column is the stable payload size; `float8_*` and `float6_*` both store 1 byte because the sub-byte formats are padded to byte alignment.

The full object footprint is interpreter-dependent.
Use `sys.getsizeof(nk.float16(1.0))` if you need the heap footprint of the Python wrapper object itself.
Use `Tensor.itemsize` and `Tensor.nbytes` for the stable payload sizes of array storage.

`ml_dtypes` matters here because NumKong explicitly interoperates with the formats that NumPy still does not model well.
The test suite compares `bfloat16`, `float8_e4m3`, `float8_e5m2`, `float6_e2m3`, and `float6_e3m2` behavior against `ml_dtypes` where that comparison is meaningful.

Promotion is intentional.
Mixed exotic floats are routed through wider compute types rather than pretending a same-width accumulator is good enough.

## ml_dtypes Interoperability

NumKong accepts `ml_dtypes` arrays directly — no `.view(np.uint8)` workaround needed:

```python
import ml_dtypes
a = np.random.randn(100, 768).astype(np.float32).astype(ml_dtypes.bfloat16)
b = np.random.randn(100, 768).astype(np.float32).astype(ml_dtypes.bfloat16)
result = nk.cdist(a, b, "dot")  # just works
```

NumKong scalars also work as NumPy dtype specifiers:

```python
arr = np.array([1.0, 2.0, 3.0], dtype=nk.bfloat16)
float(arr[0])  # → 1.0
```

Type name mapping between the two libraries:

| ml_dtypes                      | NumKong                      | Status                                    |
| ------------------------------ | ---------------------------- | ----------------------------------------- |
| `ml_dtypes.bfloat16`           | `nk.bfloat16` / `"bfloat16"` | Identical format                          |
| `ml_dtypes.float8_e4m3`        | `nk.float8_e4m3` / `"e4m3"`  | Identical (IEEE E4M3)                     |
| `ml_dtypes.float8_e4m3fn`      | `nk.float8_e4m3` / `"e4m3"`  | Identical (E4M3FN = no inf)               |
| `ml_dtypes.float8_e5m2`        | `nk.float8_e5m2` / `"e5m2"`  | Identical format                          |
| `ml_dtypes.float6_e2m3fn`      | `nk.float6_e2m3` / `"e2m3"`  | Identical (MX E2M3)                       |
| `ml_dtypes.float6_e3m2fn`      | `nk.float6_e3m2` / `"e3m2"`  | Identical (MX E3M2)                       |
| `ml_dtypes.float8_e4m3fnuz`    | —                            | Rejected: different bias, NaN, and zero   |
| `ml_dtypes.float8_e5m2fnuz`    | —                            | Rejected: different NaN and zero encoding |
| `ml_dtypes.float8_e4m3b11fnuz` | —                            | Rejected: bias=11, incompatible encoding  |
| `ml_dtypes.float8_e8m0fnu`     | `"float8_e8m0"` / `"e8m0"`   | MX block scale byte (see `ScaledTensor`)  |
| `ml_dtypes.float8_e3m4`        | —                            | Not supported: no NumKong kernel          |
| `ml_dtypes.float4_e2m1fn`      | `"float4_e2m1"` / `"e2m1"`   | NVFP4/MXFP4 element (see `ScaledTensor`)  |
| `ml_dtypes.int4`               | `"int4"`                     | Compatible via buffer protocol            |
| `ml_dtypes.uint4`              | `"uint4"`                    | Compatible via buffer protocol            |
| `ml_dtypes.int2`               | —                            | Not supported                             |
| `ml_dtypes.uint2`              | —                            | Not supported                             |

## Block-Scaled Formats (OCP MX & NVIDIA NVFP4)

Block-scaled formats group elements into fixed-size blocks, each carrying its own scale byte.

| Format         | Block | Element | Scale   | Global    |
| -------------- | ----- | ------- | ------- | --------- |
| `"nvfp4"`      | 16    | `e2m1`  | `ue4m3` | `float32` |
| `"mxfp4"`      | 32    | `e2m1`  | `ue8m0` | —         |
| `"mxfp6_e2m3"` | 32    | `e2m3`  | `ue8m0` | —         |
| `"mxfp6_e3m2"` | 32    | `e3m2`  | `ue8m0` | —         |
| `"mxfp8_e4m3"` | 32    | `e4m3`  | `ue8m0` | —         |
| `"mxfp8_e5m2"` | 32    | `e5m2`  | `ue8m0` | —         |
| `"mxint8"`     | 32    | `i8`    | `ue8m0` | —         |

Quantization happens through the same `astype` verb used for dense casts.
Passing a block-scaled dtype name to a dense `Tensor.astype(...)` returns a `ScaledTensor`; calling `astype(...)` on a `ScaledTensor` with a dense dtype materializes it back.
Quantization is always along the __last axis__, which must be a whole multiple of the block size.

A `ScaledTensor` exposes plain read attributes, but no getter/setter methods:

| Attribute       | Type             | Meaning                                                                 |
| --------------- | ---------------- | ----------------------------------------------------------------------- |
| `.elements`     | `Tensor`         | Packed sub-byte element bytes (e.g. `e2m1`), last axis in storage bytes |
| `.block_scales` | `Tensor`         | One scale byte per block (`ue4m3` / `ue8m0`), last axis = `n / block`   |
| `.tensor_scale` | `float` / `None` | NVFP4 per-tensor multiplier; `None` for the MX family                   |
| `.block_size`   | `int`            | Elements per block (16 for NVFP4, 32 for MX)                            |
| `.dtype`        | `str`            | Composite format name (e.g. `"nvfp4"`)                                  |
| `.shape`        | `tuple`          | Logical (dense) shape                                                   |

```python
import numkong, numpy as np

x = np.random.randn(64).astype(np.float32)
t = numkong.Tensor(x)

# Encode → ScaledTensor. For NVFP4 the per-tensor scale is auto-derived and surfaced as `.tensor_scale`.
q = t.astype("nvfp4")
assert q.dtype == "nvfp4" and q.block_size == 16
assert isinstance(q.tensor_scale, float)          # NVFP4 carries a per-tensor float
assert q.block_scales.shape == (4,)               # 64 / 16 blocks

# Decode / materialize back to a dense Tensor of any standard dtype.
out = q.astype("float32")                          # also "float16", "bf16", ...

# MX formats carry no per-tensor scale.
qm = t.astype("mxfp4")
assert qm.tensor_scale is None and qm.block_size == 32

# Block-aligned slicing returns a ScaledTensor view sharing storage; both children slice in lockstep.
X = numkong.Tensor(np.random.randn(3, 64).astype(np.float32)).astype("nvfp4")
row = X[1]            # → ScaledTensor of shape (64,)
cols = X[:, 16:48]    # → ScaledTensor of shape (3, 32); start/stop must be block-aligned
materialized = row.astype("float32")
```

The `.elements` and `.block_scales` tensors are DLPack-exportable for zero-copy hand-off to PyTorch,
CuPy, and friends — the `e2m1`/`e8m0`/`ue4m3` bytes exchange as `kDLFloat4_e2m1fn` /
`kDLFloat8_e8m0fnu` / `kDLFloat8_e4m3fn`:

```python
import torch
scales = torch.from_dlpack(q.block_scales)   # zero-copy view of the scale bytes
```

## Tensor Objects and Buffer Interop

`Tensor` is a memoryview-backed object with NumPy-like metadata.
It is the central container for strided views, transpose, reshape, flatten, and axis reductions.

```python
import numpy as np
import numkong as nk

t = nk.Tensor(np.arange(12, dtype=np.float32).reshape(3, 4))

print(t.shape, t.dtype, t.ndim, t.strides, t.itemsize, t.nbytes)
print(np.asarray(t))      # zero-copy array view when layout allows it
print(t.T.shape)          # transposed Tensor view
print(t.reshape(2, 6).shape)
print(t.flatten().shape)

# Slicing — row, column, and scalar access
row0 = t[0, :]            # first row, shape (4,)
col2 = t[:, 2]            # third column, strided view, shape (3,)
val  = t[1, 2]            # scalar element access → 6.0

# Reductions compose with sliced views
idx = col2.argmin()        # index of the minimum in the third column
mn, i0, mx, i1 = col2.minmax()
```

The important layout rules are:

- `Tensor` preserves shape and byte strides.
- Transpose and slicing can produce non-contiguous views.
- General reductions accept those views.
- Matrix-style packed kernels require row-contiguous left operands.
- Packed and symmetric outputs require C-contiguous `out` buffers.

### Memory Layout Requirements

| API family                                       | Input requirement                                                                                  | Output requirement                                                        |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Dense distances (`dot`, `euclidean`, etc.)       | Rows must be contiguous (`strides[last] <= itemsize`). Strided rows (sliced columns) are rejected. | `out=` can have any stride along dim 0, but inner dim must be contiguous. |
| `cdist`                                          | Same as dense distances                                                                            | `out=` must be rank-2 with shape `(a.count, b.count)`                     |
| Elementwise (`scale`, `blend`, `fma`)            | Arbitrary strides (strided views are supported)                                                    | `out=` must match input shape; strides are preserved                      |
| Packed matrix (`dots_packed`)                    | Left operand: rank-2, contiguous rows, no negative strides                                         | Output: C-contiguous with expected dtype                                  |
| Symmetric (`dots_symmetric`)                     | Contiguous rows                                                                                    | `out=`: C-contiguous square matrix                                        |
| Tensor reductions (`sum`, `min`, `argmin`, etc.) | Arbitrary strides (strided views supported)                                                        | N/A (returns scalar or reduced tensor)                                    |

## All-Pairs APIs and cdist

`cdist` is the NumPy/SciPy-shaped all-pairs entrypoint.
It handles rectangular matrix pairs and symmetric self-distance cases.

```python
import numpy as np
import numkong as nk

queries = np.random.randn(100, 768).astype(np.float32)
database = np.random.randn(10_000, 768).astype(np.float32)

pairwise = nk.angular(queries, database[:100])             # rectangular broadcasted pairwise call
all_pairs = nk.cdist(queries, database, metric="angular")  # scipy.spatial.distance.cdist analogue

assert np.asarray(pairwise).shape == (100, 100)
assert np.asarray(all_pairs).shape == (100, 10_000)
```

The intended large-scale parallel model for packed and symmetric kernels is external partitioning with row ranges, not a hidden `threads=` argument.

## Elementwise Operations

Elementwise arithmetic and fused operations are their own family.
They share the tensor infrastructure but should not be collapsed into the reduction or matrix sections.

```python
import numpy as np
import numkong as nk

a = np.arange(8, dtype=np.float32)
b = np.arange(8, dtype=np.float32)[::-1].copy()

scaled = nk.scale(a, alpha=2.0, beta=1.0)     # 2 * a + 1
blended = nk.blend(a, b, alpha=0.25, beta=0.75)
fused = nk.fma(a, b, a, alpha=1.0, beta=1.0)  # a * b + a

assert np.asarray(scaled).shape == (8,)
assert np.asarray(fused).shape == (8,)
```

## Moments Reductions

Moments reductions return `(sum, sum_of_squares)`.
The key property is that NumKong does not force you into same-storage accumulation.

```python
import numpy as np
import numkong as nk

x = np.full(4096, 255, dtype=np.uint8)

nk_sum, nk_sumsq = nk.moments(nk.Tensor(x))
naive_sum = np.sum(x, dtype=np.uint8)      # overflows immediately
naive_sumsq = np.sum(x * x, dtype=np.uint8) # also overflows

print(nk_sum, nk_sumsq, naive_sum, naive_sumsq)
assert nk_sum > int(naive_sum)
assert nk_sumsq > int(naive_sumsq)
```

Same-width accumulation is a bad default for low-precision storage.

## Min/Max Reductions

Min/max reductions are in a separate section because they cover strided reduction cases.
NumKong provides SIMD-accelerated strided reductions that are not common in other libraries.

```python
import numpy as np
import numkong as nk

matrix = nk.Tensor(np.array([
    [ 3.0,  0.0, 7.0],
    [ 1.0,  2.0, 5.0],
    [ 4.0, -1.0, 6.0],
], dtype=np.float32))

second_column = matrix[:, 1]  # strided view into a row-major Nx3 tensor

idx = second_column.argmin()
mn, i0, mx, i1 = second_column.minmax()

assert idx == 2
assert int(i0) == 2
assert float(np.asarray(mn)) == -1.0
```

Fresh measurement for the rewritten docs:
on an Apple M2 Pro, `np.argmin(matrix[:, 1])` on a row-major `2,000,000 x 3` `float32` array took about `1.63 ms` median.
The equivalent NumKong `Tensor(... )[:, 1].argmin()` took about `0.67 ms` median.
That is about `2.45x` faster on this strided reduction case.

## Sparse Operations and Intersections

Sparse helpers cover both sorted-index intersections and weighted sparse dot products.

```python
import numpy as np
import numkong as nk

idx_a, idx_b = np.array([1, 3, 5, 7], dtype=np.uint32), np.array([3, 4, 5, 8], dtype=np.uint32)
intersection_size = nk.intersect(idx_a, idx_b) # len(np.intersect1d(idx_a, idx_b))
assert intersection_size == 2, "indices 3 and 5"

val_a, val_b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
sparse_dot = nk.sparse_dot(idx_a, val_a, idx_b, val_b)
assert sparse_dot > 0, "weighted dot over shared indices"
```

## Packed Matrix Kernels for GEMM-Like Workloads

Packed matrix kernels are the right tool when the right-hand side is reused across many query batches.
This is the `GEMM`-like story.

```python
import numpy as np
import numkong as nk

left = np.random.randn(128, 768).astype(np.float32)
right = np.random.randn(10_000, 768).astype(np.float32)

right_packed = nk.dots_pack(right, dtype="float32")  # pack once, reuse many times
scores = nk.dots_packed(left, right_packed)          # equivalent to left @ right.T

assert scores.shape == (128, 10_000)
assert right_packed.nbytes == nk.PackedMatrix.packed_size(10_000, 768, dtype="float32")
```

Important runtime rules from the current implementation:

- `a` must be rank-2
- `a` must have contiguous rows
- negative strides are rejected for these matrix kernels
- `out`, when provided, must be C-contiguous with the expected dtype
- `start_row` and `end_row` split the left operand rows

The arithmetic advantages are:

- one-time packing of `B`
- one-time internal layout conversion and depth padding
- norm reuse for `angulars_packed` and `euclideans_packed`
- no repeated scan of the original right-hand-side layout

Packing itself does not require aligned caller buffers.
The packed object owns its internal payload and handles the layout under the hood.

`Tensor @ PackedMatrix` is also supported and maps to the same packed dot-product path.

## Symmetric Kernels for SYRK-Like Workloads

Symmetric kernels solve a different problem from packed cross-matrix kernels.
They compute self-similarity or self-distance matrices.
This is the `SYRK`-like story.

```python
import numpy as np
import numkong as nk

vectors = np.random.randn(1024, 768).astype(np.float32)
out = nk.zeros((1024, 1024), dtype="float64")

nk.dots_symmetric(vectors, out=out, start_row=0, end_row=256)
nk.dots_symmetric(vectors, out=out, start_row=256, end_row=512)

assert out.shape == (1024, 1024)
```

This family has different economics from packed `GEMM`-like work.
It avoids duplicate `(i, j)` and `(j, i)` evaluations.
It is naturally partitioned by row windows of one square output.

`angulars_symmetric` and `euclideans_symmetric` also benefit from reuse of dot-product-derived work inside the symmetric sweep.
That is why these APIs are faster than a nested Python loop over `angular(a[i], a[j])`.

## Geometric Mesh Alignment

Mesh alignment returns a structured result object.
The current implementation exposes `rotation`, `scale`, `rmsd`, `a_centroid`, and `b_centroid`.

```python
import numpy as np
import numkong as nk

source = np.array(
    [[0.0, 0.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float32,
)

result = nk.kabsch(source, source.copy())
assert np.asarray(result.rotation).shape == (3, 3)
assert float(np.asarray(result.scale)) == 1.0

# Umeyama with known 2x scaling
target = source * 2.0
result = nk.umeyama(source, target)
assert float(np.asarray(result.rmsd)) < 1e-6, "umeyama should recover exact alignment"
assert abs(float(np.asarray(result.scale)) - 2.0) < 0.01, "umeyama should recover 2x scale"
```

That field-level check is the right style for this API family.
It tells the reader exactly what the result object owns.

## MaxSim and ColBERT-Style Late Interaction

MaxSim is the late-interaction primitive used by systems such as [ColBERT](https://arxiv.org/abs/2004.12832).
It is not generic matrix multiplication.

```python
import numpy as np
import numkong as nk

queries = np.random.randn(32, 128).astype(np.float32)
documents = np.random.randn(192, 128).astype(np.float32)

q = nk.maxsim_pack(queries, dtype="float32")
d = nk.maxsim_pack(documents, dtype="float32")
score = nk.maxsim_packed(q, d)

assert np.isfinite(score)
assert q.nbytes == nk.MaxSimPackedMatrix.packed_size(32, 128, dtype="float32")
```

## Capabilities, GIL Behavior, and Parallel Partitioning

Capability detection is explicit:

```python
import numkong as nk

caps = nk.get_capabilities()
print({k: v for k, v in caps.items() if v})
```

The current implementation releases the GIL around the native dense metric calls and around the packed and symmetric matrix kernels.
The repository also has threading tests for packed and symmetric row-range partitioning.

`GEMM`-like packed work and `SYRK`-like symmetric work should be documented differently:

```python
import concurrent.futures
import numpy as np
import numkong as nk

left = np.random.randn(4096, 768).astype(np.float32)
right = np.random.randn(8192, 768).astype(np.float32)
packed = nk.dots_pack(right, dtype="float32")
out = nk.zeros((4096, 8192), dtype="float64")  # out must be pre-allocated with correct shape and dtype

def packed_chunk(start, end):
    nk.dots_packed(left, packed, out=out, start_row=start, end_row=end) # split left rows against one shared packed RHS

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for start in range(0, 4096, 1024):
        pool.submit(packed_chunk, start, min(start + 1024, 4096))
```

```python
import concurrent.futures
import numpy as np
import numkong as nk

vectors = np.random.randn(4096, 768).astype(np.float32)
out = nk.zeros((4096, 4096), dtype="float64")  # out must be pre-allocated with correct shape and dtype

def symmetric_chunk(start, end):
    nk.dots_symmetric(vectors, out=out, start_row=start, end_row=end) # split row windows of one square output

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for start in range(0, 4096, 1024):
        pool.submit(symmetric_chunk, start, min(start + 1024, 4096))
```

OpenMP and other native schedulers still matter in lower layers.
For Python, the intended user-facing story is external partitioning around the GIL-free kernels you actually use.

## Addressing External Memory

NumKong implements the Python buffer protocol for zero-copy interop with NumPy, PyTorch, and other buffer-aware libraries.
Two additional primitives cover pointer-level workflows: `data_ptr` reads the integer address out of any `Tensor`, and `from_pointer()` wraps any integer address back into one.

`data_ptr` returns the raw address, suitable for passing into ctypes, CUDA, or any FFI boundary.
`from_pointer(address, shape, dtype, *, strides=None, owner=None)` creates a non-owning `Tensor` view.
The optional `owner` keeps the source object alive for the lifetime of the view.

```python
import numpy as np
import numkong as nk

# Round-trip through an integer address
matrix = nk.zeros((3, 4), dtype='float32')
address = matrix.data_ptr
matrix_view = nk.from_pointer(address, (3, 4), 'float32', owner=matrix)

# Wrap a NumPy array with zero copies
embeddings = np.random.randn(1024).astype(np.float32)
embeddings_view = nk.from_pointer(embeddings.ctypes.data, (1024,), 'float32', owner=embeddings)
nk.dot(embeddings, embeddings_view)  # same underlying data
```

PyTorch tensors already implement the buffer protocol, so most functions accept them directly.
For explicit pointer-level control, or to go the other direction, the same primitives apply:

```python
import torch

query = torch.randn(512)
nk.dot(query, query)  # buffer protocol, zero copy

# Explicit pointer wrap
query_view = nk.from_pointer(query.data_ptr(), tuple(query.shape), 'float32', owner=query)

# NumKong → PyTorch: 1D via buffer protocol, N-D via numpy bridge
flat = torch.frombuffer(memoryview(nk_tensor), dtype=torch.float32)
shaped = torch.as_tensor(np.asarray(nk_tensor))
```

CUDA unified memory, pinned buffers, and mmap'd files all work the same way — any CPU-accessible pointer is valid.

```python
import ctypes, mmap

# CUDA unified memory (ensure CPU accessibility first)
cudart = ctypes.CDLL("libcudart.so")
unified_ptr = ctypes.c_void_p()
cudart.cudaMallocManaged(ctypes.byref(unified_ptr), 4096, 1)
cudart.cudaDeviceSynchronize()
unified = nk.from_pointer(unified_ptr.value, (1024,), 'float32')

# Memory-mapped file
with open("data.bin", "r+b") as f:
    mapping = mmap.mmap(f.fileno(), 0)
    mapped = nk.from_pointer(ctypes.addressof(
        ctypes.c_char.from_buffer(mapping)),
        (1024,), 'float32', owner=mapping)
```
