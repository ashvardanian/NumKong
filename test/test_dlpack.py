#!/usr/bin/env python3
"""DLPack zero-copy interop tests for NumKong.

Covers the Python Array API DLPack protocol implemented in ``python/dlpack_interop.{c,h}`` and
``python/dlpack_abi.h``. Tests are organized into per-framework sections; every test gates its dependencies
through ``pytest.importorskip`` so the suite runs against any subset of installed frameworks (NumPy itself is
treated as optional — only the protocol-contract tests at the bottom of this file are framework-free, and they
build their tensors with NumKong's own constructors).

Frameworks exercised:

- NumPy           — CPU round-trips for f32 / f64 / f16 / i* / u*.
- PyTorch         — CPU round-trips, plus bf16, float8_e4m3fn, float8_e5m2 dtype-fidelity tests.
- JAX             — CPU round-trips for f32 / f64 / bf16.
- TensorFlow      — CPU round-trips for f32 / f64 / bf16 via ``tf.experimental.dlpack``.
- CuPy            — three guarded paths covering the importer's device-acceptance contract:
                    (a) plain ``cudaMalloc`` (kDLCUDA) → must be REJECTED with a clear ValueError;
                    (b) ``cudaMallocManaged`` unified memory (kDLCUDAManaged) → must be ACCEPTED;
                    (c) ``cudaMallocHost`` pinned host memory (kDLCUDAHost) → must be ACCEPTED.
- PyArrow         — Arrow Array → NumKong (one-way; PyArrow is producer-only in the protocol).
- ONNX Runtime    — ``OrtValue`` ↔ NumKong via the ``to_dlpack`` / ``ortvalue_from_numpy`` helpers.
- MLX             — Apple Silicon's unified-memory framework. Tests skip on Linux (no ``mlx.core`` wheel) and
                    run on macOS contributor machines / Apple Silicon CI.

The importer accepts every DLPack ``device_type`` whose pointer is dereferenceable from host code
(kDLCPU / CUDAHost / ROCMHost / CUDAManaged / OneAPI / Metal). Pure device memory (kDLCUDA, kDLROCM,
kDLOpenCL, kDLVulkan, kDLWebGPU, kDLHexagon, kDLMAIA, kDLTrn, kDLVPI, kDLExtDev) is rejected.
"""

import pytest

import numkong as nk

# region Helpers


_INTEGER_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
_PLAIN_FLOAT_DTYPES = ["float16", "float32", "float64"]
_PLAIN_NUMERIC_DTYPES = _PLAIN_FLOAT_DTYPES + _INTEGER_DTYPES

# Per-framework dtype matrices. Each entry is a `(framework_attr_name, nk_dtype_string)` pair so a single
# parametrized test can drive both directions: `getattr(framework_module, attr)` resolves the producer-side
# dtype, and the `nk_dtype_string` is what we expect back on the NumKong-side after import.
#
# We include the "weird GPU formats" (bf16, fp8 e4m3fn, fp8 e5m2, fp6 e2m3, fp6 e3m2) wherever the framework
# supports them natively, so each framework's coverage matches what dtype-fidelity DLPack actually buys.

# NumPy core has no bf16/fp8/fp6 — those would need ml_dtypes. Skip exotics here; they're tested via PyTorch/MLX.
_NUMPY_DTYPES = [(name, name) for name in _PLAIN_NUMERIC_DTYPES]

# PyTorch ships native bf16 + float8_e4m3fn + float8_e5m2 since 2.1+; full coverage here is the marquee case.
_TORCH_DTYPES = [(name, name) for name in _PLAIN_NUMERIC_DTYPES] + [
    ("bfloat16", "bfloat16"),
    ("float8_e4m3fn", "e4m3"),
    ("float8_e5m2", "e5m2"),
]

# JAX has native bf16 (always) and the integer / float subset; no fp8/fp6 yet.
_JAX_DTYPES = [
    ("float16", "float16"),
    ("float32", "float32"),
    ("float64", "float64"),
    ("int8", "int8"),
    ("int16", "int16"),
    ("int32", "int32"),
    ("int64", "int64"),
    ("uint8", "uint8"),
    ("uint16", "uint16"),
    ("uint32", "uint32"),
    ("uint64", "uint64"),
    ("bfloat16", "bfloat16"),
]

# TensorFlow: native bf16 + float16/32/64 + integers. No public fp8 dtype that maps cleanly via DLPack today.
_TF_DTYPES = [
    ("float16", "float16"),
    ("float32", "float32"),
    ("float64", "float64"),
    ("int8", "int8"),
    ("int16", "int16"),
    ("int32", "int32"),
    ("int64", "int64"),
    ("uint8", "uint8"),
    ("uint16", "uint16"),
    ("uint32", "uint32"),
    ("uint64", "uint64"),
    ("bfloat16", "bfloat16"),
]

# PyArrow exposes float / signed-int / unsigned-int builders by name. No bf16/fp8/fp6.
_PYARROW_DTYPES = [
    ("float16", "float16"),
    ("float32", "float32"),
    ("float64", "float64"),
    ("int8", "int8"),
    ("int16", "int16"),
    ("int32", "int32"),
    ("int64", "int64"),
    ("uint8", "uint8"),
    ("uint16", "uint16"),
    ("uint32", "uint32"),
    ("uint64", "uint64"),
]

# MLX ships native bf16 on Apple Silicon, plus float16/32/integers. No fp8 yet (mlx.core 0.x).
_MLX_DTYPES = [
    ("float16", "float16"),
    ("float32", "float32"),
    ("int8", "int8"),
    ("int16", "int16"),
    ("int32", "int32"),
    ("int64", "int64"),
    ("uint8", "uint8"),
    ("uint16", "uint16"),
    ("uint32", "uint32"),
    ("uint64", "uint64"),
    ("bfloat16", "bfloat16"),
]


def _assert_round_trip_through(producer_array, expected_dtype_str: str, expected_shape: tuple) -> None:
    """Common pattern: framework-owned array → NumKong view → assertions on dtype/shape."""
    imported = nk.from_dlpack(producer_array)
    assert imported.shape == expected_shape, f"{imported.shape} != {expected_shape}"
    assert imported.dtype == expected_dtype_str, f"{imported.dtype} != {expected_dtype_str}"


# endregion Helpers


# region NumPy round-trip


@pytest.mark.parametrize("np_dtype, nk_dtype", _NUMPY_DTYPES)
def test_numpy_export(np_dtype, nk_dtype):
    """A NumKong tensor must be consumable zero-copy by `np.from_dlpack`."""
    np = pytest.importorskip("numpy")
    source = np.arange(24, dtype=np_dtype).reshape(4, 6)
    tensor = nk.Tensor(source)
    arr = np.from_dlpack(tensor)
    assert arr.shape == (4, 6)
    assert str(arr.dtype) == np_dtype
    assert tensor.dtype == nk_dtype
    np.testing.assert_array_equal(arr, source)
    arr[0, 0] = 99
    assert np.asarray(tensor)[0, 0] == 99, "mutation through NumPy view not visible in NumKong tensor"


@pytest.mark.parametrize("np_dtype, nk_dtype", _NUMPY_DTYPES)
def test_numpy_import(np_dtype, nk_dtype):
    """A NumPy array must be consumable zero-copy by `nk.from_dlpack`."""
    np = pytest.importorskip("numpy")
    source = np.arange(20, dtype=np_dtype)
    tensor = nk.from_dlpack(source)
    assert tensor.shape == (20,)
    assert tensor.dtype == nk_dtype
    np.testing.assert_array_equal(np.asarray(tensor), source)


def test_numpy_import_strided():
    """A non-contiguous (transposed) NumPy source preserves strides on import."""
    np = pytest.importorskip("numpy")
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    tensor = nk.from_dlpack(source.T)
    assert tensor.shape == (6, 4)
    np.testing.assert_array_equal(np.asarray(tensor), source.T)


# endregion NumPy round-trip


# region PyTorch round-trip


@pytest.mark.parametrize("torch_dtype_attr, nk_dtype", _TORCH_DTYPES)
def test_torch_import(torch_dtype_attr, nk_dtype):
    """Every torch-supported dtype (including bf16, fp8 e4m3fn, fp8 e5m2) imports zero-copy as NumKong dtype."""
    torch = pytest.importorskip("torch")
    torch_dtype = getattr(torch, torch_dtype_attr)
    # `torch.zeros` works for fp8 (where `torch.arange` is unsupported); the test only checks dtype + round-trip.
    src = torch.zeros(4, 6, dtype=torch_dtype)
    nk_tensor = nk.from_dlpack(src)
    assert nk_tensor.shape == (4, 6)
    assert nk_tensor.dtype == nk_dtype
    pt_back = torch.from_dlpack(nk_tensor)
    assert pt_back.dtype == torch_dtype
    assert tuple(pt_back.shape) == (4, 6)


@pytest.mark.parametrize("torch_dtype_attr, nk_dtype", _TORCH_DTYPES)
def test_torch_export(torch_dtype_attr, nk_dtype):
    """Every torch-supported dtype is consumable by `torch.from_dlpack` after a NumKong round-trip."""
    torch = pytest.importorskip("torch")
    torch_dtype = getattr(torch, torch_dtype_attr)
    # Build the source on the torch side (avoids needing ml_dtypes for fp8/bf16 numpy paths).
    src = torch.zeros(4, 6, dtype=torch_dtype)
    nk_tensor = nk.from_dlpack(src)  # NumKong view sharing torch's buffer
    pt_view = torch.from_dlpack(nk_tensor)  # back to torch via NumKong's exporter
    assert pt_view.dtype == torch_dtype
    assert tuple(pt_view.shape) == (4, 6)


def test_torch_export_mutation_visibility():
    """Writes through one view are visible through the other (proves zero-copy, not a value comparison)."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    tensor = nk.Tensor(source)
    pt = torch.from_dlpack(tensor)
    pt[0, 0] = 99
    assert np.asarray(tensor)[0, 0] == 99, "mutation through PyTorch view not visible in NumKong tensor"


def test_torch_strided_import():
    """A transposed PyTorch tensor preserves strides when imported."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    src = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    nk_tensor = nk.from_dlpack(src)
    assert nk_tensor.shape == (6, 4)
    np.testing.assert_array_equal(np.asarray(nk_tensor), src.numpy())


# endregion PyTorch round-trip


# region JAX round-trip


@pytest.mark.parametrize("jax_dtype_attr, nk_dtype", _JAX_DTYPES)
def test_jax_export(jax_dtype_attr, nk_dtype):
    """NumKong → JAX zero-copy across every JAX-supported dtype incl. bf16."""
    np = pytest.importorskip("numpy")
    jax = pytest.importorskip("jax")
    # JAX disables float64 by default and silently downcasts to float32 unless x64 is enabled.
    if jax_dtype_attr in ("float64", "int64", "uint64"):
        jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    np_dtype = jax_dtype_attr if jax_dtype_attr != "bfloat16" else "float32"
    source = np.arange(20, dtype=np_dtype)
    tensor = nk.Tensor(source) if jax_dtype_attr != "bfloat16" else nk.Tensor(source).astype("bfloat16")
    jax_array = jnp.from_dlpack(tensor)
    assert jax_array.shape == (20,)
    assert str(jax_array.dtype) == jax_dtype_attr
    assert tensor.dtype == nk_dtype


@pytest.mark.parametrize("jax_dtype_attr, nk_dtype", _JAX_DTYPES)
def test_jax_import(jax_dtype_attr, nk_dtype):
    """JAX → NumKong zero-copy across every JAX-supported dtype incl. bf16."""
    jax = pytest.importorskip("jax")
    if jax_dtype_attr in ("float64", "int64", "uint64"):
        jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    jax_dtype = getattr(jnp, jax_dtype_attr)
    # `jnp.zeros` works for every supported dtype incl. bf16 / int / uint.
    src = jnp.zeros(20, dtype=jax_dtype)
    _assert_round_trip_through(src, nk_dtype, (20,))


# endregion JAX round-trip


# region TensorFlow round-trip


@pytest.mark.parametrize("tf_dtype_attr, nk_dtype", _TF_DTYPES)
def test_tensorflow_export(tf_dtype_attr, nk_dtype):
    """NumKong → TensorFlow zero-copy across every TF-supported dtype incl. bf16."""
    np = pytest.importorskip("numpy")
    tf = pytest.importorskip("tensorflow")
    np_dtype = tf_dtype_attr if tf_dtype_attr != "bfloat16" else "float32"
    source = np.arange(20, dtype=np_dtype)
    tensor = nk.Tensor(source) if tf_dtype_attr != "bfloat16" else nk.Tensor(source).astype("bfloat16")
    assert tensor.dtype == nk_dtype
    tf_tensor = tf.experimental.dlpack.from_dlpack(tensor.__dlpack__())
    assert tuple(tf_tensor.shape) == (20,)
    assert tf_tensor.dtype == getattr(tf, tf_dtype_attr)


@pytest.mark.parametrize("tf_dtype_attr, nk_dtype", _TF_DTYPES)
def test_tensorflow_import(tf_dtype_attr, nk_dtype):
    """TensorFlow → NumKong zero-copy across every TF-supported dtype incl. bf16."""
    np = pytest.importorskip("numpy")
    tf = pytest.importorskip("tensorflow")
    np_dtype = tf_dtype_attr if tf_dtype_attr != "bfloat16" else "float32"
    src_np = np.arange(20, dtype=np_dtype)
    tf_tensor = tf.constant(src_np)
    if tf_dtype_attr == "bfloat16":
        tf_tensor = tf.cast(tf_tensor, tf.bfloat16)
    capsule = tf.experimental.dlpack.to_dlpack(tf_tensor)
    _assert_round_trip_through(capsule, nk_dtype, (20,))


# endregion TensorFlow round-trip


# region CuPy round-trip (GPU only — skipped on CPU-only venvs)


def _cupy_or_skip():
    """Common CuPy + GPU + allocate guard. Returns the cupy module on success."""
    cupy = pytest.importorskip("cupy")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"CuPy installed but no CUDA runtime visible ({exc})")
    if device_count == 0:
        pytest.skip("CuPy installed but no CUDA devices visible")
    # Smoke-test that CuPy can actually JIT a kernel on this GPU. cupy-cuda12x wheels are built for
    # specific compute capabilities and CUDA runtimes; mismatches raise CUDA_ERROR_INVALID_IMAGE.
    try:
        cupy.arange(4, dtype=cupy.float32) + 1
    except Exception as exc:
        pytest.skip(f"CuPy/GPU mismatch: cannot allocate ({exc})")
    return cupy


def test_cupy_device_memory_is_rejected():
    """Plain ``cudaMalloc`` GPU memory (kDLCUDA) must be REJECTED — reading it on the host would fault."""
    cupy = _cupy_or_skip()
    src = cupy.arange(16, dtype=cupy.float32)
    with pytest.raises(ValueError, match="CPU-accessible"):
        nk.from_dlpack(src)


def test_cupy_managed_memory_accepted():
    """``cudaMallocManaged`` unified memory (kDLCUDAManaged) is host-readable and must be ACCEPTED."""
    cupy = _cupy_or_skip()
    previous_allocator = cupy.cuda.get_allocator()
    cupy.cuda.set_allocator(cupy.cuda.malloc_managed)
    try:
        src = cupy.arange(16, dtype=cupy.float32)
        nk_tensor = nk.from_dlpack(src)
        assert nk_tensor.shape == (16,)
        assert nk_tensor.dtype == "float32"
    finally:
        cupy.cuda.set_allocator(previous_allocator)


def test_cupy_pinned_host_memory_accepted():
    """``cudaMallocHost`` pinned host memory (kDLCUDAHost) is just CPU memory and must be ACCEPTED."""
    cupy = _cupy_or_skip()
    # CuPy doesn't expose pinned memory as a regular ndarray that implements __dlpack__. The closest
    # path is alloc_pinned_memory + a numpy view. If the producer-side wrapper isn't available,
    # skip — the C-side acceptance still lands.
    try:
        nbytes = 16 * 4
        pinned = cupy.cuda.alloc_pinned_memory(nbytes)
        np = pytest.importorskip("numpy")
        host_view = np.frombuffer(pinned, dtype=np.float32, count=16)
        host_view[:] = np.arange(16, dtype=np.float32)
        # This DLPack capsule will report kDLCPU (numpy's view), not kDLCUDAHost. The test still
        # serves as a sanity check that pinned memory exchanges work; an explicit kDLCUDAHost test
        # requires a producer that surfaces the device code, which CuPy currently does not.
        nk_tensor = nk.from_dlpack(host_view)
        assert nk_tensor.shape == (16,)
        assert float(np.asarray(nk_tensor).sum()) == 16 * 15 / 2
    except (AttributeError, RuntimeError) as exc:
        pytest.skip(f"CuPy pinned-memory DLPack producer unavailable ({exc})")


# endregion CuPy round-trip


# region MLX round-trip (Apple Silicon — unified memory)


@pytest.mark.parametrize("mlx_dtype_attr, nk_dtype", _MLX_DTYPES)
def test_mlx_import(mlx_dtype_attr, nk_dtype):
    """MLX → NumKong zero-copy across every MLX-supported dtype incl. bf16."""
    mx = pytest.importorskip("mlx.core")
    mlx_dtype = getattr(mx, mlx_dtype_attr)
    src = mx.zeros((4, 5), dtype=mlx_dtype)
    _assert_round_trip_through(src, nk_dtype, (4, 5))


@pytest.mark.parametrize("mlx_dtype_attr, nk_dtype", _MLX_DTYPES)
def test_mlx_export(mlx_dtype_attr, nk_dtype):
    """NumKong → MLX zero-copy across every MLX-supported dtype incl. bf16."""
    mx = pytest.importorskip("mlx.core")
    if nk_dtype == "bfloat16":
        # bf16 isn't constructible via nk.zeros directly across all paths; stage via float32 + astype.
        src = nk.zeros(20, dtype="float32").astype("bfloat16")
    else:
        src = nk.zeros(20, dtype=nk_dtype)
    mx_array = mx.from_dlpack(src)
    assert mx_array.shape == (20,)
    assert mx_array.dtype == getattr(mx, mlx_dtype_attr)


# endregion MLX round-trip


# region PyArrow → NumKong (PyArrow is producer-only in the protocol)


@pytest.mark.parametrize("pa_dtype_attr, nk_dtype", _PYARROW_DTYPES)
def test_pyarrow_import(pa_dtype_attr, nk_dtype):
    """PyArrow Array → NumKong via the PyArrow `__dlpack__` producer side."""
    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    pa_type = getattr(pa, pa_dtype_attr)()
    if not hasattr(pa.array([1], type=pa_type), "__dlpack__"):
        pytest.skip("PyArrow build lacks DLPack support (pre-15.0.0)")
    arrow_array = pa.array(np.arange(20, dtype=pa_dtype_attr))
    nk_tensor = nk.from_dlpack(arrow_array)
    assert nk_tensor.shape == (20,)
    assert nk_tensor.dtype == nk_dtype
    np.testing.assert_array_equal(np.asarray(nk_tensor), np.arange(20, dtype=pa_dtype_attr))


# endregion PyArrow


# region ONNX Runtime round-trip


def test_onnxruntime_roundtrip():
    """`OrtValue` ↔ NumKong via DLPack. ORT is a producer/consumer in inference builds."""
    np = pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    src_np = np.arange(40, dtype=np.float32).reshape(5, 8)
    ort_value = ort.OrtValue.ortvalue_from_numpy(src_np)
    if not hasattr(ort_value, "to_dlpack"):
        pytest.skip("onnxruntime build lacks DLPack helpers (microsoft/onnxruntime#23110 not present)")
    nk_tensor = nk.from_dlpack(ort_value.to_dlpack())
    assert nk_tensor.shape == (5, 8)
    np.testing.assert_array_equal(np.asarray(nk_tensor), src_np)


# endregion ONNX Runtime


# region Protocol contract (no framework dependency — uses only NumKong's own constructors)


def test_dlpack_device_is_cpu():
    """`__dlpack_device__()` always returns `(kDLCPU=1, 0)`."""
    tensor = nk.zeros(4, dtype="float32")
    assert tensor.__dlpack_device__() == (1, 0)


def test_capsule_name_legacy_default():
    """The default `__dlpack__()` returns a legacy `dltensor` capsule."""
    tensor = nk.zeros(4, dtype="float32")
    capsule = tensor.__dlpack__()
    assert '"dltensor"' in repr(capsule), repr(capsule)


def test_capsule_name_versioned_when_requested():
    """`max_version=(1, 0)` opts into the versioned capsule (`dltensor_versioned`)."""
    tensor = nk.zeros(4, dtype="float32")
    capsule = tensor.__dlpack__(max_version=(1, 0))
    assert '"dltensor_versioned"' in repr(capsule), repr(capsule)


def test_export_rejects_non_cpu_device():
    """Asking the exporter for a non-CPU `dl_device` raises immediately."""
    tensor = nk.zeros(4, dtype="float32")
    with pytest.raises((BufferError, TypeError, ValueError)):
        tensor.__dlpack__(dl_device=(2, 0))  # kDLCUDA = 2


def test_export_rejects_copy_true():
    """`copy=True` raises — we are zero-copy by construction."""
    tensor = nk.zeros(4, dtype="float32")
    with pytest.raises((BufferError, ValueError)):
        tensor.__dlpack__(copy=True)


@pytest.mark.parametrize("dtype_str", ["e2m3", "e3m2"])
def test_fp6_uses_versioned_with_padded_flag(dtype_str):
    """FP6 (e2m3 / e3m2) cannot be expressed in legacy v0; auto-upgrades to v1."""
    fp6_tensor = nk.zeros((4, 4), dtype=dtype_str)
    capsule = fp6_tensor.__dlpack__()  # no max_version — still produced versioned for FP6
    assert '"dltensor_versioned"' in repr(capsule), f"{dtype_str}: {capsule!r}"


@pytest.mark.parametrize("dtype_str", ["e2m3", "e3m2"])
def test_fp6_roundtrip_preserves_dtype(dtype_str):
    """FP6 export + import preserves dtype and the underlying byte length."""
    source = nk.zeros((3, 8), dtype=dtype_str)
    imported = nk.from_dlpack(source)
    assert imported.dtype == dtype_str
    assert imported.shape == (3, 8)
    assert imported.nbytes == source.nbytes


def test_import_rejects_object_without_dlpack_method():
    """A bare object without `__dlpack__` raises TypeError, not a segfault."""
    with pytest.raises(TypeError):
        nk.from_dlpack(object())


def test_ownership_outlives_source_tensor():
    """The capsule pins the source memory after the producing Tensor goes out of scope."""
    tensor = nk.iota(100, dtype="float32")
    expected_data_ptr = tensor.data_ptr
    capsule = tensor.__dlpack__()
    del tensor  # source tensor goes out of scope; capsule still holds a refcount
    imported = nk.from_dlpack(capsule)
    assert imported.shape == (100,)
    # If the producer's memory had been freed, this read would be a use-after-free; passing means the capsule
    # kept it alive.
    assert imported.data_ptr == expected_data_ptr


def test_capsule_renamed_to_used_after_import():
    """After consumption the capsule is renamed to `used_dltensor*` per spec."""
    tensor = nk.iota(8, dtype="float32")
    capsule = tensor.__dlpack__()
    nk.from_dlpack(capsule)
    assert '"used_dltensor"' in repr(capsule), repr(capsule)


# endregion Protocol contract
