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
                    (a) plain ``cudaMalloc`` (kDLCUDA) → must be ACCEPTED as a GPU tensor keeping its device;
                    (b) ``cudaMallocManaged`` unified memory (kDLCUDAManaged) → must be ACCEPTED;
                    (c) ``cudaMallocHost`` pinned host memory (kDLCUDAHost) → must be ACCEPTED.
- PyArrow         — Arrow Array → NumKong (one-way; PyArrow is producer-only in the protocol).
- ONNX Runtime    — ``OrtValue`` ↔ NumKong via the ``to_dlpack`` / ``ortvalue_from_numpy`` helpers.
- MLX             — Apple Silicon's unified-memory framework. Tests skip on Linux (no ``mlx.core`` wheel) and
                    run on macOS contributor machines / Apple Silicon CI.

The importer accepts every DLPack ``device_type`` whose pointer is dereferenceable from host code
(kDLCPU / CUDAHost / ROCMHost / CUDAManaged / OneAPI / Metal) as a CPU tensor, and kDLCUDA / kDLROCM
as a GPU tensor keeping its device. Other device memory (kDLOpenCL, kDLVulkan, kDLWebGPU, kDLHexagon,
kDLMAIA, kDLTrn, kDLVPI, kDLExtDev) is rejected.

Run: python -m pytest test/dlpack.py

File: test/dlpack.py
Author: Ash Vardanian
Date: April 18, 2026
"""

import ctypes
import math

import pytest

import numkong as nk


# region Helpers


_INTEGER_DTYPES = ["int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"]
_PLAIN_FLOAT_DTYPES = ["float64", "float32", "float16"]
_PLAIN_NUMERIC_DTYPES = _PLAIN_FLOAT_DTYPES + _INTEGER_DTYPES

# Per-framework dtype matrices. Each entry is a `(framework_attr_name, nk_dtype_string)` pair so a
# single parametrized test can drive both directions: `getattr(framework_module, attr)` resolves the
# producer-side dtype, and `nk_dtype_string` is what we expect back on the NumKong side once NumKong
# finishes the import.
#
# We include the "weird GPU formats" — bf16, fp8 e4m3fn, fp8 e5m2, fp6 e2m3 and fp6 e3m2 — wherever
# the framework supports them natively, so each framework's coverage matches what dtype-fidelity
# DLPack actually buys.

_NUMPY_DTYPES = [(name, name) for name in _PLAIN_NUMERIC_DTYPES]
"""NumPy core has no bf16/fp8/fp6 — those would need ml_dtypes. Skip exotics here; they're tested
instead via PyTorch and MLX.
"""

_TORCH_DTYPES = [(name, name) for name in _PLAIN_NUMERIC_DTYPES] + [
    ("bfloat16", "bfloat16"),
    ("float8_e4m3fn", "e4m3"),
    ("float8_e5m2", "e5m2"),
]
"""PyTorch ships native bf16, float8_e4m3fn and float8_e5m2 since 2.1+, so this is the framework's
marquee coverage case.
"""

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
"""JAX always has native bf16 and the integer / float subset; no fp8/fp6 yet."""

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
"""TensorFlow ships native bf16 + float16/32/64 + integers, with no public fp8 dtype that maps
cleanly via DLPack today.
"""

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
"""PyArrow exposes float / signed-int / unsigned-int builders by name. No bf16/fp8/fp6."""

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
"""MLX ships native bf16 on Apple Silicon, plus float16/32/integers; mlx.core 0.x has no fp8 yet."""


def _assert_round_trip_through(producer_array, expected_dtype_str: str, expected_shape: tuple) -> None:
    """Common pattern: framework-owned array → NumKong view → assertions on dtype/shape."""
    imported = nk.from_dlpack(producer_array)
    assert imported.shape == expected_shape, f"{imported.shape} != {expected_shape}"
    assert imported.dtype == expected_dtype_str, f"{imported.dtype} != {expected_dtype_str}"


# endregion Helpers


# region NumPy round-trip


@pytest.mark.parametrize(("np_dtype", "nk_dtype"), _NUMPY_DTYPES)
def test_numpy_reads_tensors_without_a_copy(np_dtype, nk_dtype):
    """A NumKong tensor must be consumable zero-copy by `np.from_dlpack`."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    source = np.arange(24, dtype=np_dtype).reshape(4, 6)
    tensor = nk.Tensor(source)
    arr = np.from_dlpack(tensor)
    assert arr.shape == (4, 6)
    assert str(arr.dtype) == np_dtype
    assert tensor.dtype == nk_dtype
    np.testing.assert_array_equal(arr, source)
    arr[0, 0] = 99
    assert np.asarray(tensor)[0, 0] == 99, "mutation through NumPy view not visible in NumKong tensor"


@pytest.mark.parametrize(("np_dtype", "nk_dtype"), _NUMPY_DTYPES)
def test_numpy_arrays_import_without_a_copy(np_dtype, nk_dtype):
    """A NumPy array must be consumable zero-copy by `nk.from_dlpack`."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    source = np.arange(20, dtype=np_dtype)
    tensor = nk.from_dlpack(source)
    assert tensor.shape == (20,)
    assert tensor.dtype == nk_dtype
    np.testing.assert_array_equal(np.asarray(tensor), source)


def test_strided_numpy_arrays_keep_their_strides():
    """A non-contiguous (transposed) NumPy source preserves strides on import."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    tensor = nk.from_dlpack(source.T)
    assert tensor.shape == (6, 4)
    np.testing.assert_array_equal(np.asarray(tensor), source.T)


# endregion NumPy round-trip


# region PyTorch round-trip


@pytest.mark.parametrize(("torch_dtype_attr", "nk_dtype"), _TORCH_DTYPES)
def test_torch_tensors_import_with_every_dtype(torch_dtype_attr, nk_dtype):
    """Every torch-supported dtype (including bf16, fp8 e4m3fn, fp8 e5m2) imports zero-copy as NumKong dtype."""
    torch = pytest.importorskip("torch", reason="PyTorch is not installed")
    torch_dtype = getattr(torch, torch_dtype_attr)
    # `torch.zeros` works for fp8 (where `torch.arange` is unsupported); the test only checks dtype + round-trip.
    src = torch.zeros(4, 6, dtype=torch_dtype)
    nk_tensor = nk.from_dlpack(src)
    assert nk_tensor.shape == (4, 6)
    assert nk_tensor.dtype == nk_dtype
    pt_back = torch.from_dlpack(nk_tensor)
    assert pt_back.dtype == torch_dtype
    assert tuple(pt_back.shape) == (4, 6)


@pytest.mark.parametrize(("torch_dtype_attr", "nk_dtype"), _TORCH_DTYPES)
def test_torch_reads_tensors_of_every_dtype(torch_dtype_attr, nk_dtype):
    """Every torch-supported dtype is consumable by `torch.from_dlpack` after a NumKong round-trip."""
    torch = pytest.importorskip("torch", reason="PyTorch is not installed")
    torch_dtype = getattr(torch, torch_dtype_attr)
    # Build the source on the torch side (avoids needing ml_dtypes for fp8/bf16 numpy paths).
    src = torch.zeros(4, 6, dtype=torch_dtype)
    nk_tensor = nk.from_dlpack(src)  # NumKong view sharing torch's buffer
    pt_view = torch.from_dlpack(nk_tensor)  # back to torch via NumKong's exporter
    assert pt_view.dtype == torch_dtype
    assert tuple(pt_view.shape) == (4, 6)


def test_torch_and_numkong_see_each_others_writes():
    """Writes through one view are visible through the other (proves zero-copy, not a value comparison)."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    torch = pytest.importorskip("torch", reason="PyTorch is not installed")
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    tensor = nk.Tensor(source)
    pt = torch.from_dlpack(tensor)
    pt[0, 0] = 99
    assert np.asarray(tensor)[0, 0] == 99, "mutation through PyTorch view not visible in NumKong tensor"


def test_strided_torch_tensors_keep_their_strides():
    """A transposed PyTorch tensor preserves strides when imported."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    torch = pytest.importorskip("torch", reason="PyTorch is not installed")
    src = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    nk_tensor = nk.from_dlpack(src)
    assert nk_tensor.shape == (6, 4)
    np.testing.assert_array_equal(np.asarray(nk_tensor), src.numpy())


# endregion PyTorch round-trip


# region JAX round-trip


@pytest.mark.parametrize(("jax_dtype_attr", "nk_dtype"), _JAX_DTYPES)
def test_jax_reads_tensors_of_every_dtype(jax_dtype_attr, nk_dtype):
    """NumKong → JAX zero-copy across every JAX-supported dtype incl. bf16."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    jax = pytest.importorskip("jax", reason="JAX is not installed")
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


@pytest.mark.parametrize(("jax_dtype_attr", "nk_dtype"), _JAX_DTYPES)
def test_jax_arrays_import_with_every_dtype(jax_dtype_attr, nk_dtype):
    """JAX → NumKong zero-copy across every JAX-supported dtype incl. bf16."""
    jax = pytest.importorskip("jax", reason="JAX is not installed")
    if jax_dtype_attr in ("float64", "int64", "uint64"):
        jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    jax_dtype = getattr(jnp, jax_dtype_attr)
    # `jnp.zeros` works for every supported dtype incl. bf16 / int / uint.
    src = jnp.zeros(20, dtype=jax_dtype)
    _assert_round_trip_through(src, nk_dtype, (20,))


# endregion JAX round-trip


# region TensorFlow round-trip


@pytest.mark.parametrize(("tf_dtype_attr", "nk_dtype"), _TF_DTYPES)
def test_tensorflow_reads_tensors_of_every_dtype(tf_dtype_attr, nk_dtype):
    """NumKong → TensorFlow zero-copy across every TF-supported dtype incl. bf16."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    tf = pytest.importorskip("tensorflow", reason="TensorFlow is not installed")
    np_dtype = tf_dtype_attr if tf_dtype_attr != "bfloat16" else "float32"
    source = np.arange(20, dtype=np_dtype)
    tensor = nk.Tensor(source) if tf_dtype_attr != "bfloat16" else nk.Tensor(source).astype("bfloat16")
    assert tensor.dtype == nk_dtype
    tf_tensor = tf.experimental.dlpack.from_dlpack(tensor.__dlpack__())
    assert tuple(tf_tensor.shape) == (20,)
    assert tf_tensor.dtype == getattr(tf, tf_dtype_attr)


@pytest.mark.parametrize(("tf_dtype_attr", "nk_dtype"), _TF_DTYPES)
def test_tensorflow_tensors_import_with_every_dtype(tf_dtype_attr, nk_dtype):
    """TensorFlow → NumKong zero-copy across every TF-supported dtype incl. bf16."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    tf = pytest.importorskip("tensorflow", reason="TensorFlow is not installed")
    np_dtype = tf_dtype_attr if tf_dtype_attr != "bfloat16" else "float32"
    src_np = np.arange(20, dtype=np_dtype)
    tf_tensor = tf.constant(src_np)
    if tf_dtype_attr == "bfloat16":
        tf_tensor = tf.cast(tf_tensor, tf.bfloat16)
    capsule = tf.experimental.dlpack.to_dlpack(tf_tensor)
    _assert_round_trip_through(capsule, nk_dtype, (20,))


# endregion TensorFlow round-trip


# region CuPy round-trip on GPUs


_DEVICE_TENSORS = []
"""Every `_DeviceTensor`, whose structs the importer reads until its Tensor dies, so they outlive any Tensor."""


class _DeviceTensor:
    """A DLPack producer claiming host doubles live on CUDA device 0, for refusals that need no GPU."""

    class _Tensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device_type", ctypes.c_int32),
            ("device_id", ctypes.c_int32),
            ("ndim", ctypes.c_int32),
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
        ]

    def __init__(self, *shape: int):
        self.values = (ctypes.c_double * math.prod(shape))()
        self.shape = (ctypes.c_int64 * len(shape))(*shape)
        self.tensor = self._Tensor(
            data=ctypes.addressof(self.values),
            device_type=2,
            ndim=len(shape),
            code=2,
            bits=64,
            lanes=1,
            shape=self.shape,
        )
        _DEVICE_TENSORS.append(self)

    def __dlpack__(self, **_):
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.restype, capsule_new.argtypes = (
            ctypes.py_object,
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p],
        )
        return capsule_new(ctypes.addressof(self.tensor), b"dltensor", None)


@pytest.mark.parametrize("keyword", ["out", "workspace"])
def test_host_packs_refuse_gpu_tensors(keyword: str):
    """A kernel on the host never gets a GPU Tensor, so packs left on the heap raise ``BufferError``."""
    numpy = pytest.importorskip("numpy", reason="NumPy is not installed")
    packed = nk.shapes_pack(numpy.arange(12.0, dtype=numpy.float32).reshape(4, 3))
    tensor = nk.from_dlpack(_DeviceTensor(1, 1))
    assert tensor.__dlpack_device__() == (2, 0)  # kDLCUDA
    with pytest.raises(BufferError):
        nk.chamfers_packed(packed, packed, **{keyword: tensor})


def test_cupy_device_memory_keeps_its_device(cupy):
    """Plain ``cudaMalloc`` GPU memory (kDLCUDA) imports as a GPU tensor on that device."""
    src = cupy.arange(16, dtype=cupy.float32)
    assert nk.from_dlpack(src).__dlpack_device__() == (2, src.device.id)  # kDLCUDA


def test_cupy_managed_memory_imports(cupy):
    """``cudaMallocManaged`` unified memory (kDLCUDAManaged) is host-readable and must be ACCEPTED."""
    previous_allocator = cupy.cuda.get_allocator()
    cupy.cuda.set_allocator(cupy.cuda.malloc_managed)
    try:
        src = cupy.arange(16, dtype=cupy.float32)
        nk_tensor = nk.from_dlpack(src)
        assert nk_tensor.shape == (16,)
        assert nk_tensor.dtype == "float32"
    finally:
        cupy.cuda.set_allocator(previous_allocator)


def test_cupy_pinned_host_memory_imports(pinned_host_floats):
    """``cudaMallocHost`` pinned host memory (kDLCUDAHost) is just CPU memory and must be ACCEPTED."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    # The NumPy view's capsule reports kDLCPU, not kDLCUDAHost: CuPy surfaces no producer with that
    # device code.
    nk_tensor = nk.from_dlpack(pinned_host_floats)
    assert nk_tensor.shape == (16,)
    assert float(np.asarray(nk_tensor).sum()) == 16 * 15 / 2


# endregion CuPy round-trip on GPUs


# region MLX round-trip on Apple Silicon


@pytest.mark.parametrize(("mlx_dtype_attr", "nk_dtype"), _MLX_DTYPES)
def test_mlx_arrays_import_with_every_dtype(mlx_dtype_attr, nk_dtype):
    """MLX → NumKong zero-copy across every MLX-supported dtype incl. bf16."""
    mx = pytest.importorskip("mlx.core", reason="MLX is not installed")
    mlx_dtype = getattr(mx, mlx_dtype_attr)
    src = mx.zeros((4, 5), dtype=mlx_dtype)
    _assert_round_trip_through(src, nk_dtype, (4, 5))


@pytest.mark.parametrize(("mlx_dtype_attr", "nk_dtype"), _MLX_DTYPES)
def test_mlx_reads_tensors_of_every_dtype(mlx_dtype_attr, nk_dtype):
    """NumKong → MLX zero-copy across every MLX-supported dtype incl. bf16."""
    mx = pytest.importorskip("mlx.core", reason="MLX is not installed")
    if nk_dtype == "bfloat16":  # noqa: SIM108 — keep the branch to explain the bf16 staging
        # bf16 isn't constructible via nk.zeros directly across all paths; stage via float32 + astype.
        src = nk.zeros(20, dtype="float32").astype("bfloat16")
    else:
        src = nk.zeros(20, dtype=nk_dtype)
    mx_array = mx.from_dlpack(src)
    assert mx_array.shape == (20,)
    assert mx_array.dtype == getattr(mx, mlx_dtype_attr)


# endregion MLX round-trip on Apple Silicon


# region PyArrow → NumKong — producer-only in the protocol


@pytest.mark.parametrize(("pa_dtype_attr", "nk_dtype"), _PYARROW_DTYPES)
def test_pyarrow_arrays_import_without_a_copy(pyarrow, pa_dtype_attr, nk_dtype):
    """PyArrow Array → NumKong via the PyArrow `__dlpack__` producer side."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    arrow_array = pyarrow.array(np.arange(20, dtype=pa_dtype_attr))
    nk_tensor = nk.from_dlpack(arrow_array)
    assert nk_tensor.shape == (20,)
    assert nk_tensor.dtype == nk_dtype
    np.testing.assert_array_equal(np.asarray(nk_tensor), np.arange(20, dtype=pa_dtype_attr))


# endregion PyArrow


# region ONNX Runtime round-trip


def test_onnxruntime_values_round_trip(onnxruntime):
    """`OrtValue` ↔ NumKong via DLPack. ORT is a producer/consumer in inference builds."""
    np = pytest.importorskip("numpy", reason="NumPy is not installed")
    src_np = np.arange(40, dtype=np.float32).reshape(5, 8)
    ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(src_np)
    nk_tensor = nk.from_dlpack(ort_value.to_dlpack())
    assert nk_tensor.shape == (5, 8)
    np.testing.assert_array_equal(np.asarray(nk_tensor), src_np)


# endregion ONNX Runtime


# region Protocol contract without frameworks


def test_dlpack_device_is_cpu():
    """`__dlpack_device__()` always returns `(kDLCPU=1, 0)`."""
    tensor = nk.zeros(4, dtype="float32")
    assert tensor.__dlpack_device__() == (1, 0)


def test_capsules_default_to_the_legacy_name():
    """The default `__dlpack__()` returns a legacy `dltensor` capsule."""
    tensor = nk.zeros(4, dtype="float32")
    capsule = tensor.__dlpack__()
    assert '"dltensor"' in repr(capsule), repr(capsule)


def test_capsules_are_versioned_when_asked():
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


@pytest.mark.parametrize("dtype", ["e2m3", "e3m2"])
def test_fp6_uses_versioned_with_padded_flag(dtype):
    """FP6 (e2m3 / e3m2) cannot be expressed in legacy v0; auto-upgrades to v1."""
    fp6_tensor = nk.zeros((4, 4), dtype=dtype)
    capsule = fp6_tensor.__dlpack__()  # no max_version — still produced versioned for FP6
    assert '"dltensor_versioned"' in repr(capsule), f"{dtype}: {capsule!r}"


@pytest.mark.parametrize("dtype", ["e2m3", "e3m2"])
def test_fp6_round_trip_preserves_dtype(dtype):
    """FP6 export + import preserves dtype and the underlying byte length."""
    source = nk.zeros((3, 8), dtype=dtype)
    imported = nk.from_dlpack(source)
    assert imported.dtype == dtype
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


def test_capsules_are_renamed_used_after_import():
    """After consumption the capsule is renamed to `used_dltensor*` per spec."""
    tensor = nk.iota(8, dtype="float32")
    capsule = tensor.__dlpack__()
    nk.from_dlpack(capsule)
    assert '"used_dltensor"' in repr(capsule), repr(capsule)


# endregion Protocol contract without frameworks
