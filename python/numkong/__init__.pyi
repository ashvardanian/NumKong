"""Type stubs for the NumKong Python bindings.

File: python/numkong/__init__.pyi
Author: Ash Vardanian
Date: August 3, 2024
"""

from enum import IntFlag
from typing import Any, Literal, TypedDict, Unpack

# Many annotation features depend on the Python version:
# - `typing.TypeAlias` type aliases are supported in Python 3.10-3.11.
# - `type` statements are supported from Python 3.12, replacing `typing.TypeAlias`.
# - `typing.Literal` literal types are supported from Python 3.8.
#
# We cannot maintain a separate `.pyi` file for every Python version, so assume Python 3.12 with
# NumPy available.
from numpy.typing import NDArray

# region Forward Declarations and Shared Types

_IntegralTypeName = Literal[
    "uint1",
    # Sub-byte integers
    "int4",
    "uint4",
    # Signed integers
    "int64",
    "int32",
    "int16",
    "int8",
    # Unsigned integers
    "uint64",
    "uint32",
    "uint16",
    "uint8",
]
"""Integer dtype names, the first of the scalar dtype literals used throughout the API."""
_FloatTypeName = Literal[
    "f64",
    "float64",
    "f32",
    "float32",
    "bf16",  #! Not supported by NumPy
    "bfloat16",  #! Not supported by NumPy
    "f16",
    "float16",
    "e4m3",  #! FP8 E4M3 format
    "float8_e4m3",  #! FP8 E4M3 format (long-form)
    "e5m2",  #! FP8 E5M2 format
    "float8_e5m2",  #! FP8 E5M2 format (long-form)
    "e2m3",  #! FP6 E2M3 format
    "float6_e2m3",  #! FP6 E2M3 format (long-form)
    "e3m2",  #! FP6 E3M2 format
    "float6_e3m2",  #! FP6 E3M2 format (long-form)
]
_ComplexTypeName = Literal[
    "complex128",
    "complex64",
    "bcomplex32",  #! Not supported by NumPy
    "complex32",  #! Not supported by NumPy
]
_BlockScaledTypeName = Literal[
    "nvfp4",
    "mxfp4",
    "mxfp6e2m3",
    "mxfp6e3m2",
    "mxfp8e4m3",
    "mxfp8e5m2",
    "mxint8",
]
_MetricName = Literal[
    "euclidean",
    "sqeuclidean",
    "inner",
    "dot",
    "vdot",
    "angular",
    "hamming",
    "jaccard",
    "kullbackleibler",
    "kld",
    "jensenshannon",
    "jsd",
    "bilinear",
    "mahalanobis",
    "fma",
    "blend",
]

# Scalar type classes for custom float formats, registered on the module at runtime by types.c.

class bfloat16:
    """BFloat16 scalar (sign + 8-bit exponent + 7-bit mantissa)."""

    def __new__(cls, value: float | int = 0, /) -> bfloat16: ...
    def __repr__(self) -> str: ...
    def __float__(self) -> float: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __add__(self, other: float | int | bfloat16) -> bfloat16: ...
    def __sub__(self, other: float | int | bfloat16) -> bfloat16: ...
    def __mul__(self, other: float | int | bfloat16) -> bfloat16: ...
    def __truediv__(self, other: float | int | bfloat16) -> bfloat16: ...
    def __neg__(self) -> bfloat16: ...
    def __abs__(self) -> bfloat16: ...

class float16:
    """IEEE 754 half-precision scalar."""

    def __new__(cls, value: float | int = 0, /) -> float16: ...
    def __repr__(self) -> str: ...
    def __float__(self) -> float: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __add__(self, other: float | int | float16) -> float16: ...
    def __sub__(self, other: float | int | float16) -> float16: ...
    def __mul__(self, other: float | int | float16) -> float16: ...
    def __truediv__(self, other: float | int | float16) -> float16: ...
    def __neg__(self) -> float16: ...
    def __abs__(self) -> float16: ...

class float8_e4m3:
    """FP8 E4M3 scalar."""

    def __new__(cls, value: float | int = 0, /) -> float8_e4m3: ...
    def __repr__(self) -> str: ...
    def __float__(self) -> float: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __add__(self, other: float | int | float8_e4m3) -> float8_e4m3: ...
    def __sub__(self, other: float | int | float8_e4m3) -> float8_e4m3: ...
    def __mul__(self, other: float | int | float8_e4m3) -> float8_e4m3: ...
    def __truediv__(self, other: float | int | float8_e4m3) -> float8_e4m3: ...
    def __neg__(self) -> float8_e4m3: ...
    def __abs__(self) -> float8_e4m3: ...

class float8_e5m2:
    """FP8 E5M2 scalar."""

    def __new__(cls, value: float | int = 0, /) -> float8_e5m2: ...
    def __repr__(self) -> str: ...
    def __float__(self) -> float: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __add__(self, other: float | int | float8_e5m2) -> float8_e5m2: ...
    def __sub__(self, other: float | int | float8_e5m2) -> float8_e5m2: ...
    def __mul__(self, other: float | int | float8_e5m2) -> float8_e5m2: ...
    def __truediv__(self, other: float | int | float8_e5m2) -> float8_e5m2: ...
    def __neg__(self) -> float8_e5m2: ...
    def __abs__(self) -> float8_e5m2: ...

class float6_e2m3:
    """FP6 E2M3 scalar."""

    def __new__(cls, value: float | int = 0, /) -> float6_e2m3: ...
    def __repr__(self) -> str: ...
    def __float__(self) -> float: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __add__(self, other: float | int | float6_e2m3) -> float6_e2m3: ...
    def __sub__(self, other: float | int | float6_e2m3) -> float6_e2m3: ...
    def __mul__(self, other: float | int | float6_e2m3) -> float6_e2m3: ...
    def __truediv__(self, other: float | int | float6_e2m3) -> float6_e2m3: ...
    def __neg__(self) -> float6_e2m3: ...
    def __abs__(self) -> float6_e2m3: ...

class float6_e3m2:
    """FP6 E3M2 scalar."""

    def __new__(cls, value: float | int = 0, /) -> float6_e3m2: ...
    def __repr__(self) -> str: ...
    def __float__(self) -> float: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __add__(self, other: float | int | float6_e3m2) -> float6_e3m2: ...
    def __sub__(self, other: float | int | float6_e3m2) -> float6_e3m2: ...
    def __mul__(self, other: float | int | float6_e3m2) -> float6_e3m2: ...
    def __truediv__(self, other: float | int | float6_e3m2) -> float6_e3m2: ...
    def __neg__(self) -> float6_e3m2: ...
    def __abs__(self) -> float6_e3m2: ...

type _MiniFloatType = (
    type[bfloat16] | type[float16] | type[float8_e4m3] | type[float8_e5m2] | type[float6_e2m3] | type[float6_e3m2]
)

type _BufferType = NDArray[Any] | memoryview
"""Buffer-compatible tensor inputs accepted by most functions."""

class Tensor(memoryview):
    """N-dimensional tensor type returned by NumKong operations.

    Supports NumPy-like properties and buffer protocol for interoperability.
    Packed dtypes 'uint1', 'int4', 'uint4', and 'e2m1' count logical dimensions in `shape`;
    the last dimension must be a multiple of the values per byte, and the buffer protocol,
    `__array_interface__`, `nbytes`, and DLPack expose whole bytes.
    """

    def __new__(cls, array_like: _BufferType, /, *, dtype: str | _MiniFloatType | None = None) -> Tensor:
        """Construct a Tensor by copying data from a buffer-protocol object; packed dtypes read its bytes."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the tensor as a tuple of logical dimensions."""
        ...

    @property
    def dtype(self) -> _IntegralTypeName | _FloatTypeName | _ComplexTypeName:
        """Data type of the tensor elements (e.g., 'float64')."""
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        ...

    @property
    def size(self) -> int:
        """Total number of elements."""
        ...

    @property
    def nbytes(self) -> int:
        """Total number of bytes."""
        ...

    @property
    def strides(self) -> tuple[int, ...]:
        """Strides of the tensor in bytes."""
        ...

    @property
    def itemsize(self) -> int:
        """Size of each element in bytes."""
        ...

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        ...

    def __iter__(self) -> Any:
        """Iterate over the first dimension."""
        ...

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> float | Tensor:
        """Get an element or sub-tensor by index or slice."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

    def __float__(self) -> float:
        """Convert 0D tensor to Python float."""
        ...

    def __int__(self) -> int:
        """Convert 0D tensor to Python int."""
        ...

    def __str__(self) -> str:
        """Return a pretty-printed string representation."""
        ...

    def __eq__(self, other: Any) -> bool:
        """Compare tensors for equality."""
        ...

    def __ne__(self, other: Any) -> bool:
        """Compare tensors for inequality."""
        ...

    def __add__(self, other: Tensor | float | int) -> Tensor:
        """Element-wise sum with another same-shape Tensor, or a broadcast scalar."""
        ...

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        """Element-wise difference with another same-shape Tensor, or a broadcast scalar."""
        ...

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        """Element-wise product with another same-shape Tensor, or a broadcast scalar."""
        ...

    def __matmul__(self, other: PackedMatrix) -> Tensor:
        """Dot-product matrix multiplication against a pre-packed matrix; see `dots_packed`."""
        ...

    def __neg__(self) -> Tensor:
        """Return the element-wise negation."""
        ...

    def __pos__(self) -> Tensor:
        """Return a copy of the tensor."""
        ...

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """NumPy array interface dict for legacy interoperability."""
        ...

    def __dlpack__(
        self,
        *,
        stream: Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        """Return a DLPack capsule for zero-copy exchange with PyTorch, JAX, NumPy, etc."""
        ...

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the DLPack device tuple. Always (1, 0) (kDLCPU)."""
        ...

    def __array__(self, dtype: str | None = None, *, copy: bool | None = None) -> Any:
        """Convert to a NumPy array, sharing data when possible."""
        ...

    @property
    def T(self) -> Tensor:
        """Transpose of the tensor."""
        ...

    def copy(
        self, *, out: Tensor | None = None, capabilities: Capability | int | None = None, stream: int | None = None
    ) -> Tensor:
        """Return a deep copy of the tensor, or write it into `out` and return `out`."""
        ...

    def reshape(self, *shape: int) -> Tensor:
        """Return tensor reshaped to given dimensions."""
        ...

    def flatten(self, *, out: Tensor | None = None) -> Tensor:
        """Return a flattened 1D view, or a copy if non-contiguous or `out` is given."""
        ...

    def squeeze(self, /, *axes: int) -> Tensor:
        """Remove the given size-1 dimensions, or every size-1 dimension when `axes` is empty."""
        ...

    def astype(
        self,
        dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType | _BlockScaledTypeName,
        /,
        *,
        out: Tensor | None = None,
    ) -> Tensor | ScaledTensor:
        """Return a copy cast to the given dtype, quantizing to a ScaledTensor for a block-scaled target."""
        ...

    @property
    def is_contiguous(self) -> bool:
        """Whether the tensor is C-contiguous in memory."""
        ...

    @property
    def data_ptr(self) -> int:
        """Integer address of the underlying data buffer."""
        ...

    @property
    def capacity(self) -> int:
        """Allocated element capacity of the owned buffer (>= size)."""
        ...

    def resize(self, *shape: int) -> Tensor:
        """Reshape in place within capacity without moving storage; returns self."""
        ...

    def reserve(self, capacity: int) -> None:
        """Grow the allocated capacity to at least `capacity` elements (may move); preserves contents."""
        ...

    def clear(self) -> None:
        """Reset to an empty shape (size 0) while keeping the allocated capacity."""
        ...

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        *,
        keepdims: bool = False,
        out: Tensor | None = None,
        **dispatch: Unpack[_Dispatch],
    ) -> float | int | Tensor:
        """Return the sum of all elements."""
        ...

    def norm(
        self,
        axis: int | tuple[int, ...] | None = None,
        *,
        keepdims: bool = False,
        out: Tensor | None = None,
        **dispatch: Unpack[_Dispatch],
    ) -> float | Tensor:
        """Return the L2 norm."""
        ...

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        *,
        keepdims: bool = False,
        out: Tensor | None = None,
        **dispatch: Unpack[_Dispatch],
    ) -> float | int | Tensor | None:
        """Return the minimum element, or None if all elements are NaN."""
        ...

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        *,
        keepdims: bool = False,
        out: Tensor | None = None,
        **dispatch: Unpack[_Dispatch],
    ) -> float | int | Tensor | None:
        """Return the maximum element, or None if all elements are NaN."""
        ...

    def argmin(
        self,
        axis: int | None = None,
        *,
        keepdims: bool = False,
        out: Tensor | None = None,
        **dispatch: Unpack[_Dispatch],
    ) -> int | Tensor | None:
        """Return the index of the minimum element, or None if all elements are NaN."""
        ...

    def argmax(
        self,
        axis: int | None = None,
        *,
        keepdims: bool = False,
        out: Tensor | None = None,
        **dispatch: Unpack[_Dispatch],
    ) -> int | Tensor | None:
        """Return the index of the maximum element, or None if all elements are NaN."""
        ...

class ScaledTensor:
    """Block-scaled tensor: the OCP MX family plus NVIDIA NVFP4.

    Holds packed elements, per-block scales, and an optional per-tensor scale.
    Produced by `Tensor.astype('nvfp4')`, `Tensor.astype('mxfp4')`, and similar block-scaled targets.
    """

    @property
    def elements(self) -> Tensor:
        """Packed sub-byte element Tensor, DLPack-exportable."""
        ...

    @property
    def block_scales(self) -> Tensor:
        """Per-block scale Tensor, DLPack-exportable."""
        ...

    @property
    def tensor_scale(self) -> float | None:
        """Per-tensor float multiplier, None for MX formats."""
        ...

    @property
    def block_size(self) -> int:
        """Number of elements per block."""
        ...

    @property
    def dtype(self) -> _BlockScaledTypeName:
        """Composite block-scaled dtype name."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical, dense shape of the quantized tensor."""
        ...

    @property
    def capacity(self) -> int:
        """Allocated element capacity of the packed elements buffer."""
        ...

    def astype(
        self,
        dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType | _BlockScaledTypeName,
        /,
    ) -> Tensor | ScaledTensor:
        """Materialize to a dense dtype, or transcode to another block-scaled dtype."""
        ...

    def resize(self, *shape: int) -> ScaledTensor:
        """Resize the packed elements and block scales in lockstep, within capacity; returns self."""
        ...

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> ScaledTensor:
        """Slice elements and block_scales in lockstep; the last axis must stay block-aligned."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

class PackedMatrix:
    """Opaque pre-packed matrix for repeated cross operations.

    Created by dots_pack() or hammings_pack() and used with
    dots_packed(), hammings_packed(), jaccards_packed(), angulars_packed(),
    euclideans_packed(), or the @ operator.
    """

    @property
    def columns(self) -> int:
        """Number of packed vectors, the columns of the result."""
        ...

    @property
    def depth(self) -> int:
        """Number of columns in the original matrix."""
        ...

    @property
    def dtype(self) -> _IntegralTypeName | _FloatTypeName | _ComplexTypeName:
        """Data type of the packed matrix (like 'bf16' or 'i8')."""
        ...

    @property
    def nbytes(self) -> int:
        """Size of the packed buffer in bytes."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions (columns, depth) read from the packed buffer header."""
        ...

    @classmethod
    def pack_size(
        cls,
        columns: int,
        depth: int,
        /,
        dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType,
        capabilities: Capability | int | None = None,
    ) -> int:
        """Return packed buffer size in bytes for given dimensions, dtype and capabilities, GPU ones included."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

class MaxSimPackedMatrix:
    """Opaque pre-packed matrix for MaxSim late-interaction scoring."""

    @property
    def vectors(self) -> int:
        """Number of vectors."""
        ...

    @property
    def depth(self) -> int:
        """Number of dimensions per vector (depth)."""
        ...

    @property
    def dtype(self) -> _IntegralTypeName | _FloatTypeName | _ComplexTypeName:
        """Data type of the packed vectors."""
        ...

    @property
    def nbytes(self) -> int:
        """Size of the packed buffer in bytes."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions (vectors, depth) read from the packed buffer header."""
        ...

    @classmethod
    def pack_size(cls, vectors: int, depth: int, /, dtype: _FloatTypeName | _MiniFloatType) -> int:
        """Return packed buffer size in bytes for given dimensions and dtype."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

class AttentionPackedMatrix:
    """Opaque pre-packed ragged KV-cache for scaled-dot-product attention.

    Created by attention_pack() and used with attention_packed().
    """

    @property
    def segments(self) -> int:
        """Number of ragged segments."""
        ...

    @property
    def key_value_head_count(self) -> int:
        """Number of KV heads."""
        ...

    @property
    def depth(self) -> int:
        """Channels per head."""
        ...

    @property
    def tokens(self) -> int:
        """Total KV tokens across segments."""
        ...

    @property
    def shape(self) -> tuple[int, int, int]:
        """Dimensions (key_value_head_count, depth, segments) read from the packed buffer header."""
        ...

    @property
    def dtype(self) -> _IntegralTypeName | _FloatTypeName | _ComplexTypeName:
        """Data type of the packed KV-cache."""
        ...

    @property
    def nbytes(self) -> int:
        """Size of the packed buffer in bytes."""
        ...

    @classmethod
    def pack_size(
        cls,
        key_value_head_count: int,
        depth: int,
        tokens: int,
        segments: int,
        /,
        dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType,
        capabilities: Capability | int | None = None,
    ) -> int:
        """Return packed buffer size in bytes for given dimensions, dtype and capabilities, GPU ones included."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

# endregion Forward Declarations and Shared Types

# region Capabilities

class Capability(IntFlag):
    """CPU SIMD capabilities, one bit each, as the `capabilities_*` functions report and set them."""

    SERIAL = ...
    """Serial code without SIMD, available on every CPU."""
    NEON = ...
    """Arm NEON, the AArch64 Advanced SIMD baseline."""
    NEONHALF = ...
    """NEON with half-precision arithmetic, FEAT_FP16."""
    NEONBFDOT = ...
    """NEON with BFloat16 dot products, FEAT_BF16."""
    NEONFHM = ...
    """NEON with half-precision multiply-accumulate into single precision, FEAT_FHM."""
    NEONSDOT = ...
    """NEON with 8-bit integer dot products, FEAT_DotProd."""
    NEONFP8 = ...
    """NEON with FP8 dot products, FEAT_FP8DOT4."""
    SVE = ...
    """Arm Scalable Vector Extension."""
    SVEHALF = ...
    """SVE with half-precision arithmetic."""
    SVESDOT = ...
    """SVE with 8-bit integer matrix multiplies, FEAT_I8MM."""
    SVEBFDOT = ...
    """SVE with BFloat16 dot products."""
    SVE2 = ...
    """Arm SVE2."""
    SME = ...
    """Arm Scalable Matrix Extension."""
    SMEF64 = ...
    """SME with double-precision outer products, FEAT_SME_F64F64."""
    SMEBI32 = ...
    """SME with binary outer products into 32-bit integers, SME_BI32I32."""
    HASWELL = ...
    """x86 AVX2, FMA, F16C, BMI1 and BMI2, as on Intel Haswell."""
    ALDER = ...
    """`HASWELL` plus AVX-VNNI, as on Intel Alder Lake."""
    SIERRA = ...
    """`ALDER` plus AVX-VNNI-INT8, as on Intel Sierra Forest."""
    SKYLAKE = ...
    """`HASWELL` plus AVX-512 F, DQ, BW and VL, as on Intel Skylake-X."""
    ICELAKE = ...
    """`SKYLAKE` plus AVX-512 VNNI, IFMA, BITALG, VBMI, VBMI2 and VPOPCNTDQ, as on Ice Lake."""
    GENOA = ...
    """`SKYLAKE` plus AVX-512 BF16 and VNNI, as on AMD Genoa."""
    TURIN = ...
    """`GENOA` plus AVX-512 VBMI2 and VP2INTERSECT, as on AMD Turin."""
    SAPPHIRE = ...
    """`SKYLAKE` plus AVX-512 FP16, as on Intel Sapphire Rapids."""
    DIAMOND = ...
    """`SAPPHIRE` plus AVX10.2, as on Intel Diamond Rapids."""
    SAPPHIREAMX = ...
    """`SAPPHIRE` and `GENOA` plus AMX tiles with BF16 and INT8, as on Intel Sapphire Rapids."""
    GRANITEAMX = ...
    """`SAPPHIREAMX` plus AMX-FP16, as on Intel Granite Rapids."""
    DIAMONDAMX = ...
    """`SAPPHIREAMX` plus AMX-FP8 and AMX-AVX512, as on Intel Diamond Rapids."""
    RVV = ...
    """RISC-V Vector extension."""
    RVVBF16 = ...
    """RVV with BFloat16 widening multiply-accumulate, Zvfbfwma."""
    RVVHALF = ...
    """RVV with half-precision arithmetic, Zvfh."""
    RVVBB = ...
    """RVV with basic bit manipulation, Zvbb."""
    V128 = ...
    """WebAssembly 128-bit SIMD."""
    V128RELAXED = ...
    """WebAssembly Relaxed SIMD."""
    POWERVSX = ...
    """IBM Power VSX at ISA 3.0, as on POWER9."""
    LOONGSONASX = ...
    """LoongArch LASX 256-bit SIMD."""

class _Dispatch(TypedDict, total=False):
    """Keywords every function that runs kernels accepts, after its own."""

    capabilities: Capability | int | None
    """Capabilities to run within: every CPU one by default, or for packed operands the mask that packed them."""
    stream: int | None
    """A GPU stream pointer, left None on the CPU."""

# Capabilities of the CPU and each GPU, and GPU streams, the only calls taking a device ordinal.
def cpu_capabilities_detected() -> Capability:
    """What this CPU can execute."""
    ...

def cpu_capabilities_compiled() -> Capability:
    """What this binary holds CPU kernels for."""
    ...

def cpu_capabilities_enabled() -> Capability:
    """What CPU kernels run within: both of the above, settled as the library loads."""
    ...

def cpu_configure_thread(capabilities: Capability, /) -> None:
    """Prepare the calling thread for the CPU kernels of `capabilities`."""
    ...

def cuda_count_devices() -> int:
    """How many CUDA devices this process sees, zero without the runtime or its kernels."""
    ...

def cuda_capabilities_detected(ordinal: int, /) -> Capability:
    """What CUDA device `ordinal` can execute, raising ValueError past the last one."""
    ...

def cuda_capabilities_compiled() -> Capability:
    """What this binary holds CUDA kernels for."""
    ...

def cuda_capabilities_enabled(ordinal: int, /) -> Capability:
    """The mask CUDA device `ordinal` dispatches with, raising ValueError past the last one."""
    ...

def stream_init_cuda(ordinal: int, /) -> int:
    """A new `cudaStream_t` on CUDA device `ordinal` as an integer, raising ValueError past the last one."""
    ...

def stream_free_cuda(stream: int, /) -> None:
    """Free a stream `stream_init_cuda` made, once `synchronize` joined its work."""
    ...

def rocm_count_devices() -> int:
    """How many ROCm devices this process sees, zero without the runtime or its kernels."""
    ...

def rocm_capabilities_detected(ordinal: int, /) -> Capability:
    """What ROCm device `ordinal` can execute, raising ValueError past the last one."""
    ...

def rocm_capabilities_compiled() -> Capability:
    """What this binary holds ROCm kernels for."""
    ...

def rocm_capabilities_enabled(ordinal: int, /) -> Capability:
    """The mask ROCm device `ordinal` dispatches with, raising ValueError past the last one."""
    ...

def stream_init_rocm(ordinal: int, /) -> int:
    """A new `hipStream_t` on ROCm device `ordinal` as an integer, raising ValueError past the last one."""
    ...

def stream_free_rocm(stream: int, /) -> None:
    """Free a stream `stream_init_rocm` made, once `synchronize` joined its work."""
    ...

def metal_count_devices() -> int:
    """How many Metal devices this process sees, zero without the runtime or its kernels."""
    ...

def metal_capabilities_detected(ordinal: int, /) -> Capability:
    """What Metal device `ordinal` can execute, raising ValueError past the last one."""
    ...

def metal_capabilities_compiled() -> Capability:
    """What this binary holds Metal kernels for."""
    ...

def metal_capabilities_enabled(ordinal: int, /) -> Capability:
    """The mask Metal device `ordinal` dispatches with, raising ValueError past the last one."""
    ...

def stream_init_metal(ordinal: int, /) -> int:
    """A new `id<MTLCommandQueue>` on Metal device `ordinal` as an integer, raising ValueError past the last one."""
    ...

def stream_free_metal(stream: int, /) -> None:
    """Free a stream `stream_init_metal` made, once `synchronize` joined its work."""
    ...

def synchronize(capabilities: Capability | int, /, stream: int | None = None) -> None:
    """Wait for everything queued on `stream` of the device `capabilities` describes, None for its default."""
    ...

# Kernel pointer accessors, to the CPU's enabled capabilities.
# Each takes a trailing `nk_stream_t stream`, null on the CPU, and returns an `nk_status_t`.
def pointer_to_euclidean(dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType, /) -> int:
    """Return the address of the `euclidean` kernel for `dtype`, as an integer."""
    ...

def pointer_to_sqeuclidean(dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType, /) -> int:
    """Return the address of the `sqeuclidean` kernel for `dtype`, as an integer."""
    ...

def pointer_to_angular(dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType, /) -> int:
    """Return the address of the `angular` kernel for `dtype`, as an integer."""
    ...

def pointer_to_inner(dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType, /) -> int:
    """Return the address of the `inner` kernel for `dtype`, as an integer."""
    ...

def pointer_to_dot(dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType, /) -> int:
    """Return the address of the `dot` kernel for `dtype`, as an integer."""
    ...

def pointer_to_vdot(dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType, /) -> int:
    """Return the address of the `vdot` kernel for `dtype`, as an integer."""
    ...

def pointer_to_jensenshannon(dtype: _FloatTypeName | _MiniFloatType, /) -> int:
    """Return the address of the `jensenshannon` kernel for `dtype`, as an integer."""
    ...

def pointer_to_kullbackleibler(dtype: _FloatTypeName | _MiniFloatType, /) -> int:
    """Return the address of the `kullbackleibler` kernel for `dtype`, as an integer."""
    ...

# endregion Capabilities

# region Pairwise Distances

def cdist(
    a: _BufferType,
    b: _BufferType,
    /,
    metric: _MetricName = "euclidean",
    *,
    threads: int = 1,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | complex | Tensor | None:
    """Compute distances between every row of `a` and every row of `b`.

    Similar to `scipy.spatial.distance.cdist`. Two vectors give a scalar. Matrices give a
    [rows,columns] Tensor with a row per row of `a` and a column per row of `b`, or fill `out` and
    return None. GPU tensors raise BufferError; pack `b` with dots_pack() and call the `_packed`
    functions instead.

    Args:
      a: First vector or matrix.
      b: Second vector or matrix, of the dtype of `a`.
      metric: Name of the distance, 'euclidean' by default.
      threads: Worker threads.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output matrix for the distances.
      out_dtype: Result dtype, by default that of `out`, else float64, complex128 for complex input.
    """
    ...

# endregion Pairwise Distances

# region Vector Dot Products
# Vector-vector dot products for real and complex numbers.

def inner(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | complex | Tensor | None:
    """Compute the inner product of two real or complex vectors.

    Similar to `numpy.inner`. Two vectors give a scalar. Matrices pair row i of `a` with row i of
    `b`, a single row pairing with every row of the other, into a new Tensor of one result per pair,
    or into `out`, returning None.

    Args:
      a: First real or complex vector or matrix.
      b: Second real or complex vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else float64, complex128 for complex input.
    """
    ...

def dot(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | complex | Tensor | None:
    """Compute the dot product of two real or complex vectors.

    Similar to `numpy.dot`. Two vectors give a scalar. Matrices pair row i of `a` with row i of `b`,
    a single row pairing with every row of the other, into a new Tensor of one result per pair, or
    into `out`, returning None.

    Args:
      a: First real or complex vector or matrix.
      b: Second real or complex vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else float64, complex128 for complex input.
    """
    ...

def vdot(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _ComplexTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _ComplexTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> complex | Tensor | None:
    """Compute the dot product of two complex vectors, conjugating the first.

    Similar to `numpy.vdot`. Two vectors give a scalar. Matrices pair row i of `a` with row i of
    `b`, a single row pairing with every row of the other, into a new Tensor of one result per pair,
    or into `out`, returning None.

    Args:
      a: First complex vector or matrix.
      b: Second complex vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else complex128.
    """
    ...

# endregion Vector Dot Products

# region Spatial Distance Metrics
# Vector-vector spatial distance metrics for real and integer numbers.

def sqeuclidean(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor | None:
    """Compute the squared Euclidean distance between two vectors.

    Similar to `scipy.spatial.distance.sqeuclidean`. Two vectors give a scalar. Matrices pair row i
    of `a` with row i of `b`, a single row pairing with every row of the other, into a new Tensor of
    one result per pair, or into `out`, returning None.

    Args:
      a: First real or integer vector or matrix.
      b: Second real or integer vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else float64.
    """
    ...

def angular(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor | None:
    """Compute the angular distance, also known as cosine distance, between two vectors.

    Similar to `scipy.spatial.distance.cosine`. Two vectors give a scalar. Matrices pair row i of
    `a` with row i of `b`, a single row pairing with every row of the other, into a new Tensor of
    one result per pair, or into `out`, returning None.

    Args:
      a: First real or integer vector or matrix.
      b: Second real or integer vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else float64.
    """
    ...

def euclidean(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor | None:
    """Compute the Euclidean distance between two vectors.

    Similar to `scipy.spatial.distance.euclidean`. Two vectors give a scalar. Matrices pair row i of
    `a` with row i of `b`, a single row pairing with every row of the other, into a new Tensor of
    one result per pair, or into `out`, returning None.

    Args:
      a: First real or integer vector or matrix.
      b: Second real or integer vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else float64.
    """
    ...

# endregion Spatial Distance Metrics

# region Binary Similarity
# Vector-vector similarity functions for binary vectors.

def hamming(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor | None:
    """Compute the Hamming distance between two binary vectors.

    Similar to `scipy.spatial.distance.hamming`. Two vectors give a scalar. Matrices pair row i of
    `a` with row i of `b`, a single row pairing with every row of the other, into a new Tensor of
    one result per pair, or into `out`, returning None.

    Args:
      a: First binary vector or matrix.
      b: Second binary vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else float64.
    """
    ...

def jaccard(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor | None:
    """Compute the Jaccard distance, the bitwise Tanimoto metric, between two binary vectors.

    Similar to `scipy.spatial.distance.jaccard`. Two vectors give a scalar. Matrices pair row i of
    `a` with row i of `b`, a single row pairing with every row of the other, into a new Tensor of
    one result per pair, or into `out`, returning None.

    Args:
      a: First binary vector or matrix.
      b: Second binary vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else float64.
    """
    ...

# endregion Binary Similarity

# region Probability Distances
# Vector-vector similarity between probability distributions.

def jensenshannon(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor | None:
    """Compute the Jensen-Shannon distance between two probability distributions.

    Similar to `scipy.spatial.distance.jensenshannon`. Two vectors give a scalar. Matrices pair row
    i of `a` with row i of `b`, a single row pairing with every row of the other, into a new Tensor
    of one result per pair, or into `out`, returning None.

    Args:
      a: First probability vector or matrix.
      b: Second probability vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else float64.
    """
    ...

def jsd(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor | None:
    """Alias of `jensenshannon`."""
    ...

def kullbackleibler(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor | None:
    """Compute the Kullback-Leibler divergence between two probability distributions.

    Similar to `scipy.special.kl_div`. Two vectors give a scalar. Matrices pair row i of `a` with
    row i of `b`, a single row pairing with every row of the other, into a new Tensor of one result
    per pair, or into `out`, returning None.

    Args:
      a: First probability vector or matrix.
      b: Second probability vector or matrix, of the dtype of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      out: Output vector for the results.
      out_dtype: Result dtype, by default that of `out`, else float64.
    """
    ...

def kld(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor | None:
    """Alias of `kullbackleibler`."""
    ...

# endregion Probability Distances

# region Curved Space Metrics
# Vector-vector similarity between vectors in curved spaces.

def bilinear(
    a: _BufferType,
    b: _BufferType,
    metric_tensor: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float:
    """Compute the bilinear form aᵀ × M × b of two vectors under a metric tensor M.

    Similar to `numpy.dot(a, metric_tensor @ b)`.

    Args:
      a: First vector.
      b: Second vector.
      metric_tensor: Matrix M defining the bilinear form.
      dtype: Input dtype to read every buffer as, overriding the one they report.
    """
    ...

def mahalanobis(
    a: _BufferType,
    b: _BufferType,
    inverse_covariance: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float:
    """Compute the Mahalanobis distance between two vectors given an inverse covariance matrix.

    Similar to `scipy.spatial.distance.mahalanobis`.

    Args:
      a: First vector.
      b: Second vector.
      inverse_covariance: Inverse of the covariance matrix.
      dtype: Input dtype to read every buffer as, overriding the one they report.
    """
    ...

# endregion Curved Space Metrics

# region Geospatial

# Squared distances between every row and every column point in 3D, exact on int32 whenever every
# axis difference fits 32 bits, as Earth-fixed centimetres below 4,359 km of altitude do.
def sqeuclideans3d(a: _BufferType, b: _BufferType, /, *, threads: int = 1, **dispatch: Unpack[_Dispatch]) -> Tensor:
    """Compute [rows,columns] squared distances of [rows,3] and [columns,3] points, both int32 or both float64."""
    ...

class HaversinesPackedMatrix:
    """Points packed by haversines_pack() as unit vectors on the mean sphere.

    The kernels read the point count from a header ahead of the planar arrays, so the only view is
    the whole pack, and a key selecting fewer points raises ValueError.
    """

    @property
    def points(self) -> int:
        """Number of packed points."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions (points, 2) of the packed latitude and longitude pairs."""
        ...

    @property
    def nbytes(self) -> int:
        """Size of the pack in bytes."""
        ...

    @classmethod
    def pack_size(cls, points: int, /) -> int:
        """Return the bytes a pack of the given number of points takes."""
        ...

    def __len__(self) -> int:
        """Return the number of packed points."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

class VincentysPackedMatrix:
    """Points packed by vincentys_pack() as longitudes and reduced latitudes on the WGS-84 ellipsoid.

    The kernels read the point count from a header ahead of the planar arrays, so the only view is
    the whole pack, and a key selecting fewer points raises ValueError.
    """

    @property
    def points(self) -> int:
        """Number of packed points."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions (points, 2) of the packed latitude and longitude pairs."""
        ...

    @property
    def nbytes(self) -> int:
        """Size of the pack in bytes."""
        ...

    @classmethod
    def pack_size(cls, points: int, /) -> int:
        """Return the bytes a pack of the given number of points takes."""
        ...

    def __len__(self) -> int:
        """Return the number of packed points."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

# Pack [points] latitudes and longitudes in radians, cast to float64, the same on every machine.
def haversines_pack(
    latitudes: _BufferType, longitudes: _BufferType, /, **dispatch: Unpack[_Dispatch]
) -> HaversinesPackedMatrix:
    """Pack every point as a unit vector on the mean sphere, for haversines_packed()."""
    ...

def vincentys_pack(
    latitudes: _BufferType, longitudes: _BufferType, /, **dispatch: Unpack[_Dispatch]
) -> VincentysPackedMatrix:
    """Pack every point as its longitude and reduced latitude on the WGS-84 ellipsoid, for vincentys_packed()."""
    ...

# The many-to-many geodesics return [rows,columns] float64 distances in metres.
def haversines_packed(a: HaversinesPackedMatrix, b: HaversinesPackedMatrix, /, **dispatch: Unpack[_Dispatch]) -> Tensor:
    """Compute great-circle distances on the sphere of the IUGG mean Earth radius, 6,371,008.8 m."""
    ...

def vincentys_packed(a: VincentysPackedMatrix, b: VincentysPackedMatrix, /, **dispatch: Unpack[_Dispatch]) -> Tensor:
    """Compute geodesic distances on the WGS-84 ellipsoid with Vincenty's inverse formula."""
    ...

# endregion Geospatial

# region Sparse Similarity
# Vector-vector similarity between sparse vectors.

def intersect(a: _BufferType, b: _BufferType, /, **dispatch: Unpack[_Dispatch]) -> int:
    """Count the elements two sorted integer arrays share.

    Similar to `numpy.intersect1d`, returning the size of the intersection rather than its elements.
    """
    ...

def sparse_dot(
    a_indices: _BufferType,
    a_values: _BufferType,
    b_indices: _BufferType,
    b_values: _BufferType,
    /,
    **dispatch: Unpack[_Dispatch],
) -> float:
    """Compute the weighted dot product of two sparse vectors over their shared indices.

    Args:
      a_indices: First sorted index array, uint16 or uint32.
      a_values: Weights of `a_indices`, bfloat16 or float32.
      b_indices: Second sorted index array, of the dtype of `a_indices`.
      b_values: Weights of `b_indices`, of the dtype of `a_values`.
    """
    ...

# endregion Sparse Similarity

# region Tensor Constructors
type _DTypeLike = _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType

def from_pointer(
    address: int,
    shape: int | tuple[int, ...],
    dtype: _DTypeLike,
    /,
    *,
    strides: tuple[int, ...] | None = None,
    owner: Any = None,
) -> Tensor:
    """Create a zero-copy Tensor view over memory at a raw address.

    Args:
      address: Integer address of the data buffer.
      shape: Shape of the view, an int or a tuple of ints.
      dtype: Element dtype, like 'float32' or 'bf16'.
      strides: Byte strides, C-contiguous when None.
      owner: Object the view keeps alive, so the memory it owns is not collected.
    """
    ...

def from_dlpack(obj: Any, /) -> Tensor:
    """Zero-copy import a tensor from any DLPack-compatible producer.

    Accepts a PyCapsule named 'dltensor' or 'dltensor_versioned', or any object
    with a ``__dlpack__`` method (e.g. ``torch.Tensor``, ``numpy.ndarray``,
    ``jax.Array``). Only CPU tensors are supported.
    """
    ...

def empty(
    shape: int | tuple[int, ...],
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Create an uninitialized Tensor of the given shape and dtype."""
    ...

def zeros(
    shape: int | tuple[int, ...],
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Create a Tensor of the given shape and dtype filled with zeros."""
    ...

def ones(
    shape: int | tuple[int, ...],
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Create a Tensor of the given shape and dtype filled with ones."""
    ...

def full(
    shape: int | tuple[int, ...],
    fill_value: int | float,
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Create a Tensor of the given shape and dtype filled with `fill_value`."""
    ...

def iota(
    shape: int | tuple[int, ...],
    /,
    seed: int | float = 0,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Tensor filled with ``seed, seed+1, seed+2, ...`` reshaped to ``shape``."""
    ...

def diagonal(
    n: int,
    /,
    seed: int | float = 1,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Square ``n x n`` tensor with ``seed`` on the main diagonal and zeros elsewhere."""
    ...

def hash(
    shape: int | tuple[int, ...],
    /,
    seed: int = 0,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Tensor filled with deterministic pseudo-random bits (shape, dtype, seed → reproducible bytes)."""
    ...

def astype(
    a: _BufferType,
    dtype: _DTypeLike,
    /,
    *,
    out: _BufferType | None = None,
) -> Tensor:
    """Cast an N-D buffer to ``dtype`` without first copying it into a Tensor.

    A supplied ``out`` must be writable, C-contiguous, have the exact input
    shape and requested dtype, and must not overlap ``a``. Returns ``out`` when
    provided, otherwise a new Tensor.
    """
    ...

# endregion Tensor Constructors

# region Reductions

def moments(
    a: _BufferType,
    /,
    *,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> tuple[float, float]:
    """Compute the sum and the sum of squares of all elements, returned as a pair."""
    ...

def minmax(
    a: _BufferType,
    /,
    *,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> tuple[float, int, float, int] | None:
    """Find the smallest and largest elements with their indices.

    Returns `(min_value, min_index, max_value, max_index)`, or None when every element is NaN.
    """
    ...

def sum(
    a: _BufferType,
    /,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | int | Tensor:
    """Return the sum of all elements, or per-slice sums along one or more axes.

    Args:
      a: Input Tensor, NumPy array or any buffer-protocol object.
      axis: Axis or axes to reduce along, or None to reduce every element to a scalar.
      keepdims: Keep each reduced axis as a size-1 dimension.
      out: Preallocated output Tensor for the result.
      dtype: Input dtype to read `a` as, overriding the one it reports.
    """
    ...

def norm(
    a: _BufferType,
    /,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | Tensor:
    """Return the L2 norm of all elements, or per-slice norms along one or more axes.

    Args:
      a: Input Tensor, NumPy array or any buffer-protocol object.
      axis: Axis or axes to reduce along, or None to reduce every element to a scalar.
      keepdims: Keep each reduced axis as a size-1 dimension.
      out: Preallocated output Tensor for the result.
      dtype: Input dtype to read `a` as, overriding the one it reports.
    """
    ...

def min(
    a: _BufferType,
    /,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | int | Tensor | None:
    """Return the minimum element, or per-slice minimums along one or more axes.

    The scalar result is None when every element is NaN.

    Args:
      a: Input Tensor, NumPy array or any buffer-protocol object.
      axis: Axis or axes to reduce along, or None to reduce every element to a scalar.
      keepdims: Keep each reduced axis as a size-1 dimension.
      out: Preallocated output Tensor for the result.
      dtype: Input dtype to read `a` as, overriding the one it reports.
    """
    ...

def max(
    a: _BufferType,
    /,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float | int | Tensor | None:
    """Return the maximum element, or per-slice maximums along one or more axes.

    The scalar result is None when every element is NaN.

    Args:
      a: Input Tensor, NumPy array or any buffer-protocol object.
      axis: Axis or axes to reduce along, or None to reduce every element to a scalar.
      keepdims: Keep each reduced axis as a size-1 dimension.
      out: Preallocated output Tensor for the result.
      dtype: Input dtype to read `a` as, overriding the one it reports.
    """
    ...

def argmin(
    a: _BufferType,
    /,
    axis: int | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> int | Tensor | None:
    """Return the index of the minimum element, or per-slice indices along one axis.

    The scalar result is None when every element is NaN.

    Args:
      a: Input Tensor, NumPy array or any buffer-protocol object.
      axis: Axis to reduce along, or None to reduce every element to a scalar.
      keepdims: Keep each reduced axis as a size-1 dimension.
      out: Preallocated output Tensor for the result.
      dtype: Input dtype to read `a` as, overriding the one it reports.
    """
    ...

def argmax(
    a: _BufferType,
    /,
    axis: int | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> int | Tensor | None:
    """Return the index of the maximum element, or per-slice indices along one axis.

    The scalar result is None when every element is NaN.

    Args:
      a: Input Tensor, NumPy array or any buffer-protocol object.
      axis: Axis to reduce along, or None to reduce every element to a scalar.
      keepdims: Keep each reduced axis as a size-1 dimension.
      out: Preallocated output Tensor for the result.
      dtype: Input dtype to read `a` as, overriding the one it reports.
    """
    ...

# endregion Reductions

# region Vector Math

def fma(
    a: _BufferType,
    b: _BufferType,
    c: _BufferType,
    /,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    *,
    alpha: float = 1,
    beta: float = 1,
    out: _BufferType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Compute alpha × a × b + beta × c element-wise over three tensors of one shape.

    Returns a new Tensor, or None after writing into `out`.

    Args:
      a: First tensor, of any rank.
      b: Second tensor, of the shape of `a`.
      c: Third tensor, of the shape of `a`.
      dtype: Input dtype to read every buffer as, overriding the one they report.
      alpha: Scale of the product of `a` and `b`.
      beta: Scale of `c`.
      out: Output tensor for the result.
    """
    ...

def blend(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    *,
    alpha: float = 1,
    beta: float = 1,
    out: _BufferType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Compute alpha × a + beta × b element-wise over two tensors of one shape.

    Returns a new Tensor, or None after writing into `out`.

    Args:
      a: First tensor, of any rank.
      b: Second tensor, of the shape of `a`.
      dtype: Input dtype to read both buffers as, overriding the one they report.
      alpha: Scale of `a`.
      beta: Scale of `b`.
      out: Output tensor for the result.
    """
    ...

# endregion Vector Math

# region Trigonometry

def sin(
    a: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Compute the sine of every element.

    Returns a new Tensor, or None after writing into `out`.

    Args:
      a: Input tensor of any rank, angles in radians.
      dtype: Input dtype to read `a` as, overriding the one it reports.
      out: Output tensor for the result.
    """
    ...

def cos(
    a: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Compute the cosine of every element.

    Returns a new Tensor, or None after writing into `out`.

    Args:
      a: Input tensor of any rank, angles in radians.
      dtype: Input dtype to read `a` as, overriding the one it reports.
      out: Output tensor for the result.
    """
    ...

def atan(
    a: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Compute the arctangent of every element, in radians.

    Returns a new Tensor, or None after writing into `out`.

    Args:
      a: Input tensor of any rank.
      dtype: Input dtype to read `a` as, overriding the one it reports.
      out: Output tensor for the result.
    """
    ...

# endregion Trigonometry

# region Elementwise Arithmetic

def scale(
    a: _BufferType,
    /,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    *,
    alpha: float = 1,
    beta: float = 0,
    out: _BufferType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Compute alpha × a + beta element-wise over a tensor of any rank.

    Returns a new Tensor, or None after writing into `out`.

    Args:
      a: Input tensor of any rank.
      dtype: Input dtype to read `a` as, overriding the one it reports.
      alpha: Multiplicative scale.
      beta: Additive offset.
      out: Output tensor for the result.
    """
    ...

def add(
    a: _BufferType | float | int,
    b: _BufferType | float | int,
    /,
    *,
    out: _BufferType | None = None,
    a_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    b_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Add two tensors, or a tensor and a scalar, element-wise with NumPy broadcasting.

    Equivalent to `a + b`. Returns a new Tensor, or None after writing into `out`.

    Args:
      a: First operand, a tensor or a scalar.
      b: Second operand, a tensor or a scalar.
      out: Output buffer for the result.
      a_dtype: Override of the dtype of `a`.
      b_dtype: Override of the dtype of `b`.
      out_dtype: Override of the dtype of the result.
    """
    ...

def multiply(
    a: _BufferType | float | int,
    b: _BufferType | float | int,
    /,
    *,
    out: _BufferType | None = None,
    a_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    b_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Multiply two tensors, or a tensor and a scalar, element-wise with NumPy broadcasting.

    Equivalent to `a * b`. Returns a new Tensor, or None after writing into `out`.

    Args:
      a: First operand, a tensor or a scalar.
      b: Second operand, a tensor or a scalar.
      out: Output buffer for the result.
      a_dtype: Override of the dtype of `a`.
      b_dtype: Override of the dtype of `b`.
      out_dtype: Override of the dtype of the result.
    """
    ...

# endregion Elementwise Arithmetic

# region Fused Neural Primitives

def rmsnorm(
    x: _BufferType,
    /,
    gamma: _BufferType | None = None,
    *,
    out: _BufferType | None = None,
    groups: int = 1,
    epsilon: float = 1e-6,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Grouped RMSNorm, `y = x * rsqrt(mean(x^2) + epsilon) * gamma`, in place unless `out` is given.

    Each row, spanning all axes but the last, holds `groups` independent sub-vectors of length
    `x.shape[-1] // groups`, normalized separately.

    Args:
      x: Input of dtype float32, bfloat16, or e4m3, with a contiguous last axis.
      gamma: Per-column float32 gain of length `columns`, defaulting to unit scale.
      out: Output buffer, same shape and dtype as `x`, may alias `x`.
      groups: Independent sub-vectors per row.
      epsilon: Variance epsilon. For e4m3, pass it in code units and fold the output scale into `gamma`.
    """
    ...

def rmscast(
    x: _BufferType,
    /,
    gamma: _BufferType | None = None,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType,
    out: _BufferType | None = None,
    groups: int = 1,
    epsilon: float = 1e-6,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Grouped RMSNorm of dot products back into the `dtype` they multiplied, rounding like the cast.

    Each row, spanning all axes but the last, holds `groups` independent sub-vectors of length
    `x.shape[-1] // groups`, normalized separately.

    Args:
      x: Dot products with a contiguous last axis: float32 for bfloat16, float16, e4m3, e5m2, e2m3
        and e3m2 outputs, float64 for float32, int32 for int8 and uint32 for uint8.
      gamma: Per-column float32 gain of length `columns`, defaulting to unit scale.
      dtype: Output dtype, which names the dtype `x` must have.
      out: Output buffer of the shape of `x` and of `dtype`.
      groups: Independent sub-vectors per row.
      epsilon: Variance epsilon, added to the mean square of the raw input.

    Integer outputs round to nearest even and saturate, with `gamma` carrying the quantization
    scale; uint8 suits only non-negative activations.
    """
    ...

def swiglu(
    gate: _BufferType,
    /,
    up: _BufferType | None = None,
    *,
    out: _BufferType | None = None,
    gate_scale: float = 1.0,
    output_scale: float = 1.0,
    **dispatch: Unpack[_Dispatch],
) -> Tensor | None:
    """Fused SwiGLU, `y = silu(gate) * up`, in place unless `out` is given.

    The gate is read at `gate_scale * gate` and the result is multiplied by `output_scale`.
    With `up` omitted this reduces to plain SiLU.

    Args:
      gate: Gate input of dtype float32, bfloat16, or e4m3, with a contiguous last axis.
      up: Up input, same shape and dtype as `gate`, defaulting to plain SiLU.
      out: Output buffer, same shape and dtype as `gate`, may alias `gate`.
      gate_scale: Tensor scale of `gate`.
      output_scale: Multiplier on the result.
    """
    ...

# endregion Fused Neural Primitives

# region Symmetric Pairwise Operations
def dots_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute the Gram matrix, the symmetric all-pairs dot products, of the rows of `vectors`.

    Only the upper triangle of the [count,count] result is guaranteed to be initialized. Returns a
    new Tensor, or `out` when given. On a GPU the call returns before the kernel finishes.

    Args:
      vectors: Input matrix of shape [count,depth], or a CUDA or ROCm Tensor from from_dlpack() to
        run on that GPU.
      dtype: Input dtype to read `vectors` as, overriding the one it reports.
      out: C-contiguous [count,count] output, required on the same GPU for GPU `vectors`.
      rows_begin: First row to compute, 0 when None.
      rows_end: One past the last row to compute, `count` when None; only the upper triangle
        overlapping the rows in range is filled.
      threads: Worker threads, ignored on a GPU.
    """
    ...

def hammings_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute the symmetric all-pairs Hamming-distance matrix of the rows of `vectors`.

    Only the upper triangle of the [count,count] uint32 result is guaranteed to be initialized.
    Returns a new Tensor, or `out` when given.

    Args:
      vectors: Input matrix of shape [count,depth], with 'uint1' bits packed into uint8 bytes.
      dtype: Input dtype to read `vectors` as, overriding the one it reports.
      out: C-contiguous [count,count] output of dtype uint32.
      rows_begin: First row to compute, 0 when None.
      rows_end: One past the last row to compute, `count` when None; only the upper triangle
        overlapping the rows in range is filled.
      threads: Worker threads.
    """
    ...

def jaccards_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute the symmetric all-pairs Jaccard-distance matrix of the rows of `vectors`.

    Only the upper triangle of the [count,count] result is guaranteed to be initialized. Returns a
    new Tensor, or `out` when given.

    Args:
      vectors: Input matrix of shape [count,depth], with 'uint1' bits packed into uint8 bytes.
      dtype: Input dtype to read `vectors` as, overriding the one it reports.
      out: C-contiguous [count,count] output.
      rows_begin: First row to compute, 0 when None.
      rows_end: One past the last row to compute, `count` when None; only the upper triangle
        overlapping the rows in range is filled.
      threads: Worker threads.
    """
    ...

def angulars_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute the symmetric all-pairs angular-distance matrix of the rows of `vectors`.

    Only the upper triangle of the [count,count] result is guaranteed to be initialized. Returns a
    new Tensor, or `out` when given. On a GPU the call returns before the kernel finishes.

    Args:
      vectors: Input matrix of shape [count,depth], or a CUDA or ROCm Tensor from from_dlpack() to
        run on that GPU.
      dtype: Input dtype to read `vectors` as, overriding the one it reports.
      out: C-contiguous [count,count] output, required on the same GPU for GPU `vectors`.
      rows_begin: First row to compute, 0 when None.
      rows_end: One past the last row to compute, `count` when None; only the upper triangle
        overlapping the rows in range is filled.
      threads: Worker threads, ignored on a GPU.
    """
    ...

def euclideans_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute the symmetric all-pairs Euclidean-distance matrix of the rows of `vectors`.

    Only the upper triangle of the [count,count] result is guaranteed to be initialized. Returns a
    new Tensor, or `out` when given. On a GPU the call returns before the kernel finishes.

    Args:
      vectors: Input matrix of shape [count,depth], or a CUDA or ROCm Tensor from from_dlpack() to
        run on that GPU.
      dtype: Input dtype to read `vectors` as, overriding the one it reports.
      out: C-contiguous [count,count] output, required on the same GPU for GPU `vectors`.
      rows_begin: First row to compute, 0 when None.
      rows_end: One past the last row to compute, `count` when None; only the upper triangle
        overlapping the rows in range is filled.
      threads: Worker threads, ignored on a GPU.
    """
    ...

# endregion Symmetric Pairwise Operations

# region Packed Matrix Operations

def dots_pack(
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> PackedMatrix:
    """Pack a matrix with contiguous rows for repeated dot-product cross operations.

    Packing fuses no transposition, so transpose `w.T` into a contiguous copy first. The pack serves
    dots_packed(), angulars_packed(), euclideans_packed() and the `@` operator of Tensor.

    Args:
      b: Source matrix of shape [columns,depth], or a CUDA or ROCm Tensor from from_dlpack() to pack
        on that GPU.
      dtype: Packing dtype, inferred from `b` when None.
      out: C-contiguous Tensor on the GPU of `b` holding the pack, required for a GPU `b`, of at
        least PackedMatrix.pack_size() bytes.
      threads: Worker threads, ignored on a GPU.
    """
    ...

def dots_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute dot products between rows of `a` and a matrix packed by dots_pack(), like `a @ b.T`.

    Returns a [rows,columns] Tensor, or `out` when given. On a GPU the call returns before the
    kernel finishes.

    Args:
      a: Query matrix of shape [rows,depth], on the device of `b`.
      b: Matrix packed by dots_pack(), of shape [columns,depth].
      out: C-contiguous [rows,columns] output, required on the same GPU for a GPU-packed `b`.
      rows_begin: First row of `a` to process, 0 when None.
      rows_end: One past the last row of `a` to process, every row when None.
      threads: Worker threads, ignored on a GPU.
    """
    ...

def hammings_pack(
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    *,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> PackedMatrix:
    """Pack a matrix for repeated Hamming and Jaccard cross operations.

    The pack serves hammings_packed() and jaccards_packed().

    Args:
      b: Source matrix of shape [columns,depth], with 'uint1' bits packed into uint8 bytes.
      dtype: Packing dtype, 'uint1' when None.
      threads: Worker threads.
    """
    ...

def hammings_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute Hamming distances between rows of `a` and a matrix packed by hammings_pack().

    Returns a [rows,columns] uint32 Tensor, or `out` when given.

    Args:
      a: Query matrix of shape [rows,depth].
      b: Matrix packed by hammings_pack(), of shape [columns,depth].
      out: C-contiguous [rows,columns] output of dtype uint32.
      rows_begin: First row of `a` to process, 0 when None.
      rows_end: One past the last row of `a` to process, every row when None.
      threads: Worker threads.
    """
    ...

def jaccards_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute Jaccard distances between rows of `a` and a matrix packed by hammings_pack().

    Returns a [rows,columns] Tensor, or `out` when given.

    Args:
      a: Query matrix of shape [rows,depth].
      b: Matrix packed by hammings_pack(), of shape [columns,depth].
      out: C-contiguous [rows,columns] output.
      rows_begin: First row of `a` to process, 0 when None.
      rows_end: One past the last row of `a` to process, every row when None.
      threads: Worker threads.
    """
    ...

def angulars_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute angular distances between rows of `a` and a matrix packed by dots_pack().

    Returns a [rows,columns] Tensor, or `out` when given. On a GPU the call returns before the
    kernel finishes.

    Args:
      a: Query matrix of shape [rows,depth], on the device of `b`.
      b: Matrix packed by dots_pack(), of shape [columns,depth].
      out: C-contiguous [rows,columns] output, required on the same GPU for a GPU-packed `b`.
      rows_begin: First row of `a` to process, 0 when None.
      rows_end: One past the last row of `a` to process, every row when None.
      threads: Worker threads, ignored on a GPU.
    """
    ...

def euclideans_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    rows_begin: int | None = None,
    rows_end: int | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute Euclidean distances between rows of `a` and a matrix packed by dots_pack().

    Returns a [rows,columns] Tensor, or `out` when given. On a GPU the call returns before the
    kernel finishes.

    Args:
      a: Query matrix of shape [rows,depth], on the device of `b`.
      b: Matrix packed by dots_pack(), of shape [columns,depth].
      out: C-contiguous [rows,columns] output, required on the same GPU for a GPU-packed `b`.
      rows_begin: First row of `a` to process, 0 when None.
      rows_end: One past the last row of `a` to process, every row when None.
      threads: Worker threads, ignored on a GPU.
    """
    ...

# endregion Packed Matrix Operations

# region MaxSim
def maxsim_pack(
    b: _BufferType, /, dtype: _FloatTypeName | _MiniFloatType | None = None, **dispatch: Unpack[_Dispatch]
) -> MaxSimPackedMatrix:
    """Pack a matrix for MaxSim late-interaction scoring with maxsim_packed().

    Packing fuses no transposition, so transpose a depth-major matrix into a contiguous copy first.

    Args:
      b: Source matrix of shape [vectors,depth] with contiguous rows.
      dtype: Packing dtype, inferred from `b` when None.
    """
    ...

def maxsim_packed(
    queries: MaxSimPackedMatrix, documents: MaxSimPackedMatrix, /, **dispatch: Unpack[_Dispatch]
) -> float:
    """Compute the MaxSim late-interaction score of two packed matrices.

    Returns the sum of per-query minimum angular distances.

    Args:
      queries: Query vectors packed by maxsim_pack().
      documents: Document vectors packed by maxsim_pack().
    """
    ...

def attention_pack(
    k: _BufferType,
    v: _BufferType,
    /,
    key_offsets: _BufferType,
    key_lengths: _BufferType | None = None,
    depth: int | None = None,
    out: Tensor | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> AttentionPackedMatrix:
    """Pack ragged K/V token matrices into a backend-opaque KV-cache blob.

    Segment s keeps its `key_lengths[s]` keys, by default the whole slot, from row
    `key_offsets[s]`.
    """
    ...

def attention_packed(
    q: _BufferType,
    kv: AttentionPackedMatrix,
    /,
    query_offsets: _BufferType,
    out: Tensor | None = None,
    scale: float | None = None,
    keys_before: int | None = None,
    keys_after: int | None = None,
    log_sum_exp: _BufferType | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Ragged scaled-dot-product attention against a pre-packed KV-cache, under a band of keys.

    Query row r of q queries over k keys sees the keys from `r + k - q - keys_before` through
    `r + k - q + keys_after`, each side unbounded at None.
    """
    ...

def attention_packed_gradients(
    q: _BufferType,
    kv: AttentionPackedMatrix,
    /,
    query_offsets: _BufferType,
    output: _BufferType,
    output_gradient: _BufferType,
    log_sum_exp: _BufferType,
    scale: float | None = None,
    keys_before: int | None = None,
    keys_after: int | None = None,
    query_gradient: Tensor | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the bf16 query, key and value gradients of `attention_packed` under the same band."""
    ...

# NeoX split-half rotary position embedding.
def attention_rope(
    x: _BufferType,
    cos: _BufferType,
    sin: _BufferType,
    /,
    head_count: int,
    depth: int,
    *,
    out: _BufferType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> None:
    """Rotate every channel pair of each head by the per-token angle grids, in place unless `out` is given.

    Args:
      x: `[rows,head_count*depth]`, float32, bfloat16, or e4m3.
      cos: `[rows,depth/2]` float32 angle grid, shared across heads.
      sin: `[rows,depth/2]` float32 angle grid, shared across heads.
      head_count: Number of heads per token.
      depth: Even number of channels per head.
      out: Output, same shape and dtype as `x`, may alias `x`, defaulting to `x`.
    """
    ...

def maxsim(
    queries: _BufferType,
    documents: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    **dispatch: Unpack[_Dispatch],
) -> float:
    """Pack both matrices and compute their MaxSim late-interaction score in one call.

    Returns the sum of per-query minimum angular distances.

    Args:
      queries: Query matrix of shape [query_count,depth].
      documents: Document matrix of shape [document_count,depth].
      dtype: Packing dtype, inferred from the inputs when None.
    """
    ...

# endregion MaxSim

# region Shapes

class ShapesPackedMatrix:
    """Point clouds packed by shapes_pack() or shapes_pack_normalized(), centered and rotated onto principal axes.

    Indexing and slicing return views sharing the packs, which chamfers_packed() and its siblings
    score as rows or columns. The pack layout is private: the properties report every cloud's pose
    and bounds, and unpack() returns any cloud in its original pose.
    """

    @property
    def clouds(self) -> int:
        """Number of packed clouds."""
        ...

    @property
    def points(self) -> int:
        """Total points across the packed clouds."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions (clouds, points) of the packed set."""
        ...

    @property
    def centroids(self) -> Tensor:
        """Mean input point of every cloud, [clouds,3], float32 for float16 packs and float64 for float32 ones."""
        ...

    @property
    def frames(self) -> Tensor:
        """Unit quaternions (x, y, z, w) of Q rotating every centered cloud onto its principal axes, [clouds,4]."""
        ...

    @property
    def spreads(self) -> Tensor:
        """Descending eigenvalues of every packed cloud's second moments, [clouds,3]."""
        ...

    @property
    def scales(self) -> Tensor:
        """What every rotated cloud was divided by, its RMS distance from the centroid or 1, [clouds]."""
        ...

    @property
    def max_radii(self) -> Tensor:
        """Largest packed point norm of every cloud, [clouds]."""
        ...

    @property
    def dtype(self) -> Literal["float16", "float32"]:
        """Data type of the packed points."""
        ...

    @property
    def nbytes(self) -> int:
        """Size of the packs in bytes."""
        ...

    @classmethod
    def pack_size(
        cls, points: int, /, dtype: Literal["f16", "float16", "f32", "float32", "f64", "float64"] | type[float16]
    ) -> int:
        """Return the bytes one packed cloud of the given points and dtype takes."""
        ...

    def unpack(self, index: int, /) -> Tensor:
        """Return one cloud as [points,3] in the pack dtype and its original pose and scale, within one rounding."""
        ...

    def copy(self, capabilities: Capability | int, *, stream: int | None = None) -> ShapesPackedMatrix:
        """Gather the table and packs into one block of the unified memory of `capabilities`, for GPU shape functions."""
        ...

    def __len__(self) -> int:
        """Return the number of packed clouds."""
        ...

    def __getitem__(self, key: int | slice, /) -> ShapesPackedMatrix:
        """Return a view over one cloud or a slice of them, sharing the packs."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

# Pack one [points,3] cloud, or a list or tuple of clouds of any sizes, for the shape distances.
def shapes_pack(
    points: _BufferType | list[_BufferType] | tuple[_BufferType, ...],
    /,
    dtype: Literal["f16", "float16", "f32", "float32", "f64", "float64"] | type[float16] | None = None,
    *,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> ShapesPackedMatrix:
    """Center every cloud and rotate it onto its principal axes, float16 input as float16, else float32."""
    ...

# Pack like shapes_pack(), dividing each cloud by its RMS radius, which `scales` keeps.
def shapes_pack_normalized(
    points: _BufferType | list[_BufferType] | tuple[_BufferType, ...],
    /,
    dtype: Literal["f16", "float16", "f32", "float32", "f64", "float64"] | type[float16] | None = None,
    *,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> ShapesPackedMatrix:
    """Pack every cloud like shapes_pack() and divide it by its RMS radius, so every kind ignores its size."""
    ...

# The many-to-many distances return [rows,columns] distances, float32 for float16 packs and float64
# for float32 ones. A `rotations` buffer, writable C-contiguous float32 of shape [rows,columns,3,3],
# receives the rotations taking each column cloud onto each row cloud:
# aᵢ ≈ (scale_a / scale_b) × R × (bⱼ − b̄) + ā.
def rmsds_packed(
    a: ShapesPackedMatrix,
    b: ShapesPackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute superposed RMSDs of clouds with equally many points, point i matching point i."""
    ...

def hausdorffs_packed(
    a: ShapesPackedMatrix,
    b: ShapesPackedMatrix,
    /,
    *,
    rotations: _BufferType | None = None,
    limit: float | None = None,
    out: _BufferType | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute Hausdorff distances, the worst nearest-partner distance either way, best of 8 axis flips."""
    ...

def chamfers_packed(
    a: ShapesPackedMatrix,
    b: ShapesPackedMatrix,
    /,
    *,
    rotations: _BufferType | None = None,
    limit: float | None = None,
    workspace: _BufferType | None = None,
    out: _BufferType | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute Chamfer distances, the mean nearest-partner distance both ways, halved, best of 8 axis flips."""
    ...

def wassersteinprocrustes_packed(
    a: ShapesPackedMatrix,
    b: ShapesPackedMatrix,
    /,
    *,
    rotations: _BufferType | None = None,
    workspace: _BufferType | None = None,
    out: _BufferType | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute Wasserstein-Procrustes distances, the exact transport RMSD after the best rotation or reflection."""
    ...

def gromovwassersteins_packed(
    a: ShapesPackedMatrix,
    b: ShapesPackedMatrix,
    /,
    *,
    rotations: _BufferType | None = None,
    workspace: _BufferType | None = None,
    out: _BufferType | None = None,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> Tensor:
    """Compute Gromov-Wasserstein distances between the clouds' own distance matrices."""
    ...

def chamfers_workspace_size(
    a: ShapesPackedMatrix,
    b: ShapesPackedMatrix,
    /,
    *,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> int:
    """Return the bytes of the `workspace` that chamfers_packed() needs, any alignment included."""
    ...

def wassersteinprocrustes_workspace_size(
    a: ShapesPackedMatrix,
    b: ShapesPackedMatrix,
    /,
    *,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> int:
    """Return the bytes of the `workspace` that wassersteinprocrustes_packed() needs."""
    ...

def gromovwassersteins_workspace_size(
    a: ShapesPackedMatrix,
    b: ShapesPackedMatrix,
    /,
    *,
    threads: int = 1,
    **dispatch: Unpack[_Dispatch],
) -> int:
    """Return the bytes of the `workspace` that gromovwassersteins_packed() needs."""
    ...

# endregion Shapes
