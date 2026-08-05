from typing import Any, Literal, TypeAlias

# Many annotation features depend on the Python version:
# - `typing.TypeAlias` type aliases are supported in Python 3.10-3.11.
# - `type` statements are supported from Python 3.12, replacing `typing.TypeAlias`.
# - `typing.Literal` literal types are supported from Python 3.8.
#
# We cannot maintain a separate `.pyi` file for every Python version.
# Assume Python 3.11 with NumPy available.
from numpy.typing import NDArray

# region Forward Declarations and Shared Types

# Scalar dtype literals used throughout the API.
_IntegralTypeName = Literal[
    "uint1",
    # Sub-byte integers
    "int4",
    "uint4",
    # Signed integers
    "int8",
    "int16",
    "int32",
    "int64",
    # Unsigned integers
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
_FloatTypeName = Literal[
    "f32",
    "float32",
    "f16",
    "float16",
    "f64",
    "float64",
    "bf16",  #! Not supported by NumPy
    "bfloat16",  #! Not supported by NumPy
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
    "complex32",  #! Not supported by NumPy
    "bcomplex32",  #! Not supported by NumPy
    "complex64",
    "complex128",
]
_MetricName = Literal[
    "euclidean",
    "sqeuclidean",
    "inner",
    "dot",
    "angular",
    "hamming",
    "jaccard",
    "kullbackleibler",
    "kld",
    "jensenshannon",
    "jsd",
    "intersection",
    "bilinear",
    "mahalanobis",
    "fma",
    "blend",
]

# Scalar type classes for custom float formats.
# These are registered on the module at runtime (see types.c).

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

_MiniFloatType: TypeAlias = (
    type[bfloat16] | type[float16] | type[float8_e4m3] | type[float8_e5m2] | type[float6_e2m3] | type[float6_e3m2]
)

# Buffer-compatible tensor inputs accepted by most functions.
_BufferType: TypeAlias = NDArray[Any] | memoryview

class Tensor(memoryview):
    """N-dimensional tensor type returned by NumKong operations.

    Supports NumPy-like properties and buffer protocol for interoperability.
    """

    def __new__(cls, array_like: _BufferType, /, *, dtype: str | _MiniFloatType | None = None) -> Tensor:
        """Construct a Tensor by copying data from a buffer-protocol object."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the tensor as a tuple of dimensions."""
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

    @property
    def T(self) -> Tensor:
        """Transpose of the tensor."""
        ...

    def copy(self) -> Tensor:
        """Return a deep copy of the tensor."""
        ...

    def reshape(self, *shape: int) -> Tensor:
        """Return tensor reshaped to given dimensions."""
        ...

    def flatten(self) -> Tensor:
        """Return a flattened 1D view (copies if non-contiguous)."""
        ...

    def squeeze(self, axis: int | None = None) -> Tensor:
        """Remove dimensions of size 1."""
        ...

    def astype(self, dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType, /) -> Tensor:
        """Return a copy cast to the given NumKong dtype."""
        ...

    @property
    def is_contiguous(self) -> bool:
        """Whether the tensor is C-contiguous in memory."""
        ...

    @property
    def data_ptr(self) -> int:
        """Integer address of the underlying data buffer."""
        ...

    def sum(
        self, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False, out: Tensor | None = None
    ) -> float | int | Tensor:
        """Return the sum of all elements."""
        ...

    def norm(
        self, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False, out: Tensor | None = None
    ) -> float | Tensor:
        """Return the L2 norm."""
        ...

    def min(
        self, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False, out: Tensor | None = None
    ) -> float | int | None | Tensor:
        """Return the minimum element, or None if all elements are NaN."""
        ...

    def max(
        self, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False, out: Tensor | None = None
    ) -> float | int | None | Tensor:
        """Return the maximum element, or None if all elements are NaN."""
        ...

    def argmin(self, axis: int | None = None, *, out: Tensor | None = None) -> int | None | Tensor:
        """Return the index of the minimum element, or None if all elements are NaN."""
        ...

    def argmax(self, axis: int | None = None, *, out: Tensor | None = None) -> int | None | Tensor:
        """Return the index of the maximum element, or None if all elements are NaN."""
        ...

class PackedMatrix:
    """Opaque pre-packed matrix for repeated cross operations.

    Created by dots_pack() or hammings_pack() and used with
    dots_packed(), hammings_packed(), jaccards_packed(), angulars_packed(),
    euclideans_packed(), or the @ operator.
    """

    @property
    def width(self) -> int:
        """Number of rows in the original matrix."""
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

    @classmethod
    def packed_size(
        cls,
        width: int,
        depth: int,
        /,
        dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "bf16",
    ) -> int:
        """Return packed buffer size in bytes for given dimensions and dtype."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

class MaxSimPackedMatrix:
    """Opaque pre-packed matrix for MaxSim late-interaction scoring."""

    @property
    def vector_count(self) -> int:
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

    @classmethod
    def packed_size(cls, vector_count: int, depth: int, /, dtype: _FloatTypeName | _MiniFloatType = "bf16") -> int:
        """Return packed buffer size in bytes for given dimensions and dtype."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

# endregion Forward Declarations and Shared Types

# region Capabilities

# SIMD capability controls.
def get_capabilities() -> dict[str, bool]: ...
def enable_capability(capability: str, /) -> None: ...
def disable_capability(capability: str, /) -> None: ...

# Kernel pointer accessors.
def pointer_to_euclidean(dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType, /) -> int: ...
def pointer_to_sqeuclidean(dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType, /) -> int: ...
def pointer_to_angular(dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType, /) -> int: ...
def pointer_to_inner(dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType, /) -> int: ...
def pointer_to_dot(dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType, /) -> int: ...
def pointer_to_vdot(dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType, /) -> int: ...
def pointer_to_jensenshannon(dtype: _FloatTypeName | _MiniFloatType, /) -> int: ...
def pointer_to_kullbackleibler(dtype: _FloatTypeName | _MiniFloatType, /) -> int: ...

# endregion Capabilities

# region Pairwise Distances

# Pairwise distances, similar to `scipy.spatial.distance.cdist`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.cdist.html
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
) -> float | complex | Tensor | None: ...

# endregion Pairwise Distances

# region Vector Dot Products
# Vector-vector dot products for real and complex numbers.

# Inner product, similar to: `numpy.inner`.
# https://numpy.org/doc/stable/reference/generated/numpy.inner.html
def inner(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
) -> float | complex | Tensor | None: ...

# Dot product, similar to: `numpy.dot`.
# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
def dot(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _ComplexTypeName | _MiniFloatType = None,
) -> float | complex | Tensor | None: ...

# Vector-vector dot product for complex conjugates, similar to: `numpy.vdot`.
# https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
def vdot(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _ComplexTypeName | _MiniFloatType | None = None,
    *,
    out: float | complex | Tensor | None = None,
    out_dtype: _ComplexTypeName | _MiniFloatType | None = None,
) -> complex | Tensor | None: ...

# endregion Vector Dot Products

# region Spatial Distance Metrics
# Vector-vector spatial distance metrics for real and integer numbers.

# Vector-vector squared Euclidean distance, similar to: `scipy.spatial.distance.sqeuclidean`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.sqeuclidean.html
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
def sqeuclidean(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType = None,
) -> float | Tensor | None: ...

# Vector-vector angular distance (also known as cosine distance), similar to: `scipy.spatial.distance.cosine`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.cosine.html
def angular(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType = None,
) -> float | Tensor | None: ...

# Vector-vector Euclidean distance, similar to: `scipy.spatial.distance.euclidean`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.euclidean.html
def euclidean(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType | None = None,
) -> float | Tensor | None: ...

# endregion Spatial Distance Metrics

# region Binary Similarity
# Vector-vector similarity functions for binary vectors.

# Vector-vector Hamming distance, similar to: `scipy.spatial.distance.hamming`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.hamming.html
def hamming(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType = None,
) -> float | Tensor | None: ...

# Vector-vector Jaccard distance, similar to: `scipy.spatial.distance.jaccard`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.jaccard.html
def jaccard(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType = None,
) -> float | Tensor | None: ...

# endregion Binary Similarity

# region Probability Distances
# Vector-vector similarity between probability distributions.

# Vector-vector Jensen-Shannon distance, similar to: `scipy.spatial.distance.jensenshannon`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.jensenshannon.html
def jensenshannon(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType = None,
) -> float | Tensor | None: ...
def jsd(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType = None,
) -> float | Tensor | None: ...

# Vector-vector Kullback-Leibler divergence, similar to: `scipy.spatial.distance.kullback_leibler`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.kullback_leibler.html
def kullbackleibler(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType = None,
) -> float | Tensor | None: ...
def kld(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
    out_dtype: _FloatTypeName | _MiniFloatType = None,
) -> float | Tensor | None: ...

# endregion Probability Distances

# region Curved Space Metrics
# Vector-vector similarity between vectors in curved spaces.

# Vector-vector bilinear distance, similar to: `numpy.dot(a, metric_tensor @ vector2)`.
# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
def bilinear(
    a: _BufferType,
    b: _BufferType,
    metric_tensor: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
) -> float: ...

# Vector-vector Mahalanobis distance, similar to: `scipy.spatial.distance.mahalanobis`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.mahalanobis.html
def mahalanobis(
    a: _BufferType,
    b: _BufferType,
    inverse_covariance: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
) -> float: ...

# endregion Curved Space Metrics

# region Geospatial Distances

def haversine(
    a_lats: _BufferType,
    a_lons: _BufferType,
    b_lats: _BufferType,
    b_lons: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
) -> float | Tensor | None: ...
def vincenty(
    a_lats: _BufferType,
    a_lons: _BufferType,
    b_lats: _BufferType,
    b_lons: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
) -> float | Tensor | None: ...

# endregion Geospatial Distances

# region Sparse Similarity
# Vector-vector similarity between sparse vectors.

# Vector-vector intersection similarity, similar to: `numpy.intersect1d`.
# https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html
def intersect(array1: _BufferType, array2: _BufferType, /) -> float: ...
def sparse_dot(
    a_indices: _BufferType,
    a_values: _BufferType,
    b_indices: _BufferType,
    b_values: _BufferType,
    /,
) -> float: ...

# endregion Sparse Similarity

# region Tensor Constructors
_DtypeLike: TypeAlias = _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType

def from_pointer(
    address: int,
    shape: int | tuple[int, ...],
    dtype: _DtypeLike,
    *,
    strides: tuple[int, ...] | None = None,
    owner: Any = None,
) -> Tensor: ...
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
) -> Tensor: ...
def zeros(
    shape: int | tuple[int, ...],
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor: ...
def ones(
    shape: int | tuple[int, ...],
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor: ...
def full(
    shape: int | tuple[int, ...],
    fill_value: int | float,
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor: ...
def iota(
    shape: int | tuple[int, ...],
    seed: int | float = 0,
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Tensor filled with ``seed, seed+1, seed+2, ...`` reshaped to ``shape``."""
    ...
def diagonal(
    n: int,
    seed: int | float = 1,
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Square ``n x n`` tensor with ``seed`` on the main diagonal and zeros elsewhere."""
    ...
def hash(
    shape: int | tuple[int, ...],
    seed: int = 0,
    /,
    *,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType = "float32",
) -> Tensor:
    """Tensor filled with deterministic pseudo-random bits (shape, dtype, seed → reproducible bytes)."""
    ...
def astype(
    a: _BufferType,
    dtype: _DtypeLike,
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
    a: _BufferType, /, *, dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None
) -> tuple[float, float]: ...
def minmax(
    a: _BufferType, /, *, dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None
) -> tuple[float, int, float, int] | None: ...
def sum(
    a: _BufferType,
    /,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
) -> float | int | Tensor: ...
def norm(
    a: _BufferType,
    /,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
) -> float | Tensor: ...
def min(
    a: _BufferType,
    /,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
) -> float | int | None | Tensor: ...
def max(
    a: _BufferType,
    /,
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
) -> float | int | None | Tensor: ...
def argmin(
    a: _BufferType,
    /,
    axis: int | None = None,
    *,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
) -> int | None | Tensor: ...
def argmax(
    a: _BufferType,
    /,
    axis: int | None = None,
    *,
    out: Tensor | None = None,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
) -> int | None | Tensor: ...

# endregion Reductions

# region Vector Math

# Vector-vector element-wise fused multiply-add.
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
) -> Tensor | None: ...

# Vector-vector element-wise blend.
def blend(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    *,
    alpha: float = 1,
    beta: float = 1,
    out: _BufferType | None = None,
) -> Tensor | None: ...

# endregion Vector Math

# region Trigonometry

# Element-wise trigonometric sine.
def sin(
    a: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
) -> Tensor | None: ...

# Element-wise trigonometric cosine.
def cos(
    a: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
) -> Tensor | None: ...

# Element-wise trigonometric arctangent.
def atan(
    a: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
    *,
    out: _BufferType | None = None,
) -> Tensor | None: ...

# endregion Trigonometry

# region Elementwise Arithmetic

# Element-wise scale operation.
def scale(
    a: _BufferType,
    /,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    *,
    alpha: float = 1,
    beta: float = 0,
    out: _BufferType | None = None,
) -> Tensor | None: ...

# Element-wise add (NumPy-compatible with broadcasting).
def add(
    a: _BufferType | float | int,
    b: _BufferType | float | int,
    /,
    *,
    out: _BufferType | None = None,
    a_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    b_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
) -> Tensor | None: ...

# Element-wise multiply (NumPy-compatible with broadcasting).
def multiply(
    a: _BufferType | float | int,
    b: _BufferType | float | int,
    /,
    *,
    out: _BufferType | None = None,
    a_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    b_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out_dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
) -> Tensor | None: ...

# endregion Elementwise Arithmetic

# region Symmetric Pairwise Operations
def dots_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...
def hammings_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...
def jaccards_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...
def angulars_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...
def euclideans_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: _FloatTypeName | _IntegralTypeName | _MiniFloatType | None = None,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...

# endregion Symmetric Pairwise Operations

# region Packed Matrix Operations

# Pack a matrix for repeated dot-product matmul.
def dots_pack(
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
) -> PackedMatrix: ...

# Dot-product matrix multiplication with a pre-packed B matrix.
def dots_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...

# Pack a matrix for repeated Hamming distance computation.
def hammings_pack(
    b: _BufferType,
    /,
    dtype: _IntegralTypeName | _FloatTypeName | _ComplexTypeName | _MiniFloatType | None = None,
) -> PackedMatrix: ...

# Hamming distance computation with a pre-packed B matrix.
def hammings_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...

# Jaccard distance computation with a pre-packed B matrix.
def jaccards_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...

# Angular distance computation with a pre-packed B matrix.
def angulars_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...

# Euclidean distance computation with a pre-packed B matrix.
def euclideans_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: _BufferType | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
) -> Tensor: ...

# endregion Packed Matrix Operations

# region MaxSim
def maxsim_pack(b: _BufferType, /, dtype: _FloatTypeName | _MiniFloatType | None = None) -> MaxSimPackedMatrix: ...
def maxsim_packed(queries: MaxSimPackedMatrix, documents: MaxSimPackedMatrix, /) -> float: ...
def maxsim(
    queries: _BufferType, documents: _BufferType, /, dtype: _FloatTypeName | _MiniFloatType | None = None
) -> float: ...

# endregion MaxSim

# region Mesh Alignment

class MeshAlignmentResult:
    """Result of a mesh alignment operation (Kabsch, Umeyama, or RMSD)."""

    @property
    def rotation(self) -> Tensor:
        """Rotation matrix."""
        ...

    @property
    def translation(self) -> Tensor:
        """Translation vector."""
        ...

    @property
    def scale(self) -> Tensor:
        """Scale factor."""
        ...

    @property
    def rmsd(self) -> Tensor:
        """Root mean square deviation."""
        ...

def kabsch(
    source: _BufferType,
    target: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
) -> MeshAlignmentResult: ...
def umeyama(
    source: _BufferType,
    target: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
) -> MeshAlignmentResult: ...
def rmsd(
    source: _BufferType,
    target: _BufferType,
    /,
    dtype: _FloatTypeName | _MiniFloatType | None = None,
) -> float: ...

# endregion Mesh Alignment
