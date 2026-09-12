"""NumKong: portable mixed-precision BLAS-like vector, tensor, and distance kernels."""

from numkong import _numkong as _ext
from numkong._numkong import *  # noqa: F403


__version__ = _ext.__version__

del _ext
