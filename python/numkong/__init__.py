"""NumKong: portable mixed-precision BLAS-like vector, tensor, and distance kernels.

File: python/numkong/__init__.py
Author: Ash Vardanian
Date: April 12, 2026
"""

from numkong import _numkong as _ext
from numkong._numkong import *  # noqa: F403


__version__ = _ext.__version__

del _ext
