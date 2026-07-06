"""NumKong: portable mixed-precision BLAS-like vector, tensor, and distance kernels."""

# The wheel bundles its own `libomp.dylib` (via `delocate`), but NumPy, SciPy
# or PyTorch in the same process may already have loaded a different OpenMP
# runtime. LLVM's libomp detects the duplicate during its own constructor
# and aborts the process with:
#
#     OMP: Error #15: Initializing libomp.dylib, but found
#     libomp.dylib already initialized.
#
# `KMP_DUPLICATE_LIB_OK=TRUE` tells libomp to continue anyway — but only if
# the variable is set *before* libomp's constructor reads it. The load order
# when Python imports this package is:
#
#     dlopen numkong/_numkong.so
#     ├── dyld loads libomp.dylib            ← dependency resolution
#     │   └── libomp ctor reads KMP_*        ← decision point
#     └── our .so's own ctors run
#         └── PyInit__numkong                ← already too late
#
# So setting the variable from C (either `PyInit__numkong` or a constructor
# in our `.so`) cannot work. Python-level code, on the other hand, runs
# before the `from ... import *` below triggers the `dlopen`, which is why
# this lives in `__init__.py`.
#
# `setdefault` leaves any value the user deliberately set untouched.
import os


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from numkong import _numkong as _ext
from numkong._numkong import *  # noqa: F403


__version__ = _ext.__version__

del _ext, os
