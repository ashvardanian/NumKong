from numkong._numkong import *  # noqa: F401,F403
from numkong import _numkong as _ext

__version__ = _ext.__version__

del _ext
