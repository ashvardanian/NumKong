/**
 *  @file python/numpy_interop.h
 *  @author Ash Vardanian
 *  @date March 14, 2026
 *  @brief Register NumKong scalars as NumPy custom dtypes via runtime capsule API.
 *
 *  Minimal NumPy struct definitions sufficient for registering custom dtypes without including any
 *  NumPy headers. Uses runtime capsule extraction from numpy._core._multiarray_umath, requiring
 *  NumPy 2.0 or a newer release.
 */
#ifndef NUMKONG_PYTHON_NUMPY_INTEROP_H
#define NUMKONG_PYTHON_NUMPY_INTEROP_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Register NumKong scalar types as NumPy custom dtypes.
 *  @param[in] module The numkong module object.
 *  @return 0 on success, -1 on failure, with a Python exception set.
 *
 *  For each scalar type — bfloat16, float16, float8_e4m3, float8_e5m2, float6_e2m3, float6_e3m2 —
 *  this function:
 *  1. Creates a PyArray_DescrProto with appropriate ArrFuncs: getitem, setitem, copyswap,
 *     copyswapn, nonzero.
 *  2. Registers the dtype via PyArray_RegisterDataType.
 *  3. Registers cast functions to/from float32 and float64.
 *  4. Sets a `.dtype` attribute on each scalar type so `dtype=nk.bfloat16` works.
 *
 *  If NumPy is not installed or < 2.0, this function silently succeeds, returning 0, without
 *  registering anything.
 */
int nk_register_numpy_dtypes(PyObject *module);

#ifdef __cplusplus
}
#endif

#endif /* NUMKONG_PYTHON_NUMPY_INTEROP_H */
