/**
 *  @file python/dlpack_interop.h
 *  @author Ash Vardanian
 *  @date April 17, 2026
 *  @brief DLPack zero-copy interop for NumKong's Python extension.
 *
 *  Implements the Python Array API DLPack protocol on `numkong.Tensor` so it exchanges zero-copy
 *  with PyTorch, NumPy, JAX, CuPy, TensorFlow, PyArrow, ONNX Runtime, MXNet, MLX, NNabla, TVM, and
 *  any other consumer of @c __dlpack__ or @c from_dlpack. The struct layout we exchange is declared
 *  in `python/dlpack_abi.h`.
 *
 *  See python/dlpack_interop.c's Interop Partners section for the full list of upstream PRs that
 *  ratified the protocol in each of those projects.
 *
 *  @see DLPack repository: https://github.com/dmlc/dlpack
 *  @see Array API data interchange: https://data-apis.org/array-api/latest/design_topics/data_interchange.html
 */
#ifndef NK_PYTHON_DLPACK_INTEROP_H
#define NK_PYTHON_DLPACK_INTEROP_H

#include "numkong.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief `Tensor.__dlpack__(stream=None, max_version=None, dl_device=None, copy=None)`.
 *
 *  Produces a @c PyCapsule named `"dltensor"`, legacy v0, or `"dltensor_versioned"`, v1+, and
 *  reports @c kDLCPU only, so any other requested @c dl_device gets a @c BufferError; zero-copy
 *  exchange is verified with PyTorch, NumPy, JAX, CuPy, TensorFlow, PyArrow and MLX.
 */
PyObject *Tensor_dlpack(PyObject *self, PyObject *args, PyObject *kwargs);

/**
 *  @brief `Tensor.__dlpack_device__()`. Always returns `(kDLCPU=1, 0)`.
 *
 *  Part of the Array API DLPack negotiation: consumers call this before @c __dlpack__ to know
 *  whether a copy or stream sync is needed.
 */
PyObject *Tensor_dlpack_device(PyObject *self, PyObject *noargs);

/**
 *  @brief `numkong.from_dlpack(obj)` — zero-copy import.
 *
 *  Accepts a `"dltensor"` or `"dltensor_versioned"` capsule or any object implementing
 *  @c __dlpack__. Any device whose pointer is host-readable, kDLCPU, kDLCUDAHost, kDLROCMHost,
 *  kDLCUDAManaged, kDLOneAPI, kDLMetal, is accepted; pure device memory, kDLCUDA, kDLROCM,
 *  kDLOpenCL, kDLVulkan, kDLWebGPU, ..., is rejected with a clear @c ValueError naming the device
 *  code. Verified producers, zero-copy: PyTorch, NumPy, JAX, CuPy, TensorFlow, PyArrow, MLX, ONNX
 *  Runtime, training builds, MXNet.
 */
PyObject *api_from_dlpack(PyObject *self, PyObject *obj);

/** Initialize the internal @c _DLPackOwner type used to hold imported capsule ownership, called
 *  once from @c PyInit__numkong. */
int nk_dlpack_init(PyObject *module);

extern char const doc_from_dlpack[];
extern char const doc_dlpack[];
extern char const doc_dlpack_device[];

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_DLPACK_INTEROP_H
