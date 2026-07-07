/**
 *  @brief Elementwise operation declarations for NumKong Python bindings.
 *  @file python/each.h
 *  @author Ash Vardanian
 *  @date February 19, 2026
 *
 *  Forward declarations for all api_* elementwise functions and their
 *  documentation strings.
 */
#ifndef NK_PYTHON_EACH_H
#define NK_PYTHON_EACH_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Compute fused multiply-add: alpha*a*b + beta*c. */
PyObject *api_fma(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Compute blend: alpha*a + beta*b. */
PyObject *api_blend(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Scale a tensor: alpha*a + beta. */
PyObject *api_scale(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Elementwise addition of two tensors or a tensor and a scalar. */
PyObject *api_add(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Elementwise multiplication of two tensors or a tensor and a scalar. */
PyObject *api_multiply(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** @brief Grouped RMSNorm: y = x * rsqrt(mean(x^2) + eps) * gamma. */
PyObject *api_rmsnorm(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** @brief Fused SwiGLU: y = silu(gate) * up  (up=None -> plain SiLU). */
PyObject *api_swiglu(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_fma[];
extern char const doc_blend[];
extern char const doc_scale[];
extern char const doc_add[];
extern char const doc_multiply[];
extern char const doc_rmsnorm[];
extern char const doc_swiglu[];

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_EACH_H
