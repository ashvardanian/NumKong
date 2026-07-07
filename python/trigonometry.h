/**
 *  @brief Trigonometry and RoPE declarations for NumKong Python bindings.
 *  @file python/trigonometry.h
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *
 *  Forward declarations for the trigonometric (sin/cos/atan) and rotary position
 *  embedding (RoPE) api_* functions, and their documentation strings.
 */
#ifndef NK_PYTHON_TRIGONOMETRY_H
#define NK_PYTHON_TRIGONOMETRY_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Elementwise sine. */
PyObject *api_sin(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Elementwise cosine. */
PyObject *api_cos(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Elementwise arctangent. */
PyObject *api_atan(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief NeoX split-half rotary position embedding (RoPE), separate aliasable output. */
PyObject *api_rope(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_sin[];
extern char const doc_cos[];
extern char const doc_atan[];
extern char const doc_rope[];

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_TRIGONOMETRY_H
