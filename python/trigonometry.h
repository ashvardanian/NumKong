/**
 *  @file python/trigonometry.h
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *  @brief Trigonometry declarations for NumKong Python bindings.
 *
 *  Forward declarations for the trigonometric sin, cos and atan api_* functions, and their
 *  documentation strings.
 */
#ifndef NUMKONG_PYTHON_TRIGONOMETRY_H
#define NUMKONG_PYTHON_TRIGONOMETRY_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Elementwise sine. */
PyObject *api_sin(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Elementwise cosine. */
PyObject *api_cos(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** Elementwise arctangent. */
PyObject *api_atan(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_sin[];
extern char const doc_cos[];
extern char const doc_atan[];

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_PYTHON_TRIGONOMETRY_H
