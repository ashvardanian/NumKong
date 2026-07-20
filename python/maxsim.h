/**
 *  @brief MaxSim late-interaction declarations for NumKong Python bindings.
 *  @file python/maxsim.h
 *  @author Ash Vardanian
 *  @date March 9, 2026
 *
 *  Declares the MaxSimPackedMatrix type and API functions for MaxSim
 *  (ColBERT late-interaction scoring) used by the Python module.
 */
#ifndef NK_PYTHON_MAXSIM_H
#define NK_PYTHON_MAXSIM_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Pre-packed matrix for MaxSim late-interaction scoring. */
typedef struct MaxSimPackedMatrix {
    PyObject_HEAD nk_dtype_t dtype;
    nk_size_t vectors;
    nk_size_t depth;
    char start[];
} MaxSimPackedMatrix;

extern PyTypeObject MaxSimPackedMatrixType;

PyObject *api_maxsim_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_maxsim_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_maxsim_pack[];
extern char const doc_maxsim_packed[];
extern char const doc_maxsim[];

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_MAXSIM_H
