/**
 *  @file python/distance.h
 *  @author Ash Vardanian
 *  @date February 19, 2026
 *  @brief Distance metric declarations for NumKong Python bindings.
 *
 *  Forward declarations for api_* distance functions, pointer APIs, and documentation strings.
 */
#ifndef NUMKONG_PYTHON_DISTANCE_H
#define NUMKONG_PYTHON_DISTANCE_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

PyObject *api_euclidean(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_sqeuclidean(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_angular(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_dot(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_vdot(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_kld(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_jsd(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_hamming(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_jaccard(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_cdist(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_bilinear(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_mahalanobis(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_haversine(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_vincenty(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_intersect(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject *api_sparse_dot(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

PyObject *api_euclidean_pointer(PyObject *self, PyObject *dtype_obj);
PyObject *api_sqeuclidean_pointer(PyObject *self, PyObject *dtype_obj);
PyObject *api_angular_pointer(PyObject *self, PyObject *dtype_obj);
PyObject *api_dot_pointer(PyObject *self, PyObject *dtype_obj);
PyObject *api_vdot_pointer(PyObject *self, PyObject *dtype_obj);
PyObject *api_kld_pointer(PyObject *self, PyObject *dtype_obj);
PyObject *api_jsd_pointer(PyObject *self, PyObject *dtype_obj);
PyObject *api_hamming_pointer(PyObject *self, PyObject *dtype_obj);
PyObject *api_jaccard_pointer(PyObject *self, PyObject *dtype_obj);

extern char const doc_euclidean[];
extern char const doc_sqeuclidean[];
extern char const doc_angular[];
extern char const doc_dot[];
extern char const doc_vdot[];
extern char const doc_kld[];
extern char const doc_jsd[];
extern char const doc_hamming[];
extern char const doc_jaccard[];
extern char const doc_cdist[];
extern char const doc_bilinear[];
extern char const doc_mahalanobis[];
extern char const doc_haversine[];
extern char const doc_vincenty[];
extern char const doc_intersect[];
extern char const doc_sparse_dot[];
extern char const doc_euclidean_pointer[];
extern char const doc_sqeuclidean_pointer[];
extern char const doc_angular_pointer[];
extern char const doc_dot_pointer[];
extern char const doc_vdot_pointer[];
extern char const doc_kld_pointer[];
extern char const doc_jsd_pointer[];
extern char const doc_hamming_pointer[];
extern char const doc_jaccard_pointer[];

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_PYTHON_DISTANCE_H
