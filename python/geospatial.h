/**
 *  @file python/geospatial.h
 *  @author Ash Vardanian
 *  @date December 22, 2025
 *  @brief Many-to-many geospatial distance declarations for NumKong Python bindings.
 *
 *  Declares the HaversinesPackedMatrix and VincentysPackedMatrix types and the API functions
 *  packing latitude and longitude pairs and scoring every point of one set against every point of
 *  another, besides the unpacked 3D squared distances.
 */
#ifndef NUMKONG_PYTHON_GEOSPATIAL_H
#define NUMKONG_PYTHON_GEOSPATIAL_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Points packed for the Haversine or the Vincenty kernels, the type telling which. */
typedef struct GeospatialPackedMatrix {
    PyObject_HEAD

    /** Number of packed points. */
    nk_size_t points;

    /** Size of the pack in bytes. */
    nk_size_t bytes;

    /** The pack the kernels take, aligned to @c nk_default_alignment_k inside @ref storage. */
    void const *pack;

    /** The pack and the padding that aligns it. */
    char storage[];
} GeospatialPackedMatrix;

extern PyTypeObject HaversinesPackedMatrixType;
extern PyTypeObject VincentysPackedMatrixType;

PyObject *api_sqeuclideans3d(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_haversines_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_vincentys_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_haversines_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_vincentys_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_sqeuclideans3d[];
extern char const doc_haversines_pack[];
extern char const doc_vincentys_pack[];
extern char const doc_haversines_packed[];
extern char const doc_vincentys_packed[];

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_PYTHON_GEOSPATIAL_H
