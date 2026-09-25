/**
 *  @file python/mesh.h
 *  @author Ash Vardanian
 *  @date February 19, 2026
 *  @brief Mesh alignment declarations for NumKong Python bindings.
 *
 *  Forward declarations for mesh alignment, Kabsch/Umeyama/RMSD, API functions. The
 *  MeshAlignmentResultObject struct is private to mesh.c.
 */
#ifndef NUMKONG_PYTHON_MESH_H
#define NUMKONG_PYTHON_MESH_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** MeshAlignmentResult Python type object, defined in mesh.c. */
extern PyTypeObject MeshAlignmentResultType;

PyObject *api_kabsch(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_umeyama(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_rmsd(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_kabsch[];
extern char const doc_umeyama[];
extern char const doc_rmsd[];

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_PYTHON_MESH_H
