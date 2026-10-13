/**
 *  @file python/meshes.h
 *  @author Ash Vardanian
 *  @date February 19, 2026
 *  @brief Many-to-many shape distance declarations for NumKong Python bindings.
 *
 *  Declares the ShapesPacked type and the API functions packing point clouds and scoring every
 *  packed cloud of one set against every packed cloud of another.
 */
#ifndef NUMKONG_PYTHON_MESH_H
#define NUMKONG_PYTHON_MESH_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Point clouds packed one by one onto their principal axes, or a view over some of them. */
typedef struct ShapesPacked {
    PyObject_HEAD

    /** Input dtype of every pack, f16 or f32. */
    nk_dtype_t dtype;

    /** Number of packed clouds, the pointers in @ref packs. */
    nk_size_t count;

    /** The ShapesPacked whose buffer holds the packs, or NULL when they follow @ref packs here. */
    PyObject *owner;

    /** One pointer per pack, as the kernels take them, then the packs when @ref owner is NULL. */
    void const *packs[];
} ShapesPacked;

extern PyTypeObject ShapesPackedType;

PyObject *api_shapes_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_shapes_pack_normalized(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_shapes_chamfers_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_shapes_hausdorffs_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_shapes_wassersteinprocrustes_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                                  PyObject *kwnames);
PyObject *api_shapes_gromovwassersteins_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                               PyObject *kwnames);

extern char const doc_shapes_pack[];
extern char const doc_shapes_pack_normalized[];
extern char const doc_shapes_chamfers_packed[];
extern char const doc_shapes_hausdorffs_packed[];
extern char const doc_shapes_wassersteinprocrustes_packed[];
extern char const doc_shapes_gromovwassersteins_packed[];

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_PYTHON_MESH_H
