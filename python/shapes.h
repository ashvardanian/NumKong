/**
 *  @file python/shapes.h
 *  @author Ash Vardanian
 *  @date February 19, 2026
 *  @brief Many-to-many shape distance declarations for NumKong Python bindings.
 *
 *  Declares the ShapesPackedMatrix type and the API functions packing point clouds and scoring
 *  every packed cloud of one set against every packed cloud of another.
 */
#ifndef NUMKONG_PYTHON_SHAPES_H
#define NUMKONG_PYTHON_SHAPES_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Point clouds packed one by one onto their principal axes, or a view over some of them. */
typedef struct ShapesPackedMatrix {
    PyObject_HEAD

    /** Input dtype of every pack, f16, f32 or f64. */
    nk_dtype_t dtype;

    /** Number of packed clouds, the pointers in @ref packs. */
    nk_size_t clouds;

    /** The mask @c copy placed the packs with, or @c nk_cap_cpus_k on the Python heap. */
    nk_capability_t capabilities;

    /** The block @c copy gathered the table and packs into, which this matrix owns, or NULL. */
    void *unified;

    /** The bytes of @ref unified, which its release takes. */
    nk_size_t unified_bytes;

    /** The matrix whose buffer holds the packs and table, or NULL when this one does. */
    PyObject *owner;

    /** One pointer per pack, as the kernels take them: @ref table, the head of @ref unified, or a
     *  run of the table of @ref owner. */
    void const **packs;

    /** The table of a heap matrix, then its packs, or of a view in steps. */
    void const *table[];
} ShapesPackedMatrix;

extern PyTypeObject ShapesPackedMatrixType;

PyObject *api_shapes_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_shapes_pack_normalized(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_rmsds_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_hausdorffs_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_chamfers_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_wassersteinprocrustes_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_gromovwassersteins_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_chamfers_workspace_size(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_wassersteinprocrustes_workspace_size(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                                   PyObject *kwnames);
PyObject *api_gromovwassersteins_workspace_size(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                                PyObject *kwnames);

extern char const doc_shapes_pack[];
extern char const doc_shapes_pack_normalized[];
extern char const doc_rmsds_packed[];
extern char const doc_hausdorffs_packed[];
extern char const doc_chamfers_packed[];
extern char const doc_wassersteinprocrustes_packed[];
extern char const doc_gromovwassersteins_packed[];
extern char const doc_chamfers_workspace_size[];
extern char const doc_wassersteinprocrustes_workspace_size[];
extern char const doc_gromovwassersteins_workspace_size[];

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_PYTHON_SHAPES_H
