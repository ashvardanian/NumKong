/**
 *  @file python/geospatial.c
 *  @author Ash Vardanian
 *  @date December 22, 2025
 *  @brief Many-to-many geospatial distances for NumKong Python bindings.
 *
 *  This module owns:
 *  - @c sqeuclideans3d: squared distances between every row point and every column point in 3D.
 *  - @c HaversinesPackedMatrix and @c VincentysPackedMatrix: latitude and longitude pairs packed
 *    for the sphere and for the ellipsoid.
 *  - @c haversines_pack and @c vincentys_pack: pack the points into them.
 *  - @c haversines_packed and @c vincentys_packed: the metres from every row to every column point.
 *
 *  Shape naming convention used in docs and errors:
 *  - @c a: @b [rows] points, @b [rows,3] for @c sqeuclideans3d
 *  - @c b: @b [columns] points, @b [columns,3] for @c sqeuclideans3d
 *  - distances: @b [rows,columns]
 */
#include "geospatial.h"
#include "tensor.h"

#include "numkong/geospatial.h"

static PyObject *GeospatialPackedMatrix_repr(PyObject *self) {
    GeospatialPackedMatrix *packed = (GeospatialPackedMatrix *)self;
    return PyUnicode_FromFormat("<%s points=%zu nbytes=%zu>", Py_TYPE(self)->tp_name, (size_t)packed->points,
                                (size_t)packed->bytes);
}

static Py_ssize_t GeospatialPackedMatrix_length(PyObject *self) {
    return (Py_ssize_t)((GeospatialPackedMatrix *)self)->points;
}

static PySequenceMethods GeospatialPackedMatrix_as_sequence = {
    .sq_length = GeospatialPackedMatrix_length,
};

static PyObject *GeospatialPackedMatrix_get_points(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((GeospatialPackedMatrix *)self)->points);
}

static PyObject *GeospatialPackedMatrix_get_shape(PyObject *self, void *closure) {
    nk_unused_(closure);
    PyObject *points_integer = PyLong_FromSize_t(((GeospatialPackedMatrix *)self)->points);
    PyObject *axes_integer = PyLong_FromSize_t(2);
    PyObject *shape = points_integer && axes_integer ? PyTuple_Pack(2, points_integer, axes_integer) : NULL;
    Py_XDECREF(points_integer);
    Py_XDECREF(axes_integer);
    return shape;
}

static PyObject *GeospatialPackedMatrix_get_nbytes(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((GeospatialPackedMatrix *)self)->bytes);
}

static PyGetSetDef GeospatialPackedMatrix_getset[] = {
    {"points", GeospatialPackedMatrix_get_points, NULL, "Number of packed points", NULL},
    {"shape", GeospatialPackedMatrix_get_shape, NULL,
     "Dimensions (points, 2) of the packed latitude and longitude pairs", NULL},
    {"nbytes", GeospatialPackedMatrix_get_nbytes, NULL, "Size of the pack in bytes", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *GeospatialPackedMatrix_pack_size(PyObject *cls, PyObject *points_argument) {
    nk_kernel_kind_t const size_kind = cls == (PyObject *)&VincentysPackedMatrixType ? nk_kernel_vincentys_pack_size_k
                                                                                     : nk_kernel_haversines_pack_size_k;
    nk_size_t const points = (nk_size_t)PyLong_AsSize_t(points_argument);
    if (points == (nk_size_t)-1 && PyErr_Occurred()) return NULL;

    nk_geospatial_pack_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(size_kind, nk_f64_k, nk_cap_cpus_k, (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn || !cap) {
        PyErr_Format(PyExc_LookupError, "No %s kernel for this machine", nk_kernel_name(size_kind));
        return NULL;
    }
    nk_size_t bytes = 0;
    if (!check_status(size_fn(points, &bytes))) return NULL;
    return PyLong_FromSize_t(bytes);
}

static PyMethodDef GeospatialPackedMatrix_methods[] = {
    {"pack_size", GeospatialPackedMatrix_pack_size, METH_CLASS | METH_O,
     "Return the bytes a pack of the given number of points takes."},
    {NULL, NULL, 0, NULL},
};

PyTypeObject HaversinesPackedMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.HaversinesPackedMatrix",
    .tp_doc = "Points packed as unit vectors on the mean sphere, for haversines_packed()",
    .tp_basicsize = sizeof(GeospatialPackedMatrix),
    .tp_itemsize = sizeof(char),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = GeospatialPackedMatrix_getset,
    .tp_methods = GeospatialPackedMatrix_methods,
    .tp_repr = GeospatialPackedMatrix_repr,
    .tp_as_sequence = &GeospatialPackedMatrix_as_sequence,
};

PyTypeObject VincentysPackedMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.VincentysPackedMatrix",
    .tp_doc = "Points packed as longitudes and reduced latitudes on the WGS-84 ellipsoid, for vincentys_packed()",
    .tp_basicsize = sizeof(GeospatialPackedMatrix),
    .tp_itemsize = sizeof(char),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = GeospatialPackedMatrix_getset,
    .tp_methods = GeospatialPackedMatrix_methods,
    .tp_repr = GeospatialPackedMatrix_repr,
    .tp_as_sequence = &GeospatialPackedMatrix_as_sequence,
};

/** Acquires @p object as a @b [points] buffer when @p rank is 1, or a @b [points,3] one when it is
 *  2, of a known dtype, or sets a Python error. */
static int geospatial_points_buffer(PyObject *object, int rank, Py_buffer *buffer, nk_buffer_backing_t *backing,
                                    nk_dtype_t *dtype) {
    if (!nk_get_buffer(object, buffer, PyBUF_STRIDES | PyBUF_FORMAT, backing)) {
        PyErr_SetString(PyExc_TypeError, "coordinates must support buffer protocol or __array_interface__");
        return 0;
    }
    *dtype = resolve_nk_dtype_in_py_buffer(buffer);
    if (buffer->ndim != rank || (rank == 2 && buffer->shape[1] != 3))
        PyErr_SetString(PyExc_ValueError, rank == 1 ? "latitudes and longitudes must be [points] vectors"
                                                    : "points must be [points,3] matrices");
    else if (*dtype == nk_dtype_unknown_k)
        PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s'", buffer->format);
    else return 1;
    PyBuffer_Release(buffer);
    return 0;
}

/** Packs latitudes and longitudes, cast to f64, into a new @p type with the kernels of @p size_kind
 *  and @p pack_kind. */
static PyObject *geospatial_pack_common(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, PyTypeObject *type,
                                        nk_kernel_kind_t size_kind, nk_kernel_kind_t pack_kind) {
    char const *name = nk_kernel_name(pack_kind);
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "%s() requires exactly 2 positional arguments: latitudes, longitudes", name);
        return NULL;
    }
    nk_capability_t capabilities = nk_cap_cpus_k;
    nk_stream_t stream = NULL;
    Py_ssize_t const nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++)
        if (!parse_dispatch_keyword(PyTuple_GET_ITEM(kwnames, i), args[nargs + i], &capabilities, &stream)) return NULL;

    nk_geospatial_pack_size_punned_t size_fn = NULL;
    nk_geospatial_pack_punned_t pack_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(size_kind, nk_f64_k, capabilities, (nk_kernel_punned_t *)&size_fn, &cap);
    if (size_fn && cap) nk_find_kernel_punned(pack_kind, nk_f64_k, capabilities, (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!size_fn || !pack_fn || !cap) {
        PyErr_Format(PyExc_LookupError, "No %s kernel for the given capabilities", name);
        return NULL;
    }

    Py_buffer latitudes, longitudes;
    nk_buffer_backing_t latitudes_backing, longitudes_backing;
    nk_dtype_t latitudes_dtype, longitudes_dtype;
    if (!geospatial_points_buffer(args[0], 1, &latitudes, &latitudes_backing, &latitudes_dtype)) return NULL;
    if (!geospatial_points_buffer(args[1], 1, &longitudes, &longitudes_backing, &longitudes_dtype)) {
        PyBuffer_Release(&latitudes);
        return NULL;
    }

    GeospatialPackedMatrix *packed = NULL;
    char *latitudes_data = NULL, *longitudes_data = NULL;
    int latitudes_free = 0, longitudes_free = 0;
    nk_size_t const count = (nk_size_t)latitudes.shape[0];
    nk_size_t bytes = 0;
    if (longitudes.shape[0] != latitudes.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "latitudes and longitudes must have the same length");
        goto cleanup;
    }
    if (!check_status(size_fn(count, &bytes))) goto cleanup;
    latitudes_data = ensure_contiguous_buffer(latitudes.buf, latitudes_dtype, nk_f64_k, 1, latitudes.shape,
                                              latitudes.strides, count, &latitudes_free);
    if (!latitudes_data) goto cleanup;
    longitudes_data = ensure_contiguous_buffer(longitudes.buf, longitudes_dtype, nk_f64_k, 1, longitudes.shape,
                                               longitudes.strides, count, &longitudes_free);
    if (!longitudes_data) goto cleanup;

    packed = PyObject_NewVar(GeospatialPackedMatrix, type, (Py_ssize_t)(bytes + nk_default_alignment_k - 1));
    if (!packed) goto cleanup;
    packed->points = count;
    packed->bytes = bytes;
    char *pack = packed->storage;
    pack += (nk_default_alignment_k - (uintptr_t)pack % nk_default_alignment_k) % nk_default_alignment_k;
    packed->pack = pack;
    PyThreadState *save = PyEval_SaveThread();
    nk_status_t const status = pack_fn(latitudes_data, longitudes_data, count, pack, stream);
    PyEval_RestoreThread(save);
    if (!check_status(status)) Py_CLEAR(packed);

cleanup:
    if (latitudes_free) PyMem_Free(latitudes_data);
    if (longitudes_free) PyMem_Free(longitudes_data);
    PyBuffer_Release(&latitudes);
    PyBuffer_Release(&longitudes);
    return (PyObject *)packed;
}

char const doc_haversines_pack[] =                                                           //
    "haversines_pack(latitudes, longitudes, /) -> HaversinesPackedMatrix\n\n"                //
    "Pack points as unit vectors on the mean sphere for haversines_packed().\n\n"            //
    "One serial routine packs for every capability, so a pack is the same on every\n"        //
    "machine.\n\n"                                                                           //
    "Args:\n"                                                                                //
    "    latitudes (array_like): Latitudes of shape [points] in radians, cast to float64.\n" //
    "    longitudes (array_like): Longitudes of shape [points] in radians, cast to\n"        //
    "        float64.\n\n"                                                                   //
    "Returns:\n"                                                                             //
    "    HaversinesPackedMatrix: The packed points.\n\n"                                     //
    "Signature:\n"                                                                           //
    "    >>> def haversines_pack(latitudes, longitudes, /) -> HaversinesPackedMatrix: ...";

PyObject *api_haversines_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return geospatial_pack_common(args, nargs, kwnames, &HaversinesPackedMatrixType, nk_kernel_haversines_pack_size_k,
                                  nk_kernel_haversines_pack_k);
}

char const doc_vincentys_pack[] =                                                            //
    "vincentys_pack(latitudes, longitudes, /) -> VincentysPackedMatrix\n\n"                  //
    "Pack points as their longitudes and the sines and cosines of their reduced\n"           //
    "latitudes on the WGS-84 ellipsoid for vincentys_packed().\n\n"                          //
    "One serial routine packs for every capability, so a pack is the same on every\n"        //
    "machine.\n\n"                                                                           //
    "Args:\n"                                                                                //
    "    latitudes (array_like): Latitudes of shape [points] in radians, cast to float64.\n" //
    "    longitudes (array_like): Longitudes of shape [points] in radians, cast to\n"        //
    "        float64.\n\n"                                                                   //
    "Returns:\n"                                                                             //
    "    VincentysPackedMatrix: The packed points.\n\n"                                      //
    "Signature:\n"                                                                           //
    "    >>> def vincentys_pack(latitudes, longitudes, /) -> VincentysPackedMatrix: ...";

PyObject *api_vincentys_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return geospatial_pack_common(args, nargs, kwnames, &VincentysPackedMatrixType, nk_kernel_vincentys_pack_size_k,
                                  nk_kernel_vincentys_pack_k);
}

/** Scores every point of pack @p a against every point of pack @p b, both of @p type, with the
 *  kernel of @p kind. */
static PyObject *geospatial_packed_common(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                                          PyTypeObject *type, nk_kernel_kind_t kind) {
    char const *name = nk_kernel_name(kind);
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "%s() requires exactly 2 positional arguments: a, b", name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], type) || !PyObject_TypeCheck(args[1], type)) {
        PyErr_Format(PyExc_TypeError, "%s() takes a and b as %s", name, type->tp_name);
        return NULL;
    }
    GeospatialPackedMatrix *a = (GeospatialPackedMatrix *)args[0];
    GeospatialPackedMatrix *b = (GeospatialPackedMatrix *)args[1];

    nk_capability_t capabilities = nk_cap_cpus_k;
    nk_stream_t stream = NULL;
    Py_ssize_t const nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++)
        if (!parse_dispatch_keyword(PyTuple_GET_ITEM(kwnames, i), args[nargs + i], &capabilities, &stream)) return NULL;

    nk_geospatial_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(kind, nk_f64_k, capabilities, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_LookupError, "No %s kernel for the given capabilities", name);
        return NULL;
    }

    Py_ssize_t const shape[2] = {(Py_ssize_t)a->points, (Py_ssize_t)b->points};
    Tensor *distances = Tensor_new(nk_f64_k, 2, shape);
    if (!distances) return NULL;
    nk_status_t status = nk_success_k;
    // An empty Tensor holds no data pointer for the kernel to write through
    if (a->points && b->points) {
        PyThreadState *save = PyEval_SaveThread();
        status = kernel(a->pack, b->pack, distances->data, b->points * sizeof(nk_f64_t), stream);
        PyEval_RestoreThread(save);
    }
    if (check_status(status)) return (PyObject *)distances;
    Py_DECREF(distances);
    return NULL;
}

char const doc_haversines_packed[] =                                                       //
    "haversines_packed(a, b, /) -> Tensor\n\n"                                             //
    "Great-circle distances in metres between every point of a and every point of b,\n"    //
    "on the sphere of the IUGG mean Earth radius, 6,371,008.8 m.\n\n"                      //
    "Each is 2R · atan2(‖a − b‖, ‖a + b‖) over unit vectors, which stays conditioned at\n" //
    "every separation from a millimetre to the antipode, and reads within 0.5% of the\n"   //
    "ellipsoidal geodesic.\n\n"                                                            //
    "Args:\n"                                                                              //
    "    a (HaversinesPackedMatrix): Row points.\n"                                        //
    "    b (HaversinesPackedMatrix): Column points.\n\n"                                   //
    "Returns:\n"                                                                           //
    "    Tensor: The [rows,columns] float64 distances.\n\n"                                //
    "Signature:\n"                                                                         //
    "    >>> def haversines_packed(a, b, /) -> Tensor: ...";

PyObject *api_haversines_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return geospatial_packed_common(args, nargs, kwnames, &HaversinesPackedMatrixType, nk_kernel_haversines_packed_k);
}

char const doc_vincentys_packed[] =                                                       //
    "vincentys_packed(a, b, /) -> Tensor\n\n"                                             //
    "Geodesic distances in metres between every point of a and every point of b, on\n"    //
    "the WGS-84 ellipsoid.\n\n"                                                           //
    "Vincenty's inverse formula iterates until the longitude settles within 1e-12\n"      //
    "radians. A pair still unsettled after 100 iterations, which happens only near the\n" //
    "antipode, reports its last iterate.\n\n"                                             //
    "Args:\n"                                                                             //
    "    a (VincentysPackedMatrix): Row points.\n"                                        //
    "    b (VincentysPackedMatrix): Column points.\n\n"                                   //
    "Returns:\n"                                                                          //
    "    Tensor: The [rows,columns] float64 distances.\n\n"                               //
    "Signature:\n"                                                                        //
    "    >>> def vincentys_packed(a, b, /) -> Tensor: ...";

PyObject *api_vincentys_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return geospatial_packed_common(args, nargs, kwnames, &VincentysPackedMatrixType, nk_kernel_vincentys_packed_k);
}

char const doc_sqeuclideans3d[] =                                                          //
    "sqeuclideans3d(a, b, /) -> Tensor\n\n"                                                //
    "Squared Euclidean distances between every point of a and every point of b in 3D.\n\n" //
    "Integer results are exact whenever every axis difference fits 32 bits, which\n"       //
    "Earth-fixed centimetres below 4,359 km of altitude always satisfy. Float64 inputs\n"  //
    "sum their three squares in Dot2.\n\n"                                                 //
    "Args:\n"                                                                              //
    "    a (array_like): Row points of shape [rows,3], int32 or float64.\n"                //
    "    b (array_like): Column points of shape [columns,3], in the same dtype.\n\n"       //
    "Returns:\n"                                                                           //
    "    Tensor: The [rows,columns] squared distances, uint64 for int32 points and\n"      //
    "        float64 for float64 ones.\n\n"                                                //
    "Signature:\n"                                                                         //
    "    >>> def sqeuclideans3d(a, b, /) -> Tensor: ...";

PyObject *api_sqeuclideans3d(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "sqeuclideans3d() requires exactly 2 positional arguments: a, b");
        return NULL;
    }
    nk_capability_t capabilities = nk_cap_cpus_k;
    nk_stream_t stream = NULL;
    Py_ssize_t const nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++)
        if (!parse_dispatch_keyword(PyTuple_GET_ITEM(kwnames, i), args[nargs + i], &capabilities, &stream)) return NULL;

    Py_buffer a, b;
    nk_buffer_backing_t a_backing, b_backing;
    nk_dtype_t dtype, b_dtype;
    if (!geospatial_points_buffer(args[0], 2, &a, &a_backing, &dtype)) return NULL;
    if (!geospatial_points_buffer(args[1], 2, &b, &b_backing, &b_dtype)) {
        PyBuffer_Release(&a);
        return NULL;
    }

    Tensor *distances = NULL;
    char *a_data = NULL, *b_data = NULL;
    int a_free = 0, b_free = 0;
    nk_size_t const rows = (nk_size_t)a.shape[0], columns = (nk_size_t)b.shape[0];
    nk_sqeuclideans3d_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    if (dtype != b_dtype) {
        PyErr_Format(PyExc_TypeError, "dtype mismatch: a is '%s' but b is '%s'", nk_dtype_to_pybuffer_typestr(dtype),
                     nk_dtype_to_pybuffer_typestr(b_dtype));
        goto cleanup;
    }
    nk_find_kernel_punned(nk_kernel_sqeuclideans3d_k, dtype, capabilities, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_LookupError, "No sqeuclideans3d kernel for dtype '%s'", nk_dtype_to_pybuffer_typestr(dtype));
        goto cleanup;
    }
    a_data = ensure_contiguous_buffer(a.buf, dtype, dtype, 2, a.shape, a.strides, rows * 3, &a_free);
    if (!a_data) goto cleanup;
    b_data = ensure_contiguous_buffer(b.buf, dtype, dtype, 2, b.shape, b.strides, columns * 3, &b_free);
    if (!b_data) goto cleanup;

    nk_dtype_t const out_dtype = nk_kernel_output_dtype(nk_kernel_sqeuclideans3d_k, dtype);
    Py_ssize_t const shape[2] = {(Py_ssize_t)rows, (Py_ssize_t)columns};
    distances = Tensor_new(out_dtype, 2, shape);
    // An empty Tensor holds no data pointer for the kernel to write through
    if (distances && rows && columns) {
        PyThreadState *save = PyEval_SaveThread();
        nk_status_t const status = kernel(a_data, rows, b_data, columns, distances->data,
                                          columns * nk_dtype_bytes_per_value(out_dtype), stream);
        PyEval_RestoreThread(save);
        if (!check_status(status)) Py_CLEAR(distances);
    }

cleanup:
    if (a_free) PyMem_Free(a_data);
    if (b_free) PyMem_Free(b_data);
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    return (PyObject *)distances;
}
