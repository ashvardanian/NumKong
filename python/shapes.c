/**
 *  @file python/meshes.c
 *  @author Ash Vardanian
 *  @date December 24, 2025
 *  @brief Many-to-many shape distances for NumKong Python bindings.
 *
 *  This module owns:
 *  - @c ShapesPacked: point clouds packed onto their principal axes, and views over them.
 *  - @c shapes_pack: packs one cloud, or a sequence of clouds of any sizes, into it.
 *  - @c shapes_pack_normalized: packs them the same way, each divided by its RMS radius.
 *  - @c shapes_chamfers_packed and its hausdorffs, wassersteinprocrustes and gromovwassersteins
 *    siblings: the distance between every row cloud and every column cloud.
 *
 *  Shape naming convention used in docs and errors:
 *  - packed @c a: @b [rows] clouds
 *  - packed @c b: @b [columns] clouds
 *  - distances: @b [rows,columns], rotations: @b [rows,columns,3,3]
 */
#include "meshes.h"
#include "tensor.h"
#include "numkong/meshes.h"

static nk_size_t shapes_packed_points(ShapesPacked const *packed, nk_size_t index) {
    void const *pack = packed->packs[index];
    return packed->dtype == nk_f16_k ? (nk_size_t)((nk_shapes_header_f32_t const *)pack)->points
                                     : (nk_size_t)((nk_shapes_header_f64_t const *)pack)->points;
}

/** The most points in any cloud of @p packed, which sizes the transport workspace. */
static nk_size_t shapes_packed_largest(ShapesPacked const *packed) {
    nk_size_t largest = 0;
    for (nk_size_t index = 0; index != packed->count; ++index) {
        nk_size_t const points = shapes_packed_points(packed, index);
        if (points > largest) largest = points;
    }
    return largest;
}

static size_t shapes_packed_nbytes(ShapesPacked const *packed) {
    nk_shapes_pack_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_shapes_pack_size_k, packed->dtype, nk_cap_cpus_k, (nk_kernel_punned_t *)&size_fn,
                          &cap);
    nk_size_t total = 0;
    for (nk_size_t index = 0; size_fn && index != packed->count; ++index) {
        nk_size_t bytes = 0;
        size_fn(shapes_packed_points(packed, index), &bytes);
        total += bytes;
    }
    return total;
}

/** A view over @p count packs of @p source, every @p step from @p start, sharing its buffer. */
static PyObject *shapes_packed_view(ShapesPacked *source, Py_ssize_t start, Py_ssize_t step, Py_ssize_t count) {
    ShapesPacked *view = PyObject_NewVar(ShapesPacked, &ShapesPackedType, count * (Py_ssize_t)sizeof(void const *));
    if (!view) return NULL;
    view->dtype = source->dtype;
    view->count = (nk_size_t)count;
    view->owner = source->owner ? source->owner : (PyObject *)source;
    Py_INCREF(view->owner);
    for (Py_ssize_t index = 0; index != count; ++index) view->packs[index] = source->packs[start + index * step];
    return (PyObject *)view;
}

/** Reads @p key as a position in @p packed, counting negative ones from the end. */
static int shapes_packed_index(ShapesPacked const *packed, PyObject *key, nk_size_t *index) {
    Py_ssize_t position = PyNumber_AsSsize_t(key, PyExc_IndexError);
    if (position == -1 && PyErr_Occurred()) return 0;
    if (position < 0) position += (Py_ssize_t)packed->count;
    if (position < 0 || position >= (Py_ssize_t)packed->count) {
        PyErr_SetString(PyExc_IndexError, "ShapesPacked index out of range");
        return 0;
    }
    *index = (nk_size_t)position;
    return 1;
}

static void ShapesPacked_dealloc(PyObject *self) {
    Py_XDECREF(((ShapesPacked *)self)->owner);
    Py_TYPE(self)->tp_free(self);
}

static PyObject *ShapesPacked_repr(PyObject *self) {
    ShapesPacked *packed = (ShapesPacked *)self;
    return PyUnicode_FromFormat("<ShapesPacked count=%zu dtype='%s' nbytes=%zu>", (size_t)packed->count,
                                nk_dtype_python_name(packed->dtype), shapes_packed_nbytes(packed));
}

static Py_ssize_t ShapesPacked_length(PyObject *self) { return (Py_ssize_t)((ShapesPacked *)self)->count; }

static PyObject *ShapesPacked_subscript(PyObject *self, PyObject *key) {
    ShapesPacked *packed = (ShapesPacked *)self;
    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step;
        if (PySlice_Unpack(key, &start, &stop, &step) < 0) return NULL;
        Py_ssize_t const count = PySlice_AdjustIndices((Py_ssize_t)packed->count, &start, &stop, step);
        return shapes_packed_view(packed, start, step, count);
    }
    nk_size_t index;
    if (!shapes_packed_index(packed, key, &index)) return NULL;
    return shapes_packed_view(packed, (Py_ssize_t)index, 1, 1);
}

static PyMappingMethods ShapesPacked_as_mapping = {
    .mp_length = ShapesPacked_length,
    .mp_subscript = ShapesPacked_subscript,
};

static PySequenceMethods ShapesPacked_as_sequence = {
    .sq_length = ShapesPacked_length,
};

static PyObject *ShapesPacked_get_count(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((ShapesPacked *)self)->count);
}

static PyObject *ShapesPacked_get_dtype(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyUnicode_FromString(nk_dtype_python_name(((ShapesPacked *)self)->dtype));
}

static PyObject *ShapesPacked_get_nbytes(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(shapes_packed_nbytes((ShapesPacked *)self));
}

static PyObject *ShapesPacked_get_points(PyObject *self, void *closure) {
    nk_unused_(closure);
    ShapesPacked *packed = (ShapesPacked *)self;
    Py_ssize_t const shape[1] = {(Py_ssize_t)packed->count};
    Tensor *points = Tensor_new(nk_u64_k, 1, shape);
    if (!points) return NULL;
    nk_u64_t *counts = (nk_u64_t *)points->data;
    for (nk_size_t index = 0; index != packed->count; ++index) counts[index] = shapes_packed_points(packed, index);
    return (PyObject *)points;
}

/** The header fields exposed one row per pack, or one value per pack for the scales. */
typedef enum {
    shapes_centroids_k,
    shapes_frames_k,
    shapes_spreads_k,
    shapes_scales_k,
} shapes_header_field_t;

/** The @p field of every pack in @p packed, f32 for f16 packs and f64 for f32 ones. */
static PyObject *shapes_header_tensor(ShapesPacked const *packed, shapes_header_field_t field) {
    int const half = packed->dtype == nk_f16_k;
    nk_dtype_t const header_dtype = half ? nk_f32_k : nk_f64_k;
    Py_ssize_t const width = field == shapes_frames_k ? 4 : field == shapes_scales_k ? 1 : 3;
    size_t const row_bytes = (size_t)width * nk_dtype_bytes_per_value(header_dtype);
    Py_ssize_t const shape[2] = {(Py_ssize_t)packed->count, width};
    Tensor *values = Tensor_new(header_dtype, field == shapes_scales_k ? 1 : 2, shape);
    if (!values) return NULL;
    for (nk_size_t index = 0; index != packed->count; ++index) {
        nk_shapes_header_f32_t const *header_f32 = (nk_shapes_header_f32_t const *)packed->packs[index];
        nk_shapes_header_f64_t const *header_f64 = (nk_shapes_header_f64_t const *)packed->packs[index];
        void const *rows_f32[4] = {header_f32->centroid, header_f32->frame, header_f32->spreads, &header_f32->scale};
        void const *rows_f64[4] = {header_f64->centroid, header_f64->frame, header_f64->spreads, &header_f64->scale};
        memcpy(values->data + index * row_bytes, half ? rows_f32[field] : rows_f64[field], row_bytes);
    }
    return (PyObject *)values;
}

static PyObject *ShapesPacked_get_centroids(PyObject *self, void *closure) {
    nk_unused_(closure);
    return shapes_header_tensor((ShapesPacked *)self, shapes_centroids_k);
}

static PyObject *ShapesPacked_get_frames(PyObject *self, void *closure) {
    nk_unused_(closure);
    return shapes_header_tensor((ShapesPacked *)self, shapes_frames_k);
}

static PyObject *ShapesPacked_get_spreads(PyObject *self, void *closure) {
    nk_unused_(closure);
    return shapes_header_tensor((ShapesPacked *)self, shapes_spreads_k);
}

static PyObject *ShapesPacked_get_scales(PyObject *self, void *closure) {
    nk_unused_(closure);
    return shapes_header_tensor((ShapesPacked *)self, shapes_scales_k);
}

static PyGetSetDef ShapesPacked_getset[] = {
    {"count", ShapesPacked_get_count, NULL, "Number of packed clouds", NULL},
    {"points", ShapesPacked_get_points, NULL, "Points in every cloud, uint64 [count]", NULL},
    {"centroids", ShapesPacked_get_centroids, NULL, "Mean input point of every cloud, [count,3]", NULL},
    {"frames", ShapesPacked_get_frames, NULL,
     "Unit quaternions (x, y, z, w) rotating every centered cloud onto its principal axes, [count,4]", NULL},
    {"spreads", ShapesPacked_get_spreads, NULL, "Descending eigenvalues of every packed cloud's moments, [count,3]",
     NULL},
    {"scales", ShapesPacked_get_scales, NULL,
     "What every rotated cloud was divided by: its RMS distance from the centroid, or 1, [count]", NULL},
    {"dtype", ShapesPacked_get_dtype, NULL, "Data type of the packed points", NULL},
    {"nbytes", ShapesPacked_get_nbytes, NULL, "Size of the packs in bytes", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *ShapesPacked_coordinates(PyObject *self, PyObject *key) {
    ShapesPacked *packed = (ShapesPacked *)self;
    nk_size_t index;
    if (!shapes_packed_index(packed, key, &index)) return NULL;
    nk_size_t const points = shapes_packed_points(packed, index);
    Py_ssize_t const shape[2] = {(Py_ssize_t)points, 3};
    Tensor *coordinates = Tensor_new(packed->dtype, 2, shape);
    if (!coordinates) return NULL;
    size_t const value_bytes = nk_dtype_bytes_per_value(packed->dtype);
    nk_size_t axis_stride = 0;
    char const *axes = packed->dtype == nk_f16_k ? (char const *)nk_shapes_axes_f16(packed->packs[index], &axis_stride)
                                                 : (char const *)nk_shapes_axes_f32(packed->packs[index], &axis_stride);
    for (nk_size_t point = 0; point != points; ++point)
        for (nk_size_t axis = 0; axis != 3; ++axis)
            memcpy(coordinates->data + (point * 3 + axis) * value_bytes,
                   axes + (axis * axis_stride + point) * value_bytes, value_bytes);
    return (PyObject *)coordinates;
}

static PyMethodDef ShapesPacked_methods[] = {
    {"coordinates", ShapesPacked_coordinates, METH_O,
     "Return the packed points of cloud `index` as a [points,3] Tensor in the packed dtype."},
    {NULL, NULL, 0, NULL},
};

PyTypeObject ShapesPackedType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.ShapesPacked",
    .tp_doc = "Point clouds packed onto their principal axes; indexing and slicing return views sharing the packs",
    .tp_basicsize = sizeof(ShapesPacked),
    .tp_itemsize = sizeof(char),
    .tp_dealloc = ShapesPacked_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = ShapesPacked_getset,
    .tp_methods = ShapesPacked_methods,
    .tp_repr = ShapesPacked_repr,
    .tp_as_sequence = &ShapesPacked_as_sequence,
    .tp_as_mapping = &ShapesPacked_as_mapping,
};

/** Acquires @p cloud as a @b [points,3] buffer of a known dtype, or sets a Python error. */
static int shapes_cloud_buffer(PyObject *cloud, Py_buffer *buffer, nk_buffer_backing_t *backing, nk_dtype_t *dtype) {
    if (!nk_get_buffer(cloud, buffer, PyBUF_STRIDES | PyBUF_FORMAT, backing)) {
        PyErr_SetString(PyExc_TypeError, "points must support buffer protocol or __array_interface__");
        return 0;
    }
    *dtype = resolve_nk_dtype_in_py_buffer(buffer);
    if (buffer->ndim != 2 || buffer->shape[1] != 3)
        PyErr_SetString(PyExc_ValueError, "points must be a [points,3] matrix or a list of them");
    else if (*dtype == nk_dtype_unknown_k)
        PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s'", buffer->format);
    else return 1;
    PyBuffer_Release(buffer);
    return 0;
}

/** Packs one cloud or a sequence of them with the packing kernel of @p pack_kind. */
static PyObject *shapes_pack_common(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                                    nk_kernel_kind_t pack_kind) {
    char const *name = nk_kernel_name(pack_kind);
    PyObject *dtype_argument = NULL;
    nk_capability_t capabilities = nk_cap_cpus_k;
    nk_stream_t stream = NULL;

    if (nargs < 1 || nargs > 2) {
        PyErr_Format(PyExc_TypeError, "%s() requires 1-2 arguments: points, dtype=None", name);
        return NULL;
    }
    if (nargs == 2) dtype_argument = args[1];

    Py_ssize_t const nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) {
            if (nargs == 2) {
                PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument 'dtype'", name);
                return NULL;
            }
            dtype_argument = args[nargs + i];
        }
        else if (!parse_dispatch_keyword(key, args[nargs + i], &capabilities, &stream)) return NULL;
    }
    if (dtype_argument == Py_None) dtype_argument = NULL;

    nk_dtype_t dtype = dtype_argument ? py_object_to_nk_dtype(dtype_argument) : nk_dtype_unknown_k;
    if (dtype_argument && dtype == nk_dtype_unknown_k) return NULL;
    if (dtype_argument && nk_shapes_output_dtype(dtype) == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported shapes dtype '%s'", nk_dtype_to_pybuffer_typestr(dtype));
        return NULL;
    }

    // A tuple snapshot keeps the clouds fixed while the GIL is released between packs
    int const many = PyList_Check(args[0]) || PyTuple_Check(args[0]);
    PyObject *sequence = many ? PySequence_Tuple(args[0]) : NULL;
    if (many && !sequence) return NULL;
    PyObject *const *clouds = sequence ? PySequence_Fast_ITEMS(sequence) : &args[0];
    Py_ssize_t const count = sequence ? PyTuple_GET_SIZE(sequence) : 1;

    ShapesPacked *packed = NULL;
    nk_shapes_pack_size_punned_t size_fn = NULL;
    nk_shapes_pack_punned_t pack_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    Py_buffer buffer;
    nk_buffer_backing_t backing;
    nk_dtype_t source_dtype;

    // The first pass sizes the buffer, and the first cloud picks the dtype unless `dtype=` did
    nk_size_t total_bytes = 0;
    for (Py_ssize_t index = 0; index != count; ++index) {
        if (!shapes_cloud_buffer(clouds[index], &buffer, &backing, &source_dtype)) goto cleanup;
        nk_size_t const points = (nk_size_t)buffer.shape[0];
        PyBuffer_Release(&buffer);
        if (dtype == nk_dtype_unknown_k) dtype = source_dtype == nk_f16_k ? nk_f16_k : nk_f32_k;
        if (!size_fn) {
            nk_find_kernel_punned(nk_kernel_shapes_pack_size_k, dtype, capabilities, (nk_kernel_punned_t *)&size_fn,
                                  &cap);
            if (size_fn && cap)
                nk_find_kernel_punned(pack_kind, dtype, capabilities, (nk_kernel_punned_t *)&pack_fn, &cap);
            if (!size_fn || !pack_fn || !cap) {
                PyErr_Format(PyExc_LookupError, "No %s kernel for dtype '%s'", name,
                             nk_dtype_to_pybuffer_typestr(dtype));
                goto cleanup;
            }
        }
        nk_size_t bytes = 0;
        if (!check_status(size_fn(points, &bytes))) goto cleanup;
        total_bytes += bytes;
    }
    if (dtype == nk_dtype_unknown_k) dtype = nk_f32_k;

    nk_size_t const pointers_bytes = (nk_size_t)count * sizeof(void const *);
    packed = PyObject_NewVar(ShapesPacked, &ShapesPackedType,
                             (Py_ssize_t)(pointers_bytes + nk_shapes_alignment_k - 1 + total_bytes));
    if (!packed) goto cleanup;
    packed->dtype = dtype;
    packed->count = (nk_size_t)count;
    packed->owner = NULL;

    char *pack = (char *)packed->packs + pointers_bytes;
    pack += (nk_shapes_alignment_k - (uintptr_t)pack % nk_shapes_alignment_k) % nk_shapes_alignment_k;
    char const *const packs_end = pack + total_bytes;
    for (Py_ssize_t index = 0; index != count; ++index) {
        if (!shapes_cloud_buffer(clouds[index], &buffer, &backing, &source_dtype)) goto failed;
        nk_size_t const points = (nk_size_t)buffer.shape[0];
        nk_size_t bytes = 0;
        size_fn(points, &bytes);
        if (bytes > (nk_size_t)(packs_end - pack)) {
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "points changed size while packing");
            goto failed;
        }
        int needs_free = 0;
        char *contiguous = ensure_contiguous_buffer(buffer.buf, source_dtype, dtype, 2, buffer.shape, buffer.strides,
                                                    points * 3, &needs_free);
        if (!contiguous) {
            PyBuffer_Release(&buffer);
            goto failed;
        }
        PyThreadState *save = PyEval_SaveThread();
        nk_status_t const status = pack_fn(contiguous, points, pack, stream);
        PyEval_RestoreThread(save);
        if (needs_free) PyMem_Free(contiguous);
        PyBuffer_Release(&buffer);
        if (!check_status(status)) goto failed;
        packed->packs[index] = pack;
        pack += bytes;
    }
    goto cleanup;

failed:
    Py_CLEAR(packed);
cleanup:
    Py_XDECREF(sequence);
    return (PyObject *)packed;
}

char const doc_shapes_pack[] =                                                         //
    "shapes_pack(points, /, dtype=None) -> ShapesPacked\n\n"                           //
    "Pack 3D point clouds for the many-to-many shape distances.\n\n"                   //
    "Each cloud is packed on its own: centered on its centroid and rotated onto its\n" //
    "principal axes, so a pack is the same on every machine.\n\n"                      //
    "Args:\n"                                                                          //
    "    points (array_like or list): One cloud of shape [points,3], or a list or\n"   //
    "        tuple of clouds of any sizes.\n"                                          //
    "    dtype (string, optional): Packing dtype, 'float16' or 'float32'. Default:\n"  //
    "        float16 for a float16 first cloud, float32 otherwise.\n\n"                //
    "Returns:\n"                                                                       //
    "    ShapesPacked: The packed clouds, for the shapes_*_packed() functions.\n\n"    //
    "Signature:\n"                                                                     //
    "    >>> def shapes_pack(points, /, dtype=None) -> ShapesPacked: ...";

PyObject *api_shapes_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_pack_common(args, nargs, kwnames, nk_kernel_shapes_pack_k);
}

char const doc_shapes_pack_normalized[] =                                              //
    "shapes_pack_normalized(points, /, dtype=None) -> ShapesPacked\n\n"                //
    "Pack 3D point clouds like shapes_pack(), each also divided by its RMS distance\n" //
    "from the centroid, which the `scales` property keeps, so every kind compares\n"   //
    "shapes regardless of size. A cloud with no spread keeps a scale of 1. Between\n"  //
    "normalized packs, each rotation takes column cloud b onto row cloud a as:\n\n"    //
    "    aᵢ ≈ (scale_a / scale_b) × R × (bⱼ − b̄) + ā\n\n"                              //
    "Args:\n"                                                                          //
    "    points (array_like or list): One cloud of shape [points,3], or a list or\n"   //
    "        tuple of clouds of any sizes.\n"                                          //
    "    dtype (string, optional): Packing dtype, 'float16' or 'float32'. Default:\n"  //
    "        float16 for a float16 first cloud, float32 otherwise.\n\n"                //
    "Returns:\n"                                                                       //
    "    ShapesPacked: The packed clouds, for the shapes_*_packed() functions.\n\n"    //
    "Signature:\n"                                                                     //
    "    >>> def shapes_pack_normalized(points, /, dtype=None) -> ShapesPacked: ...";

PyObject *api_shapes_pack_normalized(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_pack_common(args, nargs, kwnames, nk_kernel_shapes_pack_normalized_k);
}

/** Scores every row cloud against every column cloud with the many-to-many kernel of @p kind, in a
 *  workspace sized by @p workspace_kind, or none when that is @c nk_kernel_unknown_k. */
static PyObject *shapes_packed_common(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, nk_kernel_kind_t kind,
                                      nk_kernel_kind_t workspace_kind) {
    char const *name = nk_kernel_name(kind);
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "%s() requires exactly 2 positional arguments: a, b", name);
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], &ShapesPackedType) || !PyObject_TypeCheck(args[1], &ShapesPackedType)) {
        PyErr_SetString(PyExc_TypeError, "a and b must be ShapesPacked (use shapes_pack() first)");
        return NULL;
    }
    ShapesPacked *a = (ShapesPacked *)args[0];
    ShapesPacked *b = (ShapesPacked *)args[1];

    int with_rotations = 0;
    nk_capability_t capabilities = nk_cap_cpus_k;
    nk_stream_t stream = NULL;
    Py_ssize_t const nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "rotations") == 0) {
            with_rotations = PyObject_IsTrue(args[nargs + i]);
            if (with_rotations < 0) return NULL;
        }
        else if (!parse_dispatch_keyword(key, args[nargs + i], &capabilities, &stream)) return NULL;
    }

    if (a->dtype != b->dtype) {
        PyErr_Format(PyExc_TypeError, "dtype mismatch: a is '%s' but b is '%s'", nk_dtype_to_pybuffer_typestr(a->dtype),
                     nk_dtype_to_pybuffer_typestr(b->dtype));
        return NULL;
    }
    nk_dtype_t const dtype = a->dtype;
    nk_dtype_t const out_dtype = nk_kernel_output_dtype(kind, dtype);

    nk_kernel_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(kind, dtype, capabilities, &kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_LookupError, "No %s kernel for dtype '%s'", name, nk_dtype_to_pybuffer_typestr(dtype));
        return NULL;
    }

    nk_size_t workspace_bytes = 0;
    if (workspace_kind != nk_kernel_unknown_k) {
        nk_shapes_workspace_size_punned_t size_fn = NULL;
        cap = nk_cap_serial_k;
        nk_find_kernel_punned(workspace_kind, dtype, capabilities, (nk_kernel_punned_t *)&size_fn, &cap);
        if (!size_fn || !cap) {
            PyErr_Format(PyExc_LookupError, "No %s kernel for dtype '%s'", nk_kernel_name(workspace_kind),
                         nk_dtype_to_pybuffer_typestr(dtype));
            return NULL;
        }
        if (!check_status(size_fn(shapes_packed_largest(a), shapes_packed_largest(b), &workspace_bytes))) return NULL;
    }

    Py_ssize_t const shape[4] = {(Py_ssize_t)a->count, (Py_ssize_t)b->count, 3, 3};
    Tensor *distances = Tensor_new(out_dtype, 2, shape);
    if (!distances) return NULL;
    Tensor *rotations = NULL;
    if (with_rotations && !(rotations = Tensor_new(nk_f32_k, 4, shape))) {
        Py_DECREF(distances);
        return NULL;
    }
    char *workspace = NULL, *aligned_workspace = NULL;
    if (workspace_bytes) {
        workspace = PyMem_Malloc(workspace_bytes + nk_default_alignment_k - 1);
        if (!workspace) {
            Py_DECREF(distances);
            Py_XDECREF(rotations);
            return PyErr_NoMemory();
        }
        aligned_workspace = workspace + (nk_default_alignment_k - (uintptr_t)workspace % nk_default_alignment_k) %
                                            nk_default_alignment_k;
    }

    nk_status_t status = nk_success_k;
    // An empty Tensor holds no data pointer for the kernel to write through
    if (a->count && b->count) {
        nk_size_t const results_stride = b->count * nk_dtype_bytes_per_value(out_dtype);
        nk_f32_t *rotations_data = rotations ? (nk_f32_t *)rotations->data : NULL;
        PyThreadState *save = PyEval_SaveThread();
        status = workspace_kind == nk_kernel_unknown_k
                     ? ((nk_shapes_packed_punned_t)kernel)(a->packs, a->count, b->packs, b->count, distances->data,
                                                           results_stride, rotations_data, stream)
                     : ((nk_shapes_packed_workspace_punned_t)kernel)(a->packs, a->count, b->packs, b->count,
                                                                     distances->data, results_stride, rotations_data,
                                                                     aligned_workspace, workspace_bytes, stream);
        PyEval_RestoreThread(save);
    }
    PyMem_Free(workspace);
    if (!check_status(status)) {
        Py_DECREF(distances);
        Py_XDECREF(rotations);
        return NULL;
    }
    if (!rotations) return (PyObject *)distances;
    PyObject *result = PyTuple_Pack(2, distances, rotations);
    Py_DECREF(distances);
    Py_DECREF(rotations);
    return result;
}

char const doc_shapes_chamfers_packed[] =                                                //
    "shapes_chamfers_packed(a, b, /, *, rotations=False) -> Tensor\n\n"                  //
    "Chamfer distances between every cloud of a and every cloud of b.\n\n"               //
    "Each is ½ × (meanᵢ minⱼ ‖aᵢ − S·bⱼ‖ + meanⱼ minᵢ ‖aᵢ − S·bⱼ‖), the best of the 8\n" //
    "sign flips S of the principal axes, zero between two empty clouds and infinite\n"   //
    "between an empty and a non-empty one.\n\n"                                          //
    "Args:\n"                                                                            //
    "    a (ShapesPacked): Row clouds.\n"                                                //
    "    b (ShapesPacked): Column clouds, packed in the same dtype.\n"                   //
    "    rotations (bool): Also return the rotations taking each column cloud onto\n"    //
    "        each row cloud, aᵢ ≈ R × (bⱼ − b̄) + ā. Default: False.\n\n"                 //
    "Returns:\n"                                                                         //
    "    Tensor: The [rows,columns] distances, float32 for float16 packs and float64\n"  //
    "        for float32 ones, or a tuple of them and the [rows,columns,3,3] float32\n"  //
    "        rotations when rotations=True.\n\n"                                         //
    "Signature:\n"                                                                       //
    "    >>> def shapes_chamfers_packed(a, b, /, *, rotations=False) -> Tensor: ...";

PyObject *api_shapes_chamfers_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_packed_common(args, nargs, kwnames, nk_kernel_shapes_chamfers_packed_k, nk_kernel_unknown_k);
}

char const doc_shapes_hausdorffs_packed[] =                                             //
    "shapes_hausdorffs_packed(a, b, /, *, rotations=False) -> Tensor\n\n"               //
    "Hausdorff distances between every cloud of a and every cloud of b.\n\n"            //
    "Each is max(maxᵢ minⱼ ‖aᵢ − S·bⱼ‖, maxⱼ minᵢ ‖aᵢ − S·bⱼ‖), the best of the 8\n"    //
    "sign flips S of the principal axes, with the empty-cloud convention of\n"          //
    "shapes_chamfers_packed().\n\n"                                                     //
    "Args:\n"                                                                           //
    "    a (ShapesPacked): Row clouds.\n"                                               //
    "    b (ShapesPacked): Column clouds, packed in the same dtype.\n"                  //
    "    rotations (bool): Also return the rotations taking each column cloud onto\n"   //
    "        each row cloud, aᵢ ≈ R × (bⱼ − b̄) + ā. Default: False.\n\n"                //
    "Returns:\n"                                                                        //
    "    Tensor: The [rows,columns] distances, float32 for float16 packs and float64\n" //
    "        for float32 ones, or a tuple of them and the [rows,columns,3,3] float32\n" //
    "        rotations when rotations=True.\n\n"                                        //
    "Signature:\n"                                                                      //
    "    >>> def shapes_hausdorffs_packed(a, b, /, *, rotations=False) -> Tensor: ...";

PyObject *api_shapes_hausdorffs_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_packed_common(args, nargs, kwnames, nk_kernel_shapes_hausdorffs_packed_k, nk_kernel_unknown_k);
}

char const doc_shapes_wassersteinprocrustes_packed[] =                                    //
    "shapes_wassersteinprocrustes_packed(a, b, /, *, rotations=False) -> Tensor\n\n"      //
    "Wasserstein-Procrustes distances between every cloud of a and every cloud of b.\n\n" //
    "Each is the root mean squared distance of the exact optimal transport plan\n"        //
    "between uniform weights, after the rotation or reflection minimizing it,\n"          //
    "alternated from each of the 8 sign flips of the principal axes. On equal sizes\n"    //
    "with a known correspondence it equals the Kabsch RMSD. The empty-cloud\n"            //
    "convention matches shapes_chamfers_packed(), and the workspace is allocated\n"       //
    "inside the call.\n\n"                                                                //
    "Args:\n"                                                                             //
    "    a (ShapesPacked): Row clouds.\n"                                                 //
    "    b (ShapesPacked): Column clouds, packed in the same dtype.\n"                    //
    "    rotations (bool): Also return the rotations taking each column cloud onto\n"     //
    "        each row cloud, aᵢ ≈ R × (bⱼ − b̄) + ā. Default: False.\n\n"                  //
    "Returns:\n"                                                                          //
    "    Tensor: The [rows,columns] distances, float32 for float16 packs and float64\n"   //
    "        for float32 ones, or a tuple of them and the [rows,columns,3,3] float32\n"   //
    "        rotations when rotations=True.\n\n"                                          //
    "Signature:\n"                                                                        //
    "    >>> def shapes_wassersteinprocrustes_packed(a, b, /, *, rotations=False) -> Tensor: ...";

PyObject *api_shapes_wassersteinprocrustes_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                                  PyObject *kwnames) {
    nk_unused_(self);
    return shapes_packed_common(args, nargs, kwnames, nk_kernel_shapes_wassersteinprocrustes_packed_k,
                                nk_kernel_shapes_wassersteinprocrustes_workspace_size_k);
}

char const doc_shapes_gromovwassersteins_packed[] =                                      //
    "shapes_gromovwassersteins_packed(a, b, /, *, rotations=False) -> Tensor\n\n"        //
    "Gromov-Wasserstein distances between every cloud of a and every cloud of b.\n\n"    //
    "Each compares the two clouds' own distance matrices under the best transport\n"     //
    "plan a fixed budget of Frank-Wolfe steps finds from the Wasserstein-Procrustes\n"   //
    "plans and the product plan, ½ · √Σᵢⱼₖₗ (‖aᵢ − aₖ‖ − ‖bⱼ − bₗ‖)² · Pᵢⱼ · Pₖₗ. The\n" //
    "rotation is the orthogonal Procrustes fit of that plan. The empty-cloud\n"          //
    "convention matches shapes_chamfers_packed(), and the workspace is allocated\n"      //
    "inside the call.\n\n"                                                               //
    "Args:\n"                                                                            //
    "    a (ShapesPacked): Row clouds.\n"                                                //
    "    b (ShapesPacked): Column clouds, packed in the same dtype.\n"                   //
    "    rotations (bool): Also return the rotations taking each column cloud onto\n"    //
    "        each row cloud, aᵢ ≈ R × (bⱼ − b̄) + ā. Default: False.\n\n"                 //
    "Returns:\n"                                                                         //
    "    Tensor: The [rows,columns] distances, float32 for float16 packs and float64\n"  //
    "        for float32 ones, or a tuple of them and the [rows,columns,3,3] float32\n"  //
    "        rotations when rotations=True.\n\n"                                         //
    "Signature:\n"                                                                       //
    "    >>> def shapes_gromovwassersteins_packed(a, b, /, *, rotations=False) -> Tensor: ...";

PyObject *api_shapes_gromovwassersteins_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                               PyObject *kwnames) {
    nk_unused_(self);
    return shapes_packed_common(args, nargs, kwnames, nk_kernel_shapes_gromovwassersteins_packed_k,
                                nk_kernel_shapes_gromovwassersteins_workspace_size_k);
}
