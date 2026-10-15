/**
 *  @file python/shapes.c
 *  @author Ash Vardanian
 *  @date December 24, 2025
 *  @brief Many-to-many shape distances for NumKong Python bindings.
 *
 *  This module owns:
 *  - @c ShapesPackedMatrix: point clouds packed onto their principal axes, and views over them.
 *  - @c shapes_pack: packs one cloud, or a sequence of clouds of any sizes, into it.
 *  - @c shapes_pack_normalized: packs them the same way, each divided by its RMS radius.
 *  - @c chamfers_packed and its hausdorffs, wassersteinprocrustes and gromovwassersteins siblings:
 *    the distance between every row cloud and every column cloud.
 *
 *  Shape naming convention used in docs and errors:
 *  - packed @c a: @b [rows] clouds
 *  - packed @c b: @b [columns] clouds
 *  - distances: @b [rows,columns], rotations: @b [rows,columns,3,3]
 */
#include "shapes.h"
#include "parallel.h" // `nk_parallel_for_tiles`
#include "tensor.h"

#include "numkong/shapes.h"

/** Reads the pose and bounds of pack @p index into every non-null output, f32 for f16 packs and f64
 *  for f32 ones. */
static nk_status_t shapes_packed_shape(ShapesPackedMatrix const *packed, nk_size_t index, nk_size_t *points,
                                       void *centroid, void *frame, void *scale, void *spreads, void *max_radius) {
    void const *pack = packed->packs[index];
    return packed->dtype == nk_f16_k ? nk_shapes_packed_shape_f16_best(pack, points, centroid, frame, scale, spreads,
                                                                       max_radius, nk_cap_cpus_k, NULL)
                                     : nk_shapes_packed_shape_f32_best(pack, points, centroid, frame, scale, spreads,
                                                                       max_radius, nk_cap_cpus_k, NULL);
}

static nk_size_t shapes_packed_points(ShapesPackedMatrix const *packed, nk_size_t index) {
    nk_size_t points = 0;
    shapes_packed_shape(packed, index, &points, NULL, NULL, NULL, NULL, NULL);
    return points;
}

/** The most points in any cloud of @p packed, which sizes the transport workspace. */
static nk_size_t shapes_packed_largest(ShapesPackedMatrix const *packed) {
    nk_size_t largest = 0;
    for (nk_size_t index = 0; index != packed->clouds; ++index) {
        nk_size_t const points = shapes_packed_points(packed, index);
        if (points > largest) largest = points;
    }
    return largest;
}

static nk_size_t shapes_packed_total_points(ShapesPackedMatrix const *packed) {
    nk_size_t total = 0;
    for (nk_size_t index = 0; index != packed->clouds; ++index) total += shapes_packed_points(packed, index);
    return total;
}

static size_t shapes_packed_nbytes(ShapesPackedMatrix const *packed) {
    nk_shapes_pack_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_shapes_pack_size_k, packed->dtype, nk_cap_cpus_k, (nk_kernel_punned_t *)&size_fn,
                          &cap);
    nk_size_t total = 0;
    for (nk_size_t index = 0; size_fn && index != packed->clouds; ++index) {
        nk_size_t bytes = 0;
        size_fn(shapes_packed_points(packed, index), &bytes);
        total += bytes;
    }
    return total;
}

/** A view over @p count packs of @p source, every @p step from @p start, sharing its buffer. */
static PyObject *shapes_packed_view(ShapesPackedMatrix *source, Py_ssize_t start, Py_ssize_t step, Py_ssize_t count) {
    ShapesPackedMatrix *view = PyObject_NewVar(ShapesPackedMatrix, &ShapesPackedMatrixType,
                                               count * (Py_ssize_t)sizeof(void const *));
    if (!view) return NULL;
    view->dtype = source->dtype;
    view->clouds = (nk_size_t)count;
    view->owner = source->owner ? source->owner : (PyObject *)source;
    Py_INCREF(view->owner);
    for (Py_ssize_t index = 0; index != count; ++index) view->packs[index] = source->packs[start + index * step];
    return (PyObject *)view;
}

/** Reads @p key as a position in @p packed, counting negative ones from the end. */
static int shapes_packed_index(ShapesPackedMatrix const *packed, PyObject *key, nk_size_t *index) {
    Py_ssize_t position = PyNumber_AsSsize_t(key, PyExc_IndexError);
    if (position == -1 && PyErr_Occurred()) return 0;
    if (position < 0) position += (Py_ssize_t)packed->clouds;
    if (position < 0 || position >= (Py_ssize_t)packed->clouds) {
        PyErr_SetString(PyExc_IndexError, "ShapesPackedMatrix index out of range");
        return 0;
    }
    *index = (nk_size_t)position;
    return 1;
}

static void ShapesPackedMatrix_dealloc(PyObject *self) {
    Py_XDECREF(((ShapesPackedMatrix *)self)->owner);
    Py_TYPE(self)->tp_free(self);
}

static PyObject *ShapesPackedMatrix_repr(PyObject *self) {
    ShapesPackedMatrix *packed = (ShapesPackedMatrix *)self;
    return PyUnicode_FromFormat("<ShapesPackedMatrix clouds=%zu points=%zu dtype='%s' nbytes=%zu>",
                                (size_t)packed->clouds, (size_t)shapes_packed_total_points(packed),
                                nk_dtype_python_name(packed->dtype), shapes_packed_nbytes(packed));
}

static Py_ssize_t ShapesPackedMatrix_length(PyObject *self) { return (Py_ssize_t)((ShapesPackedMatrix *)self)->clouds; }

static PyObject *ShapesPackedMatrix_subscript(PyObject *self, PyObject *key) {
    ShapesPackedMatrix *packed = (ShapesPackedMatrix *)self;
    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step;
        if (PySlice_Unpack(key, &start, &stop, &step) < 0) return NULL;
        Py_ssize_t const count = PySlice_AdjustIndices((Py_ssize_t)packed->clouds, &start, &stop, step);
        return shapes_packed_view(packed, start, step, count);
    }
    nk_size_t index;
    if (!shapes_packed_index(packed, key, &index)) return NULL;
    return shapes_packed_view(packed, (Py_ssize_t)index, 1, 1);
}

static PyMappingMethods ShapesPackedMatrix_as_mapping = {
    .mp_length = ShapesPackedMatrix_length,
    .mp_subscript = ShapesPackedMatrix_subscript,
};

static PySequenceMethods ShapesPackedMatrix_as_sequence = {
    .sq_length = ShapesPackedMatrix_length,
};

static PyObject *ShapesPackedMatrix_get_clouds(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((ShapesPackedMatrix *)self)->clouds);
}

static PyObject *ShapesPackedMatrix_get_points(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(shapes_packed_total_points((ShapesPackedMatrix *)self));
}

static PyObject *ShapesPackedMatrix_get_shape(PyObject *self, void *closure) {
    nk_unused_(closure);
    ShapesPackedMatrix *packed = (ShapesPackedMatrix *)self;
    PyObject *clouds_integer = PyLong_FromSize_t(packed->clouds);
    PyObject *points_integer = PyLong_FromSize_t(shapes_packed_total_points(packed));
    PyObject *shape = clouds_integer && points_integer ? PyTuple_Pack(2, clouds_integer, points_integer) : NULL;
    Py_XDECREF(clouds_integer);
    Py_XDECREF(points_integer);
    return shape;
}

static PyObject *ShapesPackedMatrix_get_dtype(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyUnicode_FromString(nk_dtype_python_name(((ShapesPackedMatrix *)self)->dtype));
}

static PyObject *ShapesPackedMatrix_get_nbytes(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(shapes_packed_nbytes((ShapesPackedMatrix *)self));
}

/** The pose and bounds exposed one row per pack, or one value per pack for the scales and radii,
 *  in the order @c nk_shapes_packed_shape_f16_best takes them. */
typedef enum {
    shapes_centroids_k,
    shapes_frames_k,
    shapes_scales_k,
    shapes_spreads_k,
    shapes_max_radii_k,
} shapes_packed_field_t;

/** The @p field of every pack in @p packed, f32 for f16 packs and f64 for f32 ones. */
static PyObject *shapes_packed_tensor(ShapesPackedMatrix const *packed, shapes_packed_field_t field) {
    int const scalar = field == shapes_scales_k || field == shapes_max_radii_k;
    nk_dtype_t const field_dtype = packed->dtype == nk_f16_k ? nk_f32_k : nk_f64_k;
    Py_ssize_t const width = field == shapes_frames_k ? 4 : scalar ? 1 : 3;
    size_t const row_bytes = (size_t)width * nk_dtype_bytes_per_value(field_dtype);
    Py_ssize_t const shape[2] = {(Py_ssize_t)packed->clouds, width};
    Tensor *values = Tensor_new(field_dtype, scalar ? 1 : 2, shape);
    if (!values) return NULL;
    for (nk_size_t index = 0; index != packed->clouds; ++index) {
        void *outputs[5] = {NULL, NULL, NULL, NULL, NULL};
        outputs[field] = values->data + index * row_bytes;
        nk_status_t const status = shapes_packed_shape(packed, index, NULL, outputs[0], outputs[1], outputs[2],
                                                       outputs[3], outputs[4]);
        if (!check_status(status)) {
            Py_DECREF(values);
            return NULL;
        }
    }
    return (PyObject *)values;
}

static PyObject *ShapesPackedMatrix_get_centroids(PyObject *self, void *closure) {
    nk_unused_(closure);
    return shapes_packed_tensor((ShapesPackedMatrix *)self, shapes_centroids_k);
}

static PyObject *ShapesPackedMatrix_get_frames(PyObject *self, void *closure) {
    nk_unused_(closure);
    return shapes_packed_tensor((ShapesPackedMatrix *)self, shapes_frames_k);
}

static PyObject *ShapesPackedMatrix_get_spreads(PyObject *self, void *closure) {
    nk_unused_(closure);
    return shapes_packed_tensor((ShapesPackedMatrix *)self, shapes_spreads_k);
}

static PyObject *ShapesPackedMatrix_get_scales(PyObject *self, void *closure) {
    nk_unused_(closure);
    return shapes_packed_tensor((ShapesPackedMatrix *)self, shapes_scales_k);
}

static PyObject *ShapesPackedMatrix_get_max_radii(PyObject *self, void *closure) {
    nk_unused_(closure);
    return shapes_packed_tensor((ShapesPackedMatrix *)self, shapes_max_radii_k);
}

static PyGetSetDef ShapesPackedMatrix_getset[] = {
    {"clouds", ShapesPackedMatrix_get_clouds, NULL, "Number of packed clouds", NULL},
    {"points", ShapesPackedMatrix_get_points, NULL, "Total points across the packed clouds", NULL},
    {"shape", ShapesPackedMatrix_get_shape, NULL, "Dimensions (clouds, points) of the packed set", NULL},
    {"centroids", ShapesPackedMatrix_get_centroids, NULL, "Mean input point of every cloud, [clouds,3]", NULL},
    {"frames", ShapesPackedMatrix_get_frames, NULL,
     "Unit quaternions (x, y, z, w) rotating every centered cloud onto its principal axes, [clouds,4]", NULL},
    {"spreads", ShapesPackedMatrix_get_spreads, NULL,
     "Descending eigenvalues of every packed cloud's moments, [clouds,3]", NULL},
    {"scales", ShapesPackedMatrix_get_scales, NULL,
     "What every rotated cloud was divided by: its RMS distance from the centroid, or 1, [clouds]", NULL},
    {"max_radii", ShapesPackedMatrix_get_max_radii, NULL, "Largest packed point norm of every cloud, [clouds]", NULL},
    {"dtype", ShapesPackedMatrix_get_dtype, NULL, "Data type of the packed points", NULL},
    {"nbytes", ShapesPackedMatrix_get_nbytes, NULL, "Size of the packs in bytes", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *ShapesPackedMatrix_pack_size(PyObject *cls, PyObject *const *args, Py_ssize_t nargs,
                                              PyObject *kwnames) {
    nk_unused_(cls);

    PyObject *dtype_argument = nargs == 2 ? args[1] : NULL;
    Py_ssize_t const nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs < 1 || nargs > 2 || nargs + nkw > 2) {
        PyErr_SetString(PyExc_TypeError, "pack_size(points, /, dtype)");
        return NULL;
    }
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (dtype_argument) {
                PyErr_SetString(PyExc_TypeError, "pack_size() got multiple values for argument 'dtype'");
                return NULL;
            }
            dtype_argument = args[nargs + i];
        }
        else {
            PyErr_Format(PyExc_TypeError, "pack_size() got unexpected keyword argument '%S'", name);
            return NULL;
        }
    }
    if (!dtype_argument) {
        PyErr_SetString(PyExc_TypeError, "pack_size() requires 'dtype' argument");
        return NULL;
    }

    nk_size_t const points = (nk_size_t)PyLong_AsSize_t(args[0]);
    if (points == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
    nk_dtype_t const dtype = py_object_to_nk_dtype(dtype_argument);
    if (dtype == nk_dtype_unknown_k) return NULL;

    nk_shapes_pack_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_shapes_pack_size_k, dtype, nk_cap_cpus_k, (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn || !cap) {
        PyErr_Format(PyExc_LookupError, "No shapes pack_size kernel for dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(dtype));
        return NULL;
    }
    nk_size_t bytes = 0;
    if (!check_status(size_fn(points, &bytes))) return NULL;
    return PyLong_FromSize_t(bytes);
}

static PyObject *ShapesPackedMatrix_unpack(PyObject *self, PyObject *key) {
    ShapesPackedMatrix *packed = (ShapesPackedMatrix *)self;
    nk_size_t index;
    if (!shapes_packed_index(packed, key, &index)) return NULL;
    Py_ssize_t const shape[2] = {(Py_ssize_t)shapes_packed_points(packed, index), 3};
    Tensor *cloud = Tensor_new(packed->dtype, 2, shape);
    if (!cloud) return NULL;
    void const *pack = packed->packs[index];
    nk_status_t status = nk_success_k;
    // An empty Tensor holds no data pointer for the kernel to write through
    if (shape[0])
        status = packed->dtype == nk_f16_k
                     ? nk_shapes_unpack_f16_best(pack, (nk_f16_t *)cloud->data, nk_cap_cpus_k, NULL)
                     : nk_shapes_unpack_f32_best(pack, (nk_f32_t *)cloud->data, nk_cap_cpus_k, NULL);
    if (check_status(status)) return (PyObject *)cloud;
    Py_DECREF(cloud);
    return NULL;
}

static PyMethodDef ShapesPackedMatrix_methods[] = {
    {"pack_size", (PyCFunction)ShapesPackedMatrix_pack_size, METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     "Return the bytes one packed cloud of the given points and dtype takes."},
    {"unpack", ShapesPackedMatrix_unpack, METH_O,
     "Return one cloud as [points,3] in the pack dtype and its original pose and scale, within one storage rounding."},
    {NULL, NULL, 0, NULL},
};

PyTypeObject ShapesPackedMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.ShapesPackedMatrix",
    .tp_doc = "Point clouds packed onto their principal axes; indexing and slicing return views sharing the packs",
    .tp_basicsize = sizeof(ShapesPackedMatrix),
    .tp_itemsize = sizeof(char),
    .tp_dealloc = ShapesPackedMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = ShapesPackedMatrix_getset,
    .tp_methods = ShapesPackedMatrix_methods,
    .tp_repr = ShapesPackedMatrix_repr,
    .tp_as_sequence = &ShapesPackedMatrix_as_sequence,
    .tp_as_mapping = &ShapesPackedMatrix_as_mapping,
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

/** One cloud of a pack: its acquired buffer, its points as the kernel reads them, and its pack. */
typedef struct shapes_pack_source_t {
    Py_buffer buffer;
    char *points;
    nk_size_t count;
    int owns_points;
    void *pack;
} shapes_pack_source_t;

/** Every cloud of a pack, one per tile, named as in `nk_shapes_pack_*`. */
typedef struct shapes_pack_task_t {
    nk_shapes_pack_punned_t kernel;
    shapes_pack_source_t const *sources;
    nk_stream_t stream;
} shapes_pack_task_t;

static nk_status_t shapes_pack_tile_(nk_size_t tile_index, void *context) {
    shapes_pack_task_t const *task = (shapes_pack_task_t const *)context;
    shapes_pack_source_t const *source = &task->sources[tile_index];
    return task->kernel(source->points, source->count, source->pack, task->stream);
}

/** Packs one cloud or a sequence of them with the packing kernel of @p pack_kind. */
static PyObject *shapes_pack_common(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                                    nk_kernel_kind_t pack_kind) {
    char const *name = nk_kernel_name(pack_kind);
    PyObject *dtype_argument = NULL;
    nk_size_t threads = 1;
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
        else if (PyUnicode_CompareWithASCIIString(key, "threads") == 0) {
            Py_ssize_t t = PyLong_AsSsize_t(args[nargs + i]);
            if (t == -1 && PyErr_Occurred()) return NULL;
            threads = (nk_size_t)(t >= 0 ? t : 0);
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

    // A tuple snapshot and buffers held to the end keep the clouds fixed while the GIL is released
    int const many = PyList_Check(args[0]) || PyTuple_Check(args[0]);
    PyObject *sequence = many ? PySequence_Tuple(args[0]) : NULL;
    if (many && !sequence) return NULL;
    PyObject *const *clouds = sequence ? PySequence_Fast_ITEMS(sequence) : &args[0];
    Py_ssize_t const count = sequence ? PyTuple_GET_SIZE(sequence) : 1;

    ShapesPackedMatrix *packed = NULL;
    nk_shapes_pack_size_punned_t size_fn = NULL;
    nk_shapes_pack_punned_t pack_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_buffer_backing_t backing;
    nk_dtype_t source_dtype;
    Py_ssize_t acquired = 0;
    shapes_pack_source_t *sources = PyMem_Calloc((size_t)count, sizeof(shapes_pack_source_t));
    if (!sources) {
        PyErr_NoMemory();
        goto cleanup;
    }

    // The first pass sizes the buffer, and the first cloud picks the dtype unless `dtype=` did
    nk_size_t total_bytes = 0;
    for (Py_ssize_t index = 0; index != count; ++index) {
        shapes_pack_source_t *source = &sources[index];
        if (!shapes_cloud_buffer(clouds[index], &source->buffer, &backing, &source_dtype)) goto cleanup;
        acquired = index + 1;
        source->count = (nk_size_t)source->buffer.shape[0];
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
        if (!check_status(size_fn(source->count, &bytes))) goto cleanup;
        total_bytes += nk_size_round_up_to_multiple_(bytes, nk_default_alignment_k);
        source->points = ensure_contiguous_buffer(source->buffer.buf, source_dtype, dtype, 2, source->buffer.shape,
                                                  source->buffer.strides, source->count * 3, &source->owns_points);
        if (!source->points) goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = nk_f32_k;

    nk_size_t const pointers_bytes = (nk_size_t)count * sizeof(void const *);
    packed = PyObject_NewVar(ShapesPackedMatrix, &ShapesPackedMatrixType,
                             (Py_ssize_t)(pointers_bytes + nk_default_alignment_k - 1 + total_bytes));
    if (!packed) goto cleanup;
    packed->dtype = dtype;
    packed->clouds = (nk_size_t)count;
    packed->owner = NULL;

    char *pack = (char *)packed->packs + pointers_bytes;
    pack += (nk_default_alignment_k - (uintptr_t)pack % nk_default_alignment_k) % nk_default_alignment_k;
    for (Py_ssize_t index = 0; index != count; ++index) {
        nk_size_t bytes = 0;
        size_fn(sources[index].count, &bytes);
        packed->packs[index] = sources[index].pack = pack;
        pack += nk_size_round_up_to_multiple_(bytes, nk_default_alignment_k);
    }
    shapes_pack_task_t const task = {.kernel = pack_fn, .sources = sources, .stream = stream};
    PyThreadState *save = PyEval_SaveThread();
    nk_status_t const status = nk_parallel_for_tiles((nk_size_t)count, threads, shapes_pack_tile_, (void *)&task);
    PyEval_RestoreThread(save);
    if (!check_status(status)) Py_CLEAR(packed);

cleanup:
    for (Py_ssize_t index = 0; index != acquired; ++index) {
        if (sources[index].owns_points) PyMem_Free(sources[index].points);
        PyBuffer_Release(&sources[index].buffer);
    }
    PyMem_Free(sources);
    Py_XDECREF(sequence);
    return (PyObject *)packed;
}

char const doc_shapes_pack[] =                                                         //
    "shapes_pack(points, /, dtype=None, *, threads=1) -> ShapesPackedMatrix\n\n"       //
    "Pack 3D point clouds for the many-to-many shape distances.\n\n"                   //
    "Each cloud is packed on its own: centered on its centroid and rotated onto its\n" //
    "principal axes, so a pack is the same on every machine.\n\n"                      //
    "Args:\n"                                                                          //
    "    points (array_like or list): One cloud of shape [points,3], or a list or\n"   //
    "        tuple of clouds of any sizes.\n"                                          //
    "    dtype (string, optional): Packing dtype, 'float16' or 'float32'. Default:\n"  //
    "        float16 for a float16 first cloud, float32 otherwise.\n"                  //
    "    threads (int, optional): Worker threads, defaulting to 1.\n\n"                //
    "Returns:\n"                                                                       //
    "    ShapesPackedMatrix: The packed clouds, for chamfers_packed(),\n"              //
    "        hausdorffs_packed(), wassersteinprocrustes_packed() and\n"                //
    "        gromovwassersteins_packed().\n\n"                                         //
    "Signature:\n"                                                                     //
    "    >>> def shapes_pack(points, /, dtype=None, *, threads=1) -> ShapesPackedMatrix: ...";

PyObject *api_shapes_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_pack_common(args, nargs, kwnames, nk_kernel_shapes_pack_k);
}

char const doc_shapes_pack_normalized[] =                                                   //
    "shapes_pack_normalized(points, /, dtype=None, *, threads=1) -> ShapesPackedMatrix\n\n" //
    "Pack 3D point clouds like shapes_pack(), each also divided by its RMS distance\n"      //
    "from the centroid, which the `scales` property keeps, so every kind compares\n"        //
    "shapes regardless of size. A cloud with no spread keeps a scale of 1. Between\n"       //
    "normalized packs, each rotation takes column cloud b onto row cloud a as:\n\n"         //
    "    aᵢ ≈ (scale_a / scale_b) × R × (bⱼ − b̄) + ā\n\n"                                   //
    "Args:\n"                                                                               //
    "    points (array_like or list): One cloud of shape [points,3], or a list or\n"        //
    "        tuple of clouds of any sizes.\n"                                               //
    "    dtype (string, optional): Packing dtype, 'float16' or 'float32'. Default:\n"       //
    "        float16 for a float16 first cloud, float32 otherwise.\n"                       //
    "    threads (int, optional): Worker threads, defaulting to 1.\n\n"                     //
    "Returns:\n"                                                                            //
    "    ShapesPackedMatrix: The packed clouds, for the same functions as\n"                //
    "        shapes_pack().\n\n"                                                            //
    "Signature:\n"                                                                          //
    "    >>> def shapes_pack_normalized(points, /, dtype=None, *, threads=1) -> ShapesPackedMatrix: ...";

PyObject *api_shapes_pack_normalized(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_pack_common(args, nargs, kwnames, nk_kernel_shapes_pack_normalized_k);
}

/** Acquires @p object as a writable C-contiguous f32 buffer of @p shape, or sets a Python error. */
static int shapes_rotations_buffer(PyObject *object, Py_ssize_t const *shape, Py_buffer *buffer,
                                   nk_buffer_backing_t *backing) {
    if (!nk_get_buffer(object, buffer, PyBUF_WRITABLE | PyBUF_STRIDES | PyBUF_FORMAT, backing)) {
        buffer->obj = NULL;
        return 0;
    }
    int valid = resolve_nk_dtype_in_py_buffer(buffer) == nk_f32_k && buffer->ndim == 4 &&
                strides_are_c_contiguous(nk_f32_k, 4, buffer->shape, buffer->strides);
    for (int axis = 0; valid && axis != 4; ++axis) valid = buffer->shape[axis] == shape[axis];
    if (valid) return 1;
    PyBuffer_Release(buffer);
    PyErr_Format(PyExc_ValueError, "rotations must be a writable C-contiguous float32 buffer of shape [%zd,%zd,3,3]",
                 shape[0], shape[1]);
    return 0;
}

/** A many-to-many shape call cut into windows of row clouds, named as in `nk_*_packed_*`: window
 *  @c w scores rows from w × rows ÷ windows up to (w + 1) × rows ÷ windows, in its own
 *  @c workspace_stride bytes of @c workspace when @c workspace_kernel is set. */
typedef struct shapes_packed_task_t {
    nk_shapes_packed_punned_t kernel;
    nk_shapes_packed_workspace_punned_t workspace_kernel;
    void const *const *a_packed;
    nk_size_t rows;
    void const *const *b_packed;
    nk_size_t columns;
    char *results;
    nk_size_t results_stride;
    nk_f32_t *rotations;
    char *workspace;
    nk_size_t workspace_bytes;
    nk_size_t workspace_stride;
    nk_size_t windows;
    nk_stream_t stream;
} shapes_packed_task_t;

static nk_status_t shapes_packed_tile_(nk_size_t tile_index, void *context) {
    shapes_packed_task_t const *task = (shapes_packed_task_t const *)context;
    nk_size_t const rows_begin = tile_index * task->rows / task->windows;
    nk_size_t const rows = (tile_index + 1) * task->rows / task->windows - rows_begin;
    void const *const *a_packed = task->a_packed + rows_begin;
    char *results = task->results + rows_begin * task->results_stride;
    nk_f32_t *rotations = task->rotations ? task->rotations + rows_begin * task->columns * 9 : NULL;
    if (!task->workspace_kernel)
        return task->kernel(a_packed, rows, task->b_packed, task->columns, results, task->results_stride, rotations,
                            task->stream);
    return task->workspace_kernel(a_packed, rows, task->b_packed, task->columns, results, task->results_stride,
                                  rotations, task->workspace + tile_index * task->workspace_stride,
                                  task->workspace_bytes, task->stream);
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
    if (!PyObject_TypeCheck(args[0], &ShapesPackedMatrixType) ||
        !PyObject_TypeCheck(args[1], &ShapesPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "a and b must be ShapesPackedMatrix (use shapes_pack() first)");
        return NULL;
    }
    ShapesPackedMatrix *a = (ShapesPackedMatrix *)args[0];
    ShapesPackedMatrix *b = (ShapesPackedMatrix *)args[1];

    PyObject *rotations_object = NULL;
    nk_size_t threads = 1;
    nk_capability_t capabilities = nk_cap_cpus_k;
    nk_stream_t stream = NULL;
    Py_ssize_t const nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "rotations") == 0) {
            if (args[nargs + i] != Py_None) rotations_object = args[nargs + i];
        }
        else if (PyUnicode_CompareWithASCIIString(key, "threads") == 0) {
            Py_ssize_t t = PyLong_AsSsize_t(args[nargs + i]);
            if (t == -1 && PyErr_Occurred()) return NULL;
            threads = (nk_size_t)(t >= 0 ? t : 0);
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

    Py_ssize_t const shape[4] = {(Py_ssize_t)a->clouds, (Py_ssize_t)b->clouds, 3, 3};
    Py_buffer rotations;
    nk_buffer_backing_t rotations_backing;
    rotations.obj = NULL;
    if (rotations_object && !shapes_rotations_buffer(rotations_object, shape, &rotations, &rotations_backing))
        return NULL;
    Tensor *distances = Tensor_new(out_dtype, 2, shape);
    if (!distances) {
        PyBuffer_Release(&rotations);
        return NULL;
    }
    // Each window solves in its own workspace, so there are no more windows than workers
    if (threads == 0) threads = nk_parallel_concurrency();
    nk_size_t const windows = threads < a->clouds ? threads : a->clouds;
    nk_size_t const workspace_stride = nk_size_round_up_to_multiple_(workspace_bytes, nk_default_alignment_k);
    char *workspace = NULL, *aligned_workspace = NULL;
    if (workspace_bytes) {
        workspace = PyMem_Malloc(windows * workspace_stride + nk_default_alignment_k - 1);
        if (!workspace) {
            Py_DECREF(distances);
            PyBuffer_Release(&rotations);
            return PyErr_NoMemory();
        }
        aligned_workspace = workspace + (nk_default_alignment_k - (uintptr_t)workspace % nk_default_alignment_k) %
                                            nk_default_alignment_k;
    }

    nk_status_t status = nk_success_k;
    // An empty Tensor holds no data pointer for the kernel to write through
    if (a->clouds && b->clouds) {
        int const transport = workspace_kind != nk_kernel_unknown_k;
        shapes_packed_task_t const task = {
            .kernel = transport ? NULL : (nk_shapes_packed_punned_t)kernel,
            .workspace_kernel = transport ? (nk_shapes_packed_workspace_punned_t)kernel : NULL,
            .a_packed = a->packs,
            .rows = a->clouds,
            .b_packed = b->packs,
            .columns = b->clouds,
            .results = distances->data,
            .results_stride = b->clouds * nk_dtype_bytes_per_value(out_dtype),
            .rotations = rotations_object ? (nk_f32_t *)rotations.buf : NULL,
            .workspace = aligned_workspace,
            .workspace_bytes = workspace_bytes,
            .workspace_stride = workspace_stride,
            .windows = windows,
            .stream = stream};
        PyThreadState *save = PyEval_SaveThread();
        status = nk_parallel_for_tiles(windows, threads, shapes_packed_tile_, (void *)&task);
        PyEval_RestoreThread(save);
    }
    PyMem_Free(workspace);
    PyBuffer_Release(&rotations);
    if (!check_status(status)) {
        Py_DECREF(distances);
        return NULL;
    }
    return (PyObject *)distances;
}

char const doc_chamfers_packed[] =                                                       //
    "chamfers_packed(a, b, /, *, rotations=None, threads=1) -> Tensor\n\n"               //
    "Chamfer distances between every cloud of a and every cloud of b.\n\n"               //
    "Each is ½ × (meanᵢ minⱼ ‖aᵢ − S·bⱼ‖ + meanⱼ minᵢ ‖aᵢ − S·bⱼ‖), the best of the 8\n" //
    "sign flips S of the principal axes, zero between two empty clouds and infinite\n"   //
    "between an empty and a non-empty one.\n\n"                                          //
    "Args:\n"                                                                            //
    "    a (ShapesPackedMatrix): Row clouds.\n"                                          //
    "    b (ShapesPackedMatrix): Column clouds, packed in the same dtype.\n"             //
    "    rotations (array_like, optional): Writable C-contiguous float32 buffer of\n"    //
    "        shape [rows,columns,3,3] that receives the rotations taking each column\n"  //
    "        cloud onto each row cloud, aᵢ ≈ R × (bⱼ − b̄) + ā.\n"                        //
    "    threads (int, optional): Worker threads, defaulting to 1.\n\n"                  //
    "Returns:\n"                                                                         //
    "    Tensor: The [rows,columns] distances, float32 for float16 packs and float64\n"  //
    "        for float32 ones.\n\n"                                                      //
    "Signature:\n"                                                                       //
    "    >>> def chamfers_packed(a, b, /, *, rotations=None, threads=1) -> Tensor: ...";

PyObject *api_chamfers_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_packed_common(args, nargs, kwnames, nk_kernel_chamfers_packed_k, nk_kernel_unknown_k);
}

char const doc_hausdorffs_packed[] =                                                    //
    "hausdorffs_packed(a, b, /, *, rotations=None, threads=1) -> Tensor\n\n"            //
    "Hausdorff distances between every cloud of a and every cloud of b.\n\n"            //
    "Each is max(maxᵢ minⱼ ‖aᵢ − S·bⱼ‖, maxⱼ minᵢ ‖aᵢ − S·bⱼ‖), the best of the 8\n"    //
    "sign flips S of the principal axes, with the empty-cloud convention of\n"          //
    "chamfers_packed().\n\n"                                                            //
    "Args:\n"                                                                           //
    "    a (ShapesPackedMatrix): Row clouds.\n"                                         //
    "    b (ShapesPackedMatrix): Column clouds, packed in the same dtype.\n"            //
    "    rotations (array_like, optional): As in chamfers_packed().\n"                  //
    "    threads (int, optional): As in chamfers_packed().\n\n"                         //
    "Returns:\n"                                                                        //
    "    Tensor: The [rows,columns] distances, float32 for float16 packs and float64\n" //
    "        for float32 ones.\n\n"                                                     //
    "Signature:\n"                                                                      //
    "    >>> def hausdorffs_packed(a, b, /, *, rotations=None, threads=1) -> Tensor: ...";

PyObject *api_hausdorffs_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_packed_common(args, nargs, kwnames, nk_kernel_hausdorffs_packed_k, nk_kernel_unknown_k);
}

char const doc_wassersteinprocrustes_packed[] =                                           //
    "wassersteinprocrustes_packed(a, b, /, *, rotations=None, threads=1) -> Tensor\n\n"   //
    "Wasserstein-Procrustes distances between every cloud of a and every cloud of b.\n\n" //
    "Each is the root mean squared distance of the exact optimal transport plan\n"        //
    "between uniform weights, after the rotation or reflection minimizing it,\n"          //
    "alternated from each of the 8 sign flips of the principal axes. On equal sizes\n"    //
    "it is at most the Kabsch RMSD of a known correspondence, and equals it when\n"       //
    "that correspondence is an optimal plan. The empty-cloud convention matches\n"        //
    "chamfers_packed(), and every thread's workspace is allocated inside the call.\n\n"   //
    "Args:\n"                                                                             //
    "    a (ShapesPackedMatrix): Row clouds.\n"                                           //
    "    b (ShapesPackedMatrix): Column clouds, packed in the same dtype.\n"              //
    "    rotations (array_like, optional): As in chamfers_packed().\n"                    //
    "    threads (int, optional): As in chamfers_packed().\n\n"                           //
    "Returns:\n"                                                                          //
    "    Tensor: The [rows,columns] distances, float32 for float16 packs and float64\n"   //
    "        for float32 ones.\n\n"                                                       //
    "Signature:\n"                                                                        //
    "    >>> def wassersteinprocrustes_packed(a, b, /, *, rotations=None, threads=1) -> Tensor: ...";

PyObject *api_wassersteinprocrustes_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_packed_common(args, nargs, kwnames, nk_kernel_wassersteinprocrustes_packed_k,
                                nk_kernel_wassersteinprocrustes_workspace_size_k);
}

char const doc_gromovwassersteins_packed[] =                                             //
    "gromovwassersteins_packed(a, b, /, *, rotations=None, threads=1) -> Tensor\n\n"     //
    "Gromov-Wasserstein distances between every cloud of a and every cloud of b.\n\n"    //
    "Each compares the two clouds' own distance matrices under the best transport\n"     //
    "plan a fixed budget of Frank-Wolfe steps finds from the Wasserstein-Procrustes\n"   //
    "plans and the product plan, ½ · √Σᵢⱼₖₗ (‖aᵢ − aₖ‖ − ‖bⱼ − bₗ‖)² · Pᵢⱼ · Pₖₗ. The\n" //
    "rotation is the orthogonal Procrustes fit of that plan. The empty-cloud\n"          //
    "convention matches chamfers_packed(), and every thread's workspace is allocated\n"  //
    "inside the call.\n\n"                                                               //
    "Args:\n"                                                                            //
    "    a (ShapesPackedMatrix): Row clouds.\n"                                          //
    "    b (ShapesPackedMatrix): Column clouds, packed in the same dtype.\n"             //
    "    rotations (array_like, optional): As in chamfers_packed().\n"                   //
    "    threads (int, optional): As in chamfers_packed().\n\n"                          //
    "Returns:\n"                                                                         //
    "    Tensor: The [rows,columns] distances, float32 for float16 packs and float64\n"  //
    "        for float32 ones.\n\n"                                                      //
    "Signature:\n"                                                                       //
    "    >>> def gromovwassersteins_packed(a, b, /, *, rotations=None, threads=1) -> Tensor: ...";

PyObject *api_gromovwassersteins_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return shapes_packed_common(args, nargs, kwnames, nk_kernel_gromovwassersteins_packed_k,
                                nk_kernel_gromovwassersteins_workspace_size_k);
}
