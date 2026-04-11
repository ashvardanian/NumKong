/**
 *  @brief Packed-matrix cross operations for NumKong Python bindings.
 *  @file python/matrix.c
 *  @author Ash Vardanian
 *  @date February 20, 2026
 *
 *  This module owns:
 *  - `PackedMatrix`: opaque pre-packed right-hand-side matrix representation.
 *  - Packing APIs: `dots_pack()` and `hammings_pack()`.
 *  - Packed cross APIs: `*_packed()` for dots, hammings, jaccards, angulars, euclideans.
 *  - Symmetric all-pairs APIs: `*_symmetric()` for the same metric families.
 *  - `Tensor @ PackedMatrix`: dot-product shortcut equivalent to `dots_packed`.
 *
 *  Shape naming convention used in docs and errors:
 *  - `a`: (height, depth)
 *  - packed `b`: (width, depth)
 *  - result: (height, width)
 */
#include "matrix.h"
#include "tensor.h"

#include <numkong/dots.h>

#if defined(NK_USE_OPENMP)
#include <omp.h>
#endif

static void PackedMatrix_dealloc(PyObject *self) { Py_TYPE(self)->tp_free(self); }

/** @brief Compute packed buffer size for a PackedMatrix. */
static size_t packed_matrix_nbytes(PackedMatrix *mm) {
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_packed_size_k, mm->dtype, static_capabilities, (nk_kernel_punned_t *)&size_fn,
                          &cap);
    if (!size_fn || !cap) return 0;
    return size_fn(mm->width, mm->depth);
}

static PyObject *PackedMatrix_repr(PyObject *self) {
    PackedMatrix *mm = (PackedMatrix *)self;
    size_t packed_size = packed_matrix_nbytes(mm);
    return PyUnicode_FromFormat("<PackedMatrix width=%zu depth=%zu dtype='%s' nbytes=%zu>", (size_t)mm->width,
                                (size_t)mm->depth, nk_dtype_name(mm->dtype), packed_size);
}

static PyObject *PackedMatrix_get_width(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((PackedMatrix *)self)->width);
}

static PyObject *PackedMatrix_get_depth(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((PackedMatrix *)self)->depth);
}

static PyObject *PackedMatrix_get_dtype(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyUnicode_FromString(nk_dtype_name(((PackedMatrix *)self)->dtype));
}

static PyObject *PackedMatrix_get_nbytes(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(packed_matrix_nbytes((PackedMatrix *)self));
}

static PyGetSetDef PackedMatrix_getset[] = {
    {"width", PackedMatrix_get_width, NULL, "Number of rows in the original matrix", NULL},
    {"depth", PackedMatrix_get_depth, NULL, "Number of columns in the original matrix", NULL},
    {"dtype", PackedMatrix_get_dtype, NULL, "Data type of the matrix elements", NULL},
    {"nbytes", PackedMatrix_get_nbytes, NULL, "Size of the packed buffer in bytes", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *PackedMatrix_packed_size(PyObject *cls, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(cls);

    PyObject *width_obj = NULL, *depth_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 2 || total > 3 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "packed_size(width, depth, /, dtype='bf16')");
        return NULL;
    }

    width_obj = args[0];
    depth_obj = args[1];
    if (nargs >= 3) dtype_obj = args[2];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (dtype_obj) {
                PyErr_SetString(PyExc_TypeError, "packed_size() got multiple values for argument 'dtype'");
                return NULL;
            }
            dtype_obj = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "packed_size() got unexpected keyword argument '%S'", name);
            return NULL;
        }
    }

    if (!dtype_obj) {
        PyErr_SetString(PyExc_TypeError, "packed_size() requires 'dtype' argument");
        return NULL;
    }

    nk_size_t width = (nk_size_t)PyLong_AsSize_t(width_obj);
    if (width == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
    nk_size_t depth = (nk_size_t)PyLong_AsSize_t(depth_obj);
    if (depth == (nk_size_t)-1 && PyErr_Occurred()) return NULL;

    nk_dtype_t dtype = py_object_to_nk_dtype(dtype_obj);
    if (dtype == nk_dtype_unknown_k) return NULL;

    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_packed_size_k, dtype, static_capabilities, (nk_kernel_punned_t *)&size_fn,
                          &cap);
    if (!size_fn || !cap) {
        PyErr_Format(PyExc_LookupError, "No packed_size kernel for dtype '%s'", nk_dtype_to_pybuffer_typestr(dtype));
        return NULL;
    }

    return PyLong_FromSize_t(size_fn(width, depth));
}

static PyMethodDef PackedMatrix_methods[] = {
    {"packed_size", (PyCFunction)PackedMatrix_packed_size, METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     "Return packed buffer size in bytes for a matrix shape and dtype."},
    {NULL, NULL, 0, NULL},
};

PyTypeObject PackedMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.PackedMatrix",
    .tp_doc = "Opaque pre-packed matrix for repeated cross operations",
    .tp_basicsize = sizeof(PackedMatrix),
    .tp_itemsize = sizeof(char),
    .tp_dealloc = PackedMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = PackedMatrix_getset,
    .tp_methods = PackedMatrix_methods,
    .tp_repr = PackedMatrix_repr,
};

/** @brief Matrix multiplication operator for Tensor @ PackedMatrix. */
PyObject *Tensor_matmul(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &TensorType)) { Py_RETURN_NOTIMPLEMENTED; }
    Tensor *a = (Tensor *)self;

    if (!PyObject_TypeCheck(other, &PackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "matmul requires PackedMatrix as right operand " "(use nk.dots_pack() first)");
        return NULL;
    }

    PackedMatrix *packed = (PackedMatrix *)other;

    if (a->rank != 2) {
        PyErr_SetString(PyExc_ValueError, "matmul requires 2D array as left operand");
        return NULL;
    }

    nk_size_t height = (nk_size_t)a->shape[0];
    nk_size_t depth_a = (nk_size_t)a->shape[1];

    if (depth_a != packed->depth) {
        PyErr_Format(PyExc_ValueError, "Dimension mismatch: array has depth=%zu but packed matrix has depth=%zu",
                     depth_a, packed->depth);
        return NULL;
    }

    if (a->strides[0] < 0 || a->strides[1] < 0) {
        PyErr_SetString(PyExc_ValueError, "matmul does not support negative strides");
        return NULL;
    }

    nk_size_t n = packed->width;
    nk_size_t k = packed->depth;
    nk_size_t row_stride = (nk_size_t)a->strides[0];
    nk_size_t col_stride = (nk_size_t)a->strides[1];

    // Require matching dtype and row-contiguous input
    if (a->dtype != packed->dtype) {
        PyErr_Format(PyExc_TypeError,
                     "dtype mismatch: tensor is '%s' but packed matrix is '%s'. " "Use .astype('%s') to convert first.",
                     nk_dtype_to_pybuffer_typestr(a->dtype), nk_dtype_to_pybuffer_typestr(packed->dtype),
                     nk_dtype_to_pybuffer_typestr(packed->dtype));
        return NULL;
    }
    if (col_stride != (nk_size_t)nk_dtype_bytes_per_value(packed->dtype)) {
        PyErr_SetString(PyExc_ValueError, "matmul requires row-contiguous left operand");
        return NULL;
    }

    // Determine output dtype
    nk_dtype_t out_dtype = nk_kernel_output_dtype(nk_kernel_dots_packed_k, packed->dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "Unsupported packed matrix dtype");
        return NULL;
    }

    // Find matmul kernel via punned dispatch
    nk_dots_packed_punned_t matmul_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_packed_k, packed->dtype, static_capabilities, (nk_kernel_punned_t *)&matmul_fn,
                          &cap);
    if (!matmul_fn || !cap) {
        PyErr_SetString(PyExc_LookupError, "No matmul kernel for this dtype");
        return NULL;
    }

    // Allocate output tensor
    Py_ssize_t out_shape[2] = {(Py_ssize_t)height, (Py_ssize_t)n};
    Tensor *result = Tensor_new(out_dtype, 2, out_shape);
    if (!result) return NULL;

    nk_size_t c_stride = n * nk_dtype_bytes_per_value(out_dtype);
    PyThreadState *save = PyEval_SaveThread();
    matmul_fn(a->data, packed->start, result->data, height, n, k, row_stride, c_stride);
    PyEval_RestoreThread(save);

    return (PyObject *)result;
}

typedef struct matrix_metric_spec_t {
    char const *name;
    char const *pack_name;
    nk_kernel_kind_t packed_kind;
    nk_kernel_kind_t symmetric_kind;
    nk_kernel_kind_t metric_kind;
} matrix_metric_spec_t;

static matrix_metric_spec_t const spec_dots = {
    .name = "dots",
    .pack_name = "dots_pack",
    .packed_kind = nk_kernel_dots_packed_k,
    .symmetric_kind = nk_kernel_dots_symmetric_k,
    .metric_kind = nk_kernel_dot_k,
};

static matrix_metric_spec_t const spec_angulars = {
    .name = "angulars",
    .pack_name = "dots_pack",
    .packed_kind = nk_kernel_angulars_packed_k,
    .symmetric_kind = nk_kernel_angulars_symmetric_k,
    .metric_kind = nk_kernel_angular_k,
};

static matrix_metric_spec_t const spec_euclideans = {
    .name = "euclideans",
    .pack_name = "dots_pack",
    .packed_kind = nk_kernel_euclideans_packed_k,
    .symmetric_kind = nk_kernel_euclideans_symmetric_k,
    .metric_kind = nk_kernel_euclidean_k,
};

static matrix_metric_spec_t const spec_hammings = {
    .name = "hammings",
    .pack_name = "hammings_pack",
    .packed_kind = nk_kernel_hammings_packed_k,
    .symmetric_kind = nk_kernel_hammings_symmetric_k,
    .metric_kind = nk_kernel_hamming_k,
};

static matrix_metric_spec_t const spec_jaccards = {
    .name = "jaccards",
    .pack_name = "hammings_pack",
    .packed_kind = nk_kernel_jaccards_packed_k,
    .symmetric_kind = nk_kernel_jaccards_symmetric_k,
    .metric_kind = nk_kernel_jaccard_k,
};

static int resolve_output_tensor(                                            //
    PyObject *out_obj, nk_size_t rows, nk_size_t cols, nk_dtype_t out_dtype, //
    Tensor **result, char **out_data, nk_size_t *row_stride, int *owns_result) {

    if (out_obj && out_obj != Py_None) {
        if (!PyObject_TypeCheck(out_obj, &TensorType)) {
            PyErr_SetString(PyExc_TypeError, "out must be a Tensor");
            return 0;
        }
        *result = (Tensor *)out_obj;

        if ((*result)->rank != 2 || (*result)->shape[0] != (Py_ssize_t)rows ||
            (*result)->shape[1] != (Py_ssize_t)cols) {
            PyErr_Format(PyExc_ValueError, "out has wrong shape: expected (%zu, %zu), got (%zd, %zd)", rows, cols,
                         (*result)->shape[0], (*result)->shape[1]);
            return 0;
        }

        if ((*result)->dtype != out_dtype) {
            PyErr_Format(PyExc_TypeError, "out dtype '%s' does not match expected '%s'",
                         nk_dtype_to_pybuffer_typestr((*result)->dtype), nk_dtype_to_pybuffer_typestr(out_dtype));
            return 0;
        }

        size_t out_item_size = nk_dtype_bytes_per_value(out_dtype);
        if ((*result)->strides[1] != (Py_ssize_t)out_item_size ||
            (*result)->strides[0] != (Py_ssize_t)(cols * out_item_size)) {
            PyErr_SetString(PyExc_ValueError, "out must be C-contiguous");
            return 0;
        }

        *out_data = (*result)->data;
        *row_stride = (nk_size_t)(*result)->strides[0];
        *owns_result = 0;
        return 1;
    }

    Py_ssize_t out_shape[2] = {(Py_ssize_t)rows, (Py_ssize_t)cols};
    *result = Tensor_new(out_dtype, 2, out_shape);
    if (!*result) return 0;

    *out_data = (*result)->data;
    *row_stride = cols * nk_dtype_bytes_per_value(out_dtype);
    *owns_result = 1;
    return 1;
}

static PyObject *api_packed_common( //
    PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, matrix_metric_spec_t const *spec) {

    PyObject *a_obj = NULL;
    PyObject *b_obj = NULL;
    PyObject *out_obj = NULL;
    Py_ssize_t start_row = -1, end_row = -1;
    nk_size_t threads = 1;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "%s_packed() requires exactly 2 positional arguments: a, b", spec->name);
        return NULL;
    }

    a_obj = args[0];
    b_obj = args[1];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "out") == 0) { out_obj = args[nargs + i]; }
        else if (PyUnicode_CompareWithASCIIString(name, "start_row") == 0) {
            start_row = PyLong_AsSsize_t(args[nargs + i]);
            if (start_row == -1 && PyErr_Occurred()) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "end_row") == 0) {
            end_row = PyLong_AsSsize_t(args[nargs + i]);
            if (end_row == -1 && PyErr_Occurred()) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "threads") == 0) {
            Py_ssize_t t = PyLong_AsSsize_t(args[nargs + i]);
            if (t == -1 && PyErr_Occurred()) return NULL;
            threads = (nk_size_t)(t >= 0 ? t : 0);
        }
        else {
            char const *name_str = PyUnicode_AsUTF8(name);
            PyErr_Format(PyExc_TypeError, "%s_packed() got unexpected keyword argument '%s'", spec->name, name_str);
            return NULL;
        }
    }

    if (!PyObject_TypeCheck(b_obj, &PackedMatrixType)) {
        PyErr_Format(PyExc_TypeError, "b must be a PackedMatrix (use %s() first)", spec->pack_name);
        return NULL;
    }
    PackedMatrix *packed = (PackedMatrix *)b_obj;

    Py_buffer a_buffer;
    nk_buffer_backing_t a_backing;
    if (!nk_get_buffer(a_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &a_backing)) {
        PyErr_SetString(PyExc_TypeError, "a must support buffer protocol or __array_interface__");
        return NULL;
    }

    if (a_buffer.ndim != 2) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_ValueError, "a must be a 2D matrix");
        return NULL;
    }

    nk_dtype_t src_dtype = resolve_nk_dtype_in_py_buffer(&a_buffer);
    if (src_dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s'", a_buffer.format);
        PyBuffer_Release(&a_buffer);
        return NULL;
    }
    if (a_buffer.strides[0] < 0 || a_buffer.strides[1] < 0) {
        PyBuffer_Release(&a_buffer);
        PyErr_Format(PyExc_ValueError, "%s_packed does not support negative strides", spec->name);
        return NULL;
    }

    nk_size_t height = (nk_size_t)a_buffer.shape[0];
    nk_size_t depth = (nk_size_t)a_buffer.shape[1];
    nk_size_t input_row_stride = (nk_size_t)a_buffer.strides[0];
    nk_size_t input_col_stride = (nk_size_t)a_buffer.strides[1];
    int is_subbyte = nk_dimensions_per_value(packed->dtype) > 1;
    if (is_subbyte) depth *= nk_dimensions_per_value(packed->dtype);

    if (depth != packed->depth) {
        PyBuffer_Release(&a_buffer);
        PyErr_Format(PyExc_ValueError, "Depth mismatch: a has depth=%zu but packed matrix has depth=%zu", depth,
                     packed->depth);
        return NULL;
    }

    if (src_dtype != packed->dtype && !(is_subbyte && src_dtype == nk_u8_k)) {
        PyBuffer_Release(&a_buffer);
        PyErr_Format(PyExc_TypeError, "dtype mismatch: input is '%s' but packed matrix is '%s'",
                     nk_dtype_to_pybuffer_typestr(src_dtype), nk_dtype_to_pybuffer_typestr(packed->dtype));
        return NULL;
    }

    if (input_col_stride != (nk_size_t)a_buffer.itemsize) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_ValueError, "left operand must be row-contiguous");
        return NULL;
    }

    nk_dtype_t out_dtype = nk_kernel_output_dtype(spec->packed_kind, packed->dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        PyBuffer_Release(&a_buffer);
        PyErr_Format(PyExc_ValueError, "Cannot determine output dtype for %s_packed", spec->name);
        return NULL;
    }

    nk_dots_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(spec->packed_kind, packed->dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyBuffer_Release(&a_buffer);
        PyErr_Format(PyExc_LookupError, "No %s_packed kernel for this dtype", spec->name);
        return NULL;
    }

    nk_size_t width = packed->width;
    nk_size_t depth_packed = packed->depth;

    Tensor *result = NULL;
    int owns_result = 0;
    char *out_data = NULL;
    nk_size_t output_row_stride = 0;
    if (!resolve_output_tensor(out_obj, height, width, out_dtype, &result, &out_data, &output_row_stride,
                               &owns_result)) {
        PyBuffer_Release(&a_buffer);
        return NULL;
    }

    // Apply row-range slicing
    if (start_row < 0) start_row = 0;
    if (end_row < 0) end_row = (Py_ssize_t)height;
    if (start_row > (Py_ssize_t)height || end_row > (Py_ssize_t)height || start_row > end_row) {
        PyBuffer_Release(&a_buffer);
        PyErr_Format(PyExc_ValueError, "Invalid row range [%zd, %zd) for matrix with %zu rows", start_row, end_row,
                     (size_t)height);
        return NULL;
    }
    {
        char *a_ptr = (char *)a_buffer.buf + start_row * (Py_ssize_t)input_row_stride;
        char *out_ptr = out_data + start_row * (Py_ssize_t)output_row_stride;
        nk_size_t slice_height = (nk_size_t)(end_row - start_row);
        PyThreadState *save = PyEval_SaveThread();
#if defined(NK_USE_OPENMP)
        if (threads == 0) threads = (nk_size_t)omp_get_max_threads();
        omp_set_num_threads((int)threads);
#endif
        nk_size_t const tile_count = nk_size_divide_round_up_(slice_height, NK_PARALLEL_PACKED_TILE);
#pragma omp parallel for schedule(dynamic, 1) if (threads > 1)
        for (nk_size_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
            nk_size_t row = tile_idx * NK_PARALLEL_PACKED_TILE;
            nk_size_t chunk = (row + NK_PARALLEL_PACKED_TILE <= slice_height) ? NK_PARALLEL_PACKED_TILE
                                                                              : (slice_height - row);
            kernel(a_ptr + row * input_row_stride, packed->start, out_ptr + row * output_row_stride, chunk, width,
                   depth_packed, input_row_stride, output_row_stride);
        }
        PyEval_RestoreThread(save);
    }
    PyBuffer_Release(&a_buffer);

    if (owns_result) return (PyObject *)result;
    Py_INCREF(result);
    return (PyObject *)result;
}

static PyObject *api_symmetric_common( //
    PyObject *const *args, Py_ssize_t positional_args_count, PyObject *args_names_tuple,
    matrix_metric_spec_t const *spec) {

    PyObject *vectors_obj = NULL;
    PyObject *dtype_obj = NULL;
    PyObject *out_obj = NULL;
    Py_ssize_t start_row = -1, end_row = -1;
    nk_size_t threads = 1;

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 1 || args_count > 6 || positional_args_count > 1) {
        PyErr_Format(PyExc_TypeError,
                     "%s_symmetric(vectors, *, dtype=None, out=None, start_row=None, end_row=None, threads=1)",
                     spec->name);
        return NULL;
    }

    vectors_obj = args[0];
    for (Py_ssize_t i = 0, j = positional_args_count; i < args_names_count; ++i, ++j) {
        PyObject *key = PyTuple_GetItem(args_names_tuple, i);
        PyObject *value = args[j];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = value;
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0) out_obj = value;
        else if (PyUnicode_CompareWithASCIIString(key, "start_row") == 0) {
            start_row = PyLong_AsSsize_t(value);
            if (start_row == -1 && PyErr_Occurred()) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "end_row") == 0) {
            end_row = PyLong_AsSsize_t(value);
            if (end_row == -1 && PyErr_Occurred()) return NULL;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "threads") == 0) {
            Py_ssize_t t = PyLong_AsSsize_t(value);
            if (t == -1 && PyErr_Occurred()) return NULL;
            threads = (nk_size_t)(t >= 0 ? t : 0);
        }
        else {
            PyErr_Format(PyExc_TypeError, "%s_symmetric() unexpected keyword: %S", spec->name, key);
            return NULL;
        }
    }

    Py_buffer vec_buf;
    nk_buffer_backing_t vec_backing;
    if (!nk_get_buffer(vectors_obj, &vec_buf, PyBUF_STRIDES | PyBUF_FORMAT, &vec_backing)) {
        PyErr_SetString(PyExc_TypeError, "vectors must support buffer protocol or __array_interface__");
        return NULL;
    }

    PyObject *return_obj = NULL;
    if (vec_buf.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "vectors must be a 2D matrix");
        goto cleanup;
    }

    if (vec_buf.strides[1] != vec_buf.itemsize) {
        PyErr_SetString(PyExc_ValueError, "Input rows must be contiguous");
        goto cleanup;
    }

    nk_dtype_t dtype = resolve_nk_dtype_in_py_buffer(&vec_buf);
    if (dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s'", vec_buf.format);
        goto cleanup;
    }

    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) goto cleanup;
    }

    nk_dots_symmetric_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(spec->symmetric_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_LookupError, "No %s_symmetric kernel for dtype '%s'", spec->name,
                     nk_dtype_to_pybuffer_typestr(dtype));
        goto cleanup;
    }

    nk_dtype_t out_dtype = nk_kernel_output_dtype(spec->symmetric_kind, dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Cannot determine output dtype for %s_symmetric", spec->name);
        goto cleanup;
    }

    nk_size_t n_vectors = (nk_size_t)vec_buf.shape[0];
    nk_size_t depth = (nk_size_t)vec_buf.shape[1];
    depth *= nk_dimensions_per_value(dtype);
    nk_size_t stride = (nk_size_t)vec_buf.strides[0];

    Tensor *result = NULL;
    int owns_result = 0;
    char *out_data = NULL;
    nk_size_t result_stride = 0;
    if (!resolve_output_tensor(out_obj, n_vectors, n_vectors, out_dtype, &result, &out_data, &result_stride,
                               &owns_result))
        goto cleanup;

    // Apply row-range slicing
    {
        nk_size_t row_start = (start_row >= 0) ? (nk_size_t)start_row : 0;
        nk_size_t row_end = (end_row >= 0) ? (nk_size_t)end_row : n_vectors;
        if (row_start > n_vectors || row_end > n_vectors || row_start > row_end) {
            PyErr_Format(PyExc_ValueError, "Invalid row range [%zu, %zu) for %zu vectors", (size_t)row_start,
                         (size_t)row_end, (size_t)n_vectors);
            goto cleanup;
        }
        nk_size_t row_count_val = (nk_size_t)(row_end - row_start);
        PyThreadState *save = PyEval_SaveThread();
#if defined(NK_USE_OPENMP)
        if (threads == 0) threads = (nk_size_t)omp_get_max_threads();
        omp_set_num_threads((int)threads);
#endif
        nk_size_t const tile_count = nk_size_divide_round_up_(row_count_val, NK_PARALLEL_SYMMETRIC_TILE);
#pragma omp parallel for schedule(dynamic, 1) if (threads > 1)
        for (nk_size_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
            nk_size_t tile_start = row_start + tile_idx * NK_PARALLEL_SYMMETRIC_TILE;
            nk_size_t tile_rows = (tile_start + NK_PARALLEL_SYMMETRIC_TILE <= row_end) ? NK_PARALLEL_SYMMETRIC_TILE
                                                                                       : (row_end - tile_start);
            kernel(vec_buf.buf, n_vectors, depth, stride, out_data, result_stride, tile_start, tile_rows);
        }
        PyEval_RestoreThread(save);
    }

    if (owns_result) return_obj = (PyObject *)result;
    else {
        Py_INCREF(result);
        return_obj = (PyObject *)result;
    }

cleanup:
    PyBuffer_Release(&vec_buf);
    return return_obj;
}

char const doc_dots_pack[] =                                                         //
    "dots_pack(b, /, dtype=None) -> PackedMatrix\n\n"                                //
    "Pack a 2D matrix for repeated dot-product style cross operations.\n\n"          //
    "Parameters:\n"                                                                  //
    "    b (array_like): Source matrix with shape (width, depth).\n"                 //
    "    dtype (str, optional): Packing dtype. Default: inferred from input.\n"      //
    "        Supported values: 'bf16', 'f16', 'f32', 'f64', 'i8', 'u8',\n"           //
    "        'e4m3', 'e5m2', 'e3m2', 'e2m3', 'i4', 'u4', 'u1'.\n\n"                  //
    "Returns:\n"                                                                     //
    "    PackedMatrix: Opaque packed matrix accepted by dots_packed(),\n"            //
    "        angulars_packed(), euclideans_packed(), and Tensor @ PackedMatrix.\n\n" //
    "Example:\n"                                                                     //
    "    >>> b_packed = nk.dots_pack(b, dtype=nk.bfloat16)\n"                        //
    "    >>> distances = nk.dots_packed(a, b_packed)  # shape: (100, 200)\n\n"       //
    "Signature:\n"                                                                   //
    "    >>> def dots_pack(b, /, dtype=None) -> PackedMatrix: ...";

static PyObject *api_pack_common(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, nk_dtype_t default_dtype) {

    PyObject *b_obj = NULL;
    PyObject *dtype_obj = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 1 || total > 2) {
        PyErr_SetString(PyExc_TypeError, "pack requires 1-2 arguments: b, dtype");
        return NULL;
    }

    b_obj = args[0];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (nargs >= 2) {
                PyErr_SetString(PyExc_TypeError, "got multiple values for argument 'dtype'");
                return NULL;
            }
            dtype_obj = args[nargs + i];
        }
        else {
            PyErr_Format(PyExc_TypeError, "unexpected keyword argument '%s'", PyUnicode_AsUTF8(name));
            return NULL;
        }
    }
    if (nargs >= 2) dtype_obj = args[1];

    nk_dtype_t target_dtype = dtype_obj ? py_object_to_nk_dtype(dtype_obj) : default_dtype;
    if (dtype_obj && target_dtype == nk_dtype_unknown_k) return NULL;

    Py_buffer b_buffer;
    nk_buffer_backing_t b_backing;
    if (!nk_get_buffer(b_obj, &b_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &b_backing)) {
        PyErr_SetString(PyExc_TypeError, "b must support buffer protocol or __array_interface__");
        return NULL;
    }

    if (b_buffer.ndim != 2) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "b must be a 2D matrix");
        return NULL;
    }

    nk_dtype_t src_dtype = resolve_nk_dtype_in_py_buffer(&b_buffer);
    if (src_dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s'", b_buffer.format);
        PyBuffer_Release(&b_buffer);
        return NULL;
    }
    // Auto-infer target dtype from input when no explicit dtype was provided
    if (target_dtype == nk_dtype_unknown_k) target_dtype = src_dtype;
    if (b_buffer.strides[0] < 0 || b_buffer.strides[1] < 0) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "packing does not support negative strides");
        return NULL;
    }

    nk_size_t width = (nk_size_t)b_buffer.shape[0];
    nk_size_t depth = (nk_size_t)b_buffer.shape[1];
    // For sub-byte types (e.g. uint1), shape[1] is in bytes but kernels expect logical dimensions
    depth *= nk_dimensions_per_value(target_dtype);
    nk_size_t row_stride = (nk_size_t)b_buffer.strides[0];
    nk_size_t col_stride = (nk_size_t)b_buffer.strides[1];

    // Allow uint8 input when target is a sub-byte type like uint1 (bits stored as uint8 bytes)
    int is_subbyte = nk_dimensions_per_value(target_dtype) > 1;
    if (src_dtype != target_dtype && !(is_subbyte && src_dtype == nk_u8_k)) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_TypeError, "Input dtype '%s' does not match target dtype '%s'.",
                     nk_dtype_to_pybuffer_typestr(src_dtype), nk_dtype_to_pybuffer_typestr(target_dtype));
        return NULL;
    }
    if (col_stride != (nk_size_t)b_buffer.itemsize) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "Input matrix must be row-contiguous");
        return NULL;
    }

    // Get packed size via punned dispatch
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_packed_size_k, target_dtype, static_capabilities,
                          (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn || !cap) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No packing kernel for dtype '%s'", nk_dtype_to_pybuffer_typestr(target_dtype));
        return NULL;
    }
    nk_size_t packed_size = size_fn(width, depth);

    PackedMatrix *packed = PyObject_NewVar(PackedMatrix, &PackedMatrixType, packed_size);
    if (!packed) {
        PyBuffer_Release(&b_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    packed->dtype = target_dtype;
    packed->width = width;
    packed->depth = depth;

    nk_dots_pack_punned_t pack_fn = NULL;
    cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_pack_k, target_dtype, static_capabilities, (nk_kernel_punned_t *)&pack_fn,
                          &cap);
    if (!pack_fn || !cap) {
        Py_DECREF(packed);
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No pack kernel for dtype '%s'", nk_dtype_to_pybuffer_typestr(target_dtype));
        return NULL;
    }

    {
        PyThreadState *save = PyEval_SaveThread();
        pack_fn(b_buffer.buf, width, depth, row_stride, packed->start);
        PyEval_RestoreThread(save);
    }

    PyBuffer_Release(&b_buffer);
    return (PyObject *)packed;
}

PyObject *api_dots_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return api_pack_common(args, nargs, kwnames, nk_dtype_unknown_k);
}

char const doc_dots_packed[] =                                                             //
    "dots_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor\n\n"        //
    "Compute row-wise dot products between matrix a and pre-packed matrix b.\n\n"          //
    "Parameters:\n"                                                                        //
    "    a (array_like): Query matrix with shape (height, depth).\n"                       //
    "    b (PackedMatrix): Matrix packed with dots_pack(); shape (width, depth).\n"        //
    "    out (Tensor, optional): C-contiguous output tensor with shape\n"                  //
    "        (height, width) and matching output dtype.\n"                                 //
    "    start_row (int, optional): First row of a to process (default 0).\n"              //
    "    end_row (int, optional): One-past-last row of a to process (default height).\n\n" //
    "Returns:\n"                                                                           //
    "    Tensor: Dot-product matrix with shape (height, width).\n"                         //
    "    Returns out when provided.\n\n"                                                   //
    "Note:\n"                                                                              //
    "    Equivalent to A @ B.T where B is the original unpacked matrix.\n\n"               //
    "Signature:\n"                                                                         //
    "    >>> def dots_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor: ...";

PyObject *api_dots_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return api_packed_common(args, nargs, kwnames, &spec_dots);
}

char const doc_hammings_pack[] =                                             //
    "hammings_pack(b, /, dtype='uint1') -> PackedMatrix\n\n"                 //
    "Pack a 2D matrix for repeated set-distance cross operations.\n\n"       //
    "Parameters:\n"                                                          //
    "    b (array_like): Source matrix with shape (width, depth).\n"         //
    "    dtype (str, optional): Packing dtype. Default: 'uint1'.\n"          //
    "        For 'uint1', packed bits are represented as uint8 bytes.\n\n"   //
    "Returns:\n"                                                             //
    "    PackedMatrix: Opaque packed matrix accepted by hammings_packed()\n" //
    "        and jaccards_packed().\n\n"                                     //
    "Signature:\n"                                                           //
    "    >>> def hammings_pack(b, /, dtype='uint1') -> PackedMatrix: ...";

PyObject *api_hammings_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return api_pack_common(args, nargs, kwnames, nk_u1_k);
}

char const doc_hammings_packed[] =                                                         //
    "hammings_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor\n\n"    //
    "Compute row-wise Hamming distances between matrix a and pre-packed b.\n\n"            //
    "Parameters:\n"                                                                        //
    "    a (array_like): Query matrix with shape (height, depth).\n"                       //
    "    b (PackedMatrix): Matrix packed with hammings_pack();\n"                          //
    "        shape (width, depth).\n"                                                      //
    "    out (Tensor, optional): C-contiguous output tensor with shape\n"                  //
    "        (height, width) and dtype uint32.\n"                                          //
    "    start_row (int, optional): First row of a to process (default 0).\n"              //
    "    end_row (int, optional): One-past-last row of a to process (default height).\n\n" //
    "Returns:\n"                                                                           //
    "    Tensor: Hamming-distance matrix with shape (height, width).\n"                    //
    "    Returns out when provided.\n\n"                                                   //
    "Note:\n"                                                                              //
    "    Output dtype is uint32 (Hamming distances are unsigned integer counts).\n\n"      //
    "Signature:\n"                                                                         //
    "    >>> def hammings_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor: ...";

PyObject *api_hammings_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return api_packed_common(args, nargs, kwnames, &spec_hammings);
}

char const doc_jaccards_packed[] =                                                         //
    "jaccards_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor\n\n"    //
    "Compute row-wise Jaccard distances between matrix a and pre-packed b.\n\n"            //
    "Parameters:\n"                                                                        //
    "    a (array_like): Query matrix with shape (height, depth).\n"                       //
    "    b (PackedMatrix): Matrix packed with hammings_pack();\n"                          //
    "        shape (width, depth).\n"                                                      //
    "    out (Tensor, optional): C-contiguous output tensor with\n"                        //
    "        shape (height, width) and matching output dtype.\n"                           //
    "    start_row (int, optional): First row of a to process (default 0).\n"              //
    "    end_row (int, optional): One-past-last row of a to process (default height).\n\n" //
    "Returns:\n"                                                                           //
    "    Tensor: Jaccard-distance matrix with shape (height, width).\n"                    //
    "    Returns out when provided.\n\n"                                                   //
    "Note:\n"                                                                              //
    "    Jaccard distances reuse the Hamming packing format.\n"                            //
    "    Use hammings_pack() to prepare the packed matrix.\n\n"                            //
    "Signature:\n"                                                                         //
    "    >>> def jaccards_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor: ...";

PyObject *api_jaccards_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return api_packed_common(args, nargs, kwnames, &spec_jaccards);
}

char const doc_angulars_packed[] =                                                         //
    "angulars_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor\n\n"    //
    "Compute row-wise angular distances between matrix a and pre-packed b.\n\n"            //
    "Parameters:\n"                                                                        //
    "    a (array_like): Query matrix with shape (height, depth).\n"                       //
    "    b (PackedMatrix): Matrix packed with dots_pack();\n"                              //
    "        shape (width, depth).\n"                                                      //
    "    out (Tensor, optional): C-contiguous output tensor with\n"                        //
    "        shape (height, width) and matching output dtype.\n"                           //
    "    start_row (int, optional): First row of a to process (default 0).\n"              //
    "    end_row (int, optional): One-past-last row of a to process (default height).\n\n" //
    "Returns:\n"                                                                           //
    "    Tensor: Angular-distance matrix with shape (height, width).\n"                    //
    "    Returns out when provided.\n\n"                                                   //
    "Note:\n"                                                                              //
    "    Use dots_pack() to prepare the packed matrix — angular\n"                         //
    "    distances reuse the dot-product packing layout.\n\n"                              //
    "Signature:\n"                                                                         //
    "    >>> def angulars_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor: ...";

PyObject *api_angulars_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return api_packed_common(args, nargs, kwnames, &spec_angulars);
}

char const doc_euclideans_packed[] =                                                       //
    "euclideans_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor\n\n"  //
    "Compute row-wise Euclidean distances between matrix a and pre-packed b.\n\n"          //
    "Parameters:\n"                                                                        //
    "    a (array_like): Query matrix with shape (height, depth).\n"                       //
    "    b (PackedMatrix): Matrix packed with dots_pack();\n"                              //
    "        shape (width, depth).\n"                                                      //
    "    out (Tensor, optional): C-contiguous output tensor with\n"                        //
    "        shape (height, width) and matching output dtype.\n"                           //
    "    start_row (int, optional): First row of a to process (default 0).\n"              //
    "    end_row (int, optional): One-past-last row of a to process (default height).\n\n" //
    "Returns:\n"                                                                           //
    "    Tensor: Euclidean-distance matrix with shape (height, width).\n"                  //
    "    Returns out when provided.\n\n"                                                   //
    "Note:\n"                                                                              //
    "    Use dots_pack() to prepare the packed matrix — Euclidean\n"                       //
    "    distances reuse the dot-product packing layout.\n\n"                              //
    "Signature:\n"                                                                         //
    "    >>> def euclideans_packed(a, b, /, *, out=None, start_row=None, end_row=None) -> Tensor: ...";

PyObject *api_euclideans_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    return api_packed_common(args, nargs, kwnames, &spec_euclideans);
}

char const doc_dots_symmetric[] =                                                                     //
    "dots_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor\n\n" //
    "Compute the symmetric all-pairs dot-product (Gram) matrix.\n"                                    //
    "Only the upper triangle of the output is guaranteed to be initialized.\n\n"                      //
    "Parameters:\n"                                                                                   //
    "    vectors (array_like): Input matrix with shape (count, depth).\n"                             //
    "    dtype (str, optional): Optional dtype override for kernel dispatch.\n"                       //
    "    out (Tensor, optional): C-contiguous output tensor with shape\n"                             //
    "        (count, count) and matching output dtype.\n"                                             //
    "    start_row (int, optional): First row to compute (default 0).\n"                              //
    "    end_row (int, optional): One-past-last row to compute (default count).\n"                    //
    "        Only the upper triangle overlapping with the specified row range is filled.\n\n"         //
    "Returns:\n"                                                                                      //
    "    Tensor: Symmetric dot-product matrix with shape (count, count).\n"                           //
    "    Returns out when provided.\n\n"                                                              //
    "Example:\n"                                                                                      //
    "    >>> gram = nk.dots_symmetric(vectors)  # shape: (N, N)\n\n"                                  //
    "Signature:\n"                                                                                    //
    "    >>> def dots_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor: ...";

PyObject *api_dots_symmetric( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {
    nk_unused_(self);
    return api_symmetric_common(args, positional_args_count, args_names_tuple, &spec_dots);
}

char const doc_hammings_symmetric[] =                                                                     //
    "hammings_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor\n\n" //
    "Compute the symmetric all-pairs Hamming-distance matrix.\n"                                          //
    "Only the upper triangle of the output is guaranteed to be initialized.\n\n"                          //
    "Parameters:\n"                                                                                       //
    "    vectors (array_like): Input matrix with shape (count, depth).\n"                                 //
    "        For dtype='uint1', packed bits are represented as uint8 bytes.\n"                            //
    "    dtype (str, optional): Optional dtype override for kernel dispatch.\n"                           //
    "    out (Tensor, optional): C-contiguous output tensor with shape\n"                                 //
    "        (count, count) and dtype uint32.\n"                                                          //
    "    start_row (int, optional): First row to compute (default 0).\n"                                  //
    "    end_row (int, optional): One-past-last row to compute (default count).\n"                        //
    "        Only the upper triangle overlapping with the specified row range is filled.\n\n"             //
    "Returns:\n"                                                                                          //
    "    Tensor: Symmetric Hamming-distance matrix with shape (count, count).\n"                          //
    "    Returns out when provided.\n\n"                                                                  //
    "Example:\n"                                                                                          //
    "    >>> gram = nk.hammings_symmetric(vectors)  # shape: (N, N)\n\n"                                  //
    "Signature:\n"                                                                                        //
    "    >>> def hammings_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor: ...";

PyObject *api_hammings_symmetric( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {
    nk_unused_(self);
    return api_symmetric_common(args, positional_args_count, args_names_tuple, &spec_hammings);
}

char const doc_jaccards_symmetric[] =                                                                     //
    "jaccards_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor\n\n" //
    "Compute the symmetric all-pairs Jaccard-distance matrix.\n"                                          //
    "Only the upper triangle of the output is guaranteed to be initialized.\n\n"                          //
    "Parameters:\n"                                                                                       //
    "    vectors (array_like): Input matrix with shape (count, depth).\n"                                 //
    "        For dtype='uint1', packed bits are represented as uint8 bytes.\n"                            //
    "    dtype (str, optional): Optional dtype override for kernel dispatch.\n"                           //
    "    out (Tensor, optional): C-contiguous output tensor with shape\n"                                 //
    "        (count, count) and matching output dtype.\n"                                                 //
    "    start_row (int, optional): First row to compute (default 0).\n"                                  //
    "    end_row (int, optional): One-past-last row to compute (default count).\n"                        //
    "        Only the upper triangle overlapping with the specified row range is filled.\n\n"             //
    "Returns:\n"                                                                                          //
    "    Tensor: Symmetric Jaccard-distance matrix with shape (count, count).\n"                          //
    "    Returns out when provided.\n\n"                                                                  //
    "Example:\n"                                                                                          //
    "    >>> gram = nk.jaccards_symmetric(vectors)  # shape: (N, N)\n\n"                                  //
    "Signature:\n"                                                                                        //
    "    >>> def jaccards_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor: ...";

PyObject *api_jaccards_symmetric( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {
    nk_unused_(self);
    return api_symmetric_common(args, positional_args_count, args_names_tuple, &spec_jaccards);
}

char const doc_angulars_symmetric[] =                                                                     //
    "angulars_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor\n\n" //
    "Compute the symmetric all-pairs angular-distance matrix.\n"                                          //
    "Only the upper triangle of the output is guaranteed to be initialized.\n\n"                          //
    "Parameters:\n"                                                                                       //
    "    vectors (array_like): Input matrix with shape (count, depth).\n"                                 //
    "    dtype (str, optional): Optional dtype override for kernel dispatch.\n"                           //
    "    out (Tensor, optional): C-contiguous output tensor with shape\n"                                 //
    "        (count, count) and matching output dtype.\n"                                                 //
    "    start_row (int, optional): First row to compute (default 0).\n"                                  //
    "    end_row (int, optional): One-past-last row to compute (default count).\n"                        //
    "        Only the upper triangle overlapping with the specified row range is filled.\n\n"             //
    "Returns:\n"                                                                                          //
    "    Tensor: Symmetric angular-distance matrix with shape (count, count).\n"                          //
    "    Returns out when provided.\n\n"                                                                  //
    "Example:\n"                                                                                          //
    "    >>> gram = nk.angulars_symmetric(vectors)  # shape: (N, N)\n\n"                                  //
    "Signature:\n"                                                                                        //
    "    >>> def angulars_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor: ...";

PyObject *api_angulars_symmetric( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {
    nk_unused_(self);
    return api_symmetric_common(args, positional_args_count, args_names_tuple, &spec_angulars);
}

char const doc_euclideans_symmetric[] =                                                                     //
    "euclideans_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor\n\n" //
    "Compute the symmetric all-pairs Euclidean-distance matrix.\n"                                          //
    "Only the upper triangle of the output is guaranteed to be initialized.\n\n"                            //
    "Parameters:\n"                                                                                         //
    "    vectors (array_like): Input matrix with shape (count, depth).\n"                                   //
    "    dtype (str, optional): Optional dtype override for kernel dispatch.\n"                             //
    "    out (Tensor, optional): C-contiguous output tensor with shape\n"                                   //
    "        (count, count) and matching output dtype.\n"                                                   //
    "    start_row (int, optional): First row to compute (default 0).\n"                                    //
    "    end_row (int, optional): One-past-last row to compute (default count).\n"                          //
    "        Only the upper triangle overlapping with the specified row range is filled.\n\n"               //
    "Returns:\n"                                                                                            //
    "    Tensor: Symmetric Euclidean-distance matrix with shape (count, count).\n"                          //
    "    Returns out when provided.\n\n"                                                                    //
    "Example:\n"                                                                                            //
    "    >>> gram = nk.euclideans_symmetric(vectors)  # shape: (N, N)\n\n"                                  //
    "Signature:\n"                                                                                          //
    "    >>> def euclideans_symmetric(vectors, /, *, dtype=None, out=None, start_row=None, end_row=None) -> Tensor: " "...";

PyObject *api_euclideans_symmetric( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {
    nk_unused_(self);
    return api_symmetric_common(args, positional_args_count, args_names_tuple, &spec_euclideans);
}
