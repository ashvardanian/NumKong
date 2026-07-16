/**
 *  @brief MaxSim late-interaction operations for NumKong Python bindings.
 *  @file python/maxsim.c
 *  @author Ash Vardanian
 *  @date March 9, 2026
 *
 *  This module owns:
 *  - `MaxSimPackedMatrix`: opaque pre-packed matrix for MaxSim scoring.
 *  - Packing API: `maxsim_pack()`.
 *  - Packed scoring API: `maxsim_packed()`.
 *  - Convenience API: `maxsim()` (pack + compute).
 */
#include "maxsim.h"
#include "tensor.h"

#include <numkong/maxsim.h>

static void MaxSimPackedMatrix_dealloc(PyObject *self) { Py_TYPE(self)->tp_free(self); }

static size_t maxsim_packed_matrix_nbytes(MaxSimPackedMatrix *mm) {
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, mm->dtype, (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn || !cap) return 0;
    return size_fn(mm->vector_count, mm->depth);
}

static PyObject *MaxSimPackedMatrix_repr(PyObject *self) {
    MaxSimPackedMatrix *mm = (MaxSimPackedMatrix *)self;
    size_t packed_size = maxsim_packed_matrix_nbytes(mm);
    return PyUnicode_FromFormat("<MaxSimPackedMatrix vector_count=%zu depth=%zu dtype='%s' nbytes=%zu>",
                                (size_t)mm->vector_count, (size_t)mm->depth, nk_dtype_name(mm->dtype), packed_size);
}

static PyObject *MaxSimPackedMatrix_get_vector_count(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((MaxSimPackedMatrix *)self)->vector_count);
}

static PyObject *MaxSimPackedMatrix_get_depth(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(((MaxSimPackedMatrix *)self)->depth);
}

static PyObject *MaxSimPackedMatrix_get_dtype(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyUnicode_FromString(nk_dtype_name(((MaxSimPackedMatrix *)self)->dtype));
}

static PyObject *MaxSimPackedMatrix_get_nbytes(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromSize_t(maxsim_packed_matrix_nbytes((MaxSimPackedMatrix *)self));
}

static PyGetSetDef MaxSimPackedMatrix_getset[] = {
    {"vector_count", MaxSimPackedMatrix_get_vector_count, NULL, "Number of vectors", NULL},
    {"depth", MaxSimPackedMatrix_get_depth, NULL, "Number of dimensions per vector (depth)", NULL},
    {"dtype", MaxSimPackedMatrix_get_dtype, NULL, "Data type of the packed vectors", NULL},
    {"nbytes", MaxSimPackedMatrix_get_nbytes, NULL, "Size of the packed buffer in bytes", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *MaxSimPackedMatrix_packed_size(PyObject *cls, PyObject *const *args, Py_ssize_t nargs,
                                                PyObject *kwnames) {
    nk_unused_(cls);

    PyObject *vector_count_obj = NULL, *depth_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 2 || total > 3 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "packed_size(vector_count, depth, /, dtype='bf16')");
        return NULL;
    }

    vector_count_obj = args[0];
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

    nk_size_t vector_count = (nk_size_t)PyLong_AsSize_t(vector_count_obj);
    if (vector_count == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
    nk_size_t depth = (nk_size_t)PyLong_AsSize_t(depth_obj);
    if (depth == (nk_size_t)-1 && PyErr_Occurred()) return NULL;

    nk_dtype_t dtype = py_object_to_nk_dtype(dtype_obj);
    if (dtype == nk_dtype_unknown_k) return NULL;

    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn || !cap) {
        PyErr_Format(PyExc_LookupError, "No maxsim packed_size kernel for dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(dtype));
        return NULL;
    }

    return PyLong_FromSize_t(size_fn(vector_count, depth));
}

static PyMethodDef MaxSimPackedMatrix_methods[] = {
    {"packed_size", (PyCFunction)MaxSimPackedMatrix_packed_size, METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     "Return packed buffer size in bytes for given dimensions and dtype."},
    {NULL, NULL, 0, NULL},
};

PyTypeObject MaxSimPackedMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.MaxSimPackedMatrix",
    .tp_doc = "Opaque pre-packed matrix for MaxSim late-interaction scoring",
    .tp_basicsize = sizeof(MaxSimPackedMatrix),
    .tp_itemsize = sizeof(char),
    .tp_dealloc = MaxSimPackedMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = MaxSimPackedMatrix_getset,
    .tp_methods = MaxSimPackedMatrix_methods,
    .tp_repr = MaxSimPackedMatrix_repr,
};

char const doc_maxsim_pack[] =                                                  //
    "maxsim_pack(b, /, dtype=None) -> MaxSimPackedMatrix\n\n"                   //
    "Pack a 2D matrix for MaxSim late-interaction scoring.\n\n"                 //
    "Parameters:\n"                                                             //
    "    b (array_like): Source matrix with shape (vector_count, depth).\n"     //
    "    dtype (str, optional): Packing dtype. Default: inferred from input.\n" //
    "        Supported values: 'bf16', 'f16', 'f32'.\n\n"                       //
    "Returns:\n"                                                                //
    "    MaxSimPackedMatrix: Opaque packed matrix for maxsim_packed().\n\n"     //
    "Signature:\n"                                                              //
    "    >>> def maxsim_pack(b, /, dtype=None) -> MaxSimPackedMatrix: ...";

PyObject *api_maxsim_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    PyObject *b_obj = NULL;
    PyObject *dtype_obj = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 1 || total > 2) {
        PyErr_SetString(PyExc_TypeError, "maxsim_pack() requires 1-2 arguments: b, dtype=None");
        return NULL;
    }

    b_obj = args[0];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (nargs >= 2) {
                PyErr_SetString(PyExc_TypeError, "maxsim_pack() got multiple values for argument 'dtype'");
                return NULL;
            }
            dtype_obj = args[nargs + i];
        }
        else {
            char const *name_str = PyUnicode_AsUTF8(name);
            PyErr_Format(PyExc_TypeError, "maxsim_pack() got unexpected keyword argument '%s'", name_str);
            return NULL;
        }
    }
    if (nargs >= 2) dtype_obj = args[1];

    nk_dtype_t target_dtype = dtype_obj ? py_object_to_nk_dtype(dtype_obj) : nk_dtype_unknown_k;
    if (dtype_obj && target_dtype == nk_dtype_unknown_k) return NULL;

    if (target_dtype != nk_dtype_unknown_k && nk_maxsim_output_dtype(target_dtype) == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported maxsim dtype '%s'", nk_dtype_to_pybuffer_typestr(target_dtype));
        return NULL;
    }

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
    if (target_dtype == nk_dtype_unknown_k) {
        target_dtype = src_dtype;
        if (nk_maxsim_output_dtype(target_dtype) == nk_dtype_unknown_k) {
            PyBuffer_Release(&b_buffer);
            PyErr_Format(PyExc_ValueError, "Unsupported maxsim dtype '%s' inferred from input",
                         nk_dtype_to_pybuffer_typestr(target_dtype));
            return NULL;
        }
    }

    if (b_buffer.strides[0] < 0 || b_buffer.strides[1] < 0) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "maxsim packing does not support negative strides");
        return NULL;
    }

    nk_size_t vector_count = (nk_size_t)b_buffer.shape[0];
    nk_size_t depth = (nk_size_t)b_buffer.shape[1];
    nk_size_t row_stride = (nk_size_t)b_buffer.strides[0];
    nk_size_t col_stride = (nk_size_t)b_buffer.strides[1];

    if (src_dtype != target_dtype) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_TypeError, "Input dtype '%s' does not match target dtype '%s'. Cast the input first.",
                     nk_dtype_to_pybuffer_typestr(src_dtype), nk_dtype_to_pybuffer_typestr(target_dtype));
        return NULL;
    }
    if (col_stride != (nk_size_t)nk_dtype_bytes_per_value(target_dtype)) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "Input matrix must be row-contiguous");
        return NULL;
    }

    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, target_dtype, (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn || !cap) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No maxsim packed_size kernel for dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(target_dtype));
        return NULL;
    }
    nk_size_t packed_size = size_fn(vector_count, depth);

    MaxSimPackedMatrix *packed = PyObject_NewVar(MaxSimPackedMatrix, &MaxSimPackedMatrixType, packed_size);
    if (!packed) {
        PyBuffer_Release(&b_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    packed->dtype = target_dtype;
    packed->vector_count = vector_count;
    packed->depth = depth;

    nk_dots_pack_punned_t pack_fn = NULL;
    cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, target_dtype, (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!pack_fn || !cap) {
        Py_DECREF(packed);
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No maxsim pack kernel for dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(target_dtype));
        return NULL;
    }

    {
        PyThreadState *save = PyEval_SaveThread();
        pack_fn(b_buffer.buf, vector_count, depth, row_stride, packed->start);
        PyEval_RestoreThread(save);
    }

    PyBuffer_Release(&b_buffer);
    return (PyObject *)packed;
}

char const doc_maxsim_packed[] =                                             //
    "maxsim_packed(queries, documents, /) -> float\n\n"                      //
    "Compute MaxSim late-interaction score between two packed matrices.\n\n" //
    "Parameters:\n"                                                          //
    "    queries (MaxSimPackedMatrix): Packed query vectors.\n"              //
    "    documents (MaxSimPackedMatrix): Packed document vectors.\n\n"       //
    "Returns:\n"                                                             //
    "    float: Sum of per-query minimum angular distances.\n\n"             //
    "Signature:\n"                                                           //
    "    >>> def maxsim_packed(queries, documents, /) -> float: ...";

static PyObject *maxsim_result_to_py_number(                  //
    nk_maxsim_packed_punned_t kernel, nk_dtype_t input_dtype, //
    void const *queries, void const *documents,               //
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth) {

    nk_dtype_t out_dtype = nk_kernel_output_dtype(nk_kernel_maxsim_packed_k, input_dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Cannot determine output dtype for maxsim_packed('%s')",
                     nk_dtype_to_pybuffer_typestr(input_dtype));
        return NULL;
    }

    nk_scalar_buffer_t result = {0};
    PyThreadState *save = PyEval_SaveThread();
    switch (out_dtype) {
    case nk_f64_k: kernel(queries, documents, query_count, document_count, depth, &result.f64); break;
    case nk_f32_k: kernel(queries, documents, query_count, document_count, depth, &result.f32); break;
    default:
        PyEval_RestoreThread(save);
        PyErr_Format(PyExc_ValueError, "Unsupported maxsim_packed output dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(out_dtype));
        return NULL;
    }
    PyEval_RestoreThread(save);

    return nk_scalar_buffer_to_py_number(&result, out_dtype);
}

PyObject *api_maxsim_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    if (nargs != 2 || (kwnames && PyTuple_Size(kwnames) > 0)) {
        PyErr_SetString(PyExc_TypeError, "maxsim_packed() requires exactly 2 positional arguments: queries, documents");
        return NULL;
    }

    if (!PyObject_TypeCheck(args[0], &MaxSimPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "queries must be a MaxSimPackedMatrix (use maxsim_pack() first)");
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], &MaxSimPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "documents must be a MaxSimPackedMatrix (use maxsim_pack() first)");
        return NULL;
    }

    MaxSimPackedMatrix *queries = (MaxSimPackedMatrix *)args[0];
    MaxSimPackedMatrix *documents = (MaxSimPackedMatrix *)args[1];

    if (queries->dtype != documents->dtype) {
        PyErr_Format(PyExc_TypeError, "dtype mismatch: queries is '%s' but documents is '%s'",
                     nk_dtype_to_pybuffer_typestr(queries->dtype), nk_dtype_to_pybuffer_typestr(documents->dtype));
        return NULL;
    }
    if (queries->depth != documents->depth) {
        PyErr_Format(PyExc_ValueError, "depth mismatch: queries have depth=%zu but documents have depth=%zu",
                     (size_t)queries->depth, (size_t)documents->depth);
        return NULL;
    }

    nk_maxsim_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, queries->dtype, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_LookupError, "No maxsim_packed kernel for dtype '%s'",
                     nk_dtype_to_pybuffer_typestr(queries->dtype));
        return NULL;
    }

    return maxsim_result_to_py_number(kernel, queries->dtype, queries->start, documents->start, queries->vector_count,
                                      documents->vector_count, queries->depth);
}

char const doc_maxsim[] =                                                               //
    "maxsim(queries, documents, /, dtype=None) -> float\n\n"                            //
    "Convenience MaxSim: pack both matrices and compute in one call.\n\n"               //
    "Parameters:\n"                                                                     //
    "    queries (array_like): Query matrix with shape (query_count, depth).\n"         //
    "    documents (array_like): Document matrix with shape (document_count, depth).\n" //
    "    dtype (str, optional): Packing dtype. Default: inferred from input.\n"         //
    "        Supported values: 'bf16', 'f16', 'f32'.\n\n"                               //
    "Returns:\n"                                                                        //
    "    float: Sum of per-query minimum angular distances.\n\n"                        //
    "Signature:\n"                                                                      //
    "    >>> def maxsim(queries, documents, /, dtype=None) -> float: ...";

PyObject *api_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    PyObject *queries_obj = NULL, *documents_obj = NULL;
    PyObject *dtype_obj = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 2 || total > 3) {
        PyErr_SetString(PyExc_TypeError, "maxsim() requires 2-3 arguments: queries, documents, dtype='bf16'");
        return NULL;
    }

    queries_obj = args[0];
    documents_obj = args[1];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (nargs >= 3) {
                PyErr_SetString(PyExc_TypeError, "maxsim() got multiple values for argument 'dtype'");
                return NULL;
            }
            dtype_obj = args[nargs + i];
        }
        else {
            char const *name_str = PyUnicode_AsUTF8(name);
            PyErr_Format(PyExc_TypeError, "maxsim() got unexpected keyword argument '%s'", name_str);
            return NULL;
        }
    }
    if (nargs >= 3) dtype_obj = args[2];

    nk_dtype_t target_dtype = dtype_obj ? py_object_to_nk_dtype(dtype_obj) : nk_dtype_unknown_k;
    if (dtype_obj && target_dtype == nk_dtype_unknown_k) return NULL;
    if (target_dtype != nk_dtype_unknown_k && nk_maxsim_output_dtype(target_dtype) == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported maxsim dtype '%s'", nk_dtype_to_pybuffer_typestr(target_dtype));
        return NULL;
    }

    Py_buffer queries_buffer, documents_buffer;
    nk_buffer_backing_t queries_backing, documents_backing;
    int have_queries = 0, have_documents = 0;
    PyObject *return_obj = NULL;
    MaxSimPackedMatrix *q_packed = NULL, *d_packed = NULL;

    if (!nk_get_buffer(queries_obj, &queries_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &queries_backing)) {
        PyErr_SetString(PyExc_TypeError, "queries must support buffer protocol or __array_interface__");
        return NULL;
    }
    have_queries = 1;

    if (!nk_get_buffer(documents_obj, &documents_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &documents_backing)) {
        PyErr_SetString(PyExc_TypeError, "documents must support buffer protocol or __array_interface__");
        goto cleanup;
    }
    have_documents = 1;

    if (queries_buffer.ndim != 2 || documents_buffer.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "queries and documents must be 2D matrices");
        goto cleanup;
    }

    {
        nk_dtype_t queries_dtype = resolve_nk_dtype_in_py_buffer(&queries_buffer);
        if (queries_dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s'", queries_buffer.format);
            goto cleanup;
        }
        nk_dtype_t documents_dtype = resolve_nk_dtype_in_py_buffer(&documents_buffer);
        if (documents_dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s'", documents_buffer.format);
            goto cleanup;
        }

        // Auto-infer target dtype from queries when no explicit dtype was provided
        if (target_dtype == nk_dtype_unknown_k) {
            target_dtype = queries_dtype;
            if (nk_maxsim_output_dtype(target_dtype) == nk_dtype_unknown_k) {
                PyErr_Format(PyExc_ValueError, "Unsupported maxsim dtype '%s' inferred from input",
                             nk_dtype_to_pybuffer_typestr(target_dtype));
                goto cleanup;
            }
        }

        if (queries_dtype != target_dtype) {
            PyErr_Format(PyExc_TypeError, "queries dtype '%s' does not match target dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(queries_dtype), nk_dtype_to_pybuffer_typestr(target_dtype));
            goto cleanup;
        }
        if (documents_dtype != target_dtype) {
            PyErr_Format(PyExc_TypeError, "documents dtype '%s' does not match target dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(documents_dtype), nk_dtype_to_pybuffer_typestr(target_dtype));
            goto cleanup;
        }
    }

    if (queries_buffer.strides[1] != (Py_ssize_t)nk_dtype_bytes_per_value(target_dtype) ||
        documents_buffer.strides[1] != (Py_ssize_t)nk_dtype_bytes_per_value(target_dtype)) {
        PyErr_SetString(PyExc_ValueError, "Input matrices must be row-contiguous");
        goto cleanup;
    }

    {
        nk_size_t query_count = (nk_size_t)queries_buffer.shape[0], query_depth = (nk_size_t)queries_buffer.shape[1];
        nk_size_t document_count = (nk_size_t)documents_buffer.shape[0],
                  document_depth = (nk_size_t)documents_buffer.shape[1];
        nk_size_t query_stride = (nk_size_t)queries_buffer.strides[0];
        nk_size_t document_stride = (nk_size_t)documents_buffer.strides[0];

        if (query_depth != document_depth) {
            PyErr_Format(PyExc_ValueError, "Depth mismatch: queries have depth=%zu but documents have depth=%zu",
                         (size_t)query_depth, (size_t)document_depth);
            goto cleanup;
        }

        nk_dots_packed_size_punned_t size_fn = NULL;
        nk_capability_t cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, target_dtype, (nk_kernel_punned_t *)&size_fn, &cap);
        if (!size_fn || !cap) {
            PyErr_Format(PyExc_LookupError, "No maxsim packed_size kernel for dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(target_dtype));
            goto cleanup;
        }

        nk_dots_pack_punned_t pack_fn = NULL;
        cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_maxsim_pack_k, target_dtype, (nk_kernel_punned_t *)&pack_fn, &cap);
        if (!pack_fn || !cap) {
            PyErr_Format(PyExc_LookupError, "No maxsim pack kernel for dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(target_dtype));
            goto cleanup;
        }

        nk_maxsim_packed_punned_t kernel = NULL;
        cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_maxsim_packed_k, target_dtype, (nk_kernel_punned_t *)&kernel, &cap);
        if (!kernel || !cap) {
            PyErr_Format(PyExc_LookupError, "No maxsim_packed kernel for dtype '%s'",
                         nk_dtype_to_pybuffer_typestr(target_dtype));
            goto cleanup;
        }

        nk_size_t q_packed_size = size_fn(query_count, query_depth);
        nk_size_t d_packed_size = size_fn(document_count, document_depth);

        q_packed = PyObject_NewVar(MaxSimPackedMatrix, &MaxSimPackedMatrixType, q_packed_size);
        if (!q_packed) {
            PyErr_NoMemory();
            goto cleanup;
        }
        q_packed->dtype = target_dtype;
        q_packed->vector_count = query_count;
        q_packed->depth = query_depth;

        d_packed = PyObject_NewVar(MaxSimPackedMatrix, &MaxSimPackedMatrixType, d_packed_size);
        if (!d_packed) {
            PyErr_NoMemory();
            goto cleanup;
        }
        d_packed->dtype = target_dtype;
        d_packed->vector_count = document_count;
        d_packed->depth = document_depth;

        {
            PyThreadState *save = PyEval_SaveThread();
            pack_fn(queries_buffer.buf, query_count, query_depth, query_stride, q_packed->start);
            pack_fn(documents_buffer.buf, document_count, document_depth, document_stride, d_packed->start);
            PyEval_RestoreThread(save);
        }

        return_obj = maxsim_result_to_py_number(kernel, target_dtype, q_packed->start, d_packed->start, query_count,
                                                document_count, query_depth);
    }

cleanup:
    if (have_queries) PyBuffer_Release(&queries_buffer);
    if (have_documents) PyBuffer_Release(&documents_buffer);
    Py_XDECREF(q_packed);
    Py_XDECREF(d_packed);
    return return_obj;
}
