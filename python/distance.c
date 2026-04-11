/**
 *  @brief Distance metric implementations for NumKong Python bindings.
 *  @file python/distance.c
 *  @author Ash Vardanian
 *  @date February 19, 2026
 *
 *  Extracted from numkong.c. Contains all distance-metric API functions,
 *  pointer-access wrappers, cdist, and supporting implement_* helpers.
 */
#include <math.h>

#include "distance.h"
#include "matrix.h" // NK_PARALLEL_PACKED_TILE, NK_PARALLEL_SYMMETRIC_TILE
#include "tensor.h"

#if defined(NK_USE_OPENMP)
#include <omp.h>
#endif

static PyObject *implement_dense_metric( //
    nk_kernel_kind_t metric_kind,        //
    PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;         // Required object, positional-only
    PyObject *b_obj = NULL;         // Required object, positional-only
    PyObject *dtype_obj = NULL;     // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;       // Optional object, "out" keyword-only
    PyObject *out_dtype_obj = NULL; // Optional object, "out_dtype" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    nk_dtype_t dtype = nk_dtype_unknown_k, out_dtype = nk_dtype_unknown_k;
    Py_buffer a_buffer, b_buffer, out_buffer;
    MatrixOrVectorView a_parsed, b_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    // Parse the arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 5) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-5 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "Only first 3 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
    a_obj = args[0];
    b_obj = args[1];

    // Positional or keyword arguments (dtype)
    if (positional_args_count == 3) dtype_obj = args[2];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype`
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    // Convert `out_dtype_obj` to `out_dtype`
    if (out_dtype_obj) {
        out_dtype = py_object_to_nk_dtype(out_dtype_obj);
        if (out_dtype == nk_dtype_unknown_k) return NULL;
    }

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `out_obj`.
    nk_buffer_backing_t a_backing, b_backing, out_backing;
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed, &a_backing, dtype) ||
        !parse_tensor(b_obj, &b_buffer, &b_parsed, &b_backing, dtype))
        goto cleanup;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed, &out_backing, nk_dtype_unknown_k)) goto cleanup;

    // Check dimensions
    if (a_parsed.cols != b_parsed.cols) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }
    if (a_parsed.rows == 0 || b_parsed.rows == 0) {
        PyErr_SetString(PyExc_ValueError, "Collections can't be empty");
        goto cleanup;
    }
    if (a_parsed.rows > 1 && b_parsed.rows > 1 && a_parsed.rows != b_parsed.rows) {
        PyErr_SetString(PyExc_ValueError, "Collections must have the same number of elements or just one element");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.dtype != b_parsed.dtype || //
        a_parsed.dtype == nk_dtype_unknown_k || b_parsed.dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // When a dtype override reinterprets elements at a different width, rescale dimensions.
    size_t a_cols = a_parsed.cols, b_cols = b_parsed.cols;
    {
        nk_size_t from_bits = (nk_size_t)a_buffer.itemsize * NK_BITS_PER_BYTE;
        nk_size_t to_bits = nk_dtype_bits(dtype);
        if (from_bits && to_bits && from_bits != to_bits) {
            a_cols = a_cols * from_bits / to_bits;
            b_cols = b_cols * from_bits / to_bits;
        }
    }

    // Inference order for the output type:
    // 1. `out_dtype` named argument, if defined
    // 2. `out.dtype` attribute, if `out` is passed
    // 3. double precision float (or its complex variant)
    if (out_dtype == nk_dtype_unknown_k) {
        if (out_obj) { out_dtype = out_parsed.dtype; }
        else { out_dtype = (nk_dtype_family(dtype) == nk_dtype_family_complex_float_k) ? nk_f64c_k : nk_f64_k; }
    }

    // Make sure the return dtype is complex if the input dtype is complex, and the same for real numbers
    if (out_dtype != nk_dtype_unknown_k) {
        if ((nk_dtype_family(dtype) == nk_dtype_family_complex_float_k) !=
            (nk_dtype_family(out_dtype) == nk_dtype_family_complex_float_k)) {
            PyErr_SetString(PyExc_ValueError,
                            "If the input dtype is complex, the return dtype must be complex, and same for real.");
            goto cleanup;
        }
    }

    // Check if the downcasting to provided dtype is supported
    {
        nk_scalar_buffer_t probe;
        probe.f64c.real = 0, probe.f64c.imag = 0;
        if (!nk_scalar_buffer_from_f64c(&probe.f64c, &probe, out_dtype)) {
            PyErr_SetString(PyExc_ValueError, "Exporting to the provided dtype is not supported");
            goto cleanup;
        }
    }

    // Look up the metric and the capability
    nk_metric_dense_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&metric, &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and dtype combination across vectors ('%s'/'%s' and '%s'/'%s') and " //
            "`dtype` override ('%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(a_parsed.dtype), //
            b_buffer.format ? b_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(b_parsed.dtype), //
            nk_dtype_to_pybuffer_typestr(dtype), nk_dtype_to_pybuffer_typestr(dtype));
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    nk_dtype_t const kernel_out_dtype = nk_kernel_output_dtype(metric_kind, dtype);
    if (a_parsed.rank == 1 && b_parsed.rank == 1) {
        nk_scalar_buffer_t distance;
        metric(a_parsed.data, b_parsed.data, a_cols, &distance);
        return_obj = nk_scalar_buffer_to_py_number(&distance, kernel_out_dtype);
        goto cleanup;
    }

    // Local working copies for broadcast stride override
    size_t a_row_stride = a_parsed.rows == 1 ? 0 : a_parsed.row_stride;
    size_t b_row_stride = b_parsed.rows == 1 ? 0 : b_parsed.row_stride;

    // We take the maximum of the two counts, because if only one entry is present in one of the arrays,
    // all distances will be computed against that single entry.
    size_t const count_pairs = a_parsed.rows > b_parsed.rows ? a_parsed.rows : b_parsed.rows;
    char *distances_start = NULL;
    size_t distances_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        Py_ssize_t dense_shape[1] = {(Py_ssize_t)count_pairs};
        Tensor *distances_obj = Tensor_new(out_dtype, 1, dense_shape);
        if (!distances_obj) { goto cleanup; }
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
        distances_stride_bytes = distances_obj->strides[0];
    }
    else {
        if (nk_dtype_bytes_per_value(out_parsed.dtype) != nk_dtype_bytes_per_value(out_dtype)) {
            PyErr_Format( //
                PyExc_LookupError,
                "Output tensor scalar type must be compatible with the output type ('%s' and '%s'/'%s')",
                nk_dtype_to_pybuffer_typestr(out_dtype), out_buffer.format ? out_buffer.format : "nil",
                nk_dtype_to_pybuffer_typestr(out_parsed.dtype));
            goto cleanup;
        }
        distances_start = out_parsed.data;
        distances_stride_bytes = out_buffer.strides[0];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    // Now let's release the GIL for the parallel part using the underlying mechanism of `Py_BEGIN_ALLOW_THREADS`.
    PyThreadState *save = PyEval_SaveThread();

    // Compute the distances
    for (size_t i = 0; i < count_pairs; ++i) {
        nk_scalar_buffer_t result;
        metric(                               //
            a_parsed.data + i * a_row_stride, //
            b_parsed.data + i * b_row_stride, //
            a_cols,                           //
            &result);

        // Export out:
        nk_scalar_buffer_export(&result, kernel_out_dtype, distances_start + i * distances_stride_bytes, out_dtype);
    }

    PyEval_RestoreThread(save);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static PyObject *implement_curved_metric( //
    nk_kernel_kind_t metric_kind,         //
    PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *b_obj = NULL;     // Required object, positional-only
    PyObject *c_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional

    // Once parsed, the arguments will be stored in these variables:
    nk_dtype_t dtype = nk_dtype_unknown_k;
    Py_buffer a_buffer, b_buffer, c_buffer;
    MatrixOrVectorView a_parsed, b_parsed, c_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&c_buffer, 0, sizeof(Py_buffer));

    // Parse the arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 3 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-6 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 4) {
        PyErr_Format(PyExc_TypeError, "Only first 4 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first, second, and third matrix)
    a_obj = args[0];
    b_obj = args[1];
    c_obj = args[2];

    // Positional or keyword arguments (dtype)
    if (positional_args_count == 4) dtype_obj = args[3];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype`
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `c_obj`.
    nk_buffer_backing_t a_backing, b_backing, c_backing;
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed, &a_backing, dtype) ||
        !parse_tensor(b_obj, &b_buffer, &b_parsed, &b_backing, dtype) ||
        !parse_tensor(c_obj, &c_buffer, &c_parsed, &c_backing, dtype))
        goto cleanup;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "First and second argument must be vectors");
        goto cleanup;
    }
    if (c_parsed.rank != 2) {
        PyErr_SetString(PyExc_ValueError, "Third argument must be a matrix (rank-2 tensor)");
        goto cleanup;
    }
    if (a_parsed.rows == 0 || b_parsed.rows == 0) {
        PyErr_SetString(PyExc_ValueError, "Collections can't be empty");
        goto cleanup;
    }
    if (a_parsed.rows > 1 && b_parsed.rows > 1 && a_parsed.rows != b_parsed.rows) {
        PyErr_SetString(PyExc_ValueError, "Collections must have the same number of elements or just one element");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.dtype != b_parsed.dtype || a_parsed.dtype != c_parsed.dtype || a_parsed.dtype == nk_dtype_unknown_k ||
        b_parsed.dtype == nk_dtype_unknown_k || c_parsed.dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Look up the metric and the capability
    nk_metric_curved_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&metric, &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and dtype combination across vectors ('%s'/'%s' and '%s'/'%s'), " //
            "tensor ('%s'/'%s'), and `dtype` override ('%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(a_parsed.dtype), //
            b_buffer.format ? b_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(b_parsed.dtype), //
            c_buffer.format ? c_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(c_parsed.dtype), //
            nk_dtype_to_pybuffer_typestr(dtype), nk_dtype_to_pybuffer_typestr(dtype));
        goto cleanup;
    }

    // Return a scalar
    nk_dtype_t const kernel_out_dtype = nk_kernel_output_dtype(metric_kind, dtype);
    nk_scalar_buffer_t distance;
    metric(a_parsed.data, b_parsed.data, c_parsed.data, a_parsed.cols, &distance);
    return_obj = nk_scalar_buffer_to_py_number(&distance, kernel_out_dtype);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&c_buffer);
    return return_obj;
}

static PyObject *implement_geospatial_metric( //
    nk_kernel_kind_t metric_kind,             //
    PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_lats_obj = NULL; // Required object, positional-only
    PyObject *a_lons_obj = NULL; // Required object, positional-only
    PyObject *b_lats_obj = NULL; // Required object, positional-only
    PyObject *b_lons_obj = NULL; // Required object, positional-only
    PyObject *dtype_obj = NULL;  // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;    // Optional object, "out" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    nk_dtype_t dtype = nk_dtype_unknown_k;
    Py_buffer a_lats_buffer, a_lons_buffer, b_lats_buffer, b_lons_buffer, out_buffer;
    MatrixOrVectorView a_lats_parsed, a_lons_parsed, b_lats_parsed, b_lons_parsed, out_parsed;
    memset(&a_lats_buffer, 0, sizeof(Py_buffer));
    memset(&a_lons_buffer, 0, sizeof(Py_buffer));
    memset(&b_lats_buffer, 0, sizeof(Py_buffer));
    memset(&b_lons_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    // Parse the arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 4 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 4-6 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 5) {
        PyErr_Format(PyExc_TypeError, "Only first 5 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (4 coordinate arrays)
    a_lats_obj = args[0];
    a_lons_obj = args[1];
    b_lats_obj = args[2];
    b_lons_obj = args[3];

    // Positional or keyword argument (dtype)
    if (positional_args_count == 5) dtype_obj = args[4];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype`
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    // Convert input objects to buffers
    nk_buffer_backing_t a_lats_backing, a_lons_backing, b_lats_backing, b_lons_backing, out_backing;
    if (!parse_tensor(a_lats_obj, &a_lats_buffer, &a_lats_parsed, &a_lats_backing, dtype) ||
        !parse_tensor(a_lons_obj, &a_lons_buffer, &a_lons_parsed, &a_lons_backing, dtype) ||
        !parse_tensor(b_lats_obj, &b_lats_buffer, &b_lats_parsed, &b_lats_backing, dtype) ||
        !parse_tensor(b_lons_obj, &b_lons_buffer, &b_lons_parsed, &b_lons_backing, dtype))
        goto cleanup;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed, &out_backing, nk_dtype_unknown_k)) goto cleanup;

    // Check dimensions: all inputs must be 1D vectors of equal length
    if (a_lats_parsed.rank != 1 || a_lons_parsed.rank != 1 || b_lats_parsed.rank != 1 || b_lons_parsed.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "All coordinate arrays must be 1D vectors");
        goto cleanup;
    }
    // For geospatial, n is the number of coordinate pairs (shape[0] for 1D arrays)
    size_t const n = a_lats_parsed.cols;
    if (a_lons_parsed.cols != n || b_lats_parsed.cols != n || b_lons_parsed.cols != n) {
        PyErr_SetString(PyExc_ValueError, "All coordinate arrays must have the same length");
        goto cleanup;
    }
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "Coordinate arrays can't be empty");
        goto cleanup;
    }

    // Check data types: all must match
    if (a_lats_parsed.dtype != a_lons_parsed.dtype || a_lats_parsed.dtype != b_lats_parsed.dtype ||
        a_lats_parsed.dtype != b_lons_parsed.dtype || a_lats_parsed.dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "All coordinate arrays must have the same dtype");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_lats_parsed.dtype;

    // Look up the metric kernel
    nk_metric_geospatial_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&metric, &capability);
    if (!metric || !capability) {
        PyErr_Format(PyExc_LookupError, "Unsupported metric '%c' and dtype '%s'", metric_kind,
                     nk_dtype_to_pybuffer_typestr(dtype));
        goto cleanup;
    }

    // Allocate output or use provided
    // Output dtype must match input dtype (f32 kernel writes f32, f64 kernel writes f64)
    size_t const item_size = nk_dtype_bytes_per_value(dtype);
    void *distances_start = NULL;
    if (!out_obj) {
        Py_ssize_t geo_shape[1] = {(Py_ssize_t)n};
        Tensor *distances_obj = Tensor_new(dtype, 1, geo_shape);
        if (!distances_obj) { goto cleanup; }
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
    }
    else {
        if (out_parsed.cols < n) {
            PyErr_SetString(PyExc_ValueError, "Output array is too small");
            goto cleanup;
        }
        distances_start = out_parsed.data;
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    // Call the kernel
    metric(a_lats_parsed.data, a_lons_parsed.data, b_lats_parsed.data, b_lons_parsed.data, n, distances_start);

cleanup:
    if (a_lats_buffer.buf) PyBuffer_Release(&a_lats_buffer);
    if (a_lons_buffer.buf) PyBuffer_Release(&a_lons_buffer);
    if (b_lats_buffer.buf) PyBuffer_Release(&b_lats_buffer);
    if (b_lons_buffer.buf) PyBuffer_Release(&b_lons_buffer);
    if (out_buffer.buf) PyBuffer_Release(&out_buffer);
    return return_obj;
}

static PyObject *implement_sparse_metric( //
    nk_kernel_kind_t metric_kind,         //
    PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Function expects only 2 arguments");
        return NULL;
    }

    PyObject *return_obj = NULL;
    PyObject *a_obj = args[0];
    PyObject *b_obj = args[1];

    Py_buffer a_buffer, b_buffer;
    MatrixOrVectorView a_parsed, b_parsed;
    nk_buffer_backing_t a_backing, b_backing;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed, &a_backing, nk_dtype_unknown_k) ||
        !parse_tensor(b_obj, &b_buffer, &b_parsed, &b_backing, nk_dtype_unknown_k))
        goto cleanup;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "First and second argument must be vectors");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.dtype != b_parsed.dtype && a_parsed.dtype != nk_dtype_unknown_k &&
        b_parsed.dtype != nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }

    nk_dtype_t dtype = a_parsed.dtype;
    nk_sparse_intersect_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&metric, &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError, "Unsupported metric '%c' and dtype combination ('%s'/'%s' and '%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(a_parsed.dtype), //
            b_buffer.format ? b_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(b_parsed.dtype));
        goto cleanup;
    }

    nk_size_t count = 0;
    metric(a_parsed.data, b_parsed.data, a_parsed.cols, b_parsed.cols, NULL, &count);
    return_obj = PyLong_FromSize_t(count);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    return return_obj;
}

/**
 *  @brief Map a pairwise metric kind to its batch (packed + symmetric) kernel kinds.
 *  @return 1 if batch kernels exist for this metric, 0 otherwise.
 */
static int metric_to_batch_kinds( //
    nk_kernel_kind_t metric_kind, //
    nk_kernel_kind_t *packed_kind, nk_kernel_kind_t *symmetric_kind) {
    switch (metric_kind) {
    case nk_kernel_dot_k:
        *packed_kind = nk_kernel_dots_packed_k;
        *symmetric_kind = nk_kernel_dots_symmetric_k;
        return 1;
    case nk_kernel_angular_k:
        *packed_kind = nk_kernel_angulars_packed_k;
        *symmetric_kind = nk_kernel_angulars_symmetric_k;
        return 1;
    case nk_kernel_euclidean_k:
        *packed_kind = nk_kernel_euclideans_packed_k;
        *symmetric_kind = nk_kernel_euclideans_symmetric_k;
        return 1;
    case nk_kernel_hamming_k:
        *packed_kind = nk_kernel_hammings_packed_k;
        *symmetric_kind = nk_kernel_hammings_symmetric_k;
        return 1;
    case nk_kernel_jaccard_k:
        *packed_kind = nk_kernel_jaccards_packed_k;
        *symmetric_kind = nk_kernel_jaccards_symmetric_k;
        return 1;
    default: return 0;
    }
}

/**
 *  @brief Pairwise loop fallback: compute one pair at a time via a scalar metric kernel.
 */
static void cdist_pairwise_loop(                          //
    nk_metric_dense_punned_t metric,                      //
    char const *a_start, size_t a_count, size_t a_stride, //
    char const *b_start, size_t b_count, size_t b_stride, //
    size_t dimensions,                                    //
    nk_dtype_t kernel_out_dtype, nk_dtype_t out_dtype,    //
    char *out, size_t out_row_stride, size_t out_col_stride, int const is_symmetric) {
    for (size_t i = 0; i < a_count; ++i)
        for (size_t j = 0; j < b_count; ++j) {
            if (is_symmetric && i > j) continue;
            nk_scalar_buffer_t result;
            metric(a_start + i * a_stride, b_start + j * b_stride, dimensions, &result);
            char *ptr_ij = out + i * out_row_stride + j * out_col_stride;
            nk_scalar_buffer_export(&result, kernel_out_dtype, ptr_ij, out_dtype);
            if (is_symmetric) {
                char *ptr_ji = out + j * out_row_stride + i * out_col_stride;
                nk_scalar_buffer_export(&result, kernel_out_dtype, ptr_ji, out_dtype);
            }
        }
}

/**
 *  @brief Batch symmetric path: compute A x A^T via a SIMD-optimized symmetric kernel.
 *  @return 0 on success, -1 if the kernel was not found.
 */
static int cdist_batch_symmetric(                             //
    nk_kernel_kind_t symmetric_kind, nk_dtype_t dtype,        //
    char const *vectors, size_t n_vectors, size_t dimensions, //
    size_t stride, char *out, size_t out_row_stride,          //
    nk_size_t threads) {
    nk_dots_symmetric_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(symmetric_kind, dtype, static_capabilities, //
                          (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) return -1;
#if defined(NK_USE_OPENMP)
    if (threads == 0) threads = (nk_size_t)omp_get_max_threads();
    omp_set_num_threads((int)threads);
#endif

    nk_size_t const tile_count = nk_size_divide_round_up_(n_vectors, NK_PARALLEL_SYMMETRIC_TILE);
#pragma omp parallel for schedule(dynamic, 1) if (threads > 1)
    for (nk_size_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
        nk_size_t tile_start = tile_idx * NK_PARALLEL_SYMMETRIC_TILE;
        nk_size_t tile_rows = (tile_start + NK_PARALLEL_SYMMETRIC_TILE <= n_vectors) ? NK_PARALLEL_SYMMETRIC_TILE
                                                                                     : (n_vectors - tile_start);
        kernel(vectors, n_vectors, dimensions, stride, out, out_row_stride, tile_start, tile_rows);
    }

    return 0;
}

/**
 *  @brief Batch packed path: pack B, then compute A x B_packed via a SIMD-optimized kernel.
 *  @return 0 on success, -1 on allocation failure, -2 if a kernel was not found.
 */
static int cdist_batch_packed(                                               //
    nk_kernel_kind_t packed_kind, nk_dtype_t dtype,                          //
    char const *a_start, size_t a_count, size_t a_stride,                    //
    char const *b_start, size_t b_count, size_t b_stride, size_t dimensions, //
    char *out, size_t out_row_stride, nk_size_t threads) {

    // All metric families reuse the dots pack_size / pack kernels
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_dots_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;

    nk_find_kernel_punned(nk_kernel_dots_packed_size_k, dtype, static_capabilities, (nk_kernel_punned_t *)&size_fn,
                          &cap);
    if (!size_fn || !cap) return -2;

    cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_pack_k, dtype, static_capabilities, (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!pack_fn || !cap) return -2;

    cap = nk_cap_serial_k;
    nk_find_kernel_punned(packed_kind, dtype, static_capabilities, //
                          (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) return -2;

    nk_size_t packed_size = size_fn(b_count, dimensions);
    void *b_packed = malloc(packed_size);
    if (!b_packed) return -1;

    pack_fn(b_start, b_count, dimensions, b_stride, b_packed);
#if defined(NK_USE_OPENMP)
    if (threads == 0) threads = (nk_size_t)omp_get_max_threads();
    omp_set_num_threads((int)threads);
#endif

    nk_size_t const tile_count = nk_size_divide_round_up_(a_count, NK_PARALLEL_PACKED_TILE);
#pragma omp parallel for schedule(dynamic, 1) if (threads > 1)
    for (nk_size_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
        nk_size_t row = tile_idx * NK_PARALLEL_PACKED_TILE;
        nk_size_t chunk = (row + NK_PARALLEL_PACKED_TILE <= a_count) ? NK_PARALLEL_PACKED_TILE : (a_count - row);
        kernel(a_start + row * a_stride, b_packed, out + row * out_row_stride, chunk, b_count, dimensions, a_stride,
               out_row_stride);
    }

    free(b_packed);
    return 0;
}

static PyObject *implement_cdist(                        //
    PyObject *a_obj, PyObject *b_obj, PyObject *out_obj, //
    nk_kernel_kind_t metric_kind,                        //
    nk_dtype_t dtype, nk_dtype_t out_dtype,              //
    nk_size_t threads) {

    PyObject *return_obj = NULL;

    Py_buffer a_buffer, b_buffer, out_buffer;
    MatrixOrVectorView a_parsed, b_parsed, out_parsed;
    nk_buffer_backing_t a_backing, b_backing, out_backing;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    // Error will be set by `parse_tensor` if the input is invalid
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed, &a_backing, dtype) ||
        !parse_tensor(b_obj, &b_buffer, &b_parsed, &b_backing, dtype))
        goto cleanup;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed, &out_backing, nk_dtype_unknown_k)) goto cleanup;

    // Check dimensions
    if (a_parsed.cols != b_parsed.cols) {
        PyErr_Format(PyExc_ValueError, "Vector dimensions don't match (%zu != %zu)", a_parsed.cols, b_parsed.cols);
        goto cleanup;
    }
    if (a_parsed.rows == 0 || b_parsed.rows == 0) {
        PyErr_SetString(PyExc_ValueError, "Collections can't be empty");
        goto cleanup;
    }
    if (out_obj &&
        (out_parsed.rank != 2 || out_buffer.shape[0] != a_parsed.rows || out_buffer.shape[1] != b_parsed.rows)) {
        PyErr_Format(PyExc_ValueError, "Output tensor must have shape (%zu, %zu)", a_parsed.rows, b_parsed.rows);
        goto cleanup;
    }

    // Check data types
    if (a_parsed.dtype != b_parsed.dtype || //
        a_parsed.dtype == nk_dtype_unknown_k || b_parsed.dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // When a dtype override reinterprets elements at a different width, rescale dimensions.
    size_t a_cols = a_parsed.cols, b_cols = b_parsed.cols;
    {
        nk_size_t from_bits = (nk_size_t)a_buffer.itemsize * NK_BITS_PER_BYTE;
        nk_size_t to_bits = nk_dtype_bits(dtype);
        if (from_bits && to_bits && from_bits != to_bits) {
            a_cols = a_cols * from_bits / to_bits;
            b_cols = b_cols * from_bits / to_bits;
        }
    }

    // Inference order for the output type:
    // 1. `out_dtype` named argument, if defined
    // 2. `out.dtype` attribute, if `out` is passed
    // 3. double precision float (or its complex variant)
    if (out_dtype == nk_dtype_unknown_k) {
        if (out_obj) { out_dtype = out_parsed.dtype; }
        else { out_dtype = (nk_dtype_family(dtype) == nk_dtype_family_complex_float_k) ? nk_f64c_k : nk_f64_k; }
    }

    // Make sure the return dtype is complex if the input dtype is complex, and the same for real numbers
    if (out_dtype != nk_dtype_unknown_k) {
        if ((nk_dtype_family(dtype) == nk_dtype_family_complex_float_k) !=
            (nk_dtype_family(out_dtype) == nk_dtype_family_complex_float_k)) {
            PyErr_SetString(PyExc_ValueError,
                            "If the input dtype is complex, the return dtype must be complex, and same for real.");
            goto cleanup;
        }
    }

    // Check if the downcasting to provided dtype is supported
    {
        nk_scalar_buffer_t probe;
        probe.f64c.real = 0, probe.f64c.imag = 0;
        if (!nk_scalar_buffer_from_f64c(&probe.f64c, &probe, out_dtype)) {
            PyErr_SetString(PyExc_ValueError, "Exporting to the provided dtype is not supported");
            goto cleanup;
        }
    }

    // Look up the metric and the capability
    nk_metric_dense_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&metric, &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError, "Unsupported metric '%c' and dtype combination ('%s'/'%s' and '%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(a_parsed.dtype), //
            b_buffer.format ? b_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(b_parsed.dtype));
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    nk_dtype_t const kernel_out_dtype = nk_kernel_output_dtype(metric_kind, dtype);
    if (a_parsed.rank == 1 && b_parsed.rank == 1) {
        nk_scalar_buffer_t distance;
        metric(a_parsed.data, b_parsed.data, a_cols, &distance);
        return_obj = nk_scalar_buffer_to_py_number(&distance, kernel_out_dtype);
        goto cleanup;
    }

    size_t const count_pairs = a_parsed.rows * b_parsed.rows;
    char *distances_start = NULL;
    size_t distances_rows_stride_bytes = 0;
    size_t distances_cols_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {

        Py_ssize_t cdist_shape[2] = {(Py_ssize_t)a_parsed.rows, (Py_ssize_t)b_parsed.rows};
        Tensor *distances_obj = Tensor_new(out_dtype, 2, cdist_shape);
        if (!distances_obj) { goto cleanup; }
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
        distances_rows_stride_bytes = distances_obj->strides[0];
        distances_cols_stride_bytes = distances_obj->strides[1];
    }
    else {
        if (nk_dtype_bytes_per_value(out_parsed.dtype) != nk_dtype_bytes_per_value(out_dtype)) {
            PyErr_Format( //
                PyExc_LookupError,
                "Output tensor scalar type must be compatible with the output type ('%s' and '%s'/'%s')",
                nk_dtype_to_pybuffer_typestr(out_dtype), out_buffer.format ? out_buffer.format : "nil",
                nk_dtype_to_pybuffer_typestr(out_parsed.dtype));
            goto cleanup;
        }
        distances_start = out_parsed.data;
        distances_rows_stride_bytes = out_buffer.strides[0];
        distances_cols_stride_bytes = out_buffer.strides[1];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    // Release the GIL for the compute-intensive section.
    PyThreadState *save = PyEval_SaveThread();

    int const is_symmetric = nk_kernel_is_commutative(metric_kind) && a_parsed.data == b_parsed.data &&
                             a_parsed.row_stride == b_parsed.row_stride && a_parsed.rows == b_parsed.rows;

    // Batch kernels write typed values directly into the output buffer, so we can only use them
    // when the output dtype matches the kernel's native output dtype exactly.
    nk_kernel_kind_t packed_kind = nk_kernel_unknown_k, symmetric_kind = nk_kernel_unknown_k;
    int const has_batch = metric_to_batch_kinds(metric_kind, &packed_kind, &symmetric_kind);
    int const dtype_ok = (out_dtype == kernel_out_dtype);
    int batch_result = -1;

    // Try symmetric batch path first (A x A^T, no packing needed)
    if (has_batch && dtype_ok && is_symmetric)
        batch_result = cdist_batch_symmetric(symmetric_kind, dtype, a_parsed.data, a_parsed.rows, a_cols,
                                             a_parsed.row_stride, distances_start, distances_rows_stride_bytes,
                                             threads);

    // Symmetric kernel only writes upper triangle; mirror to lower.
    if (batch_result == 0 && is_symmetric) {
        size_t const elem_size = nk_dtype_bytes_per_value(out_dtype);
        for (size_t i = 1; i < a_parsed.rows; ++i)
            for (size_t j = 0; j < i; ++j)
                memcpy(distances_start + i * distances_rows_stride_bytes + j * distances_cols_stride_bytes,
                       distances_start + j * distances_rows_stride_bytes + i * distances_cols_stride_bytes, elem_size);
    }

    // Try packed batch path (A x B_packed)
    if (has_batch && dtype_ok && !is_symmetric && batch_result != 0)
        batch_result = cdist_batch_packed(packed_kind, dtype, a_parsed.data, a_parsed.rows, a_parsed.row_stride,
                                          b_parsed.data, b_parsed.rows, b_parsed.row_stride, a_cols, distances_start,
                                          distances_rows_stride_bytes, threads);

    // Fall back to scalar pairwise loop
    if (batch_result != 0)
        cdist_pairwise_loop(metric, a_parsed.data, a_parsed.rows, a_parsed.row_stride, b_parsed.data, b_parsed.rows,
                            b_parsed.row_stride, a_cols, kernel_out_dtype, out_dtype, distances_start,
                            distances_rows_stride_bytes, distances_cols_stride_bytes, is_symmetric);

    PyEval_RestoreThread(save);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static PyObject *implement_pointer_access(nk_kernel_kind_t metric_kind, PyObject *dtype_obj) {
    nk_dtype_t dtype = py_object_to_nk_dtype(dtype_obj);
    if (!dtype) return NULL;

    nk_kernel_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, &metric, &capability);
    if (!metric || !capability) {
        PyErr_SetString(PyExc_LookupError, "No such metric");
        return NULL;
    }

    return PyLong_FromUnsignedLongLong((unsigned long long)metric);
}

char const doc_cdist[] =                                                                                          //
    "Compute pairwise distances between two input sets.\n\n"                                                      //
    "Parameters:\n"                                                                                               //
    "    a (Tensor): First matrix.\n"                                                                             //
    "    b (Tensor): Second matrix.\n"                                                                            //
    "    metric (str, optional): Distance metric to use (e.g., 'sqeuclidean', 'cosine').\n"                       //
    "    out (Tensor, optional): Output matrix to store the result.\n"                                            //
    "    dtype (Union[IntegralType, FloatType, ComplexType], optional): Override the presumed input type name.\n" //
    "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n\n"             //
    "Returns:\n"                                                                                                  //
    "    Tensor: Pairwise distances between all inputs.\n\n"                                                      //
    "Equivalent to: `scipy.spatial.distance.cdist`.\n"                                                            //
    "Signature:\n"                                                                                                //
    "    >>> def cdist(a, b, /, metric, *, dtype, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_cdist( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    // This function accepts up to six arguments, more than SciPy:
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    PyObject *a_obj = NULL;         // Required object, positional-only
    PyObject *b_obj = NULL;         // Required object, positional-only
    PyObject *metric_obj = NULL;    // Optional string, "metric" keyword or positional
    PyObject *out_obj = NULL;       // Optional object, "out" keyword-only
    PyObject *dtype_obj = NULL;     // Optional string, "dtype" keyword-only
    PyObject *out_dtype_obj = NULL; // Optional string, "out_dtype" keyword-only
    nk_size_t threads = 1;          // Optional int, "threads" keyword-only. 0 = all cores.

    // Once parsed, the arguments will be stored in these variables:
    nk_dtype_t dtype = nk_dtype_unknown_k, out_dtype = nk_dtype_unknown_k;

    /** Same default as in SciPy:
     *  https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.cdist.html */
    nk_kernel_kind_t metric_kind = nk_kernel_euclidean_k;
    char const *metric_str = NULL;

    // Parse the arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 7) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-7 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "Only first 3 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
    a_obj = args[0];
    b_obj = args[1];

    // Positional or keyword arguments (metric)
    if (positional_args_count == 3) metric_obj = args[2];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "metric") == 0 && !metric_obj) { metric_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "threads") == 0) {
            Py_ssize_t t = PyLong_AsSsize_t(value);
            if (t == -1 && PyErr_Occurred()) return NULL;
            threads = (nk_size_t)(t >= 0 ? t : 0);
        }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `metric_obj` to `metric_str` and to `metric_kind`
    if (metric_obj) {
        Py_ssize_t metric_len = 0;
        metric_str = PyUnicode_AsUTF8AndSize(metric_obj, &metric_len);
        if (!metric_str && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Expected 'metric' to be a string");
            return NULL;
        }
        metric_kind = py_string_to_nk_kernel_kind(metric_str, metric_len);
        if (metric_kind == nk_kernel_unknown_k) {
            PyErr_SetString(PyExc_LookupError, "Unsupported metric");
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype`
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    // Convert `out_dtype_obj` to `out_dtype`
    if (out_dtype_obj) {
        out_dtype = py_object_to_nk_dtype(out_dtype_obj);
        if (out_dtype == nk_dtype_unknown_k) return NULL;
    }

    return implement_cdist(a_obj, b_obj, out_obj, metric_kind, dtype, out_dtype, threads);
}

char const doc_euclidean_pointer[] = "Return an integer pointer to the `numkong.euclidean` kernel.";
PyObject *api_euclidean_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_euclidean_k, dtype_obj);
}
char const doc_sqeuclidean_pointer[] = "Return an integer pointer to the `numkong.sqeuclidean` kernel.";
PyObject *api_sqeuclidean_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_sqeuclidean_k, dtype_obj);
}
char const doc_angular_pointer[] = "Return an integer pointer to the `numkong.angular` kernel.";
PyObject *api_angular_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_angular_k, dtype_obj);
}
char const doc_dot_pointer[] = "Return an integer pointer to the `numkong.dot` kernel.";
PyObject *api_dot_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_dot_k, dtype_obj);
}
char const doc_vdot_pointer[] = "Return an integer pointer to the `numkong.vdot` kernel.";
PyObject *api_vdot_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_vdot_k, dtype_obj);
}
char const doc_kld_pointer[] = "Return an integer pointer to the `numkong.kld` kernel.";
PyObject *api_kld_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_kld_k, dtype_obj);
}
char const doc_jsd_pointer[] = "Return an integer pointer to the `numkong.jsd` kernel.";
PyObject *api_jsd_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_jsd_k, dtype_obj);
}
char const doc_hamming_pointer[] = "Return an integer pointer to the `numkong.hamming` kernel.";
PyObject *api_hamming_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_hamming_k, dtype_obj);
}
char const doc_jaccard_pointer[] = "Return an integer pointer to the `numkong.jaccard` kernel.";
PyObject *api_jaccard_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_jaccard_k, dtype_obj);
}

char const doc_euclidean[] =                                                                           //
    "Compute Euclidean distances between two matrices.\n\n"                                            //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First matrix or vector.\n"                                                        //
    "    b (Tensor): Second matrix or vector.\n"                                                       //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"   //
    "        Supported values: 'f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'e2m3', 'e3m2',\n"         //
    "        'i8', 'u8', 'i4', 'u4'.\n"                                                                //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.spatial.distance.euclidean`.\n"                                             //
    "Signature:\n"                                                                                     //
    "    >>> def euclidean(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_euclidean(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                        PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_euclidean_k, args, positional_args_count, args_names_tuple);
}

char const doc_sqeuclidean[] =                                                                         //
    "Compute squared Euclidean distances between two matrices.\n\n"                                    //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First matrix or vector.\n"                                                        //
    "    b (Tensor): Second matrix or vector.\n"                                                       //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"   //
    "        Supported values: 'f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'e2m3', 'e3m2',\n"         //
    "        'i8', 'u8', 'i4', 'u4'.\n"                                                                //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.spatial.distance.sqeuclidean`.\n"                                           //
    "Signature:\n"                                                                                     //
    "    >>> def sqeuclidean(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_sqeuclidean(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_sqeuclidean_k, args, positional_args_count, args_names_tuple);
}

char const doc_angular[] =                                                                             //
    "Compute angular distances between two matrices.\n\n"                                              //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First matrix or vector.\n"                                                        //
    "    b (Tensor): Second matrix or vector.\n"                                                       //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"   //
    "        Supported values: 'f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'e2m3', 'e3m2',\n"         //
    "        'i8', 'u8', 'i4', 'u4'.\n"                                                                //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.spatial.distance.cosine`.\n"                                                //
    "Signature:\n"                                                                                     //
    "    >>> def angular(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_angular(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                      PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_angular_k, args, positional_args_count, args_names_tuple);
}

char const doc_dot[] =                                                                                            //
    "Compute the inner (dot) product between two matrices (real or complex).\n\n"                                 //
    "Parameters:\n"                                                                                               //
    "    a (Tensor): First matrix or vector.\n"                                                                   //
    "    b (Tensor): Second matrix or vector.\n"                                                                  //
    "    dtype (Union[IntegralType, FloatType, ComplexType], optional): Override the presumed input type name.\n" //
    "        Supported values: 'f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'e2m3', 'e3m2',\n"                    //
    "        'i8', 'u8', 'i4', 'u4', 'u1', 'complex64', 'complex128', 'complex32', 'bcomplex32'.\n"               //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n"            //
    "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n\n"             //
    "Returns:\n"                                                                                                  //
    "    Tensor: The distances if `out` is not provided.\n"                                                       //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                                   //
    "Equivalent to: `numpy.inner`.\n"                                                                             //
    "Signature:\n"                                                                                                //
    "    >>> def dot(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_dot(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                  PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_dot_k, args, positional_args_count, args_names_tuple);
}

char const doc_vdot[] =                                                                                //
    "Compute the conjugate dot product between two complex matrices.\n\n"                              //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First complex matrix or vector.\n"                                                //
    "    b (Tensor): Second complex matrix or vector.\n"                                               //
    "    dtype (ComplexType, optional): Override the presumed input type name.\n"                      //
    "        Supported values: 'complex64', 'complex128', 'complex32', 'bcomplex32'.\n"                //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (Union[ComplexType], optional): Result type, default is 'float64'.\n\n"             //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `numpy.vdot`.\n"                                                                   //
    "Signature:\n"                                                                                     //
    "    >>> def vdot(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_vdot(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                   PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_vdot_k, args, positional_args_count, args_names_tuple);
}

char const doc_kld[] =                                                                                 //
    "Compute Kullback-Leibler divergences between two matrices.\n\n"                                   //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First floating-point matrix or vector.\n"                                         //
    "    b (Tensor): Second floating-point matrix or vector.\n"                                        //
    "    dtype (FloatType, optional): Override the presumed input type name.\n"                        //
    "        Supported values: 'f64', 'f32', 'f16', 'bf16'.\n"                                         //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.special.kl_div`.\n"                                                         //
    "Signature:\n"                                                                                     //
    "    >>> def kld(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_kld(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                  PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_kld_k, args, positional_args_count, args_names_tuple);
}

char const doc_jsd[] =                                                                                 //
    "Compute Jensen-Shannon distances between two matrices.\n\n"                                       //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First floating-point matrix or vector.\n"                                         //
    "    b (Tensor): Second floating-point matrix or vector.\n"                                        //
    "    dtype (FloatType, optional): Override the presumed input type name.\n"                        //
    "        Supported values: 'f64', 'f32', 'f16', 'bf16'.\n"                                         //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.spatial.distance.jensenshannon`.\n"                                         //
    "Signature:\n"                                                                                     //
    "    >>> def jsd(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_jsd(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                  PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_jsd_k, args, positional_args_count, args_names_tuple);
}

char const doc_hamming[] =                                                                             //
    "Compute Hamming distances between two matrices.\n\n"                                              //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First binary matrix or vector.\n"                                                 //
    "    b (Tensor): Second binary matrix or vector.\n"                                                //
    "    dtype (IntegralType, optional): Override the presumed input type name.\n"                     //
    "        Supported values: 'u1', 'u8'.\n"                                                          //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Similar to: `scipy.spatial.distance.hamming`.\n"                                                  //
    "Signature:\n"                                                                                     //
    "    >>> def hamming(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_hamming(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                      PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_hamming_k, args, positional_args_count, args_names_tuple);
}

char const doc_jaccard[] =                                                                             //
    "Compute Jaccard distances (bitwise Tanimoto) between two matrices.\n\n"                           //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First binary matrix or vector.\n"                                                 //
    "    b (Tensor): Second binary matrix or vector.\n"                                                //
    "    dtype (IntegralType, optional): Override the presumed input type name.\n"                     //
    "        Supported values: 'u1', 'u16', 'u32'.\n"                                                  //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Similar to: `scipy.spatial.distance.jaccard`.\n"                                                  //
    "Signature:\n"                                                                                     //
    "    >>> def jaccard(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

PyObject *api_jaccard(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                      PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_jaccard_k, args, positional_args_count, args_names_tuple);
}

char const doc_bilinear[] =                                                       //
    "Compute the bilinear form between two vectors given a metric tensor.\n\n"    //
    "Parameters:\n"                                                               //
    "    a (Tensor): First vector.\n"                                             //
    "    b (Tensor): Second vector.\n"                                            //
    "    metric_tensor (Tensor): The metric tensor defining the bilinear form.\n" //
    "    dtype (FloatType, optional): Override the presumed input type name.\n\n" //
    "Returns:\n"                                                                  //
    "    float: The bilinear form.\n\n"                                           //
    "Equivalent to: `numpy.dot` with a metric tensor.\n"                          //
    "Signature:\n"                                                                //
    "    >>> def bilinear(a, b, metric_tensor, /, dtype) -> float: ...";

PyObject *api_bilinear(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                       PyObject *args_names_tuple) {
    return implement_curved_metric(nk_kernel_bilinear_k, args, positional_args_count, args_names_tuple);
}

char const doc_mahalanobis[] =                                                                     //
    "Compute the Mahalanobis distance between two vectors given an inverse covariance matrix.\n\n" //
    "Parameters:\n"                                                                                //
    "    a (Tensor): First vector.\n"                                                              //
    "    b (Tensor): Second vector.\n"                                                             //
    "    inverse_covariance (Tensor): The inverse of the covariance matrix.\n"                     //
    "    dtype (FloatType, optional): Override the presumed input type name.\n\n"                  //
    "Returns:\n"                                                                                   //
    "    float: The Mahalanobis distance.\n\n"                                                     //
    "Equivalent to: `scipy.spatial.distance.mahalanobis`.\n"                                       //
    "Signature:\n"                                                                                 //
    "    >>> def mahalanobis(a, b, inverse_covariance, /, dtype) -> float: ...";

PyObject *api_mahalanobis(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {
    return implement_curved_metric(nk_kernel_mahalanobis_k, args, positional_args_count, args_names_tuple);
}

char const doc_haversine[] =                                                      //
    "Compute the Haversine (great-circle) distance between coordinate pairs.\n\n" //
    "Parameters:\n"                                                               //
    "    a_lats (Tensor): Latitudes of first points in radians.\n"                //
    "    a_lons (Tensor): Longitudes of first points in radians.\n"               //
    "    b_lats (Tensor): Latitudes of second points in radians.\n"               //
    "    b_lons (Tensor): Longitudes of second points in radians.\n"              //
    "    dtype (FloatType, optional): Override the presumed input type name.\n"   //
    "    out (Tensor, optional): Pre-allocated output array for distances.\n\n"   //
    "Returns:\n"                                                                  //
    "    Tensor: Distances in meters (using mean Earth radius).\n"                //
    "    None: If `out` is provided.\n\n"                                         //
    "Note: Input coordinates must be in radians. Uses spherical Earth model.\n"   //
    "Signature:\n"                                                                //
    "    >>> def haversine(a_lats, a_lons, b_lats, b_lons, /, dtype, *, out) -> Optional[Tensor]: ...";

PyObject *api_haversine(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                        PyObject *args_names_tuple) {
    return implement_geospatial_metric(nk_kernel_haversine_k, args, positional_args_count, args_names_tuple);
}

char const doc_vincenty[] =                                                                //
    "Compute the Vincenty (ellipsoidal geodesic) distance between coordinate pairs.\n\n"   //
    "Parameters:\n"                                                                        //
    "    a_lats (Tensor): Latitudes of first points in radians.\n"                         //
    "    a_lons (Tensor): Longitudes of first points in radians.\n"                        //
    "    b_lats (Tensor): Latitudes of second points in radians.\n"                        //
    "    b_lons (Tensor): Longitudes of second points in radians.\n"                       //
    "    dtype (FloatType, optional): Override the presumed input type name.\n"            //
    "    out (Tensor, optional): Pre-allocated output array for distances.\n\n"            //
    "Returns:\n"                                                                           //
    "    Tensor: Distances in meters (using WGS84 ellipsoid).\n"                           //
    "    None: If `out` is provided.\n\n"                                                  //
    "Note: Input coordinates must be in radians. Uses iterative algorithm for accuracy.\n" //
    "Signature:\n"                                                                         //
    "    >>> def vincenty(a_lats, a_lons, b_lats, b_lons, /, dtype, *, out) -> Optional[Tensor]: ...";

PyObject *api_vincenty(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                       PyObject *args_names_tuple) {
    return implement_geospatial_metric(nk_kernel_vincenty_k, args, positional_args_count, args_names_tuple);
}

char const doc_intersect[] =                                     //
    "Compute the intersection of two sorted integer arrays.\n\n" //
    "Parameters:\n"                                              //
    "    a (Tensor): First sorted integer array.\n"              //
    "    b (Tensor): Second sorted integer array.\n\n"           //
    "Returns:\n"                                                 //
    "    int: The number of intersecting elements.\n\n"          //
    "Similar to: `numpy.intersect1d`.\n"                         //
    "Signature:\n"                                               //
    "    >>> def intersect(a, b, /) -> int: ...";

PyObject *api_intersect(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    return implement_sparse_metric(nk_kernel_sparse_intersect_k, args, nargs);
}

char const doc_sparse_dot[] =                                                                       //
    "Compute the weighted sparse dot product of two sorted index arrays.\n\n"                       //
    "Parameters:\n"                                                                                 //
    "    a_indices (Tensor): First sorted index array (uint16 or uint32).\n"                        //
    "    a_values  (Tensor): Weight array corresponding to a_indices (bf16 or float32).\n"          //
    "    b_indices (Tensor): Second sorted index array (same dtype as a_indices).\n"                //
    "    b_values  (Tensor): Weight array corresponding to b_indices (same dtype as a_values).\n\n" //
    "Returns:\n"                                                                                    //
    "    float: The weighted dot product of intersecting indices.\n\n"                              //
    "Signature:\n"                                                                                  //
    "    >>> def sparse_dot(a_indices, a_values, b_indices, b_values, /) -> float: ...";

PyObject *api_sparse_dot(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    nk_unused_(self);
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError,
                        "sparse_dot() expects exactly 4 arguments: a_indices, a_values, b_indices, b_values");
        return NULL;
    }

    Py_buffer a_idx_buf, a_val_buf, b_idx_buf, b_val_buf;
    MatrixOrVectorView a_idx, a_val, b_idx, b_val;
    nk_buffer_backing_t a_idx_backing, a_val_backing, b_idx_backing, b_val_backing;
    memset(&a_idx_buf, 0, sizeof(Py_buffer));
    memset(&a_val_buf, 0, sizeof(Py_buffer));
    memset(&b_idx_buf, 0, sizeof(Py_buffer));
    memset(&b_val_buf, 0, sizeof(Py_buffer));

    PyObject *return_obj = NULL;

    if (!parse_tensor(args[0], &a_idx_buf, &a_idx, &a_idx_backing, nk_dtype_unknown_k) ||
        !parse_tensor(args[1], &a_val_buf, &a_val, &a_val_backing, nk_dtype_unknown_k) ||
        !parse_tensor(args[2], &b_idx_buf, &b_idx, &b_idx_backing, nk_dtype_unknown_k) ||
        !parse_tensor(args[3], &b_val_buf, &b_val, &b_val_backing, nk_dtype_unknown_k)) {
        // Already uses goto-safe cleanup since buffers are zero-initialized and
        // PyBuffer_Release checks .buf before releasing.
        goto cleanup;
    }

    // All must be 1D
    if (a_idx.rank != 1 || a_val.rank != 1 || b_idx.rank != 1 || b_val.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "All arguments must be 1D vectors");
        goto cleanup;
    }
    // Index lengths must match their value lengths
    if (a_idx.cols != a_val.cols || b_idx.cols != b_val.cols) {
        PyErr_SetString(PyExc_ValueError, "Index and value arrays must have the same length");
        goto cleanup;
    }
    // Index dtypes must match
    if (a_idx.dtype != b_idx.dtype) {
        PyErr_SetString(PyExc_TypeError, "Index arrays must have the same dtype");
        goto cleanup;
    }
    // Value dtypes must match
    if (a_val.dtype != b_val.dtype) {
        PyErr_SetString(PyExc_TypeError, "Value arrays must have the same dtype");
        goto cleanup;
    }

    // Determine the variant by combining index + value dtypes into the dispatch dtype
    // nk_sparse_dot_u32f32: u32 indices + f32 weights
    // nk_sparse_dot_u16bf16: u16 indices + bf16 weights
    nk_dtype_t dispatch_dtype = nk_dtype_unknown_k;
    if (a_idx.dtype == nk_u32_k && a_val.dtype == nk_f32_k) dispatch_dtype = nk_f32_k;
    else if (a_idx.dtype == nk_u16_k && a_val.dtype == nk_bf16_k) dispatch_dtype = nk_bf16_k;
    else {
        PyErr_SetString(PyExc_TypeError,
                        "sparse_dot supports (uint32 indices + float32 values) or (uint16 indices + bfloat16 values)");
        goto cleanup;
    }

    nk_sparse_dot_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_sparse_dot_k, dispatch_dtype, static_capabilities, (nk_kernel_punned_t *)&kernel,
                          &capability);
    if (!kernel || !capability) {
        PyErr_SetString(PyExc_LookupError, "No sparse_dot kernel available for this dtype combination");
        goto cleanup;
    }

    nk_scalar_buffer_t product = {0};
    nk_dtype_t product_dtype = nk_kernel_output_dtype(nk_kernel_sparse_dot_k, dispatch_dtype);
    kernel(a_idx.data, b_idx.data, a_val.data, b_val.data, a_idx.cols, b_idx.cols, &product);
    return_obj = nk_scalar_buffer_to_py_number(&product, product_dtype);

cleanup:
    if (a_idx_buf.buf) PyBuffer_Release(&a_idx_buf);
    if (a_val_buf.buf) PyBuffer_Release(&a_val_buf);
    if (b_idx_buf.buf) PyBuffer_Release(&b_idx_buf);
    if (b_val_buf.buf) PyBuffer_Release(&b_val_buf);
    return return_obj;
}
