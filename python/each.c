/**
 *  @brief Elementwise operation implementations for NumKong Python bindings.
 *  @file python/each.c
 *  @author Ash Vardanian
 *  @date February 19, 2026
 *
 *  Implements fma, blend, scale, add, multiply, and trigonometric (sin, cos, atan)
 *  element-wise operations extracted from numkong.c.
 */
#include "each.h"
#include "tensor.h"

char const doc_fma[] =                                                                                 //
    "Fused-Multiply-Add between 3 input vectors.\n\n"                                                  //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First vector.\n"                                                                  //
    "    b (Tensor): Second vector.\n"                                                                 //
    "    c (Tensor): Third vector.\n"                                                                  //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    alpha (float, optional): First scale, 1.0 by default.\n"                                      //
    "    beta (float, optional): Second scale, 1.0 by default.\n"                                      //
    "    out (Tensor, optional): Vector for resulting distances.\n\n"                                  //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `alpha * a * b + beta * c`.\n"                                                     //
    "Signature:\n"                                                                                     //
    "    >>> def fma(a, b, c, /, dtype, *, alpha, beta, out) -> Optional[Tensor]: ...";

PyObject *api_fma(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                  PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *b_obj = NULL;     // Required object, positional-only
    PyObject *c_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only
    PyObject *alpha_obj = NULL; // Optional object, "alpha" keyword-only
    PyObject *beta_obj = NULL;  // Optional object, "beta" keyword-only

    // Once parsed, the arguments will be stored in these variables:

    nk_dtype_t dtype = nk_dtype_unknown_k;

    Py_buffer a_buffer, b_buffer, c_buffer, out_buffer;
    MatrixOrVectorView a_parsed, b_parsed, c_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&c_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 3 || args_count > 7) {
        PyErr_Format(PyExc_TypeError, "Function expects 3-7 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 4) {
        PyErr_Format(PyExc_TypeError, "Only first 4 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
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
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "alpha") == 0 && !alpha_obj) { alpha_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "beta") == 0 && !beta_obj) { beta_obj = value; }
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

    // Convert inputs to buffers
    nk_buffer_backing_t a_parsed_backing, b_parsed_backing, c_parsed_backing, out_parsed_backing;
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed, &a_parsed_backing, dtype) ||
        !parse_tensor(b_obj, &b_buffer, &b_parsed, &b_parsed_backing, dtype) ||
        !parse_tensor(c_obj, &c_buffer, &c_parsed, &c_parsed_backing, dtype))
        goto cleanup;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed, &out_parsed_backing, nk_dtype_unknown_k))
        goto cleanup;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1 || c_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be vectors");
        goto cleanup;
    }
    if (a_parsed.cols != b_parsed.cols || a_parsed.cols != c_parsed.cols ||
        (out_obj && a_parsed.cols != out_parsed.cols)) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.dtype != b_parsed.dtype || a_parsed.dtype == nk_dtype_unknown_k ||
        b_parsed.dtype == nk_dtype_unknown_k || c_parsed.dtype == nk_dtype_unknown_k ||
        (out_obj && out_parsed.dtype == nk_dtype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Convert `alpha_obj` to `alpha_buf` and `beta_obj` to `beta_buf`
    nk_scalar_buffer_t alpha_buf, beta_buf;
    {
        nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(dtype);
        alpha_buf.f64 = 1.0, beta_buf.f64 = 1.0;
        if (alpha_obj) {
            if (!py_number_to_nk_scalar_buffer(alpha_obj, &alpha_buf, scalar_dtype)) goto cleanup;
        }
        else nk_scalar_buffer_from_f64(&alpha_buf.f64, &alpha_buf, scalar_dtype);
        if (beta_obj) {
            if (!py_number_to_nk_scalar_buffer(beta_obj, &beta_buf, scalar_dtype)) goto cleanup;
        }
        else nk_scalar_buffer_from_f64(&beta_buf.f64, &beta_buf, scalar_dtype);
    }

    // Look up the kernel and the capability
    nk_each_fma_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_kernel_kind_t const kernel_kind = nk_kernel_each_fma_k;
    nk_find_kernel_punned(kernel_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &capability);
    if (!kernel || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported kernel '%c' and dtype combination across vectors ('%s'/'%s') and `dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(a_parsed.dtype), //
            nk_dtype_to_pybuffer_typestr(dtype), nk_dtype_to_pybuffer_typestr(dtype));
        goto cleanup;
    }

    char *result_data = NULL;

    // nk.fma(a, b, c) → returns new Tensor with α·a·b + β·c
    if (!out_obj) {
        Py_ssize_t out_shape[1] = {a_parsed.cols};
        Tensor *result_tensor = Tensor_new(dtype, 1, out_shape);
        if (!result_tensor) goto cleanup;
        return_obj = (PyObject *)result_tensor;
        result_data = result_tensor->data;
    }
    // nk.fma(a, b, c, out=result) → writes into provided buffer, returns None
    else {
        result_data = out_parsed.data;
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    {
        PyThreadState *gil = PyEval_SaveThread();
        kernel(a_parsed.data, b_parsed.data, c_parsed.data, a_parsed.cols, &alpha_buf, &beta_buf, result_data);
        PyEval_RestoreThread(gil);
    }
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&c_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

char const doc_blend[] =                                                                               //
    "Blend of 2 input vectors.\n\n"                                                                    //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First vector.\n"                                                                  //
    "    b (Tensor): Second vector.\n"                                                                 //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    alpha (float, optional): First scale, 1.0 by default.\n"                                      //
    "    beta (float, optional): Second scale, 1.0 by default.\n"                                      //
    "    out (Tensor, optional): Vector for resulting distances.\n\n"                                  //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `alpha * a + beta * b`.\n"                                                         //
    "Signature:\n"                                                                                     //
    "    >>> def blend(a, b, /, dtype, *, alpha, beta, out) -> Optional[Tensor]: ...";

PyObject *api_blend(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                    PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *b_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only
    PyObject *alpha_obj = NULL; // Optional object, "alpha" keyword-only
    PyObject *beta_obj = NULL;  // Optional object, "beta" keyword-only

    // Once parsed, the arguments will be stored in these variables:

    nk_dtype_t dtype = nk_dtype_unknown_k;

    Py_buffer a_buffer, b_buffer, out_buffer;
    MatrixOrVectorView a_parsed, b_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-6 arguments, got %zd", args_count);
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
        else if (PyUnicode_CompareWithASCIIString(key, "alpha") == 0 && !alpha_obj) { alpha_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "beta") == 0 && !beta_obj) { beta_obj = value; }
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

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `out_obj`.
    nk_buffer_backing_t a_parsed_backing, b_parsed_backing, out_parsed_backing;
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed, &a_parsed_backing, dtype) ||
        !parse_tensor(b_obj, &b_buffer, &b_parsed, &b_parsed_backing, dtype))
        goto cleanup;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed, &out_parsed_backing, nk_dtype_unknown_k))
        goto cleanup;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be vectors");
        goto cleanup;
    }
    if (a_parsed.cols != b_parsed.cols || (out_obj && a_parsed.cols != out_parsed.cols)) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.dtype != b_parsed.dtype || a_parsed.dtype == nk_dtype_unknown_k ||
        b_parsed.dtype == nk_dtype_unknown_k || (out_obj && out_parsed.dtype == nk_dtype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Convert `alpha_obj` to `alpha_buf` and `beta_obj` to `beta_buf`
    nk_scalar_buffer_t alpha_buf, beta_buf;
    {
        nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(dtype);
        alpha_buf.f64 = 1.0, beta_buf.f64 = 1.0;
        if (alpha_obj) {
            if (!py_number_to_nk_scalar_buffer(alpha_obj, &alpha_buf, scalar_dtype)) goto cleanup;
        }
        else nk_scalar_buffer_from_f64(&alpha_buf.f64, &alpha_buf, scalar_dtype);
        if (beta_obj) {
            if (!py_number_to_nk_scalar_buffer(beta_obj, &beta_buf, scalar_dtype)) goto cleanup;
        }
        else nk_scalar_buffer_from_f64(&beta_buf.f64, &beta_buf, scalar_dtype);
    }

    // Look up the kernel and the capability
    nk_each_blend_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_kernel_kind_t const kernel_kind = nk_kernel_each_blend_k;
    nk_find_kernel_punned(kernel_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &capability);
    if (!kernel || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported kernel '%c' and dtype combination across vectors ('%s'/'%s') and `dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(a_parsed.dtype), //
            nk_dtype_to_pybuffer_typestr(dtype), nk_dtype_to_pybuffer_typestr(dtype));
        goto cleanup;
    }

    char *result_data = NULL;

    // nk.blend(a, b) → returns new Tensor with α·a + β·b
    if (!out_obj) {
        Py_ssize_t out_shape[1] = {a_parsed.cols};
        Tensor *result_tensor = Tensor_new(dtype, 1, out_shape);
        if (!result_tensor) goto cleanup;
        return_obj = (PyObject *)result_tensor;
        result_data = result_tensor->data;
    }
    // nk.blend(a, b, out=result) → writes into provided buffer, returns None
    else {
        result_data = out_parsed.data;
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    {
        PyThreadState *gil = PyEval_SaveThread();
        kernel(a_parsed.data, b_parsed.data, a_parsed.cols, &alpha_buf, &beta_buf, result_data);
        PyEval_RestoreThread(gil);
    }
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

char const doc_scale[] =                                                                               //
    "Element-wise affine transformation of a single vector.\n\n"                                       //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): Input vector.\n"                                                                  //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    alpha (float, optional): Multiplicative scale, 1.0 by default.\n"                             //
    "    beta (float, optional): Additive offset, 0.0 by default.\n"                                   //
    "    out (Tensor, optional): Vector for resulting output.\n\n"                                     //
    "Returns:\n"                                                                                       //
    "    Tensor: The result if `out` is not provided.\n"                                               //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `alpha * a + beta`.\n"                                                             //
    "Signature:\n"                                                                                     //
    "    >>> def scale(a, /, dtype, *, alpha, beta, out) -> Optional[Tensor]: ...";

PyObject *api_scale(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                    PyObject *args_names_tuple) {
    nk_unused_(self);
    PyObject *return_obj = NULL;

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only
    PyObject *alpha_obj = NULL; // Optional object, "alpha" keyword-only
    PyObject *beta_obj = NULL;  // Optional object, "beta" keyword-only

    // Once parsed, the arguments will be stored in these variables:

    nk_dtype_t dtype = nk_dtype_unknown_k;

    Py_buffer a_buffer, out_buffer;
    MatrixOrVectorView a_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 1 || args_count > 5) {
        PyErr_Format(PyExc_TypeError, "Function expects 1-5 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "Only first 2 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (input vector)
    a_obj = args[0];

    // Positional or keyword arguments (dtype)
    if (positional_args_count == 2) dtype_obj = args[1];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "alpha") == 0 && !alpha_obj) { alpha_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "beta") == 0 && !beta_obj) { beta_obj = value; }
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

    // Convert `a_obj` to `a_buffer` and to `a_parsed`.
    nk_buffer_backing_t a_parsed_backing, out_parsed_backing;
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed, &a_parsed_backing, dtype)) goto cleanup;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed, &out_parsed_backing, nk_dtype_unknown_k))
        goto cleanup;

    // Check dimensions
    if (a_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be vectors");
        goto cleanup;
    }
    if (out_obj && a_parsed.cols != out_parsed.cols) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.dtype == nk_dtype_unknown_k || (out_obj && out_parsed.dtype == nk_dtype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have known dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Convert `alpha_obj` to `alpha_buf` and `beta_obj` to `beta_buf`
    nk_scalar_buffer_t alpha_buf, beta_buf;
    {
        nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(dtype);
        alpha_buf.f64 = 1.0, beta_buf.f64 = 0.0;
        if (alpha_obj) {
            if (!py_number_to_nk_scalar_buffer(alpha_obj, &alpha_buf, scalar_dtype)) goto cleanup;
        }
        else nk_scalar_buffer_from_f64(&alpha_buf.f64, &alpha_buf, scalar_dtype);
        if (beta_obj) {
            if (!py_number_to_nk_scalar_buffer(beta_obj, &beta_buf, scalar_dtype)) goto cleanup;
        }
        else nk_scalar_buffer_from_f64(&beta_buf.f64, &beta_buf, scalar_dtype);
    }

    // Look up the kernel and the capability
    nk_each_scale_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_kernel_kind_t const kernel_kind = nk_kernel_each_scale_k;
    nk_find_kernel_punned(kernel_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &capability);
    if (!kernel || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported kernel '%c' and dtype combination across vectors ('%s'/'%s') and `dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(a_parsed.dtype), //
            nk_dtype_to_pybuffer_typestr(dtype), nk_dtype_to_pybuffer_typestr(dtype));
        goto cleanup;
    }

    char *result_data = NULL;

    // nk.scale(a, alpha=2.0, beta=1.0) → returns new Tensor with α·a + β
    if (!out_obj) {
        Py_ssize_t out_shape[1] = {a_parsed.cols};
        Tensor *result_tensor = Tensor_new(dtype, 1, out_shape);
        if (!result_tensor) goto cleanup;
        return_obj = (PyObject *)result_tensor;
        result_data = result_tensor->data;
    }
    // nk.scale(a, alpha=2.0, out=result) → writes into provided buffer, returns None
    else {
        result_data = out_parsed.data;
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    {
        PyThreadState *gil = PyEval_SaveThread();
        kernel(a_parsed.data, a_parsed.cols, &alpha_buf, &beta_buf, result_data);
        PyEval_RestoreThread(gil);
    }
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

char const doc_add[] =                                                                         //
    "Element-wise addition of two vectors or a vector and a scalar.\n\n"                       //
    "Parameters:\n"                                                                            //
    "    a (Union[Tensor, float, int]): First operand (vector or scalar).\n"                   //
    "    b (Union[Tensor, float, int]): Second operand (vector or scalar).\n"                  //
    "    out (Tensor, optional): Output buffer for the result.\n"                              //
    "    a_dtype (Union[IntegralType, FloatType], optional): Override dtype for `a`.\n"        //
    "    b_dtype (Union[IntegralType, FloatType], optional): Override dtype for `b`.\n"        //
    "    out_dtype (Union[IntegralType, FloatType], optional): Override dtype for output.\n\n" //
    "Returns:\n"                                                                               //
    "    Tensor: The sum if `out` is not provided.\n"                                          //
    "    None: If `out` is provided (in-place operation).\n\n"                                 //
    "Equivalent to: `a + b`.\n"                                                                //
    "Signature:\n"                                                                             //
    "    >>> def add(a, b, /, *, out, a_dtype, b_dtype, out_dtype) -> Optional[Tensor]: ...";

/** @brief Handle scalar + array addition: result = 1 * array + scalar. */
static PyObject *add_scalar_array(PyObject *array_obj, PyObject *scalar_obj, PyObject *out_obj,
                                  PyObject *out_dtype_obj) {
    PyObject *return_obj = NULL;
    char *cast_staging = NULL;
    Py_buffer a_buffer, out_buffer;
    nk_buffer_backing_t a_backing, out_backing;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    if (!nk_get_buffer(array_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &a_backing)) return NULL;
    if (a_buffer.ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d exceeds maximum supported rank %d", a_buffer.ndim,
                     NK_TENSOR_MAX_RANK);
        goto cleanup;
    }
    if (out_obj && !nk_get_buffer(out_obj, &out_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &out_backing)) goto cleanup;
    if (out_obj && !buffers_shapes_match(&a_buffer, &out_buffer)) goto cleanup;

    nk_dtype_t dtype = resolve_nk_dtype_in_py_buffer(&a_buffer);
    if (out_dtype_obj) { dtype = py_object_to_nk_dtype(out_dtype_obj); }
    if (dtype == nk_dtype_unknown_k) goto cleanup;

    nk_each_scale_punned_t scale_kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_each_scale_k, dtype, static_capabilities, (nk_kernel_punned_t *)&scale_kernel,
                          &capability);
    if (!scale_kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No scale kernel for dtype '%s'", nk_dtype_name(dtype));
        goto cleanup;
    }

    nk_scalar_buffer_t alpha_buf, beta_buf;
    alpha_buf.f64 = 1.0;
    nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(dtype);
    nk_scalar_buffer_from_f64(&alpha_buf.f64, &alpha_buf, scalar_dtype);
    if (!py_number_to_nk_scalar_buffer(scalar_obj, &beta_buf, scalar_dtype)) goto cleanup;

    size_t const element_size = nk_dtype_bytes_per_value(dtype);
    size_t total_elements = 1;
    for (int dim = 0; dim < a_buffer.ndim; dim++) total_elements *= (size_t)a_buffer.shape[dim];

    char *result_data = NULL;
    Py_ssize_t result_strides[NK_TENSOR_MAX_RANK];
    nk_dtype_t out_buf_dtype = nk_dtype_unknown_k;
    Py_buffer const *input_bufs[] = {&a_buffer};
    int contiguous_tail = shared_contiguous_tail_dimensions(input_bufs, 1, a_buffer.ndim);

    // nk.add(np.int16([1,2,3]), 5) → returns new Tensor(int16)
    if (!out_obj) {
        Tensor *result_tensor = Tensor_new(dtype, (size_t)a_buffer.ndim, a_buffer.shape);
        if (!result_tensor) goto cleanup;
        return_obj = (PyObject *)result_tensor;
        result_data = result_tensor->data;
        compute_contiguous_strides((size_t)a_buffer.ndim, a_buffer.shape, element_size, result_strides);
    }
    // nk.add(np.int16([1,2,3]), 5, out=np.zeros(3, dtype=np.float64))
    // → kernel computes int16, then casts int16→float64 into output buffer
    else if ((out_buf_dtype = resolve_nk_dtype_in_py_buffer(&out_buffer)) != nk_dtype_unknown_k &&
             out_buf_dtype != dtype) {
        cast_staging = PyMem_Malloc(total_elements * element_size + NK_TENSOR_PADDING_);
        if (!cast_staging) {
            PyErr_NoMemory();
            goto cleanup;
        }
        result_data = cast_staging;
        compute_contiguous_strides((size_t)a_buffer.ndim, a_buffer.shape, element_size, result_strides);
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }
    // nk.add(np.float32([1,2,3]), 5.0, out=np.zeros(3, dtype=np.float32))
    // → kernel writes float32 directly into output buffer; output may be non-contiguous
    else {
        result_data = out_buffer.buf;
        for (int dim = 0; dim < a_buffer.ndim; ++dim) result_strides[dim] = out_buffer.strides[dim];
        Py_buffer const *both_bufs[] = {&a_buffer, &out_buffer};
        contiguous_tail = shared_contiguous_tail_dimensions(both_bufs, 2, a_buffer.ndim);
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    PyThreadState *gil = PyEval_SaveThread();
    each_scale_recursive(scale_kernel, a_buffer.buf, result_data, &alpha_buf, &beta_buf, //
                         a_buffer.shape, a_buffer.strides, result_strides,               //
                         a_buffer.ndim, contiguous_tail);
    if (cast_staging) { nk_cast(cast_staging, dtype, (nk_size_t)total_elements, out_buffer.buf, out_buf_dtype); }
    PyEval_RestoreThread(gil);

cleanup:
    if (cast_staging) PyMem_Free(cast_staging);
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

/** @brief Handle array + array addition using sum kernel with dtype promotion. */
static PyObject *add_array_array(PyObject *a_obj, PyObject *b_obj, PyObject *out_obj, PyObject *out_dtype_obj) {
    PyObject *return_obj = NULL;
    char *a_promoted = NULL;
    char *b_promoted = NULL;
    char *cast_staging = NULL;
    int a_needs_free = 0, b_needs_free = 0;

    Py_buffer a_buffer, b_buffer, out_buffer;
    nk_buffer_backing_t a_backing, b_backing, out_backing;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    if (!nk_get_buffer(a_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &a_backing)) return NULL;
    if (a_buffer.ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d exceeds maximum supported rank %d", a_buffer.ndim,
                     NK_TENSOR_MAX_RANK);
        goto cleanup;
    }
    if (!nk_get_buffer(b_obj, &b_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &b_backing)) goto cleanup;
    if (out_obj && !nk_get_buffer(out_obj, &out_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &out_backing)) goto cleanup;

    if (!buffers_shapes_match(&a_buffer, &b_buffer)) goto cleanup;
    if (out_obj && !buffers_shapes_match(&a_buffer, &out_buffer)) goto cleanup;

    nk_dtype_t a_dtype = resolve_nk_dtype_in_py_buffer(&a_buffer);
    nk_dtype_t b_dtype = resolve_nk_dtype_in_py_buffer(&b_buffer);
    if (a_dtype == nk_dtype_unknown_k || b_dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Unsupported input dtype");
        goto cleanup;
    }

    nk_dtype_t dtype;
    if (a_dtype == b_dtype) { dtype = a_dtype; }
    else {
        dtype = nk_dtype_promote(a_dtype, b_dtype);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_TypeError, "Cannot promote dtypes '%s' and '%s'", nk_dtype_name(a_dtype),
                         nk_dtype_name(b_dtype));
            goto cleanup;
        }
    }

    if (out_dtype_obj) { dtype = py_object_to_nk_dtype(out_dtype_obj); }
    if (dtype == nk_dtype_unknown_k) goto cleanup;

    nk_each_sum_punned_t sum_kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_each_sum_k, dtype, static_capabilities, (nk_kernel_punned_t *)&sum_kernel,
                          &capability);
    if (!sum_kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No sum kernel for dtype '%s'", nk_dtype_name(dtype));
        goto cleanup;
    }

    int const num_dims = a_buffer.ndim;
    size_t total_elements = 1;
    for (int dim = 0; dim < num_dims; dim++) total_elements *= (size_t)a_buffer.shape[dim];

    a_promoted = ensure_contiguous_buffer(a_buffer.buf, a_dtype, dtype, num_dims, a_buffer.shape, a_buffer.strides,
                                          total_elements, &a_needs_free);
    if (!a_promoted) goto cleanup;
    b_promoted = ensure_contiguous_buffer(b_buffer.buf, b_dtype, dtype, num_dims, a_buffer.shape, b_buffer.strides,
                                          total_elements, &b_needs_free);
    if (!b_promoted) goto cleanup;

    size_t const element_size = nk_dtype_bytes_per_value(dtype);
    Py_ssize_t promoted_strides[NK_TENSOR_MAX_RANK];
    compute_contiguous_strides((size_t)num_dims, a_buffer.shape, element_size, promoted_strides);

    char *result_data = NULL;
    Py_ssize_t result_strides[NK_TENSOR_MAX_RANK];
    nk_dtype_t out_buf_dtype = nk_dtype_unknown_k;
    int contiguous_tail = num_dims;

    // nk.add(np.int16([1,2,3]), np.uint16([4,5,6]))
    // → promotes to int32, returns new Tensor(int32)
    if (!out_obj) {
        Tensor *result_tensor = Tensor_new(dtype, (size_t)num_dims, a_buffer.shape);
        if (!result_tensor) goto cleanup;
        return_obj = (PyObject *)result_tensor;
        result_data = result_tensor->data;
        memcpy(result_strides, promoted_strides, num_dims * sizeof(Py_ssize_t));
    }
    // nk.add(np.int16([1,2,3]), np.uint16([4,5,6]), out=np.zeros(3, dtype=np.float64))
    // → kernel computes int32, then casts int32→float64 into output buffer
    else if ((out_buf_dtype = resolve_nk_dtype_in_py_buffer(&out_buffer)) != nk_dtype_unknown_k &&
             out_buf_dtype != dtype) {
        cast_staging = PyMem_Malloc(total_elements * element_size + NK_TENSOR_PADDING_);
        if (!cast_staging) {
            PyErr_NoMemory();
            goto cleanup;
        }
        result_data = cast_staging;
        memcpy(result_strides, promoted_strides, num_dims * sizeof(Py_ssize_t));
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }
    // nk.add(np.float32([1,2,3]), np.float32([4,5,6]), out=np.zeros(3, dtype=np.float32))
    // → kernel writes float32 directly into output buffer; output may be non-contiguous
    else {
        result_data = out_buffer.buf;
        for (int dim = 0; dim < num_dims; dim++) result_strides[dim] = out_buffer.strides[dim];
        Py_buffer const *out_bufs[] = {&out_buffer};
        contiguous_tail = shared_contiguous_tail_dimensions(out_bufs, 1, num_dims);
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    PyThreadState *gil = PyEval_SaveThread();
    each_sum_recursive(sum_kernel, a_promoted, b_promoted, result_data, a_buffer.shape, promoted_strides,
                       promoted_strides, result_strides, num_dims, contiguous_tail);
    if (cast_staging) { nk_cast(cast_staging, dtype, (nk_size_t)total_elements, out_buffer.buf, out_buf_dtype); }
    PyEval_RestoreThread(gil);

cleanup:
    if (cast_staging) PyMem_Free(cast_staging);
    if (a_needs_free) PyMem_Free(a_promoted);
    if (b_needs_free) PyMem_Free(b_promoted);
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

PyObject *api_add(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                  PyObject *args_names_tuple) {
    nk_unused_(self);

    PyObject *a_obj = NULL, *b_obj = NULL;
    PyObject *out_obj = NULL, *a_dtype_obj = NULL, *b_dtype_obj = NULL, *out_dtype_obj = NULL;

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-6 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "Only first 2 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    a_obj = args[0];
    b_obj = args[1];

    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "a_dtype") == 0 && !a_dtype_obj) { a_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "b_dtype") == 0 && !b_dtype_obj) { b_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    int a_is_scalar = py_object_is_scalar(a_obj);
    int b_is_scalar = py_object_is_scalar(b_obj);

    if (a_is_scalar && b_is_scalar) {
        PyErr_SetString(PyExc_TypeError, "At least one argument must be an array");
        return NULL;
    }

    // nk.add(5.0, np.float32([1,2,3])) → scalar + array
    if (a_is_scalar || b_is_scalar) {
        PyObject *array_obj = a_is_scalar ? b_obj : a_obj;
        PyObject *scalar_obj = a_is_scalar ? a_obj : b_obj;
        return add_scalar_array(array_obj, scalar_obj, out_obj, out_dtype_obj);
    }

    // nk.add(np.float32([1,2,3]), np.float32([4,5,6])) → array + array
    return add_array_array(a_obj, b_obj, out_obj, out_dtype_obj);
}

char const doc_multiply[] =                                                                    //
    "Element-wise multiplication of two vectors or a vector and a scalar.\n\n"                 //
    "Parameters:\n"                                                                            //
    "    a (Union[Tensor, float, int]): First operand (vector or scalar).\n"                   //
    "    b (Union[Tensor, float, int]): Second operand (vector or scalar).\n"                  //
    "    out (Tensor, optional): Output buffer for the result.\n"                              //
    "    a_dtype (Union[IntegralType, FloatType], optional): Override dtype for `a`.\n"        //
    "    b_dtype (Union[IntegralType, FloatType], optional): Override dtype for `b`.\n"        //
    "    out_dtype (Union[IntegralType, FloatType], optional): Override dtype for output.\n\n" //
    "Returns:\n"                                                                               //
    "    Tensor: The product if `out` is not provided.\n"                                      //
    "    None: If `out` is provided (in-place operation).\n\n"                                 //
    "Equivalent to: `a * b`.\n"                                                                //
    "Signature:\n"                                                                             //
    "    >>> def multiply(a, b, /, *, out, a_dtype, b_dtype, out_dtype) -> Optional[Tensor]: ...";

/** @brief Handle scalar * array multiplication: result = scalar * array + 0. */
static PyObject *multiply_scalar_array(PyObject *array_obj, PyObject *scalar_obj, PyObject *out_obj,
                                       PyObject *out_dtype_obj) {
    PyObject *return_obj = NULL;
    char *cast_staging = NULL;
    Py_buffer a_buffer, out_buffer;
    nk_buffer_backing_t a_backing, out_backing;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    if (!nk_get_buffer(array_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &a_backing)) return NULL;
    if (a_buffer.ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d exceeds maximum supported rank %d", a_buffer.ndim,
                     NK_TENSOR_MAX_RANK);
        goto cleanup;
    }
    if (out_obj && !nk_get_buffer(out_obj, &out_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &out_backing)) goto cleanup;
    if (out_obj && !buffers_shapes_match(&a_buffer, &out_buffer)) goto cleanup;

    nk_dtype_t dtype = resolve_nk_dtype_in_py_buffer(&a_buffer);
    if (out_dtype_obj) { dtype = py_object_to_nk_dtype(out_dtype_obj); }
    if (dtype == nk_dtype_unknown_k) goto cleanup;

    nk_each_scale_punned_t scale_kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_each_scale_k, dtype, static_capabilities, (nk_kernel_punned_t *)&scale_kernel,
                          &capability);
    if (!scale_kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No scale kernel for dtype '%s'", nk_dtype_name(dtype));
        goto cleanup;
    }

    nk_scalar_buffer_t alpha_buf, beta_buf;
    beta_buf.f64 = 0.0;
    nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(dtype);
    if (!py_number_to_nk_scalar_buffer(scalar_obj, &alpha_buf, scalar_dtype)) goto cleanup;
    nk_scalar_buffer_from_f64(&beta_buf.f64, &beta_buf, scalar_dtype);

    size_t const element_size = nk_dtype_bytes_per_value(dtype);
    size_t total_elements = 1;
    for (int dim = 0; dim < a_buffer.ndim; dim++) total_elements *= (size_t)a_buffer.shape[dim];

    char *result_data = NULL;
    Py_ssize_t result_strides[NK_TENSOR_MAX_RANK];
    nk_dtype_t out_buf_dtype = nk_dtype_unknown_k;
    Py_buffer const *input_bufs[] = {&a_buffer};
    int contiguous_tail = shared_contiguous_tail_dimensions(input_bufs, 1, a_buffer.ndim);

    // nk.multiply(np.float32([1,2,3]), 5.0) → returns new Tensor(float32)
    if (!out_obj) {
        Tensor *result_tensor = Tensor_new(dtype, (size_t)a_buffer.ndim, a_buffer.shape);
        if (!result_tensor) goto cleanup;
        return_obj = (PyObject *)result_tensor;
        result_data = result_tensor->data;
        compute_contiguous_strides((size_t)a_buffer.ndim, a_buffer.shape, element_size, result_strides);
    }
    // nk.multiply(np.int16([1,2,3]), 5, out=np.zeros(3, dtype=np.float64))
    // → kernel computes int16, then casts int16→float64 into output buffer
    else if ((out_buf_dtype = resolve_nk_dtype_in_py_buffer(&out_buffer)) != nk_dtype_unknown_k &&
             out_buf_dtype != dtype) {
        cast_staging = PyMem_Malloc(total_elements * element_size + NK_TENSOR_PADDING_);
        if (!cast_staging) {
            PyErr_NoMemory();
            goto cleanup;
        }
        result_data = cast_staging;
        compute_contiguous_strides((size_t)a_buffer.ndim, a_buffer.shape, element_size, result_strides);
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }
    // nk.multiply(np.float32([1,2,3]), 5.0, out=np.zeros(3, dtype=np.float32))
    // → kernel writes float32 directly into output buffer; output may be non-contiguous
    else {
        result_data = out_buffer.buf;
        for (int dim = 0; dim < a_buffer.ndim; ++dim) result_strides[dim] = out_buffer.strides[dim];
        Py_buffer const *both_bufs[] = {&a_buffer, &out_buffer};
        contiguous_tail = shared_contiguous_tail_dimensions(both_bufs, 2, a_buffer.ndim);
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    PyThreadState *gil = PyEval_SaveThread();
    each_scale_recursive(scale_kernel, a_buffer.buf, result_data, &alpha_buf, &beta_buf, //
                         a_buffer.shape, a_buffer.strides, result_strides,               //
                         a_buffer.ndim, contiguous_tail);
    if (cast_staging) { nk_cast(cast_staging, dtype, (nk_size_t)total_elements, out_buffer.buf, out_buf_dtype); }
    PyEval_RestoreThread(gil);

cleanup:
    if (cast_staging) PyMem_Free(cast_staging);
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

/** @brief Handle array * array multiplication using fma kernel with dtype promotion. */
static PyObject *multiply_array_array(PyObject *a_obj, PyObject *b_obj, PyObject *out_obj, PyObject *out_dtype_obj) {
    PyObject *return_obj = NULL;
    char *a_promoted = NULL;
    char *b_promoted = NULL;
    char *cast_staging = NULL;
    int a_needs_free = 0, b_needs_free = 0;

    Py_buffer a_buffer, b_buffer, out_buffer;
    nk_buffer_backing_t a_backing, b_backing, out_backing;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    if (!nk_get_buffer(a_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &a_backing)) return NULL;
    if (a_buffer.ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d exceeds maximum supported rank %d", a_buffer.ndim,
                     NK_TENSOR_MAX_RANK);
        goto cleanup;
    }
    if (!nk_get_buffer(b_obj, &b_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &b_backing)) goto cleanup;
    if (out_obj && !nk_get_buffer(out_obj, &out_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &out_backing)) goto cleanup;

    if (!buffers_shapes_match(&a_buffer, &b_buffer)) goto cleanup;
    if (out_obj && !buffers_shapes_match(&a_buffer, &out_buffer)) goto cleanup;

    nk_dtype_t a_dtype = resolve_nk_dtype_in_py_buffer(&a_buffer);
    nk_dtype_t b_dtype = resolve_nk_dtype_in_py_buffer(&b_buffer);
    if (a_dtype == nk_dtype_unknown_k || b_dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Unsupported input dtype");
        goto cleanup;
    }

    nk_dtype_t dtype;
    if (a_dtype == b_dtype) { dtype = a_dtype; }
    else {
        dtype = nk_dtype_promote(a_dtype, b_dtype);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_TypeError, "Cannot promote dtypes '%s' and '%s'", nk_dtype_name(a_dtype),
                         nk_dtype_name(b_dtype));
            goto cleanup;
        }
    }

    if (out_dtype_obj) { dtype = py_object_to_nk_dtype(out_dtype_obj); }
    if (dtype == nk_dtype_unknown_k) goto cleanup;

    nk_each_fma_punned_t fma_kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_each_fma_k, dtype, static_capabilities, (nk_kernel_punned_t *)&fma_kernel,
                          &capability);
    if (!fma_kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No fma kernel for dtype '%s'", nk_dtype_name(dtype));
        goto cleanup;
    }

    nk_scalar_buffer_t alpha_buf, beta_buf;
    alpha_buf.f64 = 1.0, beta_buf.f64 = 0.0;
    nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(dtype);
    nk_scalar_buffer_from_f64(&alpha_buf.f64, &alpha_buf, scalar_dtype);
    nk_scalar_buffer_from_f64(&beta_buf.f64, &beta_buf, scalar_dtype);

    int const num_dims = a_buffer.ndim;
    size_t total_elements = 1;
    for (int dim = 0; dim < num_dims; dim++) total_elements *= (size_t)a_buffer.shape[dim];

    a_promoted = ensure_contiguous_buffer(a_buffer.buf, a_dtype, dtype, num_dims, a_buffer.shape, a_buffer.strides,
                                          total_elements, &a_needs_free);
    if (!a_promoted) goto cleanup;
    b_promoted = ensure_contiguous_buffer(b_buffer.buf, b_dtype, dtype, num_dims, a_buffer.shape, b_buffer.strides,
                                          total_elements, &b_needs_free);
    if (!b_promoted) goto cleanup;

    size_t const element_size = nk_dtype_bytes_per_value(dtype);
    Py_ssize_t promoted_strides[NK_TENSOR_MAX_RANK];
    compute_contiguous_strides((size_t)num_dims, a_buffer.shape, element_size, promoted_strides);

    char *result_data = NULL;
    Py_ssize_t result_strides[NK_TENSOR_MAX_RANK];
    nk_dtype_t out_buf_dtype = nk_dtype_unknown_k;
    int contiguous_tail = num_dims;

    // nk.multiply(np.int16([1,2,3]), np.uint16([4,5,6]))
    // → promotes to int32, returns new Tensor(int32), zero-filled to prevent 0*NaN=NaN
    if (!out_obj) {
        Tensor *result_tensor = Tensor_new(dtype, (size_t)num_dims, a_buffer.shape);
        if (!result_tensor) goto cleanup;
        memset(result_tensor->data, 0, total_elements * element_size); // prevent 0*NaN=NaN
        return_obj = (PyObject *)result_tensor;
        result_data = result_tensor->data;
        memcpy(result_strides, promoted_strides, num_dims * sizeof(Py_ssize_t));
    }
    // nk.multiply(np.int16([1,2,3]), np.uint16([4,5,6]), out=np.zeros(3, dtype=np.float64))
    // → kernel computes int32, then casts int32→float64 into output buffer
    else if ((out_buf_dtype = resolve_nk_dtype_in_py_buffer(&out_buffer)) != nk_dtype_unknown_k &&
             out_buf_dtype != dtype) {
        cast_staging = PyMem_Malloc(total_elements * element_size + NK_TENSOR_PADDING_);
        if (!cast_staging) {
            PyErr_NoMemory();
            goto cleanup;
        }
        result_data = cast_staging;
        memset(result_data, 0, total_elements * element_size); // prevent 0*NaN=NaN
        memcpy(result_strides, promoted_strides, num_dims * sizeof(Py_ssize_t));
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }
    // nk.multiply(np.float32([1,2,3]), np.float32([4,5,6]), out=np.zeros(3, dtype=np.float32))
    // → kernel writes float32 directly into output buffer; output may be non-contiguous
    else {
        result_data = out_buffer.buf;
        for (int dim = 0; dim < num_dims; dim++) result_strides[dim] = out_buffer.strides[dim];
        Py_buffer const *out_bufs[] = {&out_buffer};
        contiguous_tail = shared_contiguous_tail_dimensions(out_bufs, 1, num_dims);
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    PyThreadState *gil = PyEval_SaveThread();
    each_fma_recursive(fma_kernel, a_promoted, b_promoted, result_data, result_data, &alpha_buf, &beta_buf,
                       a_buffer.shape, promoted_strides, promoted_strides, result_strides, result_strides, num_dims,
                       contiguous_tail);
    if (cast_staging) { nk_cast(cast_staging, dtype, (nk_size_t)total_elements, out_buffer.buf, out_buf_dtype); }
    PyEval_RestoreThread(gil);

cleanup:
    if (cast_staging) PyMem_Free(cast_staging);
    if (a_needs_free) PyMem_Free(a_promoted);
    if (b_needs_free) PyMem_Free(b_promoted);
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

PyObject *api_multiply(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                       PyObject *args_names_tuple) {
    nk_unused_(self);

    PyObject *a_obj = NULL, *b_obj = NULL;
    PyObject *out_obj = NULL, *a_dtype_obj = NULL, *b_dtype_obj = NULL, *out_dtype_obj = NULL;

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-6 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "Only first 2 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    a_obj = args[0];
    b_obj = args[1];

    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "a_dtype") == 0 && !a_dtype_obj) { a_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "b_dtype") == 0 && !b_dtype_obj) { b_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    int a_is_scalar = py_object_is_scalar(a_obj);
    int b_is_scalar = py_object_is_scalar(b_obj);

    if (a_is_scalar && b_is_scalar) {
        PyErr_SetString(PyExc_TypeError, "At least one argument must be an array");
        return NULL;
    }

    // nk.multiply(5.0, np.float32([1,2,3])) → scalar * array
    if (a_is_scalar || b_is_scalar) {
        PyObject *array_obj = a_is_scalar ? b_obj : a_obj;
        PyObject *scalar_obj = a_is_scalar ? a_obj : b_obj;
        return multiply_scalar_array(array_obj, scalar_obj, out_obj, out_dtype_obj);
    }

    // nk.multiply(np.float32([1,2,3]), np.float32([4,5,6])) → array * array
    return multiply_array_array(a_obj, b_obj, out_obj, out_dtype_obj);
}

char const doc_sin[] =                                                                                 //
    "Element-wise trigonometric sine.\n\n"                                                             //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): Input vector of angles in radians.\n"                                             //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    out (Tensor, optional): Vector for resulting values.\n\n"                                     //
    "Returns:\n"                                                                                       //
    "    Tensor: The sine values if `out` is not provided.\n"                                          //
    "    None: If `out` is provided.\n\n"                                                              //
    "Signature:\n"                                                                                     //
    "    >>> def sin(a, /, dtype, *, out) -> Optional[Tensor]: ...";

char const doc_cos[] =                                                                                 //
    "Element-wise trigonometric cosine.\n\n"                                                           //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): Input vector of angles in radians.\n"                                             //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    out (Tensor, optional): Vector for resulting values.\n\n"                                     //
    "Returns:\n"                                                                                       //
    "    Tensor: The cosine values if `out` is not provided.\n"                                        //
    "    None: If `out` is provided.\n\n"                                                              //
    "Signature:\n"                                                                                     //
    "    >>> def cos(a, /, dtype, *, out) -> Optional[Tensor]: ...";

char const doc_atan[] =                                                                                //
    "Element-wise trigonometric arctangent.\n\n"                                                       //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): Input vector of values.\n"                                                        //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    out (Tensor, optional): Vector for resulting angles in radians.\n\n"                          //
    "Returns:\n"                                                                                       //
    "    Tensor: The arctangent values if `out` is not provided.\n"                                    //
    "    None: If `out` is provided.\n\n"                                                              //
    "Signature:\n"                                                                                     //
    "    >>> def atan(a, /, dtype, *, out) -> Optional[Tensor]: ...";

static PyObject *implement_trigonometry(nk_kernel_kind_t kernel_kind, PyObject *const *args,
                                        Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 3 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only

    // Once parsed, the arguments will be stored in these variables:

    nk_dtype_t dtype = nk_dtype_unknown_k;

    Py_buffer a_buffer, out_buffer;
    MatrixOrVectorView a_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 1 || args_count > 3) {
        PyErr_Format(PyExc_TypeError, "Function expects 1-3 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "Only first 2 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only argument (input array)
    a_obj = args[0];

    // Positional or keyword argument (dtype)
    if (positional_args_count == 2) dtype_obj = args[1];

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

    // Convert `a_obj` to `a_buffer` and to `a_parsed`
    nk_buffer_backing_t a_parsed_backing, out_parsed_backing;
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed, &a_parsed_backing, dtype)) goto cleanup;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed, &out_parsed_backing, nk_dtype_unknown_k))
        goto cleanup;

    // Check dimensions
    if (a_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be vectors");
        goto cleanup;
    }
    if (out_obj && a_parsed.cols != out_parsed.cols) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.dtype == nk_dtype_unknown_k || (out_obj && out_parsed.dtype == nk_dtype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError, "Input tensor must have a known dtype, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Look up the kernel and the capability
    nk_kernel_trigonometry_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(kernel_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &capability);
    if (!kernel || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported kernel '%c' and dtype combination ('%s'/'%s') and `dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", nk_dtype_to_pybuffer_typestr(a_parsed.dtype), //
            nk_dtype_to_pybuffer_typestr(dtype), nk_dtype_to_pybuffer_typestr(dtype));
        goto cleanup;
    }

    char *result_data = NULL;

    // nk.sin(np.float32([0, 1.57, 3.14])) → returns new Tensor with sine values
    if (!out_obj) {
        Py_ssize_t out_shape[1] = {(Py_ssize_t)a_parsed.cols};
        Tensor *result_tensor = Tensor_new(dtype, 1, out_shape);
        if (!result_tensor) { goto cleanup; }
        return_obj = (PyObject *)result_tensor;
        result_data = result_tensor->data;
    }
    // nk.sin(angles, out=result) → writes into provided buffer, returns None
    else {
        result_data = out_parsed.data;
        return_obj = Py_None;
        Py_INCREF(Py_None);
    }

    {
        PyThreadState *gil = PyEval_SaveThread();
        kernel(a_parsed.data, a_parsed.cols, result_data);
        PyEval_RestoreThread(gil);
    }
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

PyObject *api_sin(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                  PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_each_sin_k, args, positional_args_count, args_names_tuple);
}

PyObject *api_cos(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                  PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_each_cos_k, args, positional_args_count, args_names_tuple);
}

PyObject *api_atan(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                   PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_each_atan_k, args, positional_args_count, args_names_tuple);
}
