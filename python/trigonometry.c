/**
 *  @brief Python bindings for the trigonometry family (sin/cos/atan) and RoPE.
 *  @file python/trigonometry.c
 *
 *  Trig entry points extracted from each.c: they build on the shared elementwise binding
 *  machinery (elementwise_prepare_out, each_unary_recursive) declared in tensor.h.
 */

#include "trigonometry.h"
#include "tensor.h"

char const doc_rope[] =                                                                             //
    "NeoX split-half rotary position embedding (RoPE).\n\n"                                         //
    "Rotates every channel pair (channel i against i+half_dim) of each head by the per-token\n"     //
    "angle grids. Bake position lookup and multi-axis (M-RoPE) assignment into the `[rows,\n"       //
    "half_dim]` cos/sin grids so a single call rotates the whole head.\n\n"                         //
    "Parameters:\n"                                                                                 //
    "    x (Tensor): `[rows, heads * 2*half_dim]`, float32/bfloat16/e4m3.\n"                        //
    "    cos, sin (Tensor): `[rows, half_dim]` float32 angle grids, shared across heads.\n"         //
    "    heads (int): Number of heads per token.\n"                                                 //
    "    half_dim (int): Half the head dimension.\n"                                                //
    "    out (Tensor, optional): Output, same shape/dtype as x; may alias x. Defaults to x.\n"      //
    "    input_scale (float, optional): Scale folded onto each loaded element, 1.0 by default.\n\n" //
    "Returns:\n"                                                                                    //
    "    None: The result is written into `out` (or `x` in place).\n\n"                             //
    "Signature:\n"                                                                                  //
    "    >>> def rope(x, cos, sin, heads, half_dim, /, *, out, input_scale) -> None: ...";

PyObject *api_rope(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                   PyObject *args_names_tuple) {
    nk_unused_(self);
    PyObject *x_obj = NULL, *cos_obj = NULL, *sin_obj = NULL, *heads_obj = NULL, *half_obj = NULL;
    PyObject *out_obj = NULL, *scale_obj = NULL;

    Py_buffer x_buffer, y_buffer, cos_buffer, sin_buffer;
    nk_buffer_backing_t x_backing, y_backing, cos_backing, sin_backing;
    memset(&x_buffer, 0, sizeof(Py_buffer));
    memset(&y_buffer, 0, sizeof(Py_buffer));
    memset(&cos_buffer, 0, sizeof(Py_buffer));
    memset(&sin_buffer, 0, sizeof(Py_buffer));
    int got_x = 0, got_y = 0, got_cos = 0, got_sin = 0;

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 5 || args_count > 7) {
        PyErr_Format(PyExc_TypeError, "Function expects 5-7 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 5) {
        PyErr_Format(PyExc_TypeError, "Only first 5 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }
    PyObject *positional[5] = {NULL, NULL, NULL, NULL, NULL};
    for (Py_ssize_t i = 0; i < positional_args_count; ++i) positional[i] = args[i];
    x_obj = positional[0], cos_obj = positional[1], sin_obj = positional[2];
    heads_obj = positional[3], half_obj = positional[4];
    for (Py_ssize_t k = 0, p = positional_args_count; k < args_names_count; ++p, ++k) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, k);
        PyObject *const value = args[p];
        if (PyUnicode_CompareWithASCIIString(key, "heads") == 0 && !heads_obj) heads_obj = value;
        else if (PyUnicode_CompareWithASCIIString(key, "half_dim") == 0 && !half_obj) half_obj = value;
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) out_obj = value;
        else if (PyUnicode_CompareWithASCIIString(key, "input_scale") == 0 && !scale_obj) scale_obj = value;
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }
    if (!x_obj || !cos_obj || !sin_obj || !heads_obj || !half_obj) {
        PyErr_SetString(PyExc_TypeError, "rope requires x, cos, sin, heads, half_dim");
        return NULL;
    }

    long heads_l = PyLong_AsLong(heads_obj), half_l = PyLong_AsLong(half_obj);
    if (PyErr_Occurred()) return NULL;
    if (heads_l <= 0 || half_l <= 0) {
        PyErr_SetString(PyExc_ValueError, "heads and half_dim must be positive");
        return NULL;
    }
    nk_f32_t input_scale = 1.0f;
    if (scale_obj) {
        double s = PyFloat_AsDouble(scale_obj);
        if (PyErr_Occurred()) return NULL;
        input_scale = (nk_f32_t)s;
    }

    int const x_flags = out_obj ? (PyBUF_STRIDES | PyBUF_FORMAT) : (PyBUF_WRITABLE | PyBUF_STRIDES | PyBUF_FORMAT);
    if (!nk_get_buffer(x_obj, &x_buffer, x_flags, &x_backing)) return NULL;
    got_x = 1;
    if (x_buffer.ndim < 1 || x_buffer.ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d unsupported", x_buffer.ndim);
        goto cleanup;
    }
    nk_dtype_t dtype = resolve_nk_dtype_in_py_buffer(&x_buffer);
    if (dtype != nk_f32_k && dtype != nk_bf16_k && dtype != nk_e4m3_k) {
        PyErr_Format(PyExc_TypeError, "rope supports f32, bf16, e4m3; got '%s'", nk_dtype_name(dtype));
        goto cleanup;
    }
    int const ndim = x_buffer.ndim;
    size_t const elem = nk_dtype_bytes_per_value(dtype);
    if ((size_t)x_buffer.strides[ndim - 1] != elem) {
        PyErr_SetString(PyExc_ValueError, "rope requires the last axis to be contiguous");
        goto cleanup;
    }
    if ((nk_size_t)x_buffer.shape[ndim - 1] < (nk_size_t)(heads_l * 2 * half_l)) {
        PyErr_SetString(PyExc_ValueError, "last axis too small for heads * 2*half_dim");
        goto cleanup;
    }
    nk_size_t rows = 1;
    for (int d = 0; d < ndim - 1; ++d) rows *= (nk_size_t)x_buffer.shape[d];
    for (int d = 0; d + 2 < ndim; ++d) {
        if (x_buffer.strides[d] != x_buffer.shape[d + 1] * x_buffer.strides[d + 1]) {
            PyErr_SetString(PyExc_ValueError, "rope requires C-contiguous leading axes for rank > 2");
            goto cleanup;
        }
    }
    nk_size_t const x_row_stride = ndim >= 2 ? (nk_size_t)x_buffer.strides[ndim - 2] : 0;

    void *y_buf = x_buffer.buf;
    nk_size_t y_row_stride = x_row_stride;
    if (out_obj) {
        if (!nk_get_buffer(out_obj, &y_buffer, PyBUF_WRITABLE | PyBUF_STRIDES | PyBUF_FORMAT, &y_backing)) goto cleanup;
        got_y = 1;
        if (y_buffer.ndim != ndim || resolve_nk_dtype_in_py_buffer(&y_buffer) != dtype) {
            PyErr_SetString(PyExc_ValueError, "out must have the same shape and dtype as x");
            goto cleanup;
        }
        for (int d = 0; d < ndim; ++d)
            if (y_buffer.shape[d] != x_buffer.shape[d]) {
                PyErr_SetString(PyExc_ValueError, "out must have the same shape as x");
                goto cleanup;
            }
        if ((size_t)y_buffer.strides[ndim - 1] != elem) {
            PyErr_SetString(PyExc_ValueError, "out requires the last axis to be contiguous");
            goto cleanup;
        }
        y_buf = y_buffer.buf;
        y_row_stride = ndim >= 2 ? (nk_size_t)y_buffer.strides[ndim - 2] : 0;
    }

    if (!nk_get_buffer(cos_obj, &cos_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &cos_backing)) goto cleanup;
    got_cos = 1;
    if (!nk_get_buffer(sin_obj, &sin_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &sin_backing)) goto cleanup;
    got_sin = 1;
    if (resolve_nk_dtype_in_py_buffer(&cos_buffer) != nk_f32_k ||
        resolve_nk_dtype_in_py_buffer(&sin_buffer) != nk_f32_k) {
        PyErr_SetString(PyExc_TypeError, "cos and sin must be float32");
        goto cleanup;
    }
    if ((nk_size_t)(cos_buffer.len / (Py_ssize_t)sizeof(nk_f32_t)) < rows * (nk_size_t)half_l ||
        (nk_size_t)(sin_buffer.len / (Py_ssize_t)sizeof(nk_f32_t)) < rows * (nk_size_t)half_l) {
        PyErr_SetString(PyExc_ValueError, "cos and sin must each have at least rows * half_dim elements");
        goto cleanup;
    }

    nk_kernel_trig_rope_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_trig_rope_k, dtype, (nk_kernel_punned_t *)&kernel, &capability);
    if (!kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No rope kernel for dtype '%s'", nk_dtype_name(dtype));
        goto cleanup;
    }

    {
        PyThreadState *gil = PyEval_SaveThread();
        kernel(x_buffer.buf, y_buf, (nk_f32_t const *)cos_buffer.buf, (nk_f32_t const *)sin_buffer.buf, rows,
               (nk_size_t)heads_l, (nk_size_t)half_l, x_row_stride, y_row_stride, input_scale);
        PyEval_RestoreThread(gil);
    }
    if (got_x) PyBuffer_Release(&x_buffer);
    if (got_y) PyBuffer_Release(&y_buffer);
    if (got_cos) PyBuffer_Release(&cos_buffer);
    if (got_sin) PyBuffer_Release(&sin_buffer);
    Py_RETURN_NONE;
cleanup:
    if (got_x) PyBuffer_Release(&x_buffer);
    if (got_y) PyBuffer_Release(&y_buffer);
    if (got_cos) PyBuffer_Release(&cos_buffer);
    if (got_sin) PyBuffer_Release(&sin_buffer);
    return NULL;
}

char const doc_sin[] =                                                                                 //
    "Element-wise trigonometric sine.\n\n"                                                             //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): Input tensor of any rank, angles in radians.\n"                                   //
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
    "    a (Tensor): Input tensor of any rank, angles in radians.\n"                                   //
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
    nk_buffer_backing_t a_backing, out_backing;
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

    // Acquire the (N-D, possibly strided) input buffer.
    if (!nk_get_buffer(a_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &a_backing)) return NULL;
    if (a_buffer.ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d exceeds maximum supported rank %d", a_buffer.ndim,
                     NK_TENSOR_MAX_RANK);
        goto cleanup;
    }

    if (dtype == nk_dtype_unknown_k) dtype = resolve_nk_dtype_in_py_buffer(&a_buffer);
    if (dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Input tensor must have a known dtype, check with `X.__array_interface__`");
        goto cleanup;
    }

    // Look up the kernel and the capability
    nk_kernel_trig_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(kernel_kind, dtype, (nk_kernel_punned_t *)&kernel, &capability);
    if (!kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No '%c' kernel for dtype '%s'", kernel_kind, nk_dtype_name(dtype));
        goto cleanup;
    }

    char *result_data = NULL;
    Py_ssize_t result_strides[NK_TENSOR_MAX_RANK];
    int contiguous_tail = 0;
    Py_buffer const *inputs[] = {&a_buffer};
    if (!elementwise_prepare_out(out_obj, &out_buffer, &out_backing, inputs, 1, dtype, //
                                 &result_data, result_strides, &contiguous_tail, &return_obj))
        goto cleanup;

    {
        PyThreadState *gil = PyEval_SaveThread();
        each_unary_recursive(kernel, a_buffer.buf, result_data, //
                             a_buffer.shape, a_buffer.strides, result_strides, a_buffer.ndim, contiguous_tail);
        PyEval_RestoreThread(gil);
    }
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

PyObject *api_sin(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                  PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_trig_sin_k, args, positional_args_count, args_names_tuple);
}

PyObject *api_cos(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                  PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_trig_cos_k, args, positional_args_count, args_names_tuple);
}

PyObject *api_atan(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                   PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_trig_atan_k, args, positional_args_count, args_names_tuple);
}
