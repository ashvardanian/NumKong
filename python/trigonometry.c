/**
 *  @file python/trigonometry.c
 *  @author Ash Vardanian
 *  @date December 21, 2025
 *  @brief Python bindings for the trigonometry family: sin, cos and atan.
 *
 *  Trig entry points extracted from each.c: they build on the shared elementwise binding machinery,
 *  elementwise_prepare_out, each_unary_recursive, declared in tensor.h.
 */

#include "trigonometry.h"
#include "tensor.h"

char const doc_sin[] =                                                                                 //
    "Element-wise trigonometric sine.\n\n"                                                             //
    "Args:\n"                                                                                          //
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
    "Args:\n"                                                                                          //
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
    "Args:\n"                                                                                          //
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

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only

    // Once parsed, the arguments will be stored in these variables:

    nk_dtype_t dtype = nk_dtype_unknown_k;
    nk_capability_t capabilities = nk_cap_cpus_k;
    nk_stream_t stream = NULL;

    Py_buffer a_buffer, out_buffer;
    nk_buffer_backing_t a_backing, out_backing;
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
        else if (!parse_dispatch_keyword(key, value, &capabilities, &stream)) return NULL;
    }

    // Convert `dtype_obj` to `dtype`
    if (dtype_obj) {
        dtype = py_object_to_nk_dtype(dtype_obj);
        if (dtype == nk_dtype_unknown_k) return NULL;
    }

    // Acquire the (N-D, possibly strided) input buffer.
    if (!nk_get_buffer(a_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &a_backing)) return NULL;
    if (a_buffer.ndim > NUMKONG_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d exceeds maximum supported rank %d", a_buffer.ndim,
                     NUMKONG_TENSOR_MAX_RANK);
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
    nk_find_kernel_punned(kernel_kind, dtype, capabilities, (nk_kernel_punned_t *)&kernel, &capability);
    if (!kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No '%c' kernel for dtype '%s'", kernel_kind, nk_dtype_python_name(dtype));
        goto cleanup;
    }

    char *result_data = NULL;
    Py_ssize_t result_strides[NUMKONG_TENSOR_MAX_RANK];
    int contiguous_tail = 0;
    Py_buffer const *inputs[] = {&a_buffer};
    if (!elementwise_prepare_out(out_obj, &out_buffer, &out_backing, inputs, 1, dtype, //
                                 &result_data, result_strides, &contiguous_tail, &return_obj))
        goto cleanup;

    {
        PyThreadState *gil = PyEval_SaveThread();
        nk_status_t const status = each_unary_recursive(kernel, stream, a_buffer.buf, result_data, a_buffer.shape,
                                                        a_buffer.strides, result_strides, a_buffer.ndim,
                                                        contiguous_tail);
        PyEval_RestoreThread(gil);
        if (!check_status(status)) Py_CLEAR(return_obj);
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
