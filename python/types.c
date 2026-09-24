/**
 *  @file python/types.c
 *  @author Ash Vardanian
 *  @date December 30, 2025
 *  @brief Scalar types for NumKong low-precision types.
 *
 *  Pure Python C API implementation of bfloat16, float8_e4m3, and float8_e5m2. These work without
 *  NumPy, as standalone Python objects. All arithmetic is performed through float32 using the
 *  conversion functions from numkong/types.h.
 *
 *  @code{.py}
 *  nk.bfloat16(3.14)                    # create a bfloat16 scalar
 *  float(nk.bfloat16(3.14))             # convert back to a Python float
 *  nk.bfloat16(1.0) + nk.bfloat16(2.0)  # arithmetic
 *  @endcode
 */

#include "types.h"
#include <structmember.h>
#include <stdio.h>

static inline PyObject *richcompare_from_int(int comparison_result, int python_opcode) {
    int r;
    switch (python_opcode) {
    case Py_LT: r = comparison_result < 0; break;
    case Py_LE: r = comparison_result <= 0; break;
    case Py_EQ: r = comparison_result == 0; break;
    case Py_NE: r = comparison_result != 0; break;
    case Py_GT: r = comparison_result > 0; break;
    case Py_GE: r = comparison_result >= 0; break;
    default: Py_RETURN_NOTIMPLEMENTED;
    }
    if (r) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *NkBFloat16Scalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NkBFloat16ScalarObject *self = (NkBFloat16ScalarObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    double value = 0.0;
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs > 1) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_TypeError, "bfloat16() takes at most 1 argument");
        return NULL;
    }
    if (nargs == 1) {
        value = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 0));
        if (value == -1.0 && PyErr_Occurred()) {
            Py_DECREF(self);
            return NULL;
        }
    }

    nk_f32_t f32_val = (nk_f32_t)value;
    nk_f32_to_bf16(&f32_val, &self->value);
    return (PyObject *)self;
}

static PyObject *NkBFloat16Scalar_repr(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    char buf[64];
    snprintf(buf, sizeof(buf), "bfloat16(%.6g)", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkBFloat16Scalar_str(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkBFloat16Scalar_float(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    return PyFloat_FromDouble((double)f32_val);
}

static PyObject *NkBFloat16Scalar_int(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    return PyLong_FromDouble((double)f32_val);
}

static Py_hash_t NkBFloat16Scalar_hash(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
#if PY_VERSION_HEX >= 0x030A00F0
    return _Py_HashDouble(NULL, (double)f32_val);
#else
    return _Py_HashDouble((double)f32_val);
#endif
}

static PyObject *NkBFloat16Scalar_richcompare(PyObject *self, PyObject *other, int op) {
    if (Py_TYPE(self) == &NkBFloat16Scalar_Type && Py_TYPE(other) == &NkBFloat16Scalar_Type)
        return richcompare_from_int(
            nk_bf16_order(((NkBFloat16ScalarObject *)self)->value, ((NkBFloat16ScalarObject *)other)->value), op);

    nk_fui64_t a_fui64, b_fui64;
    nk_f32_t temporary_f32;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type)
        nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &temporary_f32), a_fui64.f = temporary_f32;
    else if (PyFloat_Check(self)) a_fui64.f = PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a_fui64.f = PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type)
        nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &temporary_f32), b_fui64.f = temporary_f32;
    else if (PyFloat_Check(other)) b_fui64.f = PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b_fui64.f = PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    return richcompare_from_int((a_fui64.f > b_fui64.f) - (a_fui64.f < b_fui64.f), op);
}

static PyObject *NkBFloat16Scalar_add(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type) nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type) nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a + b;
    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_sub(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type) nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type) nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a - b;
    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_mul(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type) nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type) nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a * b;
    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_div(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkBFloat16Scalar_Type) nk_bf16_to_f32(&((NkBFloat16ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkBFloat16Scalar_Type) nk_bf16_to_f32(&((NkBFloat16ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    if (b == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "bfloat16 division by zero");
        return NULL;
    }

    result = a / b;
    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_neg(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    f32_val = -f32_val;

    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&f32_val, &res->value);
    return (PyObject *)res;
}

static PyObject *NkBFloat16Scalar_pos(NkBFloat16ScalarObject *self) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *NkBFloat16Scalar_abs(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    if (f32_val < 0) f32_val = -f32_val;

    NkBFloat16ScalarObject *res = PyObject_New(NkBFloat16ScalarObject, &NkBFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_bf16(&f32_val, &res->value);
    return (PyObject *)res;
}

static int NkBFloat16Scalar_bool(NkBFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_bf16_to_f32(&self->value, &f32_val);
    return f32_val != 0.0f;
}

static PyNumberMethods NkBFloat16Scalar_as_number = {
    .nb_add = NkBFloat16Scalar_add,
    .nb_subtract = NkBFloat16Scalar_sub,
    .nb_multiply = NkBFloat16Scalar_mul,
    .nb_true_divide = NkBFloat16Scalar_div,
    .nb_negative = (unaryfunc)NkBFloat16Scalar_neg,
    .nb_positive = (unaryfunc)NkBFloat16Scalar_pos,
    .nb_absolute = (unaryfunc)NkBFloat16Scalar_abs,
    .nb_bool = (inquiry)NkBFloat16Scalar_bool,
    .nb_float = (unaryfunc)NkBFloat16Scalar_float,
    .nb_int = (unaryfunc)NkBFloat16Scalar_int,
};

static char const doc_bfloat16[] =                                                  //
    "BFloat16 (16-bit) floating-point scalar.\n"                                    //
    "\n"                                                                            //
    "Layout: sign(1) + exponent(8) + mantissa(7), bias=127.\n\n"                    //
    "Range: ±3.39e38, the same dynamic range as f32, epsilon at 1.0: ~7.81e-3.\n\n" //
    "32,514 of 65,280 finite values (49.8%%) fall in [-1, +1].\n\n"                 //
    "Wider dynamic range than float16 but lower precision (7 vs 10 mantissa bits).";

PyTypeObject NkBFloat16Scalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.bfloat16",
    .tp_basicsize = sizeof(NkBFloat16ScalarObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NkBFloat16Scalar_new,
    .tp_repr = (reprfunc)NkBFloat16Scalar_repr,
    .tp_str = (reprfunc)NkBFloat16Scalar_str,
    .tp_hash = (hashfunc)NkBFloat16Scalar_hash,
    .tp_richcompare = NkBFloat16Scalar_richcompare,
    .tp_as_number = &NkBFloat16Scalar_as_number,
    .tp_doc = doc_bfloat16,
};

static PyObject *NkFloat8E4M3Scalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NkFloat8E4M3ScalarObject *self = (NkFloat8E4M3ScalarObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    double value = 0.0;
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs > 1) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_TypeError, "float8_e4m3() takes at most 1 argument");
        return NULL;
    }
    if (nargs == 1) {
        value = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 0));
        if (value == -1.0 && PyErr_Occurred()) {
            Py_DECREF(self);
            return NULL;
        }
    }

    nk_f32_t f32_val = (nk_f32_t)value;
    nk_f32_to_e4m3(&f32_val, &self->value);
    return (PyObject *)self;
}

static PyObject *NkFloat8E4M3Scalar_repr(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    char buf[64];
    snprintf(buf, sizeof(buf), "float8_e4m3(%.6g)", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat8E4M3Scalar_str(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat8E4M3Scalar_float(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    return PyFloat_FromDouble((double)f32_val);
}

static PyObject *NkFloat8E4M3Scalar_int(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    return PyLong_FromDouble((double)f32_val);
}

static Py_hash_t NkFloat8E4M3Scalar_hash(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
#if PY_VERSION_HEX >= 0x030A00F0
    return _Py_HashDouble(NULL, (double)f32_val);
#else
    return _Py_HashDouble((double)f32_val);
#endif
}

static PyObject *NkFloat8E4M3Scalar_richcompare(PyObject *self, PyObject *other, int op) {
    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type && Py_TYPE(other) == &NkFloat8E4M3Scalar_Type)
        return richcompare_from_int(
            nk_e4m3_order(((NkFloat8E4M3ScalarObject *)self)->value, ((NkFloat8E4M3ScalarObject *)other)->value), op);

    nk_fui64_t a_fui64, b_fui64;
    nk_f32_t temporary_f32;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type)
        nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &temporary_f32), a_fui64.f = temporary_f32;
    else if (PyFloat_Check(self)) a_fui64.f = PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a_fui64.f = PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type)
        nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &temporary_f32), b_fui64.f = temporary_f32;
    else if (PyFloat_Check(other)) b_fui64.f = PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b_fui64.f = PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    return richcompare_from_int((a_fui64.f > b_fui64.f) - (a_fui64.f < b_fui64.f), op);
}

static PyObject *NkFloat8E4M3Scalar_add(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type) nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type) nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a + b;
    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_sub(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type) nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type) nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a - b;
    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_mul(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type) nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type) nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a * b;
    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_div(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E4M3Scalar_Type) nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E4M3Scalar_Type) nk_e4m3_to_f32(&((NkFloat8E4M3ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    if (b == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "float8_e4m3 division by zero");
        return NULL;
    }

    result = a / b;
    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_neg(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    f32_val = -f32_val;

    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&f32_val, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E4M3Scalar_pos(NkFloat8E4M3ScalarObject *self) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *NkFloat8E4M3Scalar_abs(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    if (f32_val < 0) f32_val = -f32_val;

    NkFloat8E4M3ScalarObject *res = PyObject_New(NkFloat8E4M3ScalarObject, &NkFloat8E4M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e4m3(&f32_val, &res->value);
    return (PyObject *)res;
}

static int NkFloat8E4M3Scalar_bool(NkFloat8E4M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e4m3_to_f32(&self->value, &f32_val);
    return f32_val != 0.0f;
}

static PyNumberMethods NkFloat8E4M3Scalar_as_number = {
    .nb_add = NkFloat8E4M3Scalar_add,
    .nb_subtract = NkFloat8E4M3Scalar_sub,
    .nb_multiply = NkFloat8E4M3Scalar_mul,
    .nb_true_divide = NkFloat8E4M3Scalar_div,
    .nb_negative = (unaryfunc)NkFloat8E4M3Scalar_neg,
    .nb_positive = (unaryfunc)NkFloat8E4M3Scalar_pos,
    .nb_absolute = (unaryfunc)NkFloat8E4M3Scalar_abs,
    .nb_bool = (inquiry)NkFloat8E4M3Scalar_bool,
    .nb_float = (unaryfunc)NkFloat8E4M3Scalar_float,
    .nb_int = (unaryfunc)NkFloat8E4M3Scalar_int,
};

static char const doc_float8_e4m3[] =                           //
    "8-bit E4M3, an OCP FP8 format, floating-point scalar.\n"   //
    "\n"                                                        //
    "Layout: sign(1) + exponent(4) + mantissa(3), bias=7.\n\n"  //
    "Range: ±448, no infinities (all-ones exponent = NaN).\n\n" //
    "114 of 254 finite values (44.9%%) fall in [-1, +1].\n\n"   //
    "Exact integer dot products via exponent-sum binning (29 bins).";

PyTypeObject NkFloat8E4M3Scalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.float8_e4m3",
    .tp_basicsize = sizeof(NkFloat8E4M3ScalarObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NkFloat8E4M3Scalar_new,
    .tp_repr = (reprfunc)NkFloat8E4M3Scalar_repr,
    .tp_str = (reprfunc)NkFloat8E4M3Scalar_str,
    .tp_hash = (hashfunc)NkFloat8E4M3Scalar_hash,
    .tp_richcompare = NkFloat8E4M3Scalar_richcompare,
    .tp_as_number = &NkFloat8E4M3Scalar_as_number,
    .tp_doc = doc_float8_e4m3,
};

static PyObject *NkFloat8E5M2Scalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NkFloat8E5M2ScalarObject *self = (NkFloat8E5M2ScalarObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    double value = 0.0;
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs > 1) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_TypeError, "float8_e5m2() takes at most 1 argument");
        return NULL;
    }
    if (nargs == 1) {
        value = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 0));
        if (value == -1.0 && PyErr_Occurred()) {
            Py_DECREF(self);
            return NULL;
        }
    }

    nk_f32_t f32_val = (nk_f32_t)value;
    nk_f32_to_e5m2(&f32_val, &self->value);
    return (PyObject *)self;
}

static PyObject *NkFloat8E5M2Scalar_repr(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    char buf[64];
    snprintf(buf, sizeof(buf), "float8_e5m2(%.6g)", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat8E5M2Scalar_str(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat8E5M2Scalar_float(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    return PyFloat_FromDouble((double)f32_val);
}

static PyObject *NkFloat8E5M2Scalar_int(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    return PyLong_FromDouble((double)f32_val);
}

static Py_hash_t NkFloat8E5M2Scalar_hash(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
#if PY_VERSION_HEX >= 0x030A00F0
    return _Py_HashDouble(NULL, (double)f32_val);
#else
    return _Py_HashDouble((double)f32_val);
#endif
}

static PyObject *NkFloat8E5M2Scalar_richcompare(PyObject *self, PyObject *other, int op) {
    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type && Py_TYPE(other) == &NkFloat8E5M2Scalar_Type)
        return richcompare_from_int(
            nk_e5m2_order(((NkFloat8E5M2ScalarObject *)self)->value, ((NkFloat8E5M2ScalarObject *)other)->value), op);

    nk_fui64_t a_fui64, b_fui64;
    nk_f32_t temporary_f32;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type)
        nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &temporary_f32), a_fui64.f = temporary_f32;
    else if (PyFloat_Check(self)) a_fui64.f = PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a_fui64.f = PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type)
        nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &temporary_f32), b_fui64.f = temporary_f32;
    else if (PyFloat_Check(other)) b_fui64.f = PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b_fui64.f = PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    return richcompare_from_int((a_fui64.f > b_fui64.f) - (a_fui64.f < b_fui64.f), op);
}

static PyObject *NkFloat8E5M2Scalar_add(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type) nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type) nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a + b;
    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_sub(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type) nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type) nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a - b;
    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_mul(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type) nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type) nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a * b;
    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_div(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat8E5M2Scalar_Type) nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat8E5M2Scalar_Type) nk_e5m2_to_f32(&((NkFloat8E5M2ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    if (b == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "float8_e5m2 division by zero");
        return NULL;
    }

    result = a / b;
    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_neg(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    f32_val = -f32_val;

    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&f32_val, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat8E5M2Scalar_pos(NkFloat8E5M2ScalarObject *self) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *NkFloat8E5M2Scalar_abs(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    if (f32_val < 0) f32_val = -f32_val;

    NkFloat8E5M2ScalarObject *res = PyObject_New(NkFloat8E5M2ScalarObject, &NkFloat8E5M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e5m2(&f32_val, &res->value);
    return (PyObject *)res;
}

static int NkFloat8E5M2Scalar_bool(NkFloat8E5M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e5m2_to_f32(&self->value, &f32_val);
    return f32_val != 0.0f;
}

static PyNumberMethods NkFloat8E5M2Scalar_as_number = {
    .nb_add = NkFloat8E5M2Scalar_add,
    .nb_subtract = NkFloat8E5M2Scalar_sub,
    .nb_multiply = NkFloat8E5M2Scalar_mul,
    .nb_true_divide = NkFloat8E5M2Scalar_div,
    .nb_negative = (unaryfunc)NkFloat8E5M2Scalar_neg,
    .nb_positive = (unaryfunc)NkFloat8E5M2Scalar_pos,
    .nb_absolute = (unaryfunc)NkFloat8E5M2Scalar_abs,
    .nb_bool = (inquiry)NkFloat8E5M2Scalar_bool,
    .nb_float = (unaryfunc)NkFloat8E5M2Scalar_float,
    .nb_int = (unaryfunc)NkFloat8E5M2Scalar_int,
};

static char const doc_float8_e5m2[] =                                               //
    "8-bit E5M2, an OCP FP8 format, floating-point scalar.\n"                       //
    "\n"                                                                            //
    "Layout: sign(1) + exponent(5) + mantissa(2), bias=15.\n\n"                     //
    "Range: ±57,344, supports infinities. Only 4 mantissa levels per exponent.\n\n" //
    "122 of 248 finite values (49.2%%) fall in [-1, +1].\n\n"                       //
    "High cancellation risk in dot products.";

PyTypeObject NkFloat8E5M2Scalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.float8_e5m2",
    .tp_basicsize = sizeof(NkFloat8E5M2ScalarObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NkFloat8E5M2Scalar_new,
    .tp_repr = (reprfunc)NkFloat8E5M2Scalar_repr,
    .tp_str = (reprfunc)NkFloat8E5M2Scalar_str,
    .tp_hash = (hashfunc)NkFloat8E5M2Scalar_hash,
    .tp_richcompare = NkFloat8E5M2Scalar_richcompare,
    .tp_as_number = &NkFloat8E5M2Scalar_as_number,
    .tp_doc = doc_float8_e5m2,
};

static PyObject *NkFloat16Scalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NkFloat16ScalarObject *self = (NkFloat16ScalarObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    double value = 0.0;
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs > 1) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_TypeError, "float16() takes at most 1 argument");
        return NULL;
    }
    if (nargs == 1) {
        value = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 0));
        if (value == -1.0 && PyErr_Occurred()) {
            Py_DECREF(self);
            return NULL;
        }
    }

    nk_f32_t f32_val = (nk_f32_t)value;
    nk_f32_to_f16(&f32_val, &self->value);
    return (PyObject *)self;
}

static PyObject *NkFloat16Scalar_repr(NkFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_f16_to_f32(&self->value, &f32_val);
    char buf[64];
    snprintf(buf, sizeof(buf), "float16(%.6g)", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat16Scalar_str(NkFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_f16_to_f32(&self->value, &f32_val);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat16Scalar_float(NkFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_f16_to_f32(&self->value, &f32_val);
    return PyFloat_FromDouble((double)f32_val);
}

static PyObject *NkFloat16Scalar_int(NkFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_f16_to_f32(&self->value, &f32_val);
    return PyLong_FromDouble((double)f32_val);
}

static Py_hash_t NkFloat16Scalar_hash(NkFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_f16_to_f32(&self->value, &f32_val);
#if PY_VERSION_HEX >= 0x030A00F0
    return _Py_HashDouble(NULL, (double)f32_val);
#else
    return _Py_HashDouble((double)f32_val);
#endif
}

static PyObject *NkFloat16Scalar_richcompare(PyObject *self, PyObject *other, int op) {
    if (Py_TYPE(self) == &NkFloat16Scalar_Type && Py_TYPE(other) == &NkFloat16Scalar_Type)
        return richcompare_from_int(
            nk_f16_order(((NkFloat16ScalarObject *)self)->value, ((NkFloat16ScalarObject *)other)->value), op);

    nk_fui64_t a_fui64, b_fui64;
    nk_f32_t temporary_f32;

    if (Py_TYPE(self) == &NkFloat16Scalar_Type)
        nk_f16_to_f32(&((NkFloat16ScalarObject *)self)->value, &temporary_f32), a_fui64.f = temporary_f32;
    else if (PyFloat_Check(self)) a_fui64.f = PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a_fui64.f = PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat16Scalar_Type)
        nk_f16_to_f32(&((NkFloat16ScalarObject *)other)->value, &temporary_f32), b_fui64.f = temporary_f32;
    else if (PyFloat_Check(other)) b_fui64.f = PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b_fui64.f = PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    return richcompare_from_int((a_fui64.f > b_fui64.f) - (a_fui64.f < b_fui64.f), op);
}

static PyObject *NkFloat16Scalar_add(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat16Scalar_Type) nk_f16_to_f32(&((NkFloat16ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat16Scalar_Type) nk_f16_to_f32(&((NkFloat16ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a + b;
    NkFloat16ScalarObject *res = PyObject_New(NkFloat16ScalarObject, &NkFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_f16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat16Scalar_sub(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat16Scalar_Type) nk_f16_to_f32(&((NkFloat16ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat16Scalar_Type) nk_f16_to_f32(&((NkFloat16ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a - b;
    NkFloat16ScalarObject *res = PyObject_New(NkFloat16ScalarObject, &NkFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_f16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat16Scalar_mul(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat16Scalar_Type) nk_f16_to_f32(&((NkFloat16ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat16Scalar_Type) nk_f16_to_f32(&((NkFloat16ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a * b;
    NkFloat16ScalarObject *res = PyObject_New(NkFloat16ScalarObject, &NkFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_f16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat16Scalar_div(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat16Scalar_Type) nk_f16_to_f32(&((NkFloat16ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat16Scalar_Type) nk_f16_to_f32(&((NkFloat16ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    if (b == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "float16 division by zero");
        return NULL;
    }

    result = a / b;
    NkFloat16ScalarObject *res = PyObject_New(NkFloat16ScalarObject, &NkFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_f16(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat16Scalar_neg(NkFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_f16_to_f32(&self->value, &f32_val);
    f32_val = -f32_val;

    NkFloat16ScalarObject *res = PyObject_New(NkFloat16ScalarObject, &NkFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_f16(&f32_val, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat16Scalar_pos(NkFloat16ScalarObject *self) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *NkFloat16Scalar_abs(NkFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_f16_to_f32(&self->value, &f32_val);
    if (f32_val < 0) f32_val = -f32_val;

    NkFloat16ScalarObject *res = PyObject_New(NkFloat16ScalarObject, &NkFloat16Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_f16(&f32_val, &res->value);
    return (PyObject *)res;
}

static int NkFloat16Scalar_bool(NkFloat16ScalarObject *self) {
    nk_f32_t f32_val;
    nk_f16_to_f32(&self->value, &f32_val);
    return f32_val != 0.0f;
}

static PyNumberMethods NkFloat16Scalar_as_number = {
    .nb_add = NkFloat16Scalar_add,
    .nb_subtract = NkFloat16Scalar_sub,
    .nb_multiply = NkFloat16Scalar_mul,
    .nb_true_divide = NkFloat16Scalar_div,
    .nb_negative = (unaryfunc)NkFloat16Scalar_neg,
    .nb_positive = (unaryfunc)NkFloat16Scalar_pos,
    .nb_absolute = (unaryfunc)NkFloat16Scalar_abs,
    .nb_bool = (inquiry)NkFloat16Scalar_bool,
    .nb_float = (unaryfunc)NkFloat16Scalar_float,
    .nb_int = (unaryfunc)NkFloat16Scalar_int,
};

static char const doc_float16[] =                                   //
    "IEEE 754 half-precision (16-bit) floating-point scalar.\n"     //
    "\n"                                                            //
    "Layout: sign(1) + exponent(5) + mantissa(10), bias=15.\n\n"    //
    "Range: ±65,504, epsilon at 1.0: ~9.77e-4.\n\n"                 //
    "30,722 of 63,488 finite values (48.4%%) fall in [-1, +1].\n\n" //
    "All arithmetic via f32 upcast/downcast.";

PyTypeObject NkFloat16Scalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.float16",
    .tp_basicsize = sizeof(NkFloat16ScalarObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NkFloat16Scalar_new,
    .tp_repr = (reprfunc)NkFloat16Scalar_repr,
    .tp_str = (reprfunc)NkFloat16Scalar_str,
    .tp_hash = (hashfunc)NkFloat16Scalar_hash,
    .tp_richcompare = NkFloat16Scalar_richcompare,
    .tp_as_number = &NkFloat16Scalar_as_number,
    .tp_doc = doc_float16,
};

static PyObject *NkFloat6E2M3Scalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NkFloat6E2M3ScalarObject *self = (NkFloat6E2M3ScalarObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    double value = 0.0;
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs > 1) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_TypeError, "float6_e2m3() takes at most 1 argument");
        return NULL;
    }
    if (nargs == 1) {
        value = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 0));
        if (value == -1.0 && PyErr_Occurred()) {
            Py_DECREF(self);
            return NULL;
        }
    }

    nk_f32_t f32_val = (nk_f32_t)value;
    nk_f32_to_e2m3(&f32_val, &self->value);
    return (PyObject *)self;
}

static PyObject *NkFloat6E2M3Scalar_repr(NkFloat6E2M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e2m3_to_f32(&self->value, &f32_val);
    char buf[64];
    snprintf(buf, sizeof(buf), "float6_e2m3(%.6g)", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat6E2M3Scalar_str(NkFloat6E2M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e2m3_to_f32(&self->value, &f32_val);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat6E2M3Scalar_float(NkFloat6E2M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e2m3_to_f32(&self->value, &f32_val);
    return PyFloat_FromDouble((double)f32_val);
}

static PyObject *NkFloat6E2M3Scalar_int(NkFloat6E2M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e2m3_to_f32(&self->value, &f32_val);
    return PyLong_FromDouble((double)f32_val);
}

static Py_hash_t NkFloat6E2M3Scalar_hash(NkFloat6E2M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e2m3_to_f32(&self->value, &f32_val);
#if PY_VERSION_HEX >= 0x030A00F0
    return _Py_HashDouble(NULL, (double)f32_val);
#else
    return _Py_HashDouble((double)f32_val);
#endif
}

static PyObject *NkFloat6E2M3Scalar_richcompare(PyObject *self, PyObject *other, int op) {
    if (Py_TYPE(self) == &NkFloat6E2M3Scalar_Type && Py_TYPE(other) == &NkFloat6E2M3Scalar_Type)
        return richcompare_from_int(
            nk_e2m3_order(((NkFloat6E2M3ScalarObject *)self)->value, ((NkFloat6E2M3ScalarObject *)other)->value), op);

    nk_fui64_t a_fui64, b_fui64;
    nk_f32_t temporary_f32;

    if (Py_TYPE(self) == &NkFloat6E2M3Scalar_Type)
        nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)self)->value, &temporary_f32), a_fui64.f = temporary_f32;
    else if (PyFloat_Check(self)) a_fui64.f = PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a_fui64.f = PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E2M3Scalar_Type)
        nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)other)->value, &temporary_f32), b_fui64.f = temporary_f32;
    else if (PyFloat_Check(other)) b_fui64.f = PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b_fui64.f = PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    return richcompare_from_int((a_fui64.f > b_fui64.f) - (a_fui64.f < b_fui64.f), op);
}

static PyObject *NkFloat6E2M3Scalar_add(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat6E2M3Scalar_Type) nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E2M3Scalar_Type) nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a + b;
    NkFloat6E2M3ScalarObject *res = PyObject_New(NkFloat6E2M3ScalarObject, &NkFloat6E2M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e2m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E2M3Scalar_sub(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat6E2M3Scalar_Type) nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E2M3Scalar_Type) nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a - b;
    NkFloat6E2M3ScalarObject *res = PyObject_New(NkFloat6E2M3ScalarObject, &NkFloat6E2M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e2m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E2M3Scalar_mul(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat6E2M3Scalar_Type) nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E2M3Scalar_Type) nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a * b;
    NkFloat6E2M3ScalarObject *res = PyObject_New(NkFloat6E2M3ScalarObject, &NkFloat6E2M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e2m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E2M3Scalar_div(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat6E2M3Scalar_Type) nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E2M3Scalar_Type) nk_e2m3_to_f32(&((NkFloat6E2M3ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    if (b == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "float6_e2m3 division by zero");
        return NULL;
    }

    result = a / b;
    NkFloat6E2M3ScalarObject *res = PyObject_New(NkFloat6E2M3ScalarObject, &NkFloat6E2M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e2m3(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E2M3Scalar_neg(NkFloat6E2M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e2m3_to_f32(&self->value, &f32_val);
    f32_val = -f32_val;

    NkFloat6E2M3ScalarObject *res = PyObject_New(NkFloat6E2M3ScalarObject, &NkFloat6E2M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e2m3(&f32_val, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E2M3Scalar_pos(NkFloat6E2M3ScalarObject *self) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *NkFloat6E2M3Scalar_abs(NkFloat6E2M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e2m3_to_f32(&self->value, &f32_val);
    if (f32_val < 0) f32_val = -f32_val;

    NkFloat6E2M3ScalarObject *res = PyObject_New(NkFloat6E2M3ScalarObject, &NkFloat6E2M3Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e2m3(&f32_val, &res->value);
    return (PyObject *)res;
}

static int NkFloat6E2M3Scalar_bool(NkFloat6E2M3ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e2m3_to_f32(&self->value, &f32_val);
    return f32_val != 0.0f;
}

static PyNumberMethods NkFloat6E2M3Scalar_as_number = {
    .nb_add = NkFloat6E2M3Scalar_add,
    .nb_subtract = NkFloat6E2M3Scalar_sub,
    .nb_multiply = NkFloat6E2M3Scalar_mul,
    .nb_true_divide = NkFloat6E2M3Scalar_div,
    .nb_negative = (unaryfunc)NkFloat6E2M3Scalar_neg,
    .nb_positive = (unaryfunc)NkFloat6E2M3Scalar_pos,
    .nb_absolute = (unaryfunc)NkFloat6E2M3Scalar_abs,
    .nb_bool = (inquiry)NkFloat6E2M3Scalar_bool,
    .nb_float = (unaryfunc)NkFloat6E2M3Scalar_float,
    .nb_int = (unaryfunc)NkFloat6E2M3Scalar_int,
};

static char const doc_float6_e2m3[] =                                                       //
    "6-bit E2M3 micro-float scalar, padded to 8-bit storage.\n"                             //
    "\n"                                                                                    //
    "Layout: sign(1) + exponent(2) + mantissa(3), bias=1.\n\n"                              //
    "Range: ±7.5, no infinities. Only 64 total codes.\n\n"                                  //
    "18 of 64 codes (28.1%%) fall in [-1, +1]; poor resolution for normalized vectors.\n\n" //
    "Exact integer dot products via exponent-sum binning (15 bins).";

PyTypeObject NkFloat6E2M3Scalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.float6_e2m3",
    .tp_basicsize = sizeof(NkFloat6E2M3ScalarObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NkFloat6E2M3Scalar_new,
    .tp_repr = (reprfunc)NkFloat6E2M3Scalar_repr,
    .tp_str = (reprfunc)NkFloat6E2M3Scalar_str,
    .tp_hash = (hashfunc)NkFloat6E2M3Scalar_hash,
    .tp_richcompare = NkFloat6E2M3Scalar_richcompare,
    .tp_as_number = &NkFloat6E2M3Scalar_as_number,
    .tp_doc = doc_float6_e2m3,
};

static PyObject *NkFloat6E3M2Scalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NkFloat6E3M2ScalarObject *self = (NkFloat6E3M2ScalarObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;

    double value = 0.0;
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs > 1) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_TypeError, "float6_e3m2() takes at most 1 argument");
        return NULL;
    }
    if (nargs == 1) {
        value = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 0));
        if (value == -1.0 && PyErr_Occurred()) {
            Py_DECREF(self);
            return NULL;
        }
    }

    nk_f32_t f32_val = (nk_f32_t)value;
    nk_f32_to_e3m2(&f32_val, &self->value);
    return (PyObject *)self;
}

static PyObject *NkFloat6E3M2Scalar_repr(NkFloat6E3M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e3m2_to_f32(&self->value, &f32_val);
    char buf[64];
    snprintf(buf, sizeof(buf), "float6_e3m2(%.6g)", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat6E3M2Scalar_str(NkFloat6E3M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e3m2_to_f32(&self->value, &f32_val);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", (double)f32_val);
    return PyUnicode_FromString(buf);
}

static PyObject *NkFloat6E3M2Scalar_float(NkFloat6E3M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e3m2_to_f32(&self->value, &f32_val);
    return PyFloat_FromDouble((double)f32_val);
}

static PyObject *NkFloat6E3M2Scalar_int(NkFloat6E3M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e3m2_to_f32(&self->value, &f32_val);
    return PyLong_FromDouble((double)f32_val);
}

static Py_hash_t NkFloat6E3M2Scalar_hash(NkFloat6E3M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e3m2_to_f32(&self->value, &f32_val);
#if PY_VERSION_HEX >= 0x030A00F0
    return _Py_HashDouble(NULL, (double)f32_val);
#else
    return _Py_HashDouble((double)f32_val);
#endif
}

static PyObject *NkFloat6E3M2Scalar_richcompare(PyObject *self, PyObject *other, int op) {
    if (Py_TYPE(self) == &NkFloat6E3M2Scalar_Type && Py_TYPE(other) == &NkFloat6E3M2Scalar_Type)
        return richcompare_from_int(
            nk_e3m2_order(((NkFloat6E3M2ScalarObject *)self)->value, ((NkFloat6E3M2ScalarObject *)other)->value), op);

    nk_fui64_t a_fui64, b_fui64;
    nk_f32_t temporary_f32;

    if (Py_TYPE(self) == &NkFloat6E3M2Scalar_Type)
        nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)self)->value, &temporary_f32), a_fui64.f = temporary_f32;
    else if (PyFloat_Check(self)) a_fui64.f = PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a_fui64.f = PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E3M2Scalar_Type)
        nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)other)->value, &temporary_f32), b_fui64.f = temporary_f32;
    else if (PyFloat_Check(other)) b_fui64.f = PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b_fui64.f = PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    return richcompare_from_int((a_fui64.f > b_fui64.f) - (a_fui64.f < b_fui64.f), op);
}

static PyObject *NkFloat6E3M2Scalar_add(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat6E3M2Scalar_Type) nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E3M2Scalar_Type) nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a + b;
    NkFloat6E3M2ScalarObject *res = PyObject_New(NkFloat6E3M2ScalarObject, &NkFloat6E3M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e3m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E3M2Scalar_sub(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat6E3M2Scalar_Type) nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E3M2Scalar_Type) nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a - b;
    NkFloat6E3M2ScalarObject *res = PyObject_New(NkFloat6E3M2ScalarObject, &NkFloat6E3M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e3m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E3M2Scalar_mul(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat6E3M2Scalar_Type) nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E3M2Scalar_Type) nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    result = a * b;
    NkFloat6E3M2ScalarObject *res = PyObject_New(NkFloat6E3M2ScalarObject, &NkFloat6E3M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e3m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E3M2Scalar_div(PyObject *self, PyObject *other) {
    nk_f32_t a, b, result;

    if (Py_TYPE(self) == &NkFloat6E3M2Scalar_Type) nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)self)->value, &a);
    else if (PyFloat_Check(self)) a = (nk_f32_t)PyFloat_AS_DOUBLE(self);
    else if (PyLong_Check(self)) a = (nk_f32_t)PyLong_AsDouble(self);
    else Py_RETURN_NOTIMPLEMENTED;

    if (Py_TYPE(other) == &NkFloat6E3M2Scalar_Type) nk_e3m2_to_f32(&((NkFloat6E3M2ScalarObject *)other)->value, &b);
    else if (PyFloat_Check(other)) b = (nk_f32_t)PyFloat_AS_DOUBLE(other);
    else if (PyLong_Check(other)) b = (nk_f32_t)PyLong_AsDouble(other);
    else Py_RETURN_NOTIMPLEMENTED;

    if (b == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "float6_e3m2 division by zero");
        return NULL;
    }

    result = a / b;
    NkFloat6E3M2ScalarObject *res = PyObject_New(NkFloat6E3M2ScalarObject, &NkFloat6E3M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e3m2(&result, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E3M2Scalar_neg(NkFloat6E3M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e3m2_to_f32(&self->value, &f32_val);
    f32_val = -f32_val;

    NkFloat6E3M2ScalarObject *res = PyObject_New(NkFloat6E3M2ScalarObject, &NkFloat6E3M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e3m2(&f32_val, &res->value);
    return (PyObject *)res;
}

static PyObject *NkFloat6E3M2Scalar_pos(NkFloat6E3M2ScalarObject *self) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *NkFloat6E3M2Scalar_abs(NkFloat6E3M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e3m2_to_f32(&self->value, &f32_val);
    if (f32_val < 0) f32_val = -f32_val;

    NkFloat6E3M2ScalarObject *res = PyObject_New(NkFloat6E3M2ScalarObject, &NkFloat6E3M2Scalar_Type);
    if (res == NULL) return NULL;
    nk_f32_to_e3m2(&f32_val, &res->value);
    return (PyObject *)res;
}

static int NkFloat6E3M2Scalar_bool(NkFloat6E3M2ScalarObject *self) {
    nk_f32_t f32_val;
    nk_e3m2_to_f32(&self->value, &f32_val);
    return f32_val != 0.0f;
}

static PyNumberMethods NkFloat6E3M2Scalar_as_number = {
    .nb_add = NkFloat6E3M2Scalar_add,
    .nb_subtract = NkFloat6E3M2Scalar_sub,
    .nb_multiply = NkFloat6E3M2Scalar_mul,
    .nb_true_divide = NkFloat6E3M2Scalar_div,
    .nb_negative = (unaryfunc)NkFloat6E3M2Scalar_neg,
    .nb_positive = (unaryfunc)NkFloat6E3M2Scalar_pos,
    .nb_absolute = (unaryfunc)NkFloat6E3M2Scalar_abs,
    .nb_bool = (inquiry)NkFloat6E3M2Scalar_bool,
    .nb_float = (unaryfunc)NkFloat6E3M2Scalar_float,
    .nb_int = (unaryfunc)NkFloat6E3M2Scalar_int,
};

static char const doc_float6_e3m2[] =                           //
    "6-bit E3M2 micro-float scalar, padded to 8-bit storage.\n" //
    "\n"                                                        //
    "Layout: sign(1) + exponent(3) + mantissa(2), bias=3.\n\n"  //
    "Range: ±28, supports infinities. Only 64 total codes.\n\n" //
    "26 of 64 codes (40.6%%) fall in [-1, +1].\n\n"             //
    "Exact integer dot products via exponent-sum binning (15 bins).";

PyTypeObject NkFloat6E3M2Scalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.float6_e3m2",
    .tp_basicsize = sizeof(NkFloat6E3M2ScalarObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NkFloat6E3M2Scalar_new,
    .tp_repr = (reprfunc)NkFloat6E3M2Scalar_repr,
    .tp_str = (reprfunc)NkFloat6E3M2Scalar_str,
    .tp_hash = (hashfunc)NkFloat6E3M2Scalar_hash,
    .tp_richcompare = NkFloat6E3M2Scalar_richcompare,
    .tp_as_number = &NkFloat6E3M2Scalar_as_number,
    .tp_doc = doc_float6_e3m2,
};

int nk_register_scalar_types(PyObject *module) {
    struct {
        PyTypeObject *type;
        char const *name;
    } scalar_types[] = {
        {&NkBFloat16Scalar_Type, "bfloat16"},      {&NkFloat16Scalar_Type, "float16"},
        {&NkFloat8E4M3Scalar_Type, "float8_e4m3"}, {&NkFloat8E5M2Scalar_Type, "float8_e5m2"},
        {&NkFloat6E2M3Scalar_Type, "float6_e2m3"}, {&NkFloat6E3M2Scalar_Type, "float6_e3m2"},
    };
    for (size_t i = 0; i < sizeof(scalar_types) / sizeof(scalar_types[0]); i++) {
        if (PyType_Ready(scalar_types[i].type) < 0) return -1;
        Py_INCREF(scalar_types[i].type);
        if (PyModule_AddObject(module, scalar_types[i].name, (PyObject *)scalar_types[i].type) < 0) {
            Py_DECREF(scalar_types[i].type);
            return -1;
        }
    }
    return 0;
}
