/**
 *  @brief Stateful random generator Python binding.
 *  @file python/random.c
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>

#include <numkong/random.h>

#include "numkong.h"
#include "random.h"
#include "tensor.h"

typedef struct {
    PyObject_HEAD
    nk_random_generator_t generator;
} RandomGenerator;

typedef struct {
    Tensor *owned;
    Py_buffer buffer;
    nk_buffer_backing_t backing;
    char *data;
    int has_buffer;
} RandomOutput;

static int keyword_allowed(PyObject *name, char const *const *allowed) {
    for (size_t i = 0; allowed[i]; ++i)
        if (PyUnicode_CompareWithASCIIString(name, allowed[i]) == 0) return 1;
    return 0;
}

static int reject_unknown_keywords(PyObject *kwnames, char const *const *allowed, char const *function_name) {
    Py_ssize_t const keyword_count = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t i = 0; i < keyword_count; ++i) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (!keyword_allowed(name, allowed)) {
            PyErr_Format(PyExc_TypeError, "%s() unexpected keyword: %S", function_name, name);
            return 0;
        }
    }
    return 1;
}

static int get_argument(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, Py_ssize_t position,
                        char const *name, PyObject *default_value, PyObject **value) {
    int found = 0;
    if (nargs > position) {
        *value = args[position];
        found = 1;
    }

    Py_ssize_t const keyword_count = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t i = 0; i < keyword_count; ++i) {
        PyObject *keyword = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(keyword, name) != 0) continue;
        if (found) {
            PyErr_Format(PyExc_TypeError, "got multiple values for argument '%s'", name);
            return 0;
        }
        *value = args[nargs + i];
        found = 1;
    }

    if (!found) *value = default_value;
    return 1;
}

static int parse_shape(PyObject *size_obj, Py_ssize_t *shape, size_t *rank) {
    if (size_obj == NULL || size_obj == Py_None) {
        PyErr_SetString(PyExc_TypeError, "size must be an int or tuple of ints");
        return 0;
    }
    if (PyLong_Check(size_obj)) {
        shape[0] = PyLong_AsSsize_t(size_obj);
        if (PyErr_Occurred()) return 0;
        if (shape[0] < 0) {
            PyErr_SetString(PyExc_ValueError, "size dimensions must be non-negative");
            return 0;
        }
        *rank = 1;
        return 1;
    }
    if (!PyTuple_Check(size_obj)) {
        PyErr_SetString(PyExc_TypeError, "size must be an int or tuple of ints");
        return 0;
    }

    Py_ssize_t const dimensions = PyTuple_GET_SIZE(size_obj);
    if (dimensions > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "size has %zd dimensions, max is %d", dimensions, NK_TENSOR_MAX_RANK);
        return 0;
    }
    for (Py_ssize_t i = 0; i < dimensions; ++i) {
        PyObject *dimension = PyTuple_GET_ITEM(size_obj, i);
        if (!PyLong_Check(dimension)) {
            PyErr_SetString(PyExc_TypeError, "size dimensions must be integers");
            return 0;
        }
        shape[i] = PyLong_AsSsize_t(dimension);
        if (PyErr_Occurred()) return 0;
        if (shape[i] < 0) {
            PyErr_SetString(PyExc_ValueError, "size dimensions must be non-negative");
            return 0;
        }
    }
    *rank = (size_t)dimensions;
    return 1;
}

static int count_elements(Py_ssize_t const *shape, size_t rank, size_t *total) {
    size_t count = 1;
    for (size_t i = 0; i < rank; ++i) {
        if ((size_t)shape[i] && count > SIZE_MAX / (size_t)shape[i]) {
            PyErr_SetString(PyExc_OverflowError, "size is too large");
            return 0;
        }
        count *= (size_t)shape[i];
    }
    *total = count;
    return 1;
}

static int parse_dtype(PyObject *dtype_obj, nk_dtype_t default_dtype, nk_dtype_t *dtype) {
    *dtype = dtype_obj ? py_object_to_nk_dtype(dtype_obj) : default_dtype;
    if (*dtype == nk_dtype_unknown_k) return 0;
    return 1;
}

static int parse_f64(PyObject *object, double *value, char const *name) {
    if (!object) {
        PyErr_Format(PyExc_TypeError, "%s is required", name);
        return 0;
    }
    *value = PyFloat_AsDouble(object);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError, "%s must be a real number", name);
        return 0;
    }
    return 1;
}

static int parse_i64(PyObject *object, int64_t *value, char const *name) {
    if (!object || !PyLong_Check(object)) {
        PyErr_Format(PyExc_TypeError, "%s must be an integer", name);
        return 0;
    }
    *value = PyLong_AsLongLong(object);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        PyErr_Format(PyExc_OverflowError, "%s does not fit in int64", name);
        return 0;
    }
    return 1;
}

static int prepare_output(RandomOutput *output, PyObject *out_obj, nk_dtype_t dtype, size_t rank,
                          Py_ssize_t const *shape) {
    output->owned = NULL;
    output->has_buffer = 0;
    if (out_obj == NULL || out_obj == Py_None) {
        output->owned = Tensor_new(dtype, rank, shape);
        if (!output->owned) return 0;
        output->data = output->owned->data;
        return 1;
    }

    if (!nk_get_buffer(out_obj, &output->buffer, PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_WRITABLE, &output->backing))
        return 0;
    output->has_buffer = 1;

    nk_dtype_t const out_dtype = resolve_nk_dtype_in_py_buffer(&output->buffer);
    if (out_dtype != dtype) {
        PyErr_Format(PyExc_TypeError, "out dtype '%s' does not match requested '%s'", nk_dtype_name(out_dtype),
                     nk_dtype_name(dtype));
        return 0;
    }
    if ((size_t)output->buffer.ndim != rank) {
        PyErr_Format(PyExc_ValueError, "out rank %d does not match requested rank %zu", output->buffer.ndim, rank);
        return 0;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (output->buffer.shape[i] != shape[i]) {
            PyErr_Format(PyExc_ValueError, "out shape does not match size at dimension %zu", i);
            return 0;
        }
    }
    if (!PyBuffer_IsContiguous(&output->buffer, 'C')) {
        PyErr_SetString(PyExc_ValueError, "out must be C-contiguous");
        return 0;
    }
    output->data = (char *)output->buffer.buf;
    return 1;
}

static void release_output(RandomOutput *output) {
    if (output->has_buffer) PyBuffer_Release(&output->buffer);
    Py_XDECREF(output->owned);
}

static PyObject *finish_output(RandomOutput *output) {
    if (output->has_buffer) {
        PyBuffer_Release(&output->buffer);
        output->has_buffer = 0;
        Py_RETURN_NONE;
    }
    return (PyObject *)output->owned;
}

static int floating_dtype(nk_dtype_t dtype) { return dtype == nk_f32_k || dtype == nk_f64_k; }

static int integer_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_i8_k:
    case nk_i16_k:
    case nk_i32_k:
    case nk_i64_k:
    case nk_u8_k:
    case nk_u16_k:
    case nk_u32_k:
    case nk_u64_k: return 1;
    default: return 0;
    }
}

static int validate_integer_bounds(nk_dtype_t dtype, int64_t low, int64_t high) {
    int64_t minimum = INT64_MIN, maximum_exclusive = INT64_MAX;
    switch (dtype) {
    case nk_i8_k: minimum = INT8_MIN, maximum_exclusive = INT8_MAX + 1; break;
    case nk_i16_k: minimum = INT16_MIN, maximum_exclusive = INT16_MAX + 1; break;
    case nk_i32_k: minimum = INT32_MIN, maximum_exclusive = (int64_t)INT32_MAX + 1; break;
    case nk_u8_k: minimum = 0, maximum_exclusive = UINT8_MAX + 1; break;
    case nk_u16_k: minimum = 0, maximum_exclusive = UINT16_MAX + 1; break;
    case nk_u32_k: minimum = 0, maximum_exclusive = (int64_t)UINT32_MAX + 1; break;
    case nk_u64_k: minimum = 0, maximum_exclusive = INT64_MAX; break;
    case nk_i64_k: break;
    default: return 0;
    }
    if (low < minimum || high > maximum_exclusive) {
        PyErr_Format(PyExc_ValueError, "integer bounds do not fit dtype '%s'", nk_dtype_name(dtype));
        return 0;
    }
    return 1;
}

static void fill_uniform(RandomGenerator *generator, char *data, nk_dtype_t dtype, size_t total, double low,
                         double high) {
    if (dtype == nk_f32_k) {
        float *values = (float *)data;
        float const lower = (float)low;
        float const width = (float)(high - low);
        for (size_t i = 0; i < total; ++i)
            values[i] = lower + width * nk_random_uniform_f32(&generator->generator);
    }
    else {
        double *values = (double *)data;
        for (size_t i = 0; i < total; ++i) values[i] = low + (high - low) * nk_random_uniform_f64(&generator->generator);
    }
}

static void fill_normal(RandomGenerator *generator, char *data, nk_dtype_t dtype, size_t total, double location,
                        double scale) {
    if (dtype == nk_f32_k) {
        float *values = (float *)data;
        for (size_t i = 0; i < total; ++i)
            values[i] = (float)nk_random_normal_f64(&generator->generator, location, scale);
    }
    else {
        double *values = (double *)data;
        for (size_t i = 0; i < total; ++i) values[i] = nk_random_normal_f64(&generator->generator, location, scale);
    }
}

static void write_integer(char *data, nk_dtype_t dtype, size_t index, uint64_t value) {
    switch (dtype) {
    case nk_i8_k: ((int8_t *)data)[index] = (int8_t)value; break;
    case nk_i16_k: ((int16_t *)data)[index] = (int16_t)value; break;
    case nk_i32_k: ((int32_t *)data)[index] = (int32_t)value; break;
    case nk_i64_k: ((int64_t *)data)[index] = (int64_t)value; break;
    case nk_u8_k: ((uint8_t *)data)[index] = (uint8_t)value; break;
    case nk_u16_k: ((uint16_t *)data)[index] = (uint16_t)value; break;
    case nk_u32_k: ((uint32_t *)data)[index] = (uint32_t)value; break;
    case nk_u64_k: ((uint64_t *)data)[index] = value; break;
    default: break;
    }
}

static void fill_integers(RandomGenerator *generator, char *data, nk_dtype_t dtype, size_t total, int64_t low,
                          int64_t high) {
    uint64_t const bound = (uint64_t)high - (uint64_t)low;
    uint64_t const low_bits = (uint64_t)low;
    for (size_t i = 0; i < total; ++i)
        write_integer(data, dtype, i, low_bits + nk_random_bounded_u64(&generator->generator, bound));
}

static PyObject *generator_random(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    static char const *const allowed[] = {"size", "dtype", "out", NULL};
    if (nargs > 1 || !reject_unknown_keywords(kwnames, allowed, "Generator.random")) return NULL;

    PyObject *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "size", NULL, &size_obj) ||
        !get_argument(args, nargs, kwnames, 1, "dtype", NULL, &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 2, "out", NULL, &out_obj))
        return NULL;

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    if (!parse_shape(size_obj, shape, &rank) || !count_elements(shape, rank, &total)) return NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, nk_f32_k, &dtype)) return NULL;
    if (!floating_dtype(dtype))
        return (PyErr_Format(PyExc_TypeError, "random only supports float32 and float64"), NULL);

    RandomOutput output;
    if (!prepare_output(&output, out_obj, dtype, rank, shape)) {
        release_output(&output);
        return NULL;
    }
    PyThreadState *gil = PyEval_SaveThread();
    fill_uniform((RandomGenerator *)self, output.data, dtype, total, 0.0, 1.0);
    PyEval_RestoreThread(gil);
    return finish_output(&output);
}

static PyObject *generator_uniform(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    static char const *const allowed[] = {"low", "high", "size", "dtype", "out", NULL};
    if (nargs > 3 || !reject_unknown_keywords(kwnames, allowed, "Generator.uniform")) return NULL;

    PyObject *low_obj = NULL, *high_obj = NULL, *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "low", NULL, &low_obj) ||
        !get_argument(args, nargs, kwnames, 1, "high", NULL, &high_obj) ||
        !get_argument(args, nargs, kwnames, 2, "size", NULL, &size_obj) ||
        !get_argument(args, nargs, kwnames, 3, "dtype", NULL, &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 4, "out", NULL, &out_obj))
        return NULL;
    double low, high;
    if (!low_obj) low = 0.0;
    if (!high_obj) high = 1.0;
    if ((low_obj && !parse_f64(low_obj, &low, "low")) || (high_obj && !parse_f64(high_obj, &high, "high")))
        return NULL;
    if (!isfinite(low) || !isfinite(high) || low >= high)
        return (PyErr_SetString(PyExc_ValueError, "uniform requires finite low < high"), NULL);

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    if (!parse_shape(size_obj, shape, &rank) || !count_elements(shape, rank, &total)) return NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, nk_f32_k, &dtype)) return NULL;
    if (!floating_dtype(dtype))
        return (PyErr_Format(PyExc_TypeError, "uniform only supports float32 and float64"), NULL);

    RandomOutput output;
    if (!prepare_output(&output, out_obj, dtype, rank, shape)) {
        release_output(&output);
        return NULL;
    }
    PyThreadState *gil = PyEval_SaveThread();
    fill_uniform((RandomGenerator *)self, output.data, dtype, total, low, high);
    PyEval_RestoreThread(gil);
    return finish_output(&output);
}

static PyObject *generator_normal(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    static char const *const allowed[] = {"loc", "scale", "size", "dtype", "out", NULL};
    if (nargs > 3 || !reject_unknown_keywords(kwnames, allowed, "Generator.normal")) return NULL;

    PyObject *location_obj = NULL, *scale_obj = NULL, *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "loc", NULL, &location_obj) ||
        !get_argument(args, nargs, kwnames, 1, "scale", NULL, &scale_obj) ||
        !get_argument(args, nargs, kwnames, 2, "size", NULL, &size_obj) ||
        !get_argument(args, nargs, kwnames, 3, "dtype", NULL, &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 4, "out", NULL, &out_obj))
        return NULL;
    double location, scale;
    if (location_obj) {
        if (!parse_f64(location_obj, &location, "loc")) return NULL;
    }
    else location = 0.0;
    if (scale_obj) {
        if (!parse_f64(scale_obj, &scale, "scale")) return NULL;
    }
    else scale = 1.0;
    if (!isfinite(location) || !isfinite(scale) || scale < 0)
        return (PyErr_SetString(PyExc_ValueError, "normal requires finite loc and scale >= 0"), NULL);

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    if (!parse_shape(size_obj, shape, &rank) || !count_elements(shape, rank, &total)) return NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, nk_f32_k, &dtype)) return NULL;
    if (!floating_dtype(dtype))
        return (PyErr_Format(PyExc_TypeError, "normal only supports float32 and float64"), NULL);

    RandomOutput output;
    if (!prepare_output(&output, out_obj, dtype, rank, shape)) {
        release_output(&output);
        return NULL;
    }
    PyThreadState *gil = PyEval_SaveThread();
    fill_normal((RandomGenerator *)self, output.data, dtype, total, location, scale);
    PyEval_RestoreThread(gil);
    return finish_output(&output);
}

static PyObject *generator_standard_normal(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                           PyObject *kwnames) {
    static char const *const allowed[] = {"size", "dtype", "out", NULL};
    if (nargs > 1 || !reject_unknown_keywords(kwnames, allowed, "Generator.standard_normal")) return NULL;

    PyObject *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "size", NULL, &size_obj) ||
        !get_argument(args, nargs, kwnames, 1, "dtype", NULL, &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 2, "out", NULL, &out_obj))
        return NULL;

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    if (!parse_shape(size_obj, shape, &rank) || !count_elements(shape, rank, &total)) return NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, nk_f32_k, &dtype)) return NULL;
    if (!floating_dtype(dtype))
        return (PyErr_Format(PyExc_TypeError, "standard_normal only supports float32 and float64"), NULL);

    RandomOutput output;
    if (!prepare_output(&output, out_obj, dtype, rank, shape)) {
        release_output(&output);
        return NULL;
    }
    PyThreadState *gil = PyEval_SaveThread();
    fill_normal((RandomGenerator *)self, output.data, dtype, total, 0.0, 1.0);
    PyEval_RestoreThread(gil);
    return finish_output(&output);
}

static PyObject *generator_integers(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    static char const *const allowed[] = {"low", "high", "size", "dtype", "out", NULL};
    if (nargs > 3 || !reject_unknown_keywords(kwnames, allowed, "Generator.integers")) return NULL;

    PyObject *low_obj = NULL, *high_obj = NULL, *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "low", NULL, &low_obj) ||
        !get_argument(args, nargs, kwnames, 1, "high", NULL, &high_obj) ||
        !get_argument(args, nargs, kwnames, 2, "size", NULL, &size_obj) ||
        !get_argument(args, nargs, kwnames, 3, "dtype", NULL, &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 4, "out", NULL, &out_obj))
        return NULL;
    int64_t low, high;
    if (!parse_i64(low_obj, &low, "low")) return NULL;
    if (high_obj == NULL || high_obj == Py_None) {
        high = low;
        low = 0;
    }
    else if (!parse_i64(high_obj, &high, "high")) return NULL;
    if (low >= high) return (PyErr_SetString(PyExc_ValueError, "integers requires low < high"), NULL);

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    if (!parse_shape(size_obj, shape, &rank) || !count_elements(shape, rank, &total)) return NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, nk_i64_k, &dtype)) return NULL;
    if (!integer_dtype(dtype) || !validate_integer_bounds(dtype, low, high)) return NULL;

    RandomOutput output;
    if (!prepare_output(&output, out_obj, dtype, rank, shape)) {
        release_output(&output);
        return NULL;
    }
    PyThreadState *gil = PyEval_SaveThread();
    fill_integers((RandomGenerator *)self, output.data, dtype, total, low, high);
    PyEval_RestoreThread(gil);
    return finish_output(&output);
}

static PyObject *generator_state_get(PyObject *self, void *closure) {
    nk_unused_(closure);
    return PyLong_FromUnsignedLongLong(((RandomGenerator *)self)->generator.state);
}

static int generator_state_set(PyObject *self, PyObject *value, void *closure) {
    nk_unused_(closure);
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "state must be an integer");
        return -1;
    }
    uint64_t const state = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) return -1;
    nk_random_seed(&((RandomGenerator *)self)->generator, state);
    return 0;
}

static int RandomGenerator_init(RandomGenerator *self, PyObject *args, PyObject *kwargs) {
    PyObject *seed_obj = NULL;
    static char *keywords[] = {"seed", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:Generator", keywords, &seed_obj)) return -1;
    uint64_t seed = 0;
    if (seed_obj && seed_obj != Py_None) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "seed must be an integer or None");
            return -1;
        }
        seed = PyLong_AsUnsignedLongLong(seed_obj);
        if (PyErr_Occurred()) return -1;
    }
    nk_random_seed(&self->generator, seed);
    return 0;
}

static PyMethodDef RandomGenerator_methods[] = {
    {"random", (PyCFunction)generator_random, METH_FASTCALL | METH_KEYWORDS,
     "Generate uniform samples in [0, 1)."},
    {"uniform", (PyCFunction)generator_uniform, METH_FASTCALL | METH_KEYWORDS,
     "Generate uniform samples in [low, high)."},
    {"normal", (PyCFunction)generator_normal, METH_FASTCALL | METH_KEYWORDS,
     "Generate normal samples with the requested location and scale."},
    {"standard_normal", (PyCFunction)generator_standard_normal, METH_FASTCALL | METH_KEYWORDS,
     "Generate standard normal samples."},
    {"integers", (PyCFunction)generator_integers, METH_FASTCALL | METH_KEYWORDS,
     "Generate uniformly distributed integers in [low, high)."},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef RandomGenerator_getset[] = {
    {"state", generator_state_get, generator_state_set, "Current generator state.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject RandomGeneratorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.Generator",
    .tp_doc = "Independent stateful pseudo-random number generator.",
    .tp_basicsize = sizeof(RandomGenerator),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_methods = RandomGenerator_methods,
    .tp_getset = RandomGenerator_getset,
    .tp_init = (initproc)RandomGenerator_init,
    .tp_new = PyType_GenericNew,
};
