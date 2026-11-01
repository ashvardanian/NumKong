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
    PyObject_HEAD nk_random_generator_t generator;
    PyThread_type_lock lock;
} RandomGenerator;

typedef struct {
    Tensor *owned;
    Py_buffer buffer;
    nk_buffer_backing_t backing;
    char *data;
    int has_buffer;
} RandomOutput;

static int validate_arguments(Py_ssize_t nargs, Py_ssize_t maximum, PyObject *kwnames, char const *const *allowed,
                              char const *function_name) {
    if (nargs > maximum) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most %zd positional arguments (%zd given)", function_name, maximum,
                     nargs);
        return 0;
    }
    Py_ssize_t const keyword_count = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t i = 0; i < keyword_count; ++i) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        size_t j = 0;
        while (allowed[j] && PyUnicode_CompareWithASCIIString(name, allowed[j]) != 0) ++j;
        if (!allowed[j]) {
            PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%U'", function_name, name);
            return 0;
        }
    }
    return 1;
}

static int get_argument(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, Py_ssize_t position,
                        char const *name, PyObject **value) {
    *value = nargs > position ? args[position] : NULL;

    Py_ssize_t const keyword_count = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t i = 0; i < keyword_count; ++i) {
        PyObject *keyword = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(keyword, name) != 0) continue;
        if (*value) {
            PyErr_Format(PyExc_TypeError, "got multiple values for argument '%s'", name);
            return 0;
        }
        *value = args[nargs + i];
    }

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

static int parse_integer_bounds(nk_dtype_t dtype, PyObject *low_obj, PyObject *high_obj, uint64_t *low,
                                uint64_t *bound) {
    int is_unsigned;
    switch (dtype) {
    case nk_i8_k:
    case nk_i16_k:
    case nk_i32_k:
    case nk_i64_k: is_unsigned = 0; break;
    case nk_u8_k:
    case nk_u16_k:
    case nk_u32_k:
    case nk_u64_k: is_unsigned = 1; break;
    default:
        PyErr_SetString(PyExc_TypeError, "integers only supports 8-, 16-, 32-, and 64-bit integer dtypes");
        return 0;
    }
    if (!low_obj || !PyLong_Check(low_obj) || (high_obj && high_obj != Py_None && !PyLong_Check(high_obj))) {
        PyErr_SetString(PyExc_TypeError, "low and high must be integers");
        return 0;
    }
    if (!high_obj || high_obj == Py_None) {
        high_obj = low_obj;
        low_obj = NULL;
    }

    // Convert the inclusive upper endpoint so high may be 2**63 or 2**64.
    PyObject *one = PyLong_FromLong(1);
    if (!one) return 0;
    PyObject *last_obj = PyNumber_Subtract(high_obj, one);
    Py_DECREF(one);
    if (!last_obj) return 0;
    size_t const bits = nk_dtype_bits(dtype);
    uint64_t last;
    int valid;
    if (is_unsigned) {
        *low = low_obj ? PyLong_AsUnsignedLongLong(low_obj) : 0;
        if (PyErr_Occurred()) {
            Py_DECREF(last_obj);
            return 0;
        }
        last = PyLong_AsUnsignedLongLong(last_obj);
        valid = *low <= last && last <= (UINT64_MAX >> (64 - bits));
    }
    else {
        int64_t const lower = low_obj ? PyLong_AsLongLong(low_obj) : 0;
        if (PyErr_Occurred()) {
            Py_DECREF(last_obj);
            return 0;
        }
        int64_t const upper = PyLong_AsLongLong(last_obj);
        int64_t const maximum = INT64_MAX >> (64 - bits);
        valid = lower <= upper && lower >= -maximum - 1 && upper <= maximum;
        *low = (uint64_t)lower;
        last = (uint64_t)upper;
    }
    Py_DECREF(last_obj);
    if (PyErr_Occurred()) return 0;
    if (!valid) {
        PyErr_Format(PyExc_ValueError, "integers requires low < high within dtype '%s'", nk_dtype_name(dtype));
        return 0;
    }
    *bound = last - *low + 1;
    return 1;
}

static void fill_uniform(RandomGenerator *generator, char *data, nk_dtype_t dtype, size_t total, double low,
                         double high) {
    if (dtype == nk_f32_k) {
        float *values = (float *)data;
        float const lower = (float)low;
        float const width = (float)(high - low);
        for (size_t i = 0; i < total; ++i) values[i] = lower + width * nk_random_uniform_f32(&generator->generator);
    }
    else {
        double *values = (double *)data;
        for (size_t i = 0; i < total; ++i)
            values[i] = low + (high - low) * nk_random_uniform_f64(&generator->generator);
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

static void fill_integers(RandomGenerator *generator, char *data, nk_dtype_t dtype, size_t total, uint64_t low,
                          uint64_t bound) {
    for (size_t i = 0; i < total; ++i)
        write_integer(data, dtype, i, low + nk_random_bounded_u64(&generator->generator, bound));
}

static PyObject *generate_floating_output(RandomGenerator *generator, PyObject *out_obj, nk_dtype_t dtype, size_t rank,
                                          Py_ssize_t const *shape, size_t total,
                                          void (*fill)(RandomGenerator *, char *, nk_dtype_t, size_t, double, double),
                                          double first, double second) {
    RandomOutput output;
    if (!prepare_output(&output, out_obj, dtype, rank, shape)) {
        release_output(&output);
        return NULL;
    }
    PyThreadState *gil = PyEval_SaveThread();
    PyThread_acquire_lock(generator->lock, WAIT_LOCK);
    fill(generator, output.data, dtype, total, first, second);
    PyThread_release_lock(generator->lock);
    PyEval_RestoreThread(gil);
    return finish_output(&output);
}

static PyObject *generator_random(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    static char const *const allowed[] = {"size", "dtype", "out", NULL};
    if (!validate_arguments(nargs, 1, kwnames, allowed, "Generator.random")) return NULL;

    PyObject *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "size", &size_obj) ||
        !get_argument(args, nargs, kwnames, 1, "dtype", &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 2, "out", &out_obj))
        return NULL;

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    if (!parse_shape(size_obj, shape, &rank) || !count_elements(shape, rank, &total)) return NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, nk_f32_k, &dtype)) return NULL;
    if (!floating_dtype(dtype))
        return (PyErr_Format(PyExc_TypeError, "random only supports float32 and float64"), NULL);

    return generate_floating_output((RandomGenerator *)self, out_obj, dtype, rank, shape, total, fill_uniform, 0.0,
                                    1.0);
}

static PyObject *generator_uniform(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    static char const *const allowed[] = {"low", "high", "size", "dtype", "out", NULL};
    if (!validate_arguments(nargs, 3, kwnames, allowed, "Generator.uniform")) return NULL;

    PyObject *low_obj = NULL, *high_obj = NULL, *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "low", &low_obj) ||
        !get_argument(args, nargs, kwnames, 1, "high", &high_obj) ||
        !get_argument(args, nargs, kwnames, 2, "size", &size_obj) ||
        !get_argument(args, nargs, kwnames, 3, "dtype", &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 4, "out", &out_obj))
        return NULL;
    double low, high;
    if (!low_obj) low = 0.0;
    if (!high_obj) high = 1.0;
    if ((low_obj && !parse_f64(low_obj, &low, "low")) || (high_obj && !parse_f64(high_obj, &high, "high"))) return NULL;
    if (!isfinite(low) || !isfinite(high) || low >= high)
        return (PyErr_SetString(PyExc_ValueError, "uniform requires finite low < high"), NULL);

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    if (!parse_shape(size_obj, shape, &rank) || !count_elements(shape, rank, &total)) return NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, nk_f32_k, &dtype)) return NULL;
    if (!floating_dtype(dtype))
        return (PyErr_Format(PyExc_TypeError, "uniform only supports float32 and float64"), NULL);

    return generate_floating_output((RandomGenerator *)self, out_obj, dtype, rank, shape, total, fill_uniform, low,
                                    high);
}

static PyObject *generator_normal(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    static char const *const allowed[] = {"loc", "scale", "size", "dtype", "out", NULL};
    if (!validate_arguments(nargs, 3, kwnames, allowed, "Generator.normal")) return NULL;

    PyObject *location_obj = NULL, *scale_obj = NULL, *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "loc", &location_obj) ||
        !get_argument(args, nargs, kwnames, 1, "scale", &scale_obj) ||
        !get_argument(args, nargs, kwnames, 2, "size", &size_obj) ||
        !get_argument(args, nargs, kwnames, 3, "dtype", &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 4, "out", &out_obj))
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

    return generate_floating_output((RandomGenerator *)self, out_obj, dtype, rank, shape, total, fill_normal, location,
                                    scale);
}

static PyObject *generator_standard_normal(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    static char const *const allowed[] = {"size", "dtype", "out", NULL};
    if (!validate_arguments(nargs, 1, kwnames, allowed, "Generator.standard_normal")) return NULL;

    PyObject *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "size", &size_obj) ||
        !get_argument(args, nargs, kwnames, 1, "dtype", &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 2, "out", &out_obj))
        return NULL;

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    if (!parse_shape(size_obj, shape, &rank) || !count_elements(shape, rank, &total)) return NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, nk_f32_k, &dtype)) return NULL;
    if (!floating_dtype(dtype))
        return (PyErr_Format(PyExc_TypeError, "standard_normal only supports float32 and float64"), NULL);

    return generate_floating_output((RandomGenerator *)self, out_obj, dtype, rank, shape, total, fill_normal, 0.0, 1.0);
}

static PyObject *generator_integers(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    static char const *const allowed[] = {"low", "high", "size", "dtype", "out", NULL};
    if (!validate_arguments(nargs, 3, kwnames, allowed, "Generator.integers")) return NULL;

    PyObject *low_obj = NULL, *high_obj = NULL, *size_obj = NULL, *dtype_obj = NULL, *out_obj = NULL;
    if (!get_argument(args, nargs, kwnames, 0, "low", &low_obj) ||
        !get_argument(args, nargs, kwnames, 1, "high", &high_obj) ||
        !get_argument(args, nargs, kwnames, 2, "size", &size_obj) ||
        !get_argument(args, nargs, kwnames, 3, "dtype", &dtype_obj) ||
        !get_argument(args, nargs, kwnames, 4, "out", &out_obj))
        return NULL;
    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    if (!parse_shape(size_obj, shape, &rank) || !count_elements(shape, rank, &total)) return NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, nk_i64_k, &dtype)) return NULL;
    uint64_t low, bound;
    if (!parse_integer_bounds(dtype, low_obj, high_obj, &low, &bound)) return NULL;

    RandomOutput output;
    if (!prepare_output(&output, out_obj, dtype, rank, shape)) {
        release_output(&output);
        return NULL;
    }
    RandomGenerator *generator = (RandomGenerator *)self;
    PyThreadState *gil = PyEval_SaveThread();
    PyThread_acquire_lock(generator->lock, WAIT_LOCK);
    fill_integers(generator, output.data, dtype, total, low, bound);
    PyThread_release_lock(generator->lock);
    PyEval_RestoreThread(gil);
    return finish_output(&output);
}

static void lock_generator(RandomGenerator *self) {
    if (PyThread_acquire_lock(self->lock, NOWAIT_LOCK)) return;
    PyThreadState *gil = PyEval_SaveThread();
    PyThread_acquire_lock(self->lock, WAIT_LOCK);
    PyEval_RestoreThread(gil);
}

static PyObject *RandomGenerator_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    nk_unused_(args);
    nk_unused_(kwargs);
    RandomGenerator *self = (RandomGenerator *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->lock = PyThread_allocate_lock();
    if (!self->lock) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    return (PyObject *)self;
}

static void RandomGenerator_dealloc(RandomGenerator *self) {
    if (self->lock) PyThread_free_lock(self->lock);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *generator_state_get(PyObject *self, void *closure) {
    nk_unused_(closure);
    RandomGenerator *generator = (RandomGenerator *)self;
    lock_generator(generator);
    uint64_t const state = generator->generator.state;
    PyThread_release_lock(generator->lock);
    return PyLong_FromUnsignedLongLong(state);
}

static int generator_state_set(PyObject *self, PyObject *value, void *closure) {
    nk_unused_(closure);
    if (!value || !PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "state must be an integer");
        return -1;
    }
    uint64_t const state = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) return -1;
    RandomGenerator *generator = (RandomGenerator *)self;
    lock_generator(generator);
    nk_random_seed(&generator->generator, state);
    PyThread_release_lock(generator->lock);
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
    lock_generator(self);
    nk_random_seed(&self->generator, seed);
    PyThread_release_lock(self->lock);
    return 0;
}

static PyMethodDef RandomGenerator_methods[] = {
    {"random", (PyCFunction)generator_random, METH_FASTCALL | METH_KEYWORDS, "Generate uniform samples in [0, 1)."},
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
    .tp_dealloc = (destructor)RandomGenerator_dealloc,
    .tp_methods = RandomGenerator_methods,
    .tp_getset = RandomGenerator_getset,
    .tp_init = (initproc)RandomGenerator_init,
    .tp_new = RandomGenerator_new,
};
