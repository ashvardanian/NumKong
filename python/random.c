/**
 *  @brief Stateful native sampling with buffer-protocol parameters and outputs.
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
    PyObject *array_factory;
} RandomGenerator;

typedef struct {
    PyObject *owned;
    Py_buffer buffer;
    nk_buffer_backing_t backing;
    char *data;
    int has_buffer;
} RandomOutput;

typedef struct {
    Py_buffer buffer;
    nk_buffer_backing_t backing;
    nk_dtype_t dtype;
    double scalar;
    uint64_t integer;
    int has_buffer, linear, upper_exclusive;
    size_t rank;
    Py_ssize_t shape[NK_TENSOR_MAX_RANK], strides[NK_TENSOR_MAX_RANK];
} RandomParameter;

typedef enum {
    random_uniform_k,
    random_normal_k,
    random_laplace_k,
    random_beta_k,
    random_poisson_k
} RandomDistribution;

static void lock_generator(RandomGenerator *self) {
    if (PyThread_acquire_lock(self->lock, NOWAIT_LOCK)) return;
    PyThreadState *gil = PyEval_SaveThread();
    PyThread_acquire_lock(self->lock, WAIT_LOCK);
    PyEval_RestoreThread(gil);
}

static int parse_shape(PyObject *size, Py_ssize_t *shape, size_t *rank) {
    if (PyIndex_Check(size)) {
        shape[0] = PyNumber_AsSsize_t(size, PyExc_OverflowError);
        *rank = 1;
        if (PyErr_Occurred()) return 0;
        if (shape[0] >= 0) return 1;
    }
    else if (PyTuple_Check(size) || PyList_Check(size)) {
        PyObject *sequence = PySequence_Fast(size, "size must contain integers");
        if (!sequence) return 0;
        *rank = (size_t)PySequence_Fast_GET_SIZE(sequence);
        if (*rank > NK_TENSOR_MAX_RANK) {
            Py_DECREF(sequence);
            PyErr_SetString(PyExc_ValueError, "size has too many dimensions");
            return 0;
        }
        for (size_t i = 0; i < *rank; ++i) {
            shape[i] = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(sequence, i), PyExc_OverflowError);
            if (PyErr_Occurred() || shape[i] < 0) {
                Py_DECREF(sequence);
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "size dimensions must be non-negative");
                return 0;
            }
        }
        Py_DECREF(sequence);
        return 1;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "size must be an integer or tuple of integers");
        return 0;
    }
    PyErr_SetString(PyExc_ValueError, "size dimensions must be non-negative");
    return 0;
}

static int count_elements(Py_ssize_t const *shape, size_t rank, size_t *total) {
    *total = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (shape[i] && *total > (size_t)PY_SSIZE_T_MAX / (size_t)shape[i]) {
            PyErr_SetString(PyExc_OverflowError, "size is too large");
            return 0;
        }
        *total *= (size_t)shape[i];
    }
    return 1;
}

static int parse_dtype(PyObject *object, nk_dtype_t fallback, nk_dtype_t *dtype) {
    if (object && PyType_Check(object)) {
        char const *name = ((PyTypeObject *)object)->tp_name;
        if (strncmp(name, "numpy.", 6) == 0) {
            *dtype = py_string_to_nk_dtype(name + 6, (Py_ssize_t)strlen(name + 6));
            if (*dtype != nk_dtype_unknown_k) return 1;
        }
    }
    if (object && !PyType_Check(object) && !PyUnicode_Check(object) && PyObject_HasAttrString(object, "name")) {
        PyObject *native = PyObject_GetAttrString(object, "isnative");
        if (!native) return 0;
        int const is_native = PyObject_IsTrue(native);
        Py_DECREF(native);
        if (is_native <= 0) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "dtype must use native byte order");
            return 0;
        }
        PyObject *name = PyObject_GetAttrString(object, "name");
        if (!name) return 0;
        *dtype = py_object_to_nk_dtype(name);
        Py_DECREF(name);
        return *dtype != nk_dtype_unknown_k;
    }
    *dtype = object ? py_object_to_nk_dtype(object) : fallback;
    return *dtype != nk_dtype_unknown_k;
}

static int parameter_buffer(RandomParameter *parameter, PyObject *object) {
    if (!nk_get_buffer(object, &parameter->buffer, PyBUF_FORMAT | PyBUF_STRIDES, &parameter->backing)) return 0;
    parameter->has_buffer = 1;
    parameter->dtype = resolve_nk_dtype_in_py_buffer(&parameter->buffer);
    if (parameter->buffer.ndim > NK_TENSOR_MAX_RANK || parameter->dtype == nk_dtype_unknown_k ||
        nk_dtype_bits(parameter->dtype) < 8 || nk_dtype_family(parameter->dtype) == nk_dtype_family_complex_float_k) {
        PyErr_SetString(PyExc_TypeError, "parameters must contain real fixed-width numbers");
        return 0;
    }
    parameter->rank = (size_t)parameter->buffer.ndim;
    return 1;
}

static int parameter_float(RandomParameter *parameter, PyObject *object, double fallback) {
    if (!object) {
        parameter->scalar = fallback;
        return 1;
    }
    if (PyObject_CheckBuffer(object)) return parameter_buffer(parameter, object);
    parameter->scalar = PyFloat_AsDouble(object);
    return !PyErr_Occurred();
}

static char const *parameter_pointer(RandomParameter const *parameter, size_t index) {
    if (parameter->linear) return (char const *)parameter->buffer.buf + index * parameter->buffer.itemsize;
    Py_ssize_t offset = 0;
    for (size_t i = parameter->rank; i; --i) {
        Py_ssize_t const dimension = parameter->shape[i - 1];
        if (!dimension) break;
        offset += (Py_ssize_t)(index % (size_t)dimension) * parameter->strides[i - 1];
        index /= (size_t)dimension;
    }
    return (char const *)parameter->buffer.buf + offset;
}

static double parameter_value(RandomParameter const *parameter, size_t index) {
    if (!parameter->has_buffer) return parameter->scalar;
    nk_scalar_buffer_t value;
    memcpy(&value, parameter_pointer(parameter, index), (size_t)parameter->buffer.itemsize);
    double result;
    nk_scalar_buffer_to_f64(&value, parameter->dtype, &result);
    return result;
}

static int broadcast_parameter(RandomParameter *parameter, size_t rank, Py_ssize_t const *shape) {
    if (!parameter->has_buffer) return 1;
    size_t const source_rank = (size_t)parameter->buffer.ndim;
    if (source_rank > rank) goto mismatch;
    parameter->linear = source_rank == rank && PyBuffer_IsContiguous(&parameter->buffer, 'C');
    parameter->rank = rank;
    for (size_t i = 0; i < rank; ++i) {
        Py_ssize_t const source_dimension = i < rank - source_rank ? 1
                                                                   : parameter->buffer.shape[i - (rank - source_rank)];
        if (source_dimension != 1 && source_dimension != shape[i]) goto mismatch;
        parameter->shape[i] = shape[i];
        parameter->strides[i] = source_dimension == 1 ? 0 : parameter->buffer.strides[i - (rank - source_rank)];
        if (source_dimension != shape[i]) parameter->linear = 0;
    }
    return 1;
mismatch:
    PyErr_SetString(PyExc_ValueError, "parameters cannot broadcast to the requested size");
    return 0;
}

static int output_shape(PyObject *size, PyObject *out, RandomParameter *first, RandomParameter *second,
                        Py_ssize_t *shape, size_t *rank, size_t *total, int *scalar) {
    *scalar = (!size || size == Py_None) && (!out || out == Py_None);
    *rank = 0;
    if (size && size != Py_None) {
        if (!parse_shape(size, shape, rank)) return 0;
    }
    else if (out && out != Py_None) {
        Py_buffer buffer;
        if (PyObject_GetBuffer(out, &buffer, PyBUF_ND) < 0) return 0;
        *rank = (size_t)buffer.ndim;
        if (*rank > NK_TENSOR_MAX_RANK) {
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "out has too many dimensions");
            return 0;
        }
        memcpy(shape, buffer.shape, *rank * sizeof(Py_ssize_t));
        PyBuffer_Release(&buffer);
    }
    else {
        *rank = first->rank > second->rank ? first->rank : second->rank;
        *scalar = *rank == 0;
        for (size_t i = 0; i < *rank; ++i) {
            Py_ssize_t const a = i < *rank - first->rank ? 1 : first->buffer.shape[i - (*rank - first->rank)];
            Py_ssize_t const b = i < *rank - second->rank ? 1 : second->buffer.shape[i - (*rank - second->rank)];
            if (a != b && a != 1 && b != 1) {
                PyErr_SetString(PyExc_ValueError, "parameter shapes cannot broadcast");
                return 0;
            }
            shape[i] = a == 1 ? b : a;
        }
    }
    return count_elements(shape, *rank, total) && broadcast_parameter(first, *rank, shape) &&
           broadcast_parameter(second, *rank, shape);
}

static int prepare_output(RandomGenerator *generator, RandomOutput *output, PyObject *out, nk_dtype_t dtype,
                          size_t rank, Py_ssize_t const *shape) {
    if (!out || out == Py_None) {
        if (!generator->array_factory) {
            output->owned = (PyObject *)Tensor_new(dtype, rank, shape);
            if (!output->owned) return 0;
            output->data = ((Tensor *)output->owned)->data;
            return 1;
        }
        PyObject *dimensions = PyTuple_New((Py_ssize_t)rank);
        if (!dimensions) return 0;
        for (size_t i = 0; i < rank; ++i) {
            PyObject *dimension = PyLong_FromSsize_t(shape[i]);
            if (!dimension) {
                Py_DECREF(dimensions);
                return 0;
            }
            PyTuple_SET_ITEM(dimensions, (Py_ssize_t)i, dimension);
        }
        PyObject *args = PyTuple_Pack(1, dimensions);
        PyObject *keywords = Py_BuildValue("{s:s}", "dtype", nk_dtype_name(dtype));
        Py_DECREF(dimensions);
        if (!args || !keywords) {
            Py_XDECREF(args);
            Py_XDECREF(keywords);
            return 0;
        }
        output->owned = PyObject_Call(generator->array_factory, args, keywords);
        Py_DECREF(args);
        Py_DECREF(keywords);
        if (!output->owned) return 0;
        out = output->owned;
    }
    if (!nk_get_buffer(out, &output->buffer, PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_WRITABLE, &output->backing)) return 0;
    output->has_buffer = 1;
    if (resolve_nk_dtype_in_py_buffer(&output->buffer) != dtype) {
        PyErr_SetString(PyExc_TypeError, "out dtype does not match requested dtype");
        return 0;
    }
    if ((size_t)output->buffer.ndim != rank) goto shape_error;
    for (size_t i = 0; i < rank; ++i)
        if (output->buffer.shape[i] != shape[i]) goto shape_error;
    if (!PyBuffer_IsContiguous(&output->buffer, 'C')) {
        PyErr_SetString(PyExc_ValueError, "out must be C-contiguous");
        return 0;
    }
    if ((uintptr_t)output->buffer.buf % nk_dtype_bytes_per_value(dtype)) {
        PyErr_SetString(PyExc_ValueError, "out must be aligned for its dtype");
        return 0;
    }
    output->data = output->buffer.buf;
    return 1;
shape_error:
    PyErr_SetString(PyExc_ValueError, "out shape does not match size");
    return 0;
}

static void release_output(RandomOutput *output) {
    if (output->has_buffer) PyBuffer_Release(&output->buffer);
    Py_XDECREF(output->owned);
}

static PyObject *finish_output(RandomOutput *output) {
    if (output->has_buffer) PyBuffer_Release(&output->buffer);
    if (output->owned) return output->owned;
    Py_RETURN_NONE;
}

static void release_parameter(RandomParameter *parameter) {
    if (parameter->has_buffer) PyBuffer_Release(&parameter->buffer);
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

NK_INTERNAL uint32_t bounded_u32(nk_random_generator_t *generator, uint32_t candidate, uint32_t bound,
                                 uint32_t threshold) {
    if (!bound) return candidate;
    uint64_t product = (uint64_t)candidate * bound;
    while ((uint32_t)product < threshold) { product = (uint64_t)(uint32_t)nk_random_next_u64(generator) * bound; }
    return (uint32_t)(product >> 32);
}

static double sample_floating(nk_random_generator_t *generator, RandomDistribution distribution, double first,
                              double second) {
    switch (distribution) {
    case random_uniform_k: return first + (second - first) * nk_random_uniform_f64(generator);
    case random_normal_k: return nk_random_normal_f64(generator, first, second);
    case random_laplace_k: return nk_random_laplace_f64(generator, first, second);
    case random_beta_k: return nk_random_beta_f64(generator, first, second);
    default: return 0;
    }
}

static int valid_parameters(RandomDistribution distribution, double first, double second) {
    if (!isfinite(first) || !isfinite(second)) return 0;
    switch (distribution) {
    case random_uniform_k: return first <= second && isfinite(second - first);
    case random_normal_k:
    case random_laplace_k: return second >= 0;
    case random_beta_k: return first > 0 && second > 0;
    case random_poisson_k: return first >= 0 && first <= (double)INT64_MAX - 10.0 * sqrt((double)INT64_MAX);
    }
    return 0;
}

static void fill_floating(nk_random_generator_t *generator, RandomDistribution distribution, char *output,
                          nk_dtype_t dtype, size_t total, RandomParameter const *first, RandomParameter const *second) {
    if (!first->has_buffer && !second->has_buffer) {
        double const a = first->scalar, b = second->scalar;
        if (distribution == random_uniform_k) {
            if (dtype == nk_f32_k) {
                for (size_t i = 0; i < total; ++i)
                    ((float *)output)[i] = (float)a + (float)(b - a) * nk_random_uniform_f32(generator);
            }
            else
                for (size_t i = 0; i < total; ++i)
                    ((double *)output)[i] = a + (b - a) * nk_random_uniform_f64(generator);
            return;
        }
        if (distribution == random_normal_k) {
            if (dtype == nk_f32_k) {
                for (size_t i = 0; i < total; ++i)
                    ((float *)output)[i] = (float)a + (float)b * nk_random_standard_normal_f32_(generator);
            }
            else
                for (size_t i = 0; i < total; ++i)
                    ((double *)output)[i] = a + b * nk_random_standard_normal_f64_(generator);
            return;
        }
        if (distribution == random_laplace_k) {
            if (dtype == nk_f32_k) {
                for (size_t i = 0; i < total; ++i) ((float *)output)[i] = (float)nk_random_laplace_f64(generator, a, b);
            }
            else
                for (size_t i = 0; i < total; ++i) ((double *)output)[i] = nk_random_laplace_f64(generator, a, b);
            return;
        }
        if (distribution == random_beta_k) {
            if (dtype == nk_f32_k) {
                for (size_t i = 0; i < total; ++i) ((float *)output)[i] = (float)nk_random_beta_f64(generator, a, b);
            }
            else
                for (size_t i = 0; i < total; ++i) ((double *)output)[i] = nk_random_beta_f64(generator, a, b);
            return;
        }
        if (distribution == random_poisson_k) {
            for (size_t i = 0; i < total; ++i) ((int64_t *)output)[i] = nk_random_poisson_i64(generator, a);
            return;
        }
    }
    for (size_t i = 0; i < total; ++i) {
        double const a = parameter_value(first, i), b = parameter_value(second, i);
        if (distribution == random_poisson_k) ((int64_t *)output)[i] = nk_random_poisson_i64(generator, a);
        else if (dtype == nk_f32_k && distribution == random_normal_k)
            ((float *)output)[i] = (float)a + (float)b * nk_random_standard_normal_f32_(generator);
        else if (dtype == nk_f32_k && distribution == random_uniform_k)
            ((float *)output)[i] = (float)a + (float)(b - a) * nk_random_uniform_f32(generator);
        else {
            double const value = sample_floating(generator, distribution, a, b);
            if (dtype == nk_f32_k) ((float *)output)[i] = (float)value;
            else ((double *)output)[i] = value;
        }
    }
}

static PyObject *generate_distribution(RandomGenerator *self, RandomDistribution distribution, PyObject *first_obj,
                                       PyObject *second_obj, double first_default, double second_default,
                                       PyObject *size, PyObject *dtype_obj, PyObject *out) {
    RandomParameter first = {0}, second = {0};
    RandomOutput output = {0};
    PyObject *result = NULL;
    nk_dtype_t dtype;
    if (!parse_dtype(dtype_obj, distribution == random_poisson_k ? nk_i64_k : nk_f64_k, &dtype)) goto cleanup;
    if ((distribution == random_poisson_k && dtype != nk_i64_k) ||
        (distribution != random_poisson_k && dtype != nk_f32_k && dtype != nk_f64_k)) {
        PyErr_SetString(PyExc_TypeError, "floating sampling supports float32 and float64; poisson returns int64");
        goto cleanup;
    }
    if (!parameter_float(&first, first_obj, first_default) || !parameter_float(&second, second_obj, second_default))
        goto cleanup;
    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    int scalar;
    if (!output_shape(size, out, &first, &second, shape, &rank, &total, &scalar)) goto cleanup;
    size_t const checks = first.has_buffer || second.has_buffer ? total : 1;
    for (size_t i = 0; i < checks; ++i) {
        if (!valid_parameters(distribution, parameter_value(&first, i), parameter_value(&second, i))) {
            PyErr_SetString(PyExc_ValueError, "distribution parameters are outside their finite domain");
            goto cleanup;
        }
    }
    if (scalar) {
        lock_generator(self);
        if (distribution == random_poisson_k)
            result = PyLong_FromLongLong(nk_random_poisson_i64(&self->generator, parameter_value(&first, 0)));
        else {
            double const a = parameter_value(&first, 0), b = parameter_value(&second, 0);
            double value;
            if (dtype == nk_f32_k && distribution == random_normal_k)
                value = (float)a + (float)b * nk_random_standard_normal_f32_(&self->generator);
            else if (dtype == nk_f32_k && distribution == random_uniform_k)
                value = (float)a + (float)(b - a) * nk_random_uniform_f32(&self->generator);
            else {
                value = sample_floating(&self->generator, distribution, a, b);
                if (dtype == nk_f32_k) value = (float)value;
            }
            result = PyFloat_FromDouble(value);
        }
        PyThread_release_lock(self->lock);
        goto cleanup;
    }
    if (!prepare_output(self, &output, out, dtype, rank, shape)) goto cleanup;
    PyThreadState *gil = PyEval_SaveThread();
    PyThread_acquire_lock(self->lock, WAIT_LOCK);
    fill_floating(&self->generator, distribution, output.data, dtype, total, &first, &second);
    PyThread_release_lock(self->lock);
    PyEval_RestoreThread(gil);
    result = finish_output(&output);
    output.owned = NULL;
    output.has_buffer = 0;
cleanup:
    release_output(&output);
    release_parameter(&first);
    release_parameter(&second);
    return result;
}

static PyObject *generator_random(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *size = NULL, *dtype = NULL, *out = NULL;
    static char *names[] = {"size", "dtype", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO:random", names, &size, &dtype, &out)) return NULL;
    return generate_distribution((RandomGenerator *)object, random_uniform_k, NULL, NULL, 0, 1, size, dtype, out);
}

static PyObject *generator_standard_normal(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *size = NULL, *dtype = NULL, *out = NULL;
    static char *names[] = {"size", "dtype", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO:standard_normal", names, &size, &dtype, &out)) return NULL;
    return generate_distribution((RandomGenerator *)object, random_normal_k, NULL, NULL, 0, 1, size, dtype, out);
}

static PyObject *generator_uniform(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *first = NULL, *second = NULL, *size = NULL, *dtype = NULL, *out = NULL;
    static char *names[] = {"low", "high", "size", "dtype", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO$OO:uniform", names, &first, &second, &size, &dtype, &out))
        return NULL;
    return generate_distribution((RandomGenerator *)object, random_uniform_k, first, second, 0, 1, size, dtype, out);
}

static PyObject *generator_normal(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *first = NULL, *second = NULL, *size = NULL, *dtype = NULL, *out = NULL;
    static char *names[] = {"loc", "scale", "size", "dtype", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO$OO:normal", names, &first, &second, &size, &dtype, &out))
        return NULL;
    return generate_distribution((RandomGenerator *)object, random_normal_k, first, second, 0, 1, size, dtype, out);
}

static PyObject *generator_laplace(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *first = NULL, *second = NULL, *size = NULL, *dtype = NULL, *out = NULL;
    static char *names[] = {"loc", "scale", "size", "dtype", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO$OO:laplace", names, &first, &second, &size, &dtype, &out))
        return NULL;
    return generate_distribution((RandomGenerator *)object, random_laplace_k, first, second, 0, 1, size, dtype, out);
}

static PyObject *generator_beta(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *first = NULL, *second = NULL, *size = NULL, *dtype = NULL, *out = NULL;
    static char *names[] = {"a", "b", "size", "dtype", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O$OO:beta", names, &first, &second, &size, &dtype, &out))
        return NULL;
    return generate_distribution((RandomGenerator *)object, random_beta_k, first, second, 0, 0, size, dtype, out);
}

static PyObject *generator_poisson(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *lambda = NULL, *size = NULL, *out = NULL;
    static char *names[] = {"lam", "size", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO$O:poisson", names, &lambda, &size, &out)) return NULL;
    return generate_distribution((RandomGenerator *)object, random_poisson_k, lambda, NULL, 1, 0, size, NULL, out);
}

static int integer_dtype(nk_dtype_t dtype, int *is_unsigned) {
    switch (dtype) {
    case nk_i8_k:
    case nk_i16_k:
    case nk_i32_k:
    case nk_i64_k: *is_unsigned = 0; return 1;
    case nk_u8_k:
    case nk_u16_k:
    case nk_u32_k:
    case nk_u64_k: *is_unsigned = 1; return 1;
    default: PyErr_SetString(PyExc_TypeError, "integers requires an 8-, 16-, 32-, or 64-bit integer dtype"); return 0;
    }
}

static int parameter_integer(RandomParameter *parameter, PyObject *object, int is_unsigned, int upper_exclusive) {
    if (PyObject_CheckBuffer(object)) {
        if (!parameter_buffer(parameter, object)) return 0;
        int input_unsigned;
        if (!integer_dtype(parameter->dtype, &input_unsigned)) return 0;
        parameter->upper_exclusive = upper_exclusive;
        return 1;
    }
    PyObject *value = PyNumber_Index(object);
    if (!value) return 0;
    if (upper_exclusive) {
        PyObject *one = PyLong_FromLong(1);
        PyObject *last = one ? PyNumber_Subtract(value, one) : NULL;
        Py_DECREF(value);
        Py_XDECREF(one);
        if (!last) return 0;
        value = last;
    }
    parameter->integer = is_unsigned ? PyLong_AsUnsignedLongLong(value) : (uint64_t)PyLong_AsLongLong(value);
    Py_DECREF(value);
    return !PyErr_Occurred();
}

static int integer_value(RandomParameter const *parameter, size_t index, int target_unsigned, uint64_t *result) {
    if (!parameter->has_buffer) {
        *result = parameter->integer;
        return 1;
    }
    nk_scalar_buffer_t value;
    memcpy(&value, parameter_pointer(parameter, index), (size_t)parameter->buffer.itemsize);
    uint64_t raw;
    int source_unsigned = 0;
    switch (parameter->dtype) {
    case nk_i8_k: raw = (uint64_t)value.i8; break;
    case nk_i16_k: raw = (uint64_t)value.i16; break;
    case nk_i32_k: raw = (uint64_t)value.i32; break;
    case nk_i64_k: raw = (uint64_t)value.i64; break;
    case nk_u8_k:
        raw = value.u8;
        source_unsigned = 1;
        break;
    case nk_u16_k:
        raw = value.u16;
        source_unsigned = 1;
        break;
    case nk_u32_k:
        raw = value.u32;
        source_unsigned = 1;
        break;
    case nk_u64_k:
        raw = value.u64;
        source_unsigned = 1;
        break;
    default: return 0;
    }
    if (parameter->upper_exclusive) {
        if ((source_unsigned && raw == 0) || (!source_unsigned && raw == (UINT64_C(1) << 63))) return 0;
        --raw;
    }
    if (target_unsigned && !source_unsigned && (int64_t)raw < 0) return 0;
    if (!target_unsigned && source_unsigned && raw > INT64_MAX) return 0;
    *result = raw;
    return 1;
}

static int integer_bounds(RandomParameter const *first, RandomParameter const *second, size_t index, nk_dtype_t dtype,
                          int is_unsigned, uint64_t *low, uint64_t *bound) {
    uint64_t last;
    if (!integer_value(first, index, is_unsigned, low) || !integer_value(second, index, is_unsigned, &last)) return 0;
    unsigned int const bits = nk_dtype_bits(dtype);
    if (is_unsigned) {
        if (*low > last || last > (UINT64_MAX >> (64 - bits))) return 0;
    }
    else {
        int64_t const maximum = INT64_MAX >> (64 - bits);
        if ((int64_t)*low > (int64_t)last || (int64_t)*low < -maximum - 1 || (int64_t)last > maximum) return 0;
    }
    *bound = last - *low + 1;
    return 1;
}

static PyObject *generator_integers(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *low_obj, *high_obj = Py_None, *size = NULL, *dtype_obj = NULL, *out = NULL;
    int endpoint = 0;
    static char *names[] = {"low", "high", "size", "dtype", "endpoint", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOOp$O:integers", names, &low_obj, &high_obj, &size, &dtype_obj,
                                     &endpoint, &out))
        return NULL;
    RandomGenerator *self = (RandomGenerator *)object;
    RandomParameter low = {0}, high = {0};
    RandomOutput output = {0};
    PyObject *result = NULL, *zero = NULL;
    nk_dtype_t dtype;
    int is_unsigned;
    if (!parse_dtype(dtype_obj, nk_i64_k, &dtype) || !integer_dtype(dtype, &is_unsigned)) goto cleanup;
    if (high_obj == Py_None) {
        high_obj = low_obj;
        zero = PyLong_FromLong(0);
        if (!zero) goto cleanup;
        low_obj = zero;
    }
    if (!parameter_integer(&low, low_obj, is_unsigned, 0) ||
        !parameter_integer(&high, high_obj, is_unsigned, !endpoint))
        goto cleanup;
    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank, total;
    int scalar;
    if (!output_shape(size, out, &low, &high, shape, &rank, &total, &scalar)) goto cleanup;
    uint64_t lower, bound;
    size_t const checks = low.has_buffer || high.has_buffer ? total : 1;
    for (size_t i = 0; i < checks; ++i) {
        if (!integer_bounds(&low, &high, i, dtype, is_unsigned, &lower, &bound)) {
            PyErr_SetString(PyExc_ValueError, "integer bounds are outside dtype or low exceeds the upper endpoint");
            goto cleanup;
        }
    }
    if (scalar) {
        integer_bounds(&low, &high, 0, dtype, is_unsigned, &lower, &bound);
        lock_generator(self);
        uint64_t const value = lower + nk_random_bounded_u64(&self->generator, bound);
        PyThread_release_lock(self->lock);
        result = is_unsigned ? PyLong_FromUnsignedLongLong(value) : PyLong_FromLongLong((int64_t)value);
        goto cleanup;
    }
    if (!prepare_output(self, &output, out, dtype, rank, shape)) goto cleanup;
    PyThreadState *gil = PyEval_SaveThread();
    PyThread_acquire_lock(self->lock, WAIT_LOCK);
    if (!low.has_buffer && !high.has_buffer) {
        integer_bounds(&low, &high, 0, dtype, is_unsigned, &lower, &bound);
        if (dtype == nk_u8_k && lower == 0 && bound == 256) {
            size_t i = 0;
            for (; total - i >= sizeof(uint64_t); i += sizeof(uint64_t)) {
                uint64_t const bits = nk_random_next_u64(&self->generator);
                memcpy(output.data + i, &bits, sizeof(bits));
            }
            if (i < total) {
                uint64_t const bits = nk_random_next_u64(&self->generator);
                memcpy(output.data + i, &bits, total - i);
            }
        }
        else if (nk_dtype_bits(dtype) <= 32) {
            uint32_t const bound32 = (uint32_t)bound;
            uint32_t const threshold = bound32 ? (uint32_t)(0 - bound32) % bound32 : 0;
            for (size_t i = 0; i < total; i += 2) {
                uint64_t const bits = nk_random_next_u64(&self->generator);
                uint32_t const first = bounded_u32(&self->generator, (uint32_t)bits, bound32, threshold);
                write_integer(output.data, dtype, i, lower + first);
                if (i + 1 < total) {
                    uint32_t const second = bounded_u32(&self->generator, (uint32_t)(bits >> 32), bound32, threshold);
                    write_integer(output.data, dtype, i + 1, lower + second);
                }
            }
        }
        else
            for (size_t i = 0; i < total; ++i)
                write_integer(output.data, dtype, i, lower + nk_random_bounded_u64(&self->generator, bound));
    }
    else
        for (size_t i = 0; i < total; ++i) {
            integer_bounds(&low, &high, i, dtype, is_unsigned, &lower, &bound);
            write_integer(output.data, dtype, i, lower + nk_random_bounded_u64(&self->generator, bound));
        }
    PyThread_release_lock(self->lock);
    PyEval_RestoreThread(gil);
    result = finish_output(&output);
    output.owned = NULL;
    output.has_buffer = 0;
cleanup:
    Py_XDECREF(zero);
    release_output(&output);
    release_parameter(&low);
    release_parameter(&high);
    return result;
}

static PyObject *generator_shuffle(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *values;
    Py_ssize_t axis = 0;
    static char *names[] = {"x", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|n:shuffle", names, &values, &axis)) return NULL;
    RandomGenerator *self = (RandomGenerator *)object;
    if (PyObject_CheckBuffer(values)) {
        Py_buffer buffer;
        if (PyObject_GetBuffer(values, &buffer, PyBUF_STRIDES | PyBUF_WRITABLE) < 0) return NULL;
        if (axis < 0) axis += buffer.ndim;
        if (!buffer.ndim || axis != 0 || !PyBuffer_IsContiguous(&buffer, 'C')) {
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "shuffle currently requires a C-contiguous array and axis=0");
            return NULL;
        }
        size_t const rows = (size_t)buffer.shape[0];
        size_t const row_bytes = rows ? (size_t)buffer.len / rows : 0;
        if (rows < 2 || !row_bytes) {
            PyBuffer_Release(&buffer);
            Py_RETURN_NONE;
        }
        char *temporary = PyMem_Malloc(row_bytes ? row_bytes : 1);
        if (!temporary) {
            PyBuffer_Release(&buffer);
            return PyErr_NoMemory();
        }
        PyThreadState *gil = PyEval_SaveThread();
        PyThread_acquire_lock(self->lock, WAIT_LOCK);
        for (size_t i = rows; i > 1; --i) {
            size_t const selected = (size_t)nk_random_bounded_u64(&self->generator, i);
            if (selected == i - 1) continue;
            char *last = (char *)buffer.buf + (i - 1) * row_bytes, *other = (char *)buffer.buf + selected * row_bytes;
            memcpy(temporary, last, row_bytes);
            memcpy(last, other, row_bytes);
            memcpy(other, temporary, row_bytes);
        }
        PyThread_release_lock(self->lock);
        PyEval_RestoreThread(gil);
        PyMem_Free(temporary);
        PyBuffer_Release(&buffer);
        Py_RETURN_NONE;
    }
    if (axis != 0 && axis != -1) {
        PyErr_SetString(PyExc_ValueError, "sequence shuffle requires axis=0");
        return NULL;
    }
    if (!PyList_Check(values)) {
        PyErr_SetString(PyExc_TypeError, "shuffle requires a list or writable array");
        return NULL;
    }
    lock_generator(self);
    for (Py_ssize_t i = PyList_GET_SIZE(values); i > 1; --i) {
        Py_ssize_t const selected = (Py_ssize_t)nk_random_bounded_u64(&self->generator, (uint64_t)i);
        PyObject *last = PyList_GET_ITEM(values, i - 1), *other = PyList_GET_ITEM(values, selected);
        PyList_SET_ITEM(values, i - 1, other);
        PyList_SET_ITEM(values, selected, last);
    }
    PyThread_release_lock(self->lock);
    Py_RETURN_NONE;
}

static size_t choice_index(nk_random_generator_t *generator, double const *weights, size_t population, double total) {
    if (!weights) return (size_t)nk_random_bounded_u64(generator, population);
    double const target = nk_random_uniform_f64(generator) * total;
    double cumulative = 0;
    size_t last_positive = 0;
    for (size_t i = 0; i < population; ++i) {
        cumulative += weights[i];
        if (weights[i] > 0) last_positive = i;
        if (target < cumulative) return i;
    }
    return last_positive;
}

static int choice_insert(size_t *table, size_t capacity, size_t value) {
    size_t slot = (size_t)((uint64_t)value * UINT64_C(0x9E3779B97F4A7C15)) & (capacity - 1);
    while (table[slot] != SIZE_MAX) {
        if (table[slot] == value) return 0;
        slot = (slot + 1) & (capacity - 1);
    }
    table[slot] = value;
    return 1;
}

static PyObject *generator_choice(PyObject *object, PyObject *args, PyObject *kwargs) {
    PyObject *a, *size = NULL, *p = Py_None;
    int replace = 1, shuffle = 1;
    Py_ssize_t axis = 0;
    static char *names[] = {"a", "size", "replace", "p", "axis", "shuffle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OpOnp:choice", names, &a, &size, &replace, &p, &axis, &shuffle))
        return NULL;
    if (axis != 0 && axis != -1) {
        PyErr_SetString(PyExc_ValueError, "choice currently supports one-dimensional populations");
        return NULL;
    }
    RandomGenerator *self = (RandomGenerator *)object;
    PyObject *sequence = NULL, *probabilities = NULL, *result = NULL;
    double *weights = NULL;
    size_t *pool = NULL, *indices = NULL;
    size_t hash_capacity = 0;
    RandomOutput output = {0};
    Py_ssize_t population;
    int const integer_population = PyIndex_Check(a) && !PySequence_Check(a);
    if (integer_population) population = PyNumber_AsSsize_t(a, PyExc_OverflowError);
    else {
        sequence = PySequence_Fast(a, "a must be an integer or one-dimensional sequence");
        if (!sequence) goto cleanup;
        population = PySequence_Fast_GET_SIZE(sequence);
    }
    if (PyErr_Occurred()) goto cleanup;
    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank = 0, total = 1;
    int const scalar = !size || size == Py_None;
    if (!scalar && (!parse_shape(size, shape, &rank) || !count_elements(shape, rank, &total))) goto cleanup;
    if (population < 0 || (!population && total) || (!replace && total > (size_t)population)) {
        PyErr_SetString(PyExc_ValueError, "choice population is empty or too small for sampling without replacement");
        goto cleanup;
    }
    double weight_sum = 0;
    if (p != Py_None) {
        probabilities = PySequence_Fast(p, "p must be a sequence of probabilities");
        if (!probabilities) goto cleanup;
        if (PySequence_Fast_GET_SIZE(probabilities) != population) {
            PyErr_SetString(PyExc_ValueError, "a and p must have equal lengths");
            goto cleanup;
        }
        if ((size_t)population > SIZE_MAX / sizeof(double)) {
            PyErr_NoMemory();
            goto cleanup;
        }
        weights = PyMem_Malloc((size_t)population * sizeof(double));
        if (!weights && population) {
            PyErr_NoMemory();
            goto cleanup;
        }
        size_t positive = 0;
        for (Py_ssize_t i = 0; i < population; ++i) {
            weights[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(probabilities, i));
            if (PyErr_Occurred()) goto cleanup;
            if (!isfinite(weights[i]) || weights[i] < 0) {
                PyErr_SetString(PyExc_ValueError, "p must contain finite non-negative probabilities");
                goto cleanup;
            }
            positive += weights[i] > 0;
            weight_sum += weights[i];
        }
        if (fabs(weight_sum - 1.0) > 1e-8 || (!replace && total > positive)) {
            PyErr_SetString(PyExc_ValueError, "p must sum to one and contain enough positive probabilities");
            goto cleanup;
        }
    }
    if (total > SIZE_MAX / sizeof(size_t)) {
        PyErr_NoMemory();
        goto cleanup;
    }
    indices = PyMem_Malloc((total ? total : 1) * sizeof(size_t));
    if (!indices) {
        PyErr_NoMemory();
        goto cleanup;
    }
    if (!replace && !weights) {
        if (total < (size_t)population / 4) {
            hash_capacity = 2;
            while (hash_capacity / 2 < total) {
                if (hash_capacity > SIZE_MAX / 2 / sizeof(size_t)) {
                    PyErr_NoMemory();
                    goto cleanup;
                }
                hash_capacity *= 2;
            }
        }
        size_t const capacity = hash_capacity ? hash_capacity : (size_t)population;
        if (capacity > SIZE_MAX / sizeof(size_t)) {
            PyErr_NoMemory();
            goto cleanup;
        }
        pool = PyMem_Malloc((capacity ? capacity : 1) * sizeof(size_t));
        if (!pool) {
            PyErr_NoMemory();
            goto cleanup;
        }
        for (size_t i = 0; i < capacity; ++i) pool[i] = hash_capacity ? SIZE_MAX : i;
    }
    nk_dtype_t dtype = nk_i64_k;
    if (sequence && !scalar && population && !PyIndex_Check(PySequence_Fast_GET_ITEM(sequence, 0))) dtype = nk_f64_k;
    if (!scalar && !prepare_output(self, &output, NULL, dtype, rank, shape)) goto cleanup;
    PyThreadState *gil = PyEval_SaveThread();
    PyThread_acquire_lock(self->lock, WAIT_LOCK);
    for (size_t i = 0; i < total; ++i) {
        size_t selected;
        if (hash_capacity) {
            size_t const upper = (size_t)population - total + i;
            selected = (size_t)nk_random_bounded_u64(&self->generator, upper + 1);
            if (!choice_insert(pool, hash_capacity, selected)) {
                selected = upper;
                choice_insert(pool, hash_capacity, selected);
            }
            indices[i] = selected;
        }
        else if (pool) {
            size_t const remaining = (size_t)population - i;
            selected = (size_t)nk_random_bounded_u64(&self->generator, remaining);
            indices[i] = pool[selected];
            pool[selected] = pool[remaining - 1];
        }
        else {
            selected = choice_index(&self->generator, weights, (size_t)population, weight_sum);
            indices[i] = selected;
            if (!replace) {
                weight_sum -= weights[selected];
                weights[selected] = 0;
            }
        }
    }
    if (hash_capacity && shuffle)
        for (size_t i = total; i > 1; --i) {
            size_t const selected = (size_t)nk_random_bounded_u64(&self->generator, i);
            size_t const temporary = indices[i - 1];
            indices[i - 1] = indices[selected];
            indices[selected] = temporary;
        }
    PyThread_release_lock(self->lock);
    PyEval_RestoreThread(gil);
    if (scalar) {
        if (integer_population) result = PyLong_FromSize_t(indices[0]);
        else {
            result = PySequence_Fast_GET_ITEM(sequence, (Py_ssize_t)indices[0]);
            Py_INCREF(result);
        }
    }
    else {
        for (size_t i = 0; i < total; ++i) {
            if (integer_population) ((int64_t *)output.data)[i] = (int64_t)indices[i];
            else if (dtype == nk_i64_k) {
                PyObject *value = PyNumber_Index(PySequence_Fast_GET_ITEM(sequence, (Py_ssize_t)indices[i]));
                if (!value) goto cleanup;
                ((int64_t *)output.data)[i] = PyLong_AsLongLong(value);
                Py_DECREF(value);
                if (PyErr_Occurred()) goto cleanup;
            }
            else {
                ((double *)output.data)[i] = PyFloat_AsDouble(
                    PySequence_Fast_GET_ITEM(sequence, (Py_ssize_t)indices[i]));
                if (PyErr_Occurred()) goto cleanup;
            }
        }
        result = finish_output(&output);
        output.owned = NULL;
        output.has_buffer = 0;
    }
cleanup:
    release_output(&output);
    PyMem_Free(weights);
    PyMem_Free(pool);
    PyMem_Free(indices);
    Py_XDECREF(sequence);
    Py_XDECREF(probabilities);
    return result;
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

static int RandomGenerator_traverse(RandomGenerator *self, visitproc visit, void *arg) {
    Py_VISIT(self->array_factory);
    return 0;
}

static int RandomGenerator_clear(RandomGenerator *self) {
    Py_CLEAR(self->array_factory);
    return 0;
}

static void RandomGenerator_dealloc(RandomGenerator *self) {
    PyObject_GC_UnTrack(self);
    RandomGenerator_clear(self);
    if (self->lock) PyThread_free_lock(self->lock);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *generator_state_get(PyObject *object, void *closure) {
    nk_unused_(closure);
    RandomGenerator *self = (RandomGenerator *)object;
    lock_generator(self);
    uint64_t const state = self->generator.state;
    PyThread_release_lock(self->lock);
    return PyLong_FromUnsignedLongLong(state);
}

static int generator_state_set(PyObject *object, PyObject *value, void *closure) {
    nk_unused_(closure);
    if (!value || !PyIndex_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "state must be an integer");
        return -1;
    }
    PyObject *integer = PyNumber_Index(value);
    if (!integer) return -1;
    uint64_t const state = PyLong_AsUnsignedLongLong(integer);
    Py_DECREF(integer);
    if (PyErr_Occurred()) return -1;
    RandomGenerator *self = (RandomGenerator *)object;
    lock_generator(self);
    nk_random_seed(&self->generator, state);
    PyThread_release_lock(self->lock);
    return 0;
}

static int RandomGenerator_init(RandomGenerator *self, PyObject *args, PyObject *kwargs) {
    PyObject *seed = Py_None, *factory = Py_None;
    static char *names[] = {"seed", "array_factory", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O$O:Generator", names, &seed, &factory)) return -1;
    if (factory != Py_None && !PyCallable_Check(factory)) {
        PyErr_SetString(PyExc_TypeError, "array_factory must be callable or None");
        return -1;
    }
    PyObject *integer;
    if (seed == Py_None) {
        PyObject *os = PyImport_ImportModule("os");
        PyObject *bytes = os ? PyObject_CallMethod(os, "urandom", "i", 8) : NULL;
        Py_XDECREF(os);
        if (!bytes) return -1;
        uint64_t value;
        memcpy(&value, PyBytes_AS_STRING(bytes), sizeof(value));
        Py_DECREF(bytes);
        integer = PyLong_FromUnsignedLongLong(value);
    }
    else integer = PyNumber_Index(seed);
    if (!integer) return -1;
    int const status = generator_state_set((PyObject *)self, integer, NULL);
    Py_DECREF(integer);
    if (status < 0) return -1;
    if (factory == Py_None) Py_CLEAR(self->array_factory);
    else {
        Py_INCREF(factory);
        Py_XSETREF(self->array_factory, factory);
    }
    return 0;
}

static PyObject *generator_reduce(PyObject *object, PyObject *unused) {
    nk_unused_(unused);
    RandomGenerator *self = (RandomGenerator *)object;
    PyObject *state = generator_state_get(object, NULL);
    if (!state) return NULL;
    PyObject *result = Py_BuildValue("(O(O)O)", Py_TYPE(self), state,
                                     self->array_factory ? self->array_factory : Py_None);
    Py_DECREF(state);
    return result;
}

static PyObject *generator_setstate(PyObject *object, PyObject *factory) {
    if (factory != Py_None && !PyCallable_Check(factory)) {
        PyErr_SetString(PyExc_TypeError, "array_factory must be callable or None");
        return NULL;
    }
    RandomGenerator *self = (RandomGenerator *)object;
    if (factory == Py_None) Py_CLEAR(self->array_factory);
    else {
        Py_INCREF(factory);
        Py_XSETREF(self->array_factory, factory);
    }
    Py_RETURN_NONE;
}

static PyMethodDef RandomGenerator_methods[] = {
    {"random", (PyCFunction)generator_random, METH_VARARGS | METH_KEYWORDS, "Uniform samples in [0, 1)."},
    {"uniform", (PyCFunction)generator_uniform, METH_VARARGS | METH_KEYWORDS,
     "Uniform samples with broadcastable bounds."},
    {"normal", (PyCFunction)generator_normal, METH_VARARGS | METH_KEYWORDS,
     "Normal samples with broadcastable location and scale."},
    {"standard_normal", (PyCFunction)generator_standard_normal, METH_VARARGS | METH_KEYWORDS,
     "Standard normal samples."},
    {"integers", (PyCFunction)generator_integers, METH_VARARGS | METH_KEYWORDS,
     "Unbiased integers with broadcastable bounds."},
    {"laplace", (PyCFunction)generator_laplace, METH_VARARGS | METH_KEYWORDS,
     "Laplace samples with broadcastable location and scale."},
    {"beta", (PyCFunction)generator_beta, METH_VARARGS | METH_KEYWORDS,
     "Beta samples with broadcastable shape parameters."},
    {"poisson", (PyCFunction)generator_poisson, METH_VARARGS | METH_KEYWORDS,
     "Poisson counts with broadcastable rates."},
    {"choice", (PyCFunction)generator_choice, METH_VARARGS | METH_KEYWORDS,
     "Sample a one-dimensional population, optionally weighted and without replacement."},
    {"shuffle", (PyCFunction)generator_shuffle, METH_VARARGS | METH_KEYWORDS,
     "Shuffle a list or C-contiguous array along its first axis."},
    {"__reduce__", generator_reduce, METH_NOARGS, NULL},
    {"__setstate__", generator_setstate, METH_O, NULL},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef RandomGenerator_getset[] = {
    {"state", generator_state_get, generator_state_set, "Current uint64 stream state.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject RandomGeneratorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.Generator",
    .tp_doc = "Independent native generator. array_factory(shape, dtype=...) optionally owns array allocation.",
    .tp_basicsize = sizeof(RandomGenerator),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)RandomGenerator_traverse,
    .tp_clear = (inquiry)RandomGenerator_clear,
    .tp_dealloc = (destructor)RandomGenerator_dealloc,
    .tp_methods = RandomGenerator_methods,
    .tp_getset = RandomGenerator_getset,
    .tp_init = (initproc)RandomGenerator_init,
    .tp_new = RandomGenerator_new,
};
