/**
 *  @brief Mesh alignment (Kabsch, Umeyama, RMSD) for NumKong Python bindings.
 *  @file python/mesh.c
 *  @author Ash Vardanian
 *  @date February 19, 2026
 *
 *  Implements the MeshAlignmentResult type and the three mesh-alignment API
 *  functions (kabsch, umeyama, rmsd).  The MeshAlignmentResultObject struct
 *  is opaque — only the PyTypeObject is exported.
 */
#include "mesh.h"
#include "tensor.h"

#include <structmember.h> // `PyMemberDef`, `T_OBJECT_EX`, `READONLY`, `offsetof`

/** @brief Mesh alignment result type — structured return for kabsch/umeyama/rmsd. */
typedef struct {
    PyObject_HEAD
    /** (3,3) rotation matrix Tensor. */
    PyObject *rotation;
    /** 0-D scale factor Tensor. */
    PyObject *scale;
    /** 0-D RMSD Tensor. */
    PyObject *rmsd;
    /** (3,) centroid of first point cloud. */
    PyObject *a_centroid;
    /** (3,) centroid of second point cloud. */
    PyObject *b_centroid;
} MeshAlignmentResultObject;

static void MeshAlignmentResult_dealloc(MeshAlignmentResultObject *self) {
    Py_XDECREF(self->rotation);
    Py_XDECREF(self->scale);
    Py_XDECREF(self->rmsd);
    Py_XDECREF(self->a_centroid);
    Py_XDECREF(self->b_centroid);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *MeshAlignmentResult_repr(MeshAlignmentResultObject *self) {
    PyObject *scale_repr = PyObject_Repr(self->scale);
    PyObject *rmsd_repr = PyObject_Repr(self->rmsd);
    if (!scale_repr || !rmsd_repr) {
        Py_XDECREF(scale_repr);
        Py_XDECREF(rmsd_repr);
        return NULL;
    }
    PyObject *result = PyUnicode_FromFormat("MeshAlignmentResult(scale=%U, rmsd=%U)", scale_repr, rmsd_repr);
    Py_DECREF(scale_repr);
    Py_DECREF(rmsd_repr);
    return result;
}

static Py_ssize_t MeshAlignmentResult_length(MeshAlignmentResultObject *self) {
    nk_unused_(self);
    return 5;
}

static PyObject *MeshAlignmentResult_item(MeshAlignmentResultObject *self, Py_ssize_t index) {
    PyObject *items[5] = {self->rotation, self->scale, self->rmsd, self->a_centroid, self->b_centroid};
    if (index < 0 || index >= 5) {
        PyErr_SetString(PyExc_IndexError, "MeshAlignmentResult index out of range");
        return NULL;
    }
    Py_INCREF(items[index]);
    return items[index];
}

static PySequenceMethods MeshAlignmentResult_as_sequence = {
    .sq_length = (lenfunc)MeshAlignmentResult_length,
    .sq_item = (ssizeargfunc)MeshAlignmentResult_item,
};

static PyMethodDef MeshAlignmentResult_methods[] = {
    {NULL, NULL, 0, NULL},
};

static PyMemberDef MeshAlignmentResult_members[] = {
    {"rotation", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, rotation), READONLY, "(3,3) rotation matrix"},
    {"scale", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, scale), READONLY, "Scale factor"},
    {"rmsd", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, rmsd), READONLY, "Root mean square deviation"},
    {"a_centroid", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, a_centroid), READONLY, "Centroid of first cloud"},
    {"b_centroid", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, b_centroid), READONLY, "Centroid of second cloud"},
    {NULL, 0, 0, 0, NULL},
};

static char const doc_mesh_alignment_result[] =                //
    "Result of mesh alignment (Kabsch, Umeyama, RMSD).\n\n"    //
    "Fields: rotation, scale, rmsd, a_centroid, b_centroid.\n" //
    "Supports iteration and indexing for backward-compatible destructuring.";

PyTypeObject MeshAlignmentResultType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.MeshAlignmentResult",
    .tp_basicsize = sizeof(MeshAlignmentResultObject),
    .tp_dealloc = (destructor)MeshAlignmentResult_dealloc,
    .tp_repr = (reprfunc)MeshAlignmentResult_repr,
    .tp_as_sequence = &MeshAlignmentResult_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_mesh_alignment_result,
    .tp_methods = MeshAlignmentResult_methods,
    .tp_members = MeshAlignmentResult_members,
};

char const doc_kabsch[] =                                                                                        //
    "Compute optimal rigid transformation (Kabsch algorithm) between two point clouds.\n\n"                      //
    "Finds the optimal rotation matrix that minimizes RMSD between point clouds.\n"                              //
    "The transformation aligns point cloud A to point cloud B:\n"                                                //
    "    a'_i = scale * R * (a_i - a_centroid) + b_centroid\n\n"                                                 //
    "Supports both single-pair and batched inputs:\n"                                                            //
    "    - Single pair: (N, 3) -> rotation (3,3), scale (), rmsd (), centroids (3,)\n"                           //
    "    - Batched: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)\n\n"                   //
    "Parameters:\n"                                                                                              //
    "    a (Tensor): First point cloud(s), shape (N, 3) or (B, N, 3), float16, bfloat16, float32, or float64.\n" //
    "    b (Tensor): Second point cloud(s), shape (N, 3) or (B, N, 3), same dtype as a.\n\n"                     //
    "Returns:\n"                                                                                                 //
    "    MeshAlignmentResult: rotation, scale, rmsd, a_centroid, b_centroid fields.\n\n"                         //
    "Example:\n"                                                                                                 //
    "    >>> result = numkong.kabsch(a, b)\n"                                                                    //
    "    >>> np.asarray(result.rotation)  # (3, 3) rotation matrix\n"                                            //
    "    >>> float(result.scale)          # scale factor (always 1.0 for Kabsch)\n";

char const doc_umeyama[] =                                                                                       //
    "Compute optimal similarity transformation (Umeyama algorithm) between two point clouds.\n\n"                //
    "Finds the optimal rotation matrix and uniform scaling factor that minimize RMSD.\n"                         //
    "The transformation aligns point cloud A to point cloud B:\n"                                                //
    "    a'_i = scale * R * (a_i - a_centroid) + b_centroid\n\n"                                                 //
    "Supports both single-pair and batched inputs:\n"                                                            //
    "    - Single pair: (N, 3) -> rotation (3,3), scale (), rmsd (), centroids (3,)\n"                           //
    "    - Batched: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)\n\n"                   //
    "Parameters:\n"                                                                                              //
    "    a (Tensor): First point cloud(s), shape (N, 3) or (B, N, 3), float16, bfloat16, float32, or float64.\n" //
    "    b (Tensor): Second point cloud(s), shape (N, 3) or (B, N, 3), same dtype as a.\n\n"                     //
    "Returns:\n"                                                                                                 //
    "    MeshAlignmentResult: rotation, scale, rmsd, a_centroid, b_centroid fields.\n\n"                         //
    "Example:\n"                                                                                                 //
    "    >>> result = numkong.umeyama(a, b)\n"                                                                   //
    "    >>> float(result.scale)  # Will differ from 1.0 if point clouds have different scales\n";

char const doc_rmsd[] =                                                                                          //
    "Compute raw RMSD between two point clouds without centering or alignment.\n\n"                              //
    "Computes √(Σ‖aᵢ − bᵢ‖² / n) directly on raw point differences.\n"                                           //
    "Returns identity rotation, scale=1.0, and zeroed centroids.\n\n"                                            //
    "Supports both single-pair and batched inputs:\n"                                                            //
    "    - Single pair: (N, 3) -> rotation (3,3), scale (), rmsd (), centroids (3,)\n"                           //
    "    - Batched: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)\n\n"                   //
    "Parameters:\n"                                                                                              //
    "    a (Tensor): First point cloud(s), shape (N, 3) or (B, N, 3), float16, bfloat16, float32, or float64.\n" //
    "    b (Tensor): Second point cloud(s), shape (N, 3) or (B, N, 3), same dtype as a.\n\n"                     //
    "Returns:\n"                                                                                                 //
    "    MeshAlignmentResult: rotation, scale, rmsd, a_centroid, b_centroid fields.\n";

static PyObject *implement_mesh_alignment(nk_kernel_kind_t metric_kind, PyObject *const *args,
                                          Py_ssize_t positional_args_count) {
    // We expect exactly 2 positional arguments: a and b
    if (positional_args_count != 2) {
        PyErr_SetString(PyExc_TypeError, "Expected exactly 2 positional arguments (a, b)");
        return NULL;
    }

    Py_buffer a_buffer, b_buffer;
    nk_buffer_backing_t a_backing, b_backing;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));

    // Get buffer for array a
    if (!nk_get_buffer(args[0], &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &a_backing)) {
        PyErr_SetString(PyExc_TypeError, "First argument must support buffer protocol or __array_interface__");
        return NULL;
    }

    // Get buffer for array b
    if (!nk_get_buffer(args[1], &b_buffer, PyBUF_STRIDES | PyBUF_FORMAT, &b_backing)) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_TypeError, "Second argument must support buffer protocol or __array_interface__");
        return NULL;
    }

    PyObject *result = NULL;
    Tensor *rot_tensor = NULL;
    Tensor *scale_tensor = NULL;
    Tensor *rmsd_tensor = NULL;
    Tensor *a_cent_tensor = NULL;
    Tensor *b_cent_tensor = NULL;

    // Validate shapes: accept (N, 3) for single pair or (B, N, 3) for batch
    int is_batched = 0;
    Py_ssize_t batch_size = 1;
    Py_ssize_t n_points, last_dim_a, last_dim_b;

    if (a_buffer.ndim == 2 && b_buffer.ndim == 2) {
        // Single pair: (N, 3) shape
        n_points = a_buffer.shape[0];
        last_dim_a = a_buffer.shape[1];
        last_dim_b = b_buffer.shape[1];
    }
    else if (a_buffer.ndim == 3 && b_buffer.ndim == 3) {
        // Batched: (B, N, 3) shape
        is_batched = 1;
        batch_size = a_buffer.shape[0];
        n_points = a_buffer.shape[1];
        last_dim_a = a_buffer.shape[2];
        last_dim_b = b_buffer.shape[2];
        if (a_buffer.shape[0] != b_buffer.shape[0]) {
            PyErr_SetString(PyExc_ValueError, "Batch sizes must match");
            goto cleanup;
        }
        if (a_buffer.shape[1] != b_buffer.shape[1]) {
            PyErr_SetString(PyExc_ValueError, "Point clouds must have the same number of points");
            goto cleanup;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Point clouds must be 2D (N,3) or 3D (B,N,3) arrays");
        goto cleanup;
    }

    if (last_dim_a != 3 || last_dim_b != 3) {
        PyErr_SetString(PyExc_ValueError, "Point clouds must have 3 columns (x, y, z coordinates)");
        goto cleanup;
    }
    if (!is_batched && a_buffer.shape[0] != b_buffer.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "Point clouds must have the same number of points");
        goto cleanup;
    }
    if (n_points < 3) {
        PyErr_SetString(PyExc_ValueError, "Point clouds must have at least 3 points");
        goto cleanup;
    }

    // Check data types and get kernel
    nk_dtype_t dtype = resolve_nk_dtype_in_py_buffer(&a_buffer);
    if (dtype != nk_f64_k && dtype != nk_f32_k && dtype != nk_f16_k && dtype != nk_bf16_k) {
        PyErr_SetString(PyExc_TypeError, "Point clouds must be float16, bfloat16, float32, or float64");
        goto cleanup;
    }

    // Find the appropriate kernel
    nk_metric_mesh_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, (nk_kernel_punned_t *)&kernel, &capability);
    if (!kernel || !capability) {
        PyErr_SetString(PyExc_RuntimeError, "No suitable mesh kernel found for this data type");
        goto cleanup;
    }

    // Check contiguity - we need row-major contiguous data for the innermost 2 dimensions
    Py_ssize_t const elem_size = (Py_ssize_t)nk_dtype_bytes_per_value(dtype);
    Py_ssize_t const inner_stride_a = is_batched ? a_buffer.strides[2] : a_buffer.strides[1];
    Py_ssize_t const inner_stride_b = is_batched ? b_buffer.strides[2] : b_buffer.strides[1];
    if (inner_stride_a != elem_size || inner_stride_b != elem_size) {
        PyErr_SetString(PyExc_ValueError, "Point clouds must be C-contiguous (row-major)");
        goto cleanup;
    }

    // Calculate strides between batches
    Py_ssize_t const batch_stride_a = is_batched ? a_buffer.strides[0] : 0;
    Py_ssize_t const batch_stride_b = is_batched ? b_buffer.strides[0] : 0;
    nk_size_t num_points = (nk_size_t)n_points;

    nk_dtype_t metric_dtype = nk_kernel_output_dtype(metric_kind, dtype);
    nk_dtype_t transform_dtype = nk_mesh_transform_dtype(dtype);

    if (!is_batched) {
        // Single pair case - return 0D scalars for scale/rmsd, (3,3) for rotation, (3,) for centroids
        Py_ssize_t rot_shape[2] = {3, 3};
        Py_ssize_t cent_shape[1] = {3};

        rot_tensor = Tensor_new(transform_dtype, 2, rot_shape);
        scale_tensor = Tensor_new(transform_dtype, 0, NULL);
        rmsd_tensor = Tensor_new(metric_dtype, 0, NULL);
        a_cent_tensor = Tensor_new(transform_dtype, 1, cent_shape);
        b_cent_tensor = Tensor_new(transform_dtype, 1, cent_shape);

        if (!rot_tensor || !scale_tensor || !rmsd_tensor || !a_cent_tensor || !b_cent_tensor) goto cleanup;

        kernel(a_buffer.buf, b_buffer.buf, num_points, a_cent_tensor->data, b_cent_tensor->data, rot_tensor->data,
               scale_tensor->data, rmsd_tensor->data);
    }
    else {
        // Batched case: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)
        Py_ssize_t rot_shape[3] = {batch_size, 3, 3};
        Py_ssize_t scalar_shape[1] = {batch_size};
        Py_ssize_t cent_shape[2] = {batch_size, 3};

        rot_tensor = Tensor_new(transform_dtype, 3, rot_shape);
        scale_tensor = Tensor_new(transform_dtype, 1, scalar_shape);
        rmsd_tensor = Tensor_new(metric_dtype, 1, scalar_shape);
        a_cent_tensor = Tensor_new(transform_dtype, 2, cent_shape);
        b_cent_tensor = Tensor_new(transform_dtype, 2, cent_shape);

        if (!rot_tensor || !scale_tensor || !rmsd_tensor || !a_cent_tensor || !b_cent_tensor) goto cleanup;

        char *a_ptr = (char *)a_buffer.buf;
        char *b_ptr = (char *)b_buffer.buf;
        size_t const transform_bytes = nk_dtype_bytes_per_value(transform_dtype);
        size_t const metric_bytes = nk_dtype_bytes_per_value(metric_dtype);

        for (Py_ssize_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            kernel(a_ptr + batch_idx * batch_stride_a, b_ptr + batch_idx * batch_stride_b, num_points,
                   a_cent_tensor->data + batch_idx * 3 * transform_bytes,
                   b_cent_tensor->data + batch_idx * 3 * transform_bytes,
                   rot_tensor->data + batch_idx * 9 * transform_bytes, scale_tensor->data + batch_idx * transform_bytes,
                   rmsd_tensor->data + batch_idx * metric_bytes);
        }
    }

    // Build MeshAlignmentResult
    MeshAlignmentResultObject *mesh_result = PyObject_New(MeshAlignmentResultObject, &MeshAlignmentResultType);
    if (mesh_result) {
        mesh_result->rotation = (PyObject *)rot_tensor;
        Py_INCREF(rot_tensor);
        mesh_result->scale = (PyObject *)scale_tensor;
        Py_INCREF(scale_tensor);
        mesh_result->rmsd = (PyObject *)rmsd_tensor;
        Py_INCREF(rmsd_tensor);
        mesh_result->a_centroid = (PyObject *)a_cent_tensor;
        Py_INCREF(a_cent_tensor);
        mesh_result->b_centroid = (PyObject *)b_cent_tensor;
        Py_INCREF(b_cent_tensor);
        result = (PyObject *)mesh_result;
    }

cleanup:
    // Individual tensors are always decref'd; if result was created, it holds its own references
    Py_XDECREF(rot_tensor);
    Py_XDECREF(scale_tensor);
    Py_XDECREF(rmsd_tensor);
    Py_XDECREF(a_cent_tensor);
    Py_XDECREF(b_cent_tensor);
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    return result;
}

PyObject *api_kabsch(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                     PyObject *args_names_tuple) {
    nk_unused_(self);
    nk_unused_(args_names_tuple);
    return implement_mesh_alignment(nk_kernel_kabsch_k, args, positional_args_count);
}

PyObject *api_umeyama(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                      PyObject *args_names_tuple) {
    nk_unused_(self);
    nk_unused_(args_names_tuple);
    return implement_mesh_alignment(nk_kernel_umeyama_k, args, positional_args_count);
}

PyObject *api_rmsd(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                   PyObject *args_names_tuple) {
    nk_unused_(self);
    nk_unused_(args_names_tuple);
    return implement_mesh_alignment(nk_kernel_rmsd_k, args, positional_args_count);
}
