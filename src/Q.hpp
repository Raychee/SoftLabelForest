# ifndef _Q_HPP
# define _Q_HPP


# include <iostream>
# include <fstream>
# include <algorithm>

const size_t SIZEOF_LINE = 1048576;
const size_t SIZEOF_PATH = 4096;

// type of the value to be computed (parameters, training samples, etc)
// alternatives: double / float
typedef float  COMP_T;
// type of the supervising information (classes, labels, etc)
// alternatives: any type of signed integer
typedef int    SUPV_T;
// type of the number of the data set
// alternatives: any type of signed integer
typedef int    N_DAT_T;
// type of the dimension of the data
// alternatives: any type of integer
typedef int    DAT_DIM_T;

template <typename _TYPE> _TYPE strto(const char* str);
template <typename _TYPE> _TYPE strto(const char* str, char*& str_end);
template <typename _TYPE> _TYPE strto(const char* str, int base);
template <typename _TYPE> _TYPE strto(const char* str, char*& str_end, int base);
char* strtostr(char* str);

/**
 * @brief   Read 2-dimensional matrix data
 * @details Load a file containing a 2-dimensional matrix into the memory. The
 *   first two units of data (of type _M and _N) indicates number of rows and
 *   columns; the rest of the file should contain exactly M * N units of data
 *   (of type _X), which stores the matrix in a column-wise manner.
 *
 * @param  data_file Matrix data file name.
 * @param  X         Memory space for holding data.
 * @param  M         Number of rows.
 * @param  N         Number of columns.
 * @return           Status code (0 = normal)
 */
template <typename _X, typename _M, typename _N>
int read_2d_matrix(const char* data_file, _X*& X, _M& M, _N& N) {
    std::ifstream file(data_file, std::ios_base::in|std::ios_base::binary);
    if (!file.is_open()) {
        std::cerr << "**ERROR** -> read_2d_matrix(data_file=\""
                  << data_file << "\") -> Failed opening file \""
                  << data_file << "\"." << std::endl;
        return -1;
    }
    M = 0; N = 0; X = NULL;
    file.read((char*)&M, sizeof(_M));
    file.read((char*)&N, sizeof(_N));
    if (M > 0 && N > 0) X = new _X[M * N];
    else {
        std::cerr << "**ERROR** -> read_2d_matrix(data_file=\""
                  << data_file << "\") -> Corrupted file \"" << data_file
                  << "\": Number of either rows or colums is zero." << std::endl;
        file.close();
        return -1;
    }
    file.read((char*)X, sizeof(_X) * M * N);
    if (!file) {
        std::cerr << "**ERROR** -> read_2d_matrix(data_file=\""
                  << data_file << "\") -> Corrupted file \"" << data_file
                  << "\": Cannot load matrix data." << std::endl;
        file.close();
        return -1;
    }
    file.close();
    return 0;
}

/**
 * @brief   Write 2-dimensional matrix data
 * @details Write a 2-dimensional matrix into a file.
 *
 * @param  data_file Matrix data file.
 * @param  X         Matrix data.
 * @param  M         Number of rows.
 * @param  N         Number of columns.
 * @return           Status code (0 = normal)
 */
template <typename _X, typename _M, typename _N>
int write_2d_matrix(const char* data_file, const _X* X, const _M& M, const _N& N) {
    if (X == NULL || M == 0 || N == 0) {
        std::cerr << "**ERROR** -> write_2d_matrix(data_file=\""
                  << data_file << "\", _X=" << X << ") -> Invalid matrix data." << std::endl;
        return -1;
    }
    std::ofstream file(data_file, std::ios_base::out|std::ios_base::binary);
    if (!file.is_open()) {
        std::cerr << "\nFailed opening file \"" << data_file << "\"." << std::endl;
        return -1;
    }
    file.write((char*)&M, sizeof(_M));
    file.write((char*)&N, sizeof(_N));
    file.write((char*)X, sizeof(_X) * M * N);
    if (!file) {
        std::cerr << "\nCorrupted file \"" << data_file
                  << "\": Cannot write matrix data." << std::endl;
        file.close();
        return -1;
    }
    file.close();
    return 0;
}


/************************* Boost.Python & Numpy C-API *************************/

# include <boost/python.hpp>

namespace bpy = boost::python;

template <typename _VALUE, typename _LENGTH>
bpy::list c_array_to_python_list(const _VALUE* c_array, const _LENGTH length) {
    bpy::list py_list;
    for (_LENGTH i = 0; i < length; ++i) {
        py_list.append(c_array[i]);
    }
    return py_list;
}

template <typename _VALUE, typename _LENGTH>
int python_list_to_c_array(const bpy::list& py_list,
                           _VALUE*& c_array, _LENGTH& length) {
    length = bpy::len(py_list);
    c_array = new _VALUE[length];
    for (_LENGTH i = 0; i < length; ++i) {
        c_array[i] = bpy::extract<_VALUE>(py_list[i]);
    }
    return 0;
}


# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
// # define NPY_ARRAY_IN_ARRAY    NPY_IN_ARRAY
// # define NPY_ARRAY_OWNDATA     NPY_OWNDATA
# include <numpy/arrayobject.h>

// Numpy API Definitions
template <typename _T> struct NumPy_Type;
template <> struct NumPy_Type<float> {
    static const int typenum = NPY_FLOAT;
};
template <> struct NumPy_Type<double> {
    static const int typenum = NPY_DOUBLE;
};
template <> struct NumPy_Type<int> {
    static const int typenum = NPY_INT;
};

template <typename _X>
int parse_numpy_array(const bpy::object& bpy_x,
                      PyObject*& py_x, _X*& x, int& ndim, npy_intp*& shape) {
    PyArrayObject* _py_x = reinterpret_cast<PyArrayObject*>(
                           PyArray_FROM_OTF(bpy_x.ptr(), NumPy_Type<_X>::typenum,
                                            NPY_ARRAY_IN_ARRAY));
    if (_py_x == NULL) {
        std::cerr << "**ERROR** "
                     "-> parse_numpy_array(bpy::object& bpy_x@" << &bpy_x
                  << ", PyObject*& py_x, _X*& x, int& ndim, npy_intp*& shape) "
                  << "-> Cannot convert bpy_x." << std::endl;
        return -1;
    }
    int       _ndim    = PyArray_NDIM(_py_x);
    npy_intp* _shape   = PyArray_DIMS(_py_x);
    npy_intp* _strides = PyArray_STRIDES(_py_x);
    if (_strides[_ndim-1] != sizeof(_X)) {
        std::cerr << "**ERROR** "
                     "-> parse_numpy_array(bpy::object& bpy_x@" << &bpy_x
                  << ", PyObject*& py_x, _X*& x, int& ndim, npy_intp*& shape) "
                  << "-> Stride of x (" << _strides[_ndim-1]
                  << ") is not compatible with its C type (" << sizeof(_X)
                  << ")." << std::endl;
        Py_DECREF(_py_x);
        return -1;
    }
    for (int i = _ndim-1; i > 0; --i) {
        if (_strides[i-1] != _shape[i] * _strides[i]) {
            std::cerr << "**ERROR** "
                         "-> parse_numpy_array(bpy::object& bpy_x@" << &bpy_x
                      << ", PyObject*& py_x, _X*& x, int& ndim, npy_intp*& shape) "
                      << "-> Data of x is not contiguous." << std::endl;
            Py_DECREF(_py_x);
            return -1;
        }
    }
    py_x  = reinterpret_cast<PyObject*>(_py_x);
    x     = reinterpret_cast<_X*>(PyArray_DATA(_py_x));
    ndim  = _ndim;
    shape = _shape;
    return 0;
}

template<typename _VALUE, typename _LENGTH> inline
int numpy_1d_array_to_c_array(const bpy::object& bpy_npy_array,
                              _VALUE*& c_array, _LENGTH& length,
                              bool copy = true, bool transfer = false) {
    PyObject* py_npy_array;
    int ndim;
    npy_intp* shape;
    if (parse_numpy_array(bpy_npy_array, py_npy_array, c_array, ndim, shape)) {
        std::cerr << "**ERROR** -> numpy_1d_array_to_c_array(...) "
                     "-> Cannot parse indices." << std::endl;
        return -1;
    }
    length = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] > length) {
            if (length > 1) {
                std::cerr << "**ERROR** -> numpy_1d_array_to_c_array(...) "
                             "-> The array is not 1d." << std::endl;
                Py_DECREF(py_npy_array);
                return -1;
            }
            length = shape[i];
        }
    }
    if (transfer) {
        PyArrayObject* py_npyarr =
            reinterpret_cast<PyArrayObject*>(py_npy_array);
        PyArrayObject* py_npyarr_base =
            reinterpret_cast<PyArrayObject*>(PyArray_BASE(py_npyarr));
        if (py_npyarr_base != NULL) {
            PyArray_CLEARFLAGS(py_npyarr_base, NPY_ARRAY_OWNDATA);
        } else {
            PyArray_CLEARFLAGS(py_npyarr, NPY_ARRAY_OWNDATA);
        }
    } else if (copy) {
        _VALUE* _c_array = new _VALUE[length];
        std::memcpy(_c_array, c_array, sizeof(_VALUE) * length);
        c_array = _c_array;
    }
    Py_DECREF(py_npy_array);
    return 0;
}

template<typename _VALUE> inline
bpy::object c_array_to_numpy_1d_array(_VALUE*  c_array, npy_intp length,
                                      bool copy = true) {
    PyObject* py_npy_array;
    if (copy) {
        py_npy_array =
            PyArray_SimpleNew(1, &length, NumPy_Type<_VALUE>::typenum);
        _VALUE* _c_array =
            reinterpret_cast<_VALUE*>(
                PyArray_DATA(reinterpret_cast<PyArrayObject*>(py_npy_array)));
        std::memcpy(_c_array, c_array, sizeof(_VALUE) * length);
    } else {
        py_npy_array = PyArray_SimpleNewFromData(
                           1, &length, NumPy_Type<_VALUE>::typenum, c_array);
    }
    return bpy::object(bpy::handle<>(py_npy_array));
}


/******************************* OpenMP *******************************/

# include <omp.h>

template<typename _INT> inline
_INT omp_num_of_threads(_INT num_of_iter, _INT min_iter_per_thread = 1) {
    return std::min((num_of_iter - 1) / min_iter_per_thread + 1,
                    omp_get_max_threads());
}


# endif
