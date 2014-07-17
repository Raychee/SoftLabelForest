# include <boost/python.hpp>
// # include <numpy/arrayobject.h>
# include <cstring>
# include "Qdefs.hpp"
// # include "Data.hpp"
// # include "Some.hpp"
// # include "SoftDecisionModel.hpp"

using namespace boost::python;

// npy_float& get(char* arr_data, npy_intp* strides, npy_intp* ind, int ndim) {
//     for (int i_dim = 0; i_dim < ndim; ++i_dim) {
//         arr_data += ind[i_dim] * strides[i_dim];
//     }
//     return *reinterpret_cast<npy_float*>(arr_data);
// }

// bool end_of_ndarray(npy_intp* shape, npy_intp* ind, int ndim) {
//     for (int i_dim = 0; i_dim < ndim; ++i_dim) {
//         if (ind[i_dim] < shape[i_dim] - 1) {
//             return false;
//         }
//     }
//     return true;
// }

// void get_next_ind(npy_intp* shape, npy_intp* ind, int ndim) {
//     for (int i_dim = ndim - 1; i_dim > 0; --i_dim) {
//         ++ind[i_dim];
//         if (ind[i_dim] >= shape[i_dim]) {
//             ind[i_dim] = 0;
//         } else {
//             return;
//         }
//     }
//     ++ind[0];
// }

// void test_raw_ndarray(object& seq_object) {
//     std::cout << "0" << std::endl;
//     std::cout << seq_object.ptr() << std::endl;
//     PyObject* pyarr = PyArray_FROM_OTF(seq_object.ptr(), NPY_COMP_T, NPY_IN_ARRAY);
//     char* arr_data = PyArray_BYTES(pyarr);
//     int ndim = PyArray_NDIM(pyarr);
//     npy_intp* strides = PyArray_STRIDES(pyarr);
//     npy_intp* shape = PyArray_DIMS(pyarr);
//     npy_intp* ind = new npy_intp[ndim];
//     std::memset(ind, 0, sizeof(npy_intp) * ndim);

//     std::cout << "Strides:";
//     for (int i = 0; i < ndim; ++i) {
//         std::cout << " " << strides[i];
//     }
//     std::cout << std::endl;
//     std::cout << "Strides in unit (int):";
//     for (int i = 0; i < ndim; ++i) {
//         std::cout << " " << strides[i] / sizeof(COMP_T);
//     }
//     std::cout << std::endl;
//     std::cout << "Shape:";
//     for (int i = 0; i < ndim; ++i) {
//         std::cout << " " << shape[i];
//     }
//     std::cout << std::endl;

//     std::cout << "Data:" << std::endl;
//     while (ind[0] < shape[0]) {
//         std::cout << "[";
//         for (int i_dim = 0; i_dim < ndim; ++i_dim) {
//             std::cout << " " << ind[i_dim];
//         }
//         std::cout << " ] = " << get(arr_data, strides, ind, ndim) << std::endl;
//         get_next_ind(shape, ind, ndim);
//     }
//     std::memset(ind, 0, sizeof(npy_intp) * ndim);
//     get(arr_data, strides, ind, ndim) = -3.1415;
//     std::cout << "Modified [ 0 0 ]" << std::endl;

//     std::cout << "Data:" << std::endl;
//     while (ind[0] < shape[0]) {
//         std::cout << "[";
//         for (int i_dim = 0; i_dim < ndim; ++i_dim) {
//             std::cout << " " << ind[i_dim];
//         }
//         std::cout << " ] = " << get(arr_data, strides, ind, ndim) << std::endl;
//         get_next_ind(shape, ind, ndim);
//     }

//     std::cerr << "Test Error" << std::endl;

//     Py_XDECREF(pyarr);
//     delete[] ind;
// }

// std::ostream& operator<<(std::ostream& out, Some& some) {
//     out << "Cout << Some;" << std::endl;
//     return out;
// }

class Some {
  public:
    Some(int _n = 10) : n(_n) {
        indices = new N_DAT_T[n];
        for (int i = 0; i < n; ++i) {
            indices[i] = -i;
        }
    }
    ~Some() {
        delete[] indices;
    }
    list gen_list() {
        list l;
        for (int i = 0; i < n; ++i) {
            l.append(indices[i]);
        }
        return l;
    }
  private:
    N_DAT_T* indices;
    int n;
};

BOOST_PYTHON_MODULE(tp) {
    class_<Some>("Some")
        .def("gen_list", &Some::gen_list);
}

