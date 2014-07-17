# ifndef _Q_DEFS_H
# define _Q_DEFS_H

const size_t SIZEOF_LINE = 1048576;
const size_t SIZEOF_PATH = 4096;


// type of the value to be computed (parameters, training samples, etc)
// alternatives: double / float
typedef double COMP_T;
// type of the supervising information (classes, labels, etc)
// alternatives: any type of signed integer
typedef int   SUPV_T;
// type of the number of the data set
// alternatives: any type of signed integer
typedef int   N_DAT_T;
// type of the dimension of the data
// alternatives: any type of integer
typedef int   DAT_DIM_T;


// Numpy API Definitions

# include <numpy/arrayobject.h>

template <typename _T> struct NumPy_Traits;
template <> struct NumPy_Traits<float> {
    static const int typenum = NPY_FLOAT;
};
template <> struct NumPy_Traits<double> {
    static const int typenum = NPY_DOUBLE;
};
template <> struct NumPy_Traits<int> {
    static const int typenum = NPY_INT;
};

# endif
