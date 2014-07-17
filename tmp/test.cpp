# include "my_typedefs.h"
# include "GDSolver.hpp"
# include "Data.hpp"
# include "my_lib.hpp"
# include <iostream>

extern "C" {


typedef struct {
    int*   i;
    float* f;
} mystruct;

mystruct* new_mystruct(int ni = 5, int nf = 5) {
    mystruct* mys = new mystruct;
    mys->i = new int[ni];
    mys->f = new float[nf];
    std::cout << "Created " << ni << "i[]s, " << nf << "f[]s." << std::endl;
    return mys;
}

int del_mystruct(mystruct* mys) {
    delete[] mys->i;
    delete[] mys->f;
    delete mys;
    std::cout << "Deleted mys." << std::endl;
    return 0;
}

}

void read_data(const char* data_file, const char* y_file,
               COMP_T*& X, DAT_DIM_T& D, N_DAT_T& N, SUPV_T*& Y) {
    std::cout << "Scanning examples..." << std::flush;
    DAT_DIM_T D_y = 0;
    N_DAT_T   N_y = 0;
    if (read_2d_matrix(data_file, X, D, N)) std::exit(-1);
    if (read_2d_matrix(y_file, Y, D_y, N_y)) std::exit(-1);
    if (N != N_y || D_y != 1) {
        std::cerr << "\nNumber of samples mismatch. Abort." << std::endl;
        std::exit(-1);
    }
    std::cout << "Done." << std::endl;
}

using namespace std;

int main_aux() {
    COMP_T* X; DAT_DIM_T D; N_DAT_T N; SUPV_T* Y;
    Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> data, subdata, subsubdata;
    SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> solver;

    read_data("../dat/x", "../dat/y", X, D, N, Y);
    data.load(X, Y, D, N);
    typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::permute_iterator sample(data);

    cout << "data.num_of_samples(): " << data.num_of_samples() << endl;
    for (N_DAT_T i = 0; i < data.num_of_samples(); ++i) {
        cout << "data.y(" << i << "): " << data.y(i)
             << "\ndata.x(" << i << "):";
        COMP_T* x = data.x(i);
        for (DAT_DIM_T j = 0; j < D; ++j) {
            cout << " " << x[j];
        }
        cout << endl;
    }
    for (; sample; ++sample) {
        cout << "sample.y() = " << sample.y() << ":";
        for (DAT_DIM_T j = 0; j < D; ++j) {
            cout << " " << sample[j];
        }
        cout << endl;
    }

    N_DAT_T index[] = {0, 1, 3, 5};
    subdata.load(data, index, 4);
    cout << "subdata.num_of_samples(): " << subdata.num_of_samples() << endl;
    for (N_DAT_T i = 0; i < subdata.num_of_samples(); ++i) {
        cout << "subdata.y(" << i << "): " << subdata.y(i)
             << "\nsubdata.x(" << i << "):";
        COMP_T* x = subdata.x(i);
        for (DAT_DIM_T j = 0; j < D; ++j) {
            cout << " " << x[j];
        }
        cout << endl;
    }
    for (sample.begin(subdata); sample; ++sample) {
        cout << "sample.y() = " << sample.y() << ":";
        for (DAT_DIM_T j = 0; j < D; ++j) {
            cout << " " << sample[j];
        }
        cout << endl;
    }

    N_DAT_T index2[] = {0, 2};
    subsubdata.load(subdata, index2, 2);
    cout << "subsubdata.num_of_samples(): " << subsubdata.num_of_samples() << endl;
    for (N_DAT_T i = 0; i < subsubdata.num_of_samples(); ++i) {
        cout << "subsubdata.y(" << i << "): " << subsubdata.y(i)
             << "\nsubdata.x(" << i << "):";
        COMP_T* x = subsubdata.x(i);
        for (DAT_DIM_T j = 0; j < D; ++j) {
            cout << " " << x[j];
        }
        cout << endl;
    }
    for (sample.begin(subsubdata); sample; ++sample) {
        cout << "sample.y() = " << sample.y() << ":";
        for (DAT_DIM_T j = 0; j < D; ++j) {
            cout << " " << sample[j];
        }
        cout << endl;
    }
    return 0;
}

int main(int argc, const char** argv) {
    int r = main_aux();
    cout << "Exit status: " << r << "\nThis is the end of program." << endl;
    return 0;
}
