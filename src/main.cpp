# include "my_typedefs.h"
# include "my_lib.hpp"
# include "SoftDecisionModel.hpp"
# include <iostream>

int read_data(const char* data_file, const char* y_file,
               COMP_T*& X, DAT_DIM_T& D, N_DAT_T& N, SUPV_T*& Y) {
    std::cout << "Scanning examples..." << std::flush;
    DAT_DIM_T D_y = 0;
    N_DAT_T   N_y = 0;
    if (int r = read_2d_matrix(data_file, X, D, N)) return r;
    if (int r = read_2d_matrix(y_file, Y, D_y, N_y)) return r;
    if (N != N_y || D_y != 1) {
        std::cerr << "\nNumber of samples mismatch. Abort." << std::endl;
        return -1;
    }
    std::cout << "Done." << std::endl;
    return 0;
}

int main(int argc, const char** argv) {
    COMP_T* X; DAT_DIM_T D; N_DAT_T N; SUPV_T* Y;

    GDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam gd_param;
    SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam sgd_param;
    SoftDecisionSolver::SDParam sd_param;
    SoftDecisionModel::ModelParam model_param;

    Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> data;

    read_data("dat/data.X", "dat/data.Y", X, D, N, Y);
    data.load(X, Y, D, N);

    std::ofstream log_file("dat/log.txt");
    if (!log_file.is_open()) {
        return -1;
    }
    gd_param.out_training_proc = &log_file;
    sd_param.out_training_proc = &log_file;

    SoftDecisionModel model(data, model_param);
    SoftDecisionSolver solver;
    solver.set_up(gd_param, sgd_param, sd_param);

    solver.train(data, model);

    log_file.close();

    return 0;
}
