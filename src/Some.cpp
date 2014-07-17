# include <iostream>
# include <cstring>
# include <fstream>
# include "Some.hpp"

SoftDecisionSolver::SDParam::SDParam()
      : verbosity(0),
        show_p_each_iter(false),
        num_of_trials(1),
        num_of_iter_update_p_per_train(1),
        num_of_iter_update_p_per_epoch(100),
        num_of_iter_update_p_per_batch(100),
        out_stream(NULL),
        out_name("") {
}

const char* SoftDecisionSolver::SDParam::out() const {
    return out_name;
}

SoftDecisionSolver::SDParam&
SoftDecisionSolver::SDParam::out(const char* _out_name) {
    if (strncmp(_out_name, out_name, sizeof(out_name)) == 0) {
        std::cout << "out_name unchanged." << std::endl;
        return *this;
    }
    if (out_stream != &std::cout && out_stream != NULL) {
        static_cast<std::ofstream*>(out_stream)->close();
        delete out_stream;
        std::cout << "static_cast<std::ofstream*>(out_stream)->close();\ndelete out_stream;" << std::endl;
    }
    if (strcmp(_out_name, "stdout") == 0) {
        out_stream = &std::cout;
        std::cout << "out_stream = &std::cout;" << std::endl;
    } else if (_out_name[0] == '\0') {
        out_stream = NULL;
        std::cout << "out_stream = NULL;" << std::endl;
    } else {
        out_stream = new std::ofstream(_out_name, std::ios_base::out|std::ios_base::app);
        if (!static_cast<std::ofstream*>(out_stream)->is_open()) {
            std::cerr << "***ERROR*** -> SoftDecisionSolver::SDParam::out("
                      << _out_name << ") -> File open failed." << std::endl;
            return *this;
        }
    }
    strncpy(out_name, _out_name, sizeof(out_name));
    return *this;
}
