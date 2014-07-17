# ifndef GDSOLVER_INL_
# define GDSOLVER_INL_

// # include "GDSolver.hpp"
# include <limits>
# include <cstring>
# include <fstream>
# include <cstdio>

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_COMP_T GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
learning_rate(_COMP_T init_learning_rate, unsigned int t) {
    return init_learning_rate;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_COMP_T GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
objective(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data) {
    typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator sample(data);
    return objective(sample, data.num_of_samples());
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam::
GDParam() : verbosity(0),
            show_obj_each_iter(false),
            show_learning_rate_each_iter(false),
            num_of_iter(100),
            optimal_error(1e-8),
            init_learning_rate(0),
            init_learning_rate_try_1st(1),
            init_learning_rate_try_factor(2),
            init_learning_rate_try_subsample_rate(1),
            init_learning_rate_try_min_sample(0),
            out_stream(NULL) {
    out_name[0] = '\0';
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam::
GDParam(const GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& some)
      : verbosity(some.verbosity),
        show_obj_each_iter(some.show_obj_each_iter),
        show_learning_rate_each_iter(some.show_learning_rate_each_iter),
        num_of_iter(some.num_of_iter),
        optimal_error(some.optimal_error),
        init_learning_rate(some.init_learning_rate),
        init_learning_rate_try_1st(some.init_learning_rate_try_1st),
        init_learning_rate_try_factor(some.init_learning_rate_try_factor),
        init_learning_rate_try_subsample_rate(some.init_learning_rate_try_subsample_rate),
        init_learning_rate_try_min_sample(some.init_learning_rate_try_min_sample),
        out_stream(NULL) {
    out_name[0] = '\0';
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam::
~GDParam() {
    if (out_stream != &std::cout && out_stream != NULL) {
        static_cast<std::ofstream*>(out_stream)->close();
        delete out_stream;
    }
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
const char* GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam::
out() const {
    return out_name;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam&
GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam::
out(const char* _out_name) {
    if (strncmp(_out_name, out_name, sizeof(out_name)) == 0) {
        return *this;
    }
    if (out_stream != &std::cout && out_stream != NULL) {
        static_cast<std::ofstream*>(out_stream)->close();
        delete out_stream;
    }
    if (strcmp(_out_name, "stdout") == 0) {
        out_stream = &std::cout;
    } else if (_out_name[0] == '\0') {
        out_stream = NULL;
    } else {
        out_stream = new std::ofstream(_out_name, std::ios_base::out|std::ios_base::app);
        if (!static_cast<std::ofstream*>(out_stream)->is_open()) {
            std::cerr << "***ERROR*** -> GDSolver::GDParam::out("
                      << _out_name << ") -> File open failed." << std::endl;
            return *this;
        }
    }
    strncpy(out_name, _out_name, sizeof(out_name));
    return *this;
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
GDSolver() : t(0),
             init_learning_rate(0),
             gd_param(NULL) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
GDSolver(const GDParam& _gd_param)
      : t(0),
        init_learning_rate(0),
        gd_param(&_gd_param) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
int GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
set_up(const GDParam& _gd_param) {
    gd_param = &_gd_param;
    return 0;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
bool GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
is_set_up() const {
    return gd_param != NULL;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
int GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
reset() {
    t                  = 0;
    init_learning_rate = gd_param->init_learning_rate;
    return 0;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
int GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data,
      GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem) {
    if (!is_set_up()) {
        std::cerr << "Error in GDSolver::train(): Parameters have not been specified. Training process is skipped." << std::endl;
        return -1;
    }
    if (data.dimension() <= 0) {
        std::cerr << "Error in GDSolver::train(): Dimensionality has not been specified. Training process is skipped." << std::endl;
        return -1;
    }
    if (gd_param->verbosity >= 1) {
        std::cout << "GD Training: \n    Data: " << data.num_of_samples()
                  << " samples, " << data.dimension()
                  << " feature dimensions.\n    Stopping Criterion: "
                  << gd_param->num_of_iter << " iterations";
        if (gd_param->optimal_error > 0)
            std::cout << " or accuracy higher than " << gd_param->optimal_error;
        std::cout <<  ".\nGD Training: begin." << std::endl;
    }
    reset();
    if (init_learning_rate == 0)
        if (int r = try_learning_rate(data, problem)) return r;
    if (init_learning_rate < gd_param->optimal_error) {
        if (gd_param->verbosity >= 1) {
            std::cout << "GD Training: finished.\n"
                      << "    Accuracy already satisfied." << std::endl;
        }
        return 0;
    }
    _COMP_T       obj0 = std::numeric_limits<_COMP_T>::max();
    _COMP_T       obj1 = 0;
    std::ostream* out  = gd_param->out_stream;
    unsigned int i;
    typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator* sample = train_iterator(data);
    if (gd_param->verbosity >= 1) {
        std::cout << "Training ... ";
        if (gd_param->verbosity > 1) std::cout << std::endl;
        else std::cout.flush();
    }
    if (out != NULL) {
        problem.ostream_var(*out);
        *out << " " << problem.objective(sample->pos(0), data.num_of_samples())
             << std::endl;
    }
    for (i = 0; i < gd_param->num_of_iter; ++i) {
        if (gd_param->verbosity >= 2) {
            std::cout << "    Iteration " << i + 1 << " ... ";
            if (gd_param->verbosity > 2) {
                std::cout << std::endl;
            } else {
                std::cout << std::flush;
            }
        }
        train_iteration(sample->begin(), data.num_of_samples(), problem);
        if (gd_param->optimal_error > 0 || gd_param->show_obj_each_iter) {
            obj1 = problem.objective(sample->pos(0), data.num_of_samples());
        }
        if (gd_param->verbosity == 2) {
            std::cout << "Done.";
            if (gd_param->show_learning_rate_each_iter) {
                std::cout << " Learning rate = "
                          << problem.learning_rate(init_learning_rate, t);
                if (gd_param->show_obj_each_iter) std::cout << ",";
                else std::cout << ".";
            }
            if (gd_param->show_obj_each_iter) {
                std::cout << " Objective = " << obj1 << ".";
            }
            std::cout << std::endl;
        }
        if (out) {
            problem.ostream_var(*out);
            *out << " ";
            *out << obj1 << std::endl;
        }
        if (gd_param->optimal_error > 0 &&
            obj0 - obj1 < gd_param->optimal_error) break;
        obj0 = obj1;
    }
    delete sample;
    if (gd_param->verbosity >= 1) {
        if (gd_param->verbosity == 1) std::cout << "Done.\n";
        std::cout << "GD Training: finished. \n";
        if (i < gd_param->num_of_iter)
            std::cout << "    Training stopped at iteration " << t + 1
                      << " with convergence.";
        else
            std::cout << "    Max number of iterations has been reached.";
        std::cout << std::endl;
    }
    return 0;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
int GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
test(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data,
     GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem) {
    if (data.dimension() <= 0) {
        std::cerr << "Error in GDSolver::test(): Dimensionality has not been specified. Testing process is skipped." << std::endl;
        return -1;
    }
    if (gd_param->verbosity >= 1)
        std::cout << "GD Testing: \n  Data: " << data.num_of_samples()
                  << " samples.\nTesting ... " << std::flush;
    typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator sample;
    for (sample.begin(data); sample; ++sample) {
        sample.ry() = problem.test_one(sample);
    }
    if (gd_param->verbosity >= 1)
        std::cout << "Done.\nGD Testing: finished." << std::endl;
    return 0;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
int GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
try_learning_rate(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data,
                  GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem) {
    _COMP_T eta0_try1, eta0_try2, obj_try1, obj_try2, eta0_try_factor;
    GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>* try1 = NULL, *try2 = NULL, *try_temp = NULL;
    _N_DAT_T num_of_subsamples;
    unsigned int origin_t = t;
    typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator sample;

    if (gd_param->verbosity >= 1) {
        std::cout << "Initializing ... ";
        if (gd_param->verbosity >= 2) {
            std::cout << "\n    No initial learning rate \"eta0\" specified. "
                      << "Deciding automatically ... ";
            if (gd_param->verbosity > 2) std::cout << std::endl;
            else std::cout.flush();
        }
        std::cout.flush();
    }
    num_of_subsamples = gd_param->init_learning_rate_try_subsample_rate * data.num_of_samples();
    if (num_of_subsamples < gd_param->init_learning_rate_try_min_sample) {
        num_of_subsamples = gd_param->init_learning_rate_try_min_sample;
    }
    if (num_of_subsamples > data.num_of_samples()) {
        num_of_subsamples = data.num_of_samples();
    }
    sample.begin(data, num_of_subsamples);
    eta0_try1 = gd_param->init_learning_rate_try_1st;
    eta0_try_factor = gd_param->init_learning_rate_try_factor;
    if (eta0_try1 < gd_param->optimal_error) {
        init_learning_rate = eta0_try1;
        return 0;
    }
    if (gd_param->verbosity >= 3)
        std::cout << "        Trying eta0 = " << eta0_try1 << " ... " << std::flush;
    try1 = problem.copy();
    init_learning_rate = eta0_try1;
    train_iteration(sample.pos(0), num_of_subsamples, *try1);
    obj_try1 = try1->objective(sample.pos(0), num_of_subsamples);
    if (gd_param->verbosity >= 3)
        std::cout << "Done. Obj = " << obj_try1 << "." << std::endl;
    eta0_try2 = eta0_try1 * eta0_try_factor;
    if (eta0_try2 < gd_param->optimal_error) {
        delete try1;
        init_learning_rate = eta0_try2;
        return 0;
    }
    if (gd_param->verbosity >= 3)
        std::cout << "        Trying eta0 = " << eta0_try2 << "... " << std::flush;
    try2 = problem.copy();
    init_learning_rate = eta0_try2;
    t = origin_t;
    train_iteration(sample.pos(0), num_of_subsamples, *try2);
    obj_try2 = try2->objective(sample.pos(0), num_of_subsamples);
    if (gd_param->verbosity >= 3)
        std::cout << "Done. Obj = " << obj_try2 << "." << std::endl;
    if (obj_try1 < obj_try2) {
        eta0_try_factor = 1 / eta0_try_factor;
        obj_try2        = obj_try1;
        eta0_try2       = eta0_try1;
        try_temp        = try2;
        try2            = try1;
        try1            = try_temp;
    }
    do {
        eta0_try1 = eta0_try2;
        obj_try1  = obj_try2;
        try_temp  = try1;
        try1      = try2;
        try2      = try_temp;
        eta0_try2 = eta0_try1 * eta0_try_factor;
        if (eta0_try2 < gd_param->optimal_error) break;
        if (gd_param->verbosity >= 3)
            std::cout << "        Trying eta0 = " << eta0_try2 << "... " << std::flush;
        try2->assign_from(problem);
        init_learning_rate = eta0_try2;
        t = origin_t;
        train_iteration(sample.pos(0), num_of_subsamples, *try2);
        obj_try2 = try2->objective(sample.pos(0), num_of_subsamples);
        if (gd_param->verbosity >= 3)
            std::cout << "Done. Obj = " << obj_try2 << "." << std::endl;
    } while (obj_try1 > obj_try2);
    delete try2;
    init_learning_rate = eta0_try1;// * (eta0_try_factor > 1 ? 1.0 / eta0_try_factor : eta0_try_factor);
    problem.assign_from(*try1);
    delete try1;
    if (gd_param->verbosity == 1) std::cout << "Done." << std::endl;
    if (gd_param->verbosity >= 2) {
        if (gd_param->verbosity == 2) std::cout << "Done.\n";
        std::cout << "    Setting eta0 = " << init_learning_rate << "." << std::endl;
    }
    return 0;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator*
GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train_iterator(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data) {
    return new typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator(data);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
int GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train_iteration(typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator& sample,
                _N_DAT_T num_of_samples,
                GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem) {
    _COMP_T learning_rate = problem.learning_rate(init_learning_rate, t++);
    return problem.train_batch(sample, num_of_samples, learning_rate);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::SGDParam::
SGDParam() : size_of_batch(1) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGDSolver() : sgd_param(NULL) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGDSolver(const typename GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& _gd_param,
          const SGDParam& _sgd_param)
      : GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>(_gd_param),
        sgd_param(&_sgd_param) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
int SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
set_up(const typename GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& _gd_param,
       const SGDParam& _sgd_param) {
    this->gd_param = &_gd_param;
    sgd_param      = &_sgd_param;
    return 0;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
bool SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
is_set_up() const {
    return this->gd_param != NULL && sgd_param != NULL;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator*
SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train_iterator(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data) {
    return new typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator(data);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
int SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train_iteration(typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator& sample,
                _N_DAT_T num_of_samples,
                GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem) {
    _N_DAT_T origin_pos = sample.pos();
    for (_N_DAT_T i = 0; i < num_of_samples && sample; i += sgd_param->size_of_batch, ++this->t) {
        _COMP_T learning_rate = problem.learning_rate(this->init_learning_rate, this->t);
        _N_DAT_T size_of_next_batch = num_of_samples - i;
        if (size_of_next_batch > sgd_param->size_of_batch) {
            size_of_next_batch = sgd_param->size_of_batch;
        }
        if (this->gd_param->verbosity >= 3) {
            std::cout << "        Iterating through samples "
                      << i + 1 << " ~ " << i + size_of_next_batch
                      << " ... " << std::flush;
        }
        if (int r =
            problem.train_batch(sample, size_of_next_batch, learning_rate))
            return r;
        if (this->gd_param->verbosity >= 3) {
            std::cout << "Done.";
            if (this->gd_param->show_learning_rate_each_iter) {
                std::cout << " Learning rate = "
                          << problem.learning_rate(this->init_learning_rate, this->t);
                if (this->gd_param->show_obj_each_iter) std::cout << ",";
                else std::cout << ".";
            }
            if (this->gd_param->show_obj_each_iter) {
                _N_DAT_T cur_pos = sample.pos();
                std::cout << " Objective = "
                          << problem.objective(sample.pos(origin_pos), num_of_samples) << ".";
                sample.pos(cur_pos);
            }
            std::cout << std::endl;
        }
    }
    return 0;
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
bpy::tuple GDSolver_GDParam_bpy_pickle<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
getstate(const typename GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& param) {
    return bpy::make_tuple(param.verbosity,
                           param.show_obj_each_iter,
                           param.show_learning_rate_each_iter,
                           param.num_of_iter,
                           param.optimal_error,
                           param.init_learning_rate,
                           param.init_learning_rate_try_1st,
                           param.init_learning_rate_try_factor,
                           param.init_learning_rate_try_subsample_rate,
                           param.init_learning_rate_try_min_sample);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
void GDSolver_GDParam_bpy_pickle<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
setstate(typename GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& param, bpy::tuple& state) {
    if (bpy::len(state) != 10) {
        char err_msg[SIZEOF_LINE];
        snprintf(err_msg, SIZEOF_LINE, "Expected 10-item tuple in call to " \
                 "GDParam.__setstate__; got %zd items", bpy::len(state));
        PyErr_SetString(PyExc_ValueError, err_msg);
        bpy::throw_error_already_set();
    }
    param.verbosity                             = bpy::extract<char>(state[0]);
    param.show_obj_each_iter                    = bpy::extract<bool>(state[1]);
    param.show_learning_rate_each_iter          = bpy::extract<bool>(state[2]);
    param.num_of_iter                           = bpy::extract<unsigned int>(state[3]);
    param.optimal_error                         = bpy::extract<_COMP_T>(state[4]);
    param.init_learning_rate                    = bpy::extract<_COMP_T>(state[5]);
    param.init_learning_rate_try_1st            = bpy::extract<_COMP_T>(state[6]);
    param.init_learning_rate_try_factor         = bpy::extract<_COMP_T>(state[7]);
    param.init_learning_rate_try_subsample_rate = bpy::extract<_COMP_T>(state[8]);
    param.init_learning_rate_try_min_sample     = bpy::extract<_N_DAT_T>(state[9]);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
bpy::tuple SGDSolver_SGDParam_bpy_pickle<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
getstate(const typename SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::SGDParam& param) {
    return bpy::make_tuple(param.size_of_batch);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
void SGDSolver_SGDParam_bpy_pickle<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
setstate(typename SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::SGDParam& param, bpy::tuple& state) {
    if (bpy::len(state) != 1) {
        char err_msg[SIZEOF_LINE];
        snprintf(err_msg, SIZEOF_LINE, "Expected 1-item tuple in call to " \
                 "SGDParam.__setstate__; got %zd items", bpy::len(state));
        PyErr_SetString(PyExc_ValueError, err_msg);
        bpy::throw_error_already_set();
    }
    param.size_of_batch = bpy::extract<_N_DAT_T>(state[0]);
}

# endif
