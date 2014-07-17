# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
# define PY_ARRAY_UNIQUE_SYMBOL SOFTLABELTREE_MODULE
# define NO_IMPORT_ARRAY

# include <cstring>
# include <iostream>
# include <iomanip>
# include <limits>
# include <utility>
# include <fstream>
# include <algorithm>
# include <omp.h>
# include <boost/python.hpp>
# include <numpy/arrayobject.h>
# include "Q.hpp"
# include "SoftDecisionModel.hpp"

const DAT_DIM_T iter_per_thread = 512;

SoftDecisionModel::ModelParam::ModelParam()
      : reg_coeff(1e-5),
        reg_l1_ratio(0),
        b_learning_rate_factor(0.01),
        init_var_subsample_rate(1),
        init_var_subsample_min(1) {
}

SoftDecisionModel::SoftDecisionModel()
      : b(0),
        n_w_i(0),
        w_i(NULL),
        w(NULL),
        p(NULL),
        param(NULL),
        index_of_pos_sample(NULL),
        index_of_neg_sample(NULL),
        num_of_pos_sample(0),
        num_of_neg_sample(0),
        dimension(0),
        num_of_labels(0),
        num_of_samples(0) {
}

SoftDecisionModel::SoftDecisionModel(const ModelParam& _param)
      : b(0),
        n_w_i(0),
        w_i(NULL),
        w(NULL),
        p(NULL),
        param(&_param),
        index_of_pos_sample(NULL),
        index_of_neg_sample(NULL),
        num_of_pos_sample(0),
        num_of_neg_sample(0),
        dimension(0),
        num_of_labels(0),
        num_of_samples(0) {
}

SoftDecisionModel::SoftDecisionModel(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data,
                                     ModelParam& _param)
      : b(0),
        n_w_i(data.dimension()),
        param(&_param),
        num_of_pos_sample(0),
        num_of_neg_sample(0),
        dimension(data.dimension()),
        num_of_labels(data.num_of_labels()),
        num_of_samples(data.num_of_samples()) {
    w_i = new DAT_DIM_T[dimension];
    w   = new COMP_T[dimension];
    p   = new COMP_T[num_of_labels];
    index_of_pos_sample = new N_DAT_T[num_of_samples];
    index_of_neg_sample = new N_DAT_T[num_of_samples];
    for (DAT_DIM_T i = 0; i < n_w_i; ++i) {
        w_i[i] = i;
    }
}

SoftDecisionModel::SoftDecisionModel(const SoftDecisionModel& some)
      : b(some.b),
        n_w_i(some.n_w_i),
        w_i(NULL),
        w(NULL),
        p(NULL),
        param(some.param),
        index_of_pos_sample(NULL),
        index_of_neg_sample(NULL),
        num_of_pos_sample(some.num_of_pos_sample),
        num_of_neg_sample(some.num_of_neg_sample),
        dimension(some.dimension),
        num_of_labels(some.num_of_labels),
        num_of_samples(some.num_of_samples) {
    if (some.w_i != NULL) {
        w_i = new DAT_DIM_T[dimension];
        std::memcpy(w_i, some.w_i, sizeof(DAT_DIM_T) * n_w_i);
    }
    if (some.w != NULL) {
        w = new COMP_T[dimension];
        std::memcpy(w, some.w, sizeof(COMP_T) * dimension);
    }
    if (some.p != NULL) {
        p = new COMP_T[num_of_labels];
        std::memcpy(p, some.p, sizeof(COMP_T) * num_of_labels);
    }
    if (some.index_of_pos_sample != NULL) {
        index_of_pos_sample = new N_DAT_T[num_of_samples];
        std::memcpy(index_of_pos_sample, some.index_of_pos_sample,
                    sizeof(N_DAT_T) * num_of_pos_sample);
    }
    if (some.index_of_neg_sample != NULL) {
        index_of_neg_sample = new N_DAT_T[num_of_samples];
        std::memcpy(index_of_neg_sample, some.index_of_neg_sample,
                    sizeof(N_DAT_T) * num_of_neg_sample);
    }
}

SoftDecisionModel::SoftDecisionModel(SoftDecisionModel&& some)
      : b(some.b),
        n_w_i(some.n_w_i),
        w_i(some.w_i),
        w(some.w),
        p(some.p),
        param(some.param),
        index_of_pos_sample(some.index_of_pos_sample),
        index_of_neg_sample(some.index_of_neg_sample),
        num_of_pos_sample(some.num_of_pos_sample),
        num_of_neg_sample(some.num_of_neg_sample),
        dimension(some.dimension),
        num_of_labels(some.num_of_labels),
        num_of_samples(some.num_of_samples) {
    some.w_i = NULL;
    some.w   = NULL;
    some.p   = NULL;
    some.index_of_pos_sample = NULL;
    some.index_of_neg_sample = NULL;
}


SoftDecisionModel::~SoftDecisionModel() {
    delete[] w_i;
    delete[] w;
    delete[] p;
    delete[] index_of_pos_sample;
    delete[] index_of_neg_sample;
}

SoftDecisionModel& SoftDecisionModel::operator=(SoftDecisionModel& some) {
    if (&some == this) return *this;
    b = some.b;
    n_w_i = some.n_w_i;
    if (dimension != some.dimension) {
        delete[] w_i;
        delete[] w;
        dimension = some.dimension;
        if (dimension > 0) {
            w_i = new DAT_DIM_T[dimension];
            w   = new COMP_T[dimension];
        } else {
            w_i = NULL;
            w   = NULL;
        }
    }
    if (num_of_labels != some.num_of_labels) {
        delete[] p;
        num_of_labels = some.num_of_labels;
        if (num_of_labels > 0) {
            p = new COMP_T[num_of_labels];
        } else {
            p = NULL;
        }
    }
    param = some.param;
    if (num_of_samples != some.num_of_samples) {
        delete[] index_of_pos_sample;
        delete[] index_of_neg_sample;
        num_of_samples = some.num_of_samples;
        if (num_of_samples > 0) {
            index_of_pos_sample = new N_DAT_T[num_of_samples];
            index_of_neg_sample = new N_DAT_T[num_of_samples];
        } else {
            index_of_pos_sample = NULL;
            index_of_neg_sample = NULL;
        }
    }
    num_of_pos_sample = some.num_of_pos_sample;
    num_of_neg_sample = some.num_of_neg_sample;
    if (w_i != NULL) std::memcpy(w_i, some.w_i, sizeof(DAT_DIM_T) * n_w_i);
    if (w != NULL) std::memcpy(w, some.w, sizeof(COMP_T) * dimension);
    if (p != NULL) std::memcpy(p, some.p, sizeof(COMP_T) * num_of_labels);
    if (index_of_pos_sample != NULL)
        std::memcpy(index_of_pos_sample, some.index_of_pos_sample,
                    sizeof(N_DAT_T) * num_of_pos_sample);
    if (index_of_neg_sample != NULL)
        std::memcpy(index_of_neg_sample, some.index_of_neg_sample,
                    sizeof(N_DAT_T) * num_of_neg_sample);
    return *this;
}

SoftDecisionModel& SoftDecisionModel::operator=(SoftDecisionModel&& some) {
    if (&some == this) return *this;
    delete[] w_i;
    delete[] w;
    delete[] p;
    delete[] index_of_pos_sample;
    delete[] index_of_neg_sample;
    b                        = some.b;
    n_w_i                    = some.n_w_i;
    w_i                      = some.w_i;
    w                        = some.w;
    p                        = some.p;
    param                    = some.param;
    index_of_pos_sample      = some.index_of_pos_sample;
    index_of_neg_sample      = some.index_of_neg_sample;
    num_of_pos_sample        = some.num_of_pos_sample;
    num_of_neg_sample        = some.num_of_neg_sample;
    dimension                = some.dimension;
    num_of_labels            = some.num_of_labels;
    num_of_samples           = some.num_of_samples;
    some.w_i                 = NULL;
    some.w                   = NULL;
    some.p                   = NULL;
    some.index_of_pos_sample = NULL;
    some.index_of_neg_sample = NULL;
    return *this;
}

int SoftDecisionModel::init_var(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data) {
    N_DAT_T num_of_subsamples = param->init_var_subsample_rate * data.num_of_samples();
    if (num_of_subsamples < param->init_var_subsample_min) {
        num_of_subsamples = param->init_var_subsample_min;
    }
    if (num_of_subsamples > data.num_of_samples()) {
        num_of_subsamples = data.num_of_samples();
    }
    typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::permute_iterator
        sample(data, num_of_subsamples);
    COMP_T* subcenter = new COMP_T[dimension];
    n_w_i = dimension;
    # pragma omp parallel for num_threads(omp_num_of_threads(dimension, iter_per_thread))
    for (DAT_DIM_T d = 0; d < dimension; ++d) {
        subcenter[d] = 0;
        w_i[d] = d;
    }
    for (N_DAT_T i = 0; sample && i < num_of_subsamples; ++i, ++sample) {
        # pragma omp parallel for num_threads(omp_num_of_threads(dimension, iter_per_thread))
        for (DAT_DIM_T d = 0; d < dimension; ++d) {
            subcenter[d] += sample[d];
        }
    }
    b = 0;
    COMP_T neg_b = 0;
    sample.begin(data, 1);
    # pragma omp parallel for reduction(+:neg_b) num_threads(omp_num_of_threads(dimension, iter_per_thread))
    for (DAT_DIM_T d = 0; d < dimension; ++d) {
        subcenter[d] /= num_of_subsamples;
        w[d] = sample[d] - subcenter[d];
        neg_b += w[d] * subcenter[d];
    }
    b = -neg_b;
    delete[] subcenter;
    return 0;
}

N_DAT_T SoftDecisionModel::update_p(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data) {
    typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::label_iterator sample(data);
    N_DAT_T num_of_pos_sample_old = num_of_pos_sample;
    num_of_pos_sample  = 0;
    num_of_neg_sample  = 0;
    for (SUPV_T l = 0; l < data.num_of_labels(); ++l) {
        N_DAT_T num_of_pos_sample_with_label = 0;
        for (sample.begin(l); sample; ++sample) {
            if (score(sample) > 0) {
                ++num_of_pos_sample_with_label;
                index_of_pos_sample[num_of_pos_sample++] = sample.index();
            } else {
                index_of_neg_sample[num_of_neg_sample++] = sample.index();
            }
        }
        p[l] = static_cast<COMP_T>(num_of_pos_sample_with_label) /
               data.num_of_samples_with_label(l);
    }
    return num_of_pos_sample - num_of_pos_sample_old;
}

COMP_T SoftDecisionModel::learning_rate(COMP_T init_learning_rate,
                                        unsigned int t) {
    return init_learning_rate / (1 + param->reg_coeff * init_learning_rate * t);
}

COMP_T SoftDecisionModel::objective(typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample,
                                    N_DAT_T _num_of_samples) {
    COMP_T loss_term = loss(sample, _num_of_samples);
    return loss_term + param->reg_coeff * (
                            param->reg_l1_ratio * l1_norm() +
                            0.5 * (1 - param->reg_l1_ratio) * l2_norm());
}

int SoftDecisionModel::train_batch(typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample,
                                   N_DAT_T _num_of_samples, COMP_T learning_rate) {
    COMP_T     temp_term   = 0;
    COMP_T*    temp_term_d = new COMP_T[n_w_i];
    DAT_DIM_T* w_i_zero    = new DAT_DIM_T[n_w_i];
    DAT_DIM_T  n_w_i_zero  = 0;
    # pragma omp parallel for num_threads(omp_num_of_threads(n_w_i, iter_per_thread))
    for (DAT_DIM_T i = 0; i < n_w_i; ++i) {
        temp_term_d[i] = 0;
    }
    COMP_T weight_pos, weight_neg;
    weight_pos = 1.0 * (num_of_pos_sample + num_of_neg_sample) /
                 (_num_of_samples *
                  (num_of_pos_sample > 0 ? num_of_pos_sample : 1.0));
    weight_neg = 1.0 * (num_of_pos_sample + num_of_neg_sample) /
                 (_num_of_samples *
                  (num_of_neg_sample > 0 ? num_of_neg_sample : 1.0));
    for (N_DAT_T count = 0; sample && count < _num_of_samples; ++count, ++sample) {
        COMP_T sample_score = score(sample);
        COMP_T coeff;
        if (sample_score > 1.0)
            coeff = (p[sample->index_of_label(sample.y())] - 1.0) * weight_neg;
        else if (sample_score < -1.0)
            coeff = p[sample->index_of_label(sample.y())] * weight_pos;
        else
            coeff = p[sample->index_of_label(sample.y())] *
                    (weight_pos + weight_neg) - weight_neg;
        # pragma omp parallel for num_threads(omp_num_of_threads(n_w_i, iter_per_thread))
        for (DAT_DIM_T i = 0; i < n_w_i; ++i) {
            temp_term_d[i] += coeff * sample[w_i[i]];
        }
        temp_term += coeff;
    }
    # pragma omp parallel for num_threads(omp_num_of_threads(n_w_i, iter_per_thread))
    for (DAT_DIM_T i = 0; i < n_w_i; ++i) {
        w[w_i[i]] += learning_rate * (temp_term_d[i] +
                     param->reg_coeff * (param->reg_l1_ratio - 1) * w[w_i[i]]);
        COMP_T w_d   = w[w_i[i]];
        COMP_T delta = learning_rate * param->reg_coeff * param->reg_l1_ratio;
        if (delta > (w_d > 0 ? w_d : -w_d)) {
            # pragma omp critical (add_zero)
            w_i_zero[n_w_i_zero++] = i;
            w[w_i[i]] = 0;
        } else {
            delta = w_d > 0 ? -delta : delta;
            w[w_i[i]] = w_d + delta;
        }
    }
    std::sort(w_i_zero, w_i_zero + n_w_i_zero);
    DAT_DIM_T offset = 1;
    DAT_DIM_T i_w_i = w_i_zero[0];
    for (DAT_DIM_T i_zero = 1; i_zero <= n_w_i_zero; ++i_zero) {
        DAT_DIM_T next_i_zero = (i_zero == n_w_i_zero) ? n_w_i : w_i_zero[i_zero];
        for (; i_w_i + offset < next_i_zero; ++i_w_i) {
            w_i[i_w_i] = w_i[i_w_i + offset];
        }
        offset++;
    }
    n_w_i -= n_w_i_zero;
    b += param->b_learning_rate_factor * learning_rate * temp_term;
    delete[] temp_term_d;
    delete[] w_i_zero;
    return 0;
}

SUPV_T SoftDecisionModel::test_one(
        typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample) {
    return score(sample) > 0 ? 1 : -1;
}

COMP_T SoftDecisionModel::test_one(bpy::object& bpy_sample) {
    PyObject* py_sample;
    COMP_T*   sample;
    int       ndim;
    npy_intp* shape;
    if (parse_numpy_array(bpy_sample, py_sample, sample, ndim, shape)) {
        std::cerr << "**ERROR** -> SoftDecisionModel::test_one(sample) "
                     "-> Cannot parse sample." << std::endl;
        return 0;
    }
    if (ndim > 2) {
        std::cerr << "**ERROR** -> SoftDecisionModel::test_one(sample) "
                     "-> sample is an ndarray which has more "
                     "than 2 dimensions." << std::endl;
        return 0;
    }
    DAT_DIM_T sample_dimension = (ndim == 1 ?
                                  shape[0] : (shape[0] > shape[1] ?
                                              shape[0] : shape[1]));
    if (sample_dimension != dimension) {
        std::cerr << "**ERROR** -> SoftDecisionModel::test_one(sample) "
                  << "-> sample has incompatible dimensions with the model."
                  << std::endl;
        return 0;
    }
    COMP_T score = 0;
    # pragma omp parallel for reduction(+:score) num_threads(omp_num_of_threads(n_w_i, iter_per_thread))
    for (DAT_DIM_T i = 0; i < n_w_i; ++i) {
        score += sample[w_i[i]] * w[w_i[i]];
    }
    Py_DECREF(py_sample);
    return score + b;
}

GDProblem<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>*
SoftDecisionModel::copy() {
    return new SoftDecisionModel(*this);
}

int SoftDecisionModel::assign_from(GDProblem& some) {
    SoftDecisionModel& model = static_cast<SoftDecisionModel&>(some);
    *this = model;
    return 0;
}

int SoftDecisionModel::ostream_var(std::ostream& out) {
    for (DAT_DIM_T d = 0; d < dimension; ++d) {
        out << w[d] << " ";
    }
    out << b;
    return 0;
}

int SoftDecisionModel::ostream_p(
        std::ostream& out,
        Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data) {
    for (SUPV_T l = 0; l < num_of_labels; ++l) {
        out << "[" << std::setw(6) << std::right << data.label(l) << "|"
            << std::setprecision(5) << std::setw(7) << std::left << p[l] << "]";
    }
    return 0;
}

COMP_T SoftDecisionModel::p_mean() {
    COMP_T p_sum = 0;
    for (SUPV_T i = 0; i < num_of_labels; ++i) {
        p_sum += p[i];
    }
    return p_sum / num_of_labels;
}

COMP_T SoftDecisionModel::pos_percentage() {
    return static_cast<COMP_T>(num_of_pos_sample) / (num_of_pos_sample + num_of_neg_sample);
}

bpy::object SoftDecisionModel::ws() const {
    return c_array_to_numpy_1d_array(w, dimension, false);
}

bpy::object SoftDecisionModel::ps() const {
    return c_array_to_numpy_1d_array(p, num_of_labels, false);
}

COMP_T SoftDecisionModel::bs() const {
    return b;
}

DAT_DIM_T SoftDecisionModel::num_of_nonzeros() const {
    return n_w_i;
}

const SoftDecisionModel::ModelParam& SoftDecisionModel::model_param() const {
    return *param;
}

bpy::object SoftDecisionModel::indices_of_pos_samples() {
    return c_array_to_numpy_1d_array(index_of_pos_sample, num_of_pos_sample, false);
}

bpy::object SoftDecisionModel::indices_of_neg_samples() {
    return c_array_to_numpy_1d_array(index_of_neg_sample, num_of_neg_sample, false);
}

COMP_T SoftDecisionModel::loss(
        typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample,
        N_DAT_T _num_of_samples) const {
    COMP_T* loss_pos_p = new COMP_T[num_of_labels];
    COMP_T* loss_neg_p = new COMP_T[num_of_labels];
    for (SUPV_T i = 0; i < num_of_labels; ++i) {
        loss_pos_p[i] = 0;
        loss_neg_p[i] = 0;
    }
    for (N_DAT_T i = 0; sample && i < _num_of_samples; ++sample, ++i) {
        COMP_T sample_score = score(sample);
        COMP_T sample_pos_loss = sample_score < 1.0 ? 1.0 - sample_score : 0;
        COMP_T sample_neg_loss = sample_score > -1.0 ? 1.0 + sample_score : 0;
        loss_pos_p[sample->index_of_label(sample.y())] += sample_pos_loss;
        loss_neg_p[sample->index_of_label(sample.y())] += sample_neg_loss;
    }
    COMP_T loss_pos = 0;
    COMP_T loss_neg = 0;
    for (SUPV_T i = 0; i < num_of_labels; ++i) {
        loss_pos += p[i] * loss_pos_p[i];
        loss_neg += (1.0 - p[i]) * loss_neg_p[i];
    }
    delete[] loss_pos_p;
    delete[] loss_neg_p;
    return loss_pos / num_of_pos_sample + loss_neg / num_of_neg_sample;
}

COMP_T SoftDecisionModel::score(
        typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample) const {
    COMP_T score = 0;
    # pragma omp parallel for reduction(+:score) num_threads(omp_num_of_threads(n_w_i, iter_per_thread))
    for (DAT_DIM_T i = 0; i < n_w_i; ++i) {
        score += sample[w_i[i]] * w[w_i[i]];
    }
    return score + b;
}

COMP_T SoftDecisionModel::l1_norm() const {
    COMP_T norm = 0;
    for (DAT_DIM_T i = 0; i < n_w_i; ++i) {
        norm += (w[w_i[i]] > 0 ? w[w_i[i]] : -w[w_i[i]]);
    }
    return norm;
}

COMP_T SoftDecisionModel::l2_norm() const {
    COMP_T norm = 0;
    for (DAT_DIM_T i = 0; i < n_w_i; ++i) {
        norm += w[w_i[i]] * w[w_i[i]];
    }
    return norm;
}

SoftDecisionSolver::SDParam::SDParam()
      : verbosity(0),
        show_p_each_iter(false),
        num_of_trials(1),
        num_of_iter_update_p_per_train(1),
        num_of_iter_update_p_per_epoch(100),
        num_of_iter_update_p_per_batch(100),
        num_of_iter_confirm_converge(0),
        out_stream(NULL) {
    out_name[0] = '\0';
}

SoftDecisionSolver::SDParam::SDParam(const SoftDecisionSolver::SDParam& some)
      : verbosity(some.verbosity),
        show_p_each_iter(some.show_p_each_iter),
        num_of_trials(some.num_of_trials),
        num_of_iter_update_p_per_train(some.num_of_iter_update_p_per_train),
        num_of_iter_update_p_per_epoch(some.num_of_iter_update_p_per_epoch),
        num_of_iter_update_p_per_batch(some.num_of_iter_update_p_per_batch),
        num_of_iter_confirm_converge(some.num_of_iter_confirm_converge),
        out_stream(NULL) {
    out_name[0] = '\0';
}

SoftDecisionSolver::SDParam::~SDParam() {
    if (out_stream != &std::cout && out_stream != NULL) {
        static_cast<std::ofstream*>(out_stream)->close();
        delete out_stream;
    }
}

const char* SoftDecisionSolver::SDParam::out() const {
    return out_name;
}

SoftDecisionSolver::SDParam&
SoftDecisionSolver::SDParam::out(const char* _out_name) {
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
            std::cerr << "***ERROR*** -> SoftDecisionSolver::SDParam::out("
                      << _out_name << ") -> File open failed." << std::endl;
            return *this;
        }
    }
    strncpy(out_name, _out_name, sizeof(out_name));
    return *this;
}

SoftDecisionSolver::SoftDecisionSolver()
      : sd_param(NULL) {
}

int SoftDecisionSolver::set_up(
        const typename GDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam& _gd_param,
        const typename SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
        const SDParam& _sd_param) {
    set_up(_gd_param, _sgd_param);
    sd_param = &_sd_param;
    return 0;
}

bool SoftDecisionSolver::is_set_up() const {
    return SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::is_set_up() &&
           sd_param != NULL;
}

int SoftDecisionSolver::train(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data,
                              SoftDecisionModel& model) {
    if (!is_set_up()) {
        std::cerr << "**ERROR** -> SoftDecisionSolver::train(...) " \
                  << "-> Solver is not set up." << std::endl;
        return -1;
    }
    if (sd_param->verbosity >= 1) {
        std::cout << "Soft-Decision Training: \n    Data: "
                  << data.num_of_samples() << " samples, "
                  << data.dimension() << " feature dimensions.\n    Stopping Criterion: "
                  << sd_param->num_of_iter_update_p_per_train << " - "
                  << sd_param->num_of_iter_update_p_per_epoch << " - "
                  << sd_param->num_of_iter_update_p_per_batch
                  << " stage-epoch-batch-wise iterations";
        std::cout <<  ".\nSoft-Decision Training: begin." << std::endl;
    }

    SoftDecisionModel try_model = model;
    Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator sample(data);
    COMP_T best_obj = std::numeric_limits<COMP_T>::max();
    init_learning_rate = gd_param->init_learning_rate;
    for (unsigned int i_trial = 0; i_trial < sd_param->num_of_trials; ++i_trial) {
        if (sd_param->verbosity >= 1) {
            std::cout << "Trial " << i_trial + 1
                      << "/" << sd_param->num_of_trials << " ... \n"
                      << "Initializing ... " << std::flush;
        }
        try_model.init_var(data);
        try_model.update_p(data);
        if (sd_param->verbosity >= 1) {
            std::cout << "Done. \n    Average p = " << try_model.p_mean()
                      << ", Pos samples = " << try_model.pos_percentage()
                      << "." << std::endl;
            if (sd_param->show_p_each_iter) {
                try_model.ostream_p(std::cout, data);
                std::cout << std::endl;
            }
        }
        if (std::ostream* out = sd_param->out_stream) {
            try_model.ostream_var(*out);
            *out << std::endl;
        }
        if (sd_param->num_of_iter_update_p_per_train > 0)
            train_alter(data, try_model);
        if (sd_param->num_of_iter_update_p_per_epoch > 0)
            train_alter_epoch(data, try_model);
        if (sd_param->num_of_iter_update_p_per_batch > 0)
            train_alter_batch(data, try_model);
        COMP_T try_obj = try_model.objective(sample.begin(), data.num_of_samples());
        if (try_obj < best_obj) {
            if (sd_param->verbosity >= 1) {
                std::cout << "    Current trial " << i_trial + 1
                          << "/" << sd_param->num_of_trials
                          << " is the best so far. (Loss: Old Best = "
                          << best_obj << ", New = " << try_obj << ")"
                          << std::endl;
            }
            std::swap(model, try_model);
            best_obj = try_obj;
        } else {
            if (sd_param->verbosity >= 1) {
                std::cout << "    Current trial " << i_trial + 1
                          << "/" << sd_param->num_of_trials
                          << " is not the best. (Loss: Old Best = "
                          << best_obj << ", New = " << try_obj << ")"
                          << std::endl;
            }
        }
    }
    return 0;
}

int SoftDecisionSolver::train_alter(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data,
                                    SoftDecisionModel& model) {
    if (sd_param->verbosity >= 1) {
        std::cout << "Stage-wise training ... ";
        if (sd_param->verbosity > 1 || gd_param->verbosity > 0)
            std::cout << std::endl;
        else
            std::cout << std::flush;
    }
    unsigned int num_of_iter_p_unchange = 0;
    for (unsigned int i_train = 0;
         i_train < sd_param->num_of_iter_update_p_per_train;
         ++i_train) {
        if (sd_param->verbosity >= 2) {
            std::cout << "    Iteration (stage) " << i_train + 1
                      << "/" << sd_param->num_of_iter_update_p_per_train
                      << " ... ";
            if (sd_param->verbosity >= 3)
                std::cout << "\n        Updating {W, b} ... ";
            if (gd_param->verbosity > 0) std::cout << std::endl;
            else                         std::cout << std::flush;
        }
        SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::train(data, model);
        if (sd_param->verbosity >= 3) {
            if (gd_param->verbosity <= 0)
                std::cout << "Done." << std::endl;
            std::cout << "        Updating p ... " << std::flush;
        }
        if (model.update_p(data) == 0) num_of_iter_p_unchange++;
        else                           num_of_iter_p_unchange = 0;
        if (sd_param->verbosity >= 2) {
            if (gd_param->verbosity <= 0) std::cout << "Done." << std::endl;
            std::cout << "        Average p = " << model.p_mean()
                      << ", Pos samples = " << model.pos_percentage();
            if (gd_param->show_learning_rate_each_iter)
                std::cout << ", Learning rate = "
                          << model.learning_rate(init_learning_rate, t);
            if (gd_param->show_obj_each_iter)
                std::cout << ", Objective = " << model.objective(data);
            std::cout << "." << std::endl;
            if (sd_param->show_p_each_iter) {
                model.ostream_p(std::cout, data);
                std::cout << std::endl;
            }
        }
        if (std::ostream* out = sd_param->out_stream) {
            model.ostream_var(*out);
            *out << std::endl;
        }
        if (sd_param->num_of_iter_confirm_converge > 0 &&
            num_of_iter_p_unchange >= sd_param->num_of_iter_confirm_converge) {
            if (sd_param->verbosity >= 2) {
                std::cout << "    Nothing has changed in "
                          << num_of_iter_p_unchange << " iterations. Done."
                          << std::endl;
            }
            break;
        }
    }
    if (sd_param->verbosity == 1 && gd_param->verbosity <= 0)
        std::cout << "Done." << std::endl;
    return 0;
}

int SoftDecisionSolver::train_alter_epoch(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data,
                                          SoftDecisionModel& model) {
    if (sd_param->verbosity >= 1) {
        std::cout << "Epoch-wise training ... ";
        if (sd_param->verbosity > 1 || gd_param->verbosity > 0)
            std::cout << std::endl;
        else
            std::cout << std::flush;
    }
    reset();
    if (init_learning_rate == 0)
        if (int r = try_learning_rate(data, model)) return r;
    if (init_learning_rate < gd_param->optimal_error) {
        if (sd_param->verbosity >= 1)
            std::cout << "Epoch-wise Training: finished.\n"
                      << "    Accuracy already satisfied." << std::endl;
        return 0;
    }
    Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::permute_iterator sample(data);
    for (unsigned int i_epoch = 0;
         i_epoch < sd_param->num_of_iter_update_p_per_epoch;
         ++i_epoch) {
        if (sd_param->verbosity >= 2) {
            std::cout << "    Iteration (epoch) " << i_epoch + 1
                      << "/" << sd_param->num_of_iter_update_p_per_epoch
                      << " ... ";
            if (sd_param->verbosity >= 3)
                std::cout << "\n        Updating {W, b} ... ";
            if (gd_param->verbosity > 0) std::cout << std::endl;
            else                         std::cout << std::flush;
        }
        SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::
            train_iteration(sample.begin(), data.num_of_samples(), model);
        if (sd_param->verbosity >= 3) {
            if (gd_param->verbosity <= 0) std::cout << "Done." << std::endl;
            std::cout << "        Updating p ... " << std::flush;
        }
        model.update_p(data);
        if (sd_param->verbosity >= 2) {
            if (gd_param->verbosity <= 0) std::cout << "Done." << std::endl;
            std::cout << "        Average p = " << model.p_mean()
                      << ", Pos samples = " << model.pos_percentage();
            if (gd_param->show_learning_rate_each_iter)
                std::cout << ", Learning rate = "
                          << model.learning_rate(init_learning_rate, t);
            if (gd_param->show_obj_each_iter)
                std::cout << ", Objective = " << model.objective(data);
            std::cout << "." << std::endl;
            if (sd_param->show_p_each_iter) {
                model.ostream_p(std::cout, data);
                std::cout << std::endl;
            }
        }
        if (std::ostream* out = sd_param->out_stream) {
            model.ostream_var(*out);
            *out << std::endl;
        }
    }
    if (sd_param->verbosity == 1 && gd_param->verbosity <= 0)
        std::cout << "Done." << std::endl;
    return 0;
}

int SoftDecisionSolver::train_alter_batch(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data,
                                          SoftDecisionModel& model) {
    if (sd_param->verbosity >= 1) {
        std::cout << "Batch-wise training ... ";
        if (sd_param->verbosity > 1 || gd_param->verbosity > 0)
            std::cout << std::endl;
        else
            std::cout << std::flush;
    }
    reset();
    if (init_learning_rate == 0)
        if (int r = try_learning_rate(data, model)) return r;
    if (init_learning_rate < gd_param->optimal_error) {
        if (sd_param->verbosity >= 1)
            std::cout << "Batch-wise Training: finished.\n"
                      << "    Accuracy already satisfied." << std::endl;
        return 0;
    }
    Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::permute_iterator sample(data);
    for (unsigned int i_batch = 0;
         i_batch < sd_param->num_of_iter_update_p_per_batch;
         ++i_batch) {
        if (sd_param->verbosity >= 2)
            std::cout << "    Iteration (batch) " << i_batch + 1
                      << "/" << sd_param->num_of_iter_update_p_per_batch
                      << " ... " << std::flush;
        sample.begin();
        for (N_DAT_T i = 0;
             i < data.num_of_samples() && sample;
             i+= sgd_param->size_of_batch, ++t) {
            COMP_T learning_rate = model.learning_rate(init_learning_rate, t);
            N_DAT_T size_of_next_batch = data.num_of_samples() - i;
            if (size_of_next_batch > sgd_param->size_of_batch) {
                size_of_next_batch = sgd_param->size_of_batch;
            }
            if (int r =
                model.train_batch(sample, size_of_next_batch, learning_rate))
                return r;
            model.update_p(data);
        }
        if (sd_param->verbosity >= 2) {
            std::cout << "Done.\n        Average p = " << model.p_mean()
                      << ", Pos samples = " << model.pos_percentage();
            if (gd_param->show_learning_rate_each_iter)
                std::cout << ", Learning rate = "
                          << model.learning_rate(init_learning_rate, t);
            if (gd_param->show_obj_each_iter)
                std::cout << ", Objective = " << model.objective(data);
            std::cout << "." << std::endl;
            if (sd_param->show_p_each_iter) {
                model.ostream_p(std::cout, data);
                std::cout << std::endl;
            }
        }
        if (std::ostream* out = sd_param->out_stream) {
            model.ostream_var(*out);
            *out << std::endl;
        }
    }
    if (sd_param->verbosity == 1 && gd_param->verbosity <= 0)
        std::cout << "Done." << std::endl;
    return 0;
}


bpy::tuple SoftDecisionModel_ModelParam_bpy_pickle::
getstate(const SoftDecisionModel::ModelParam& param) {
    return bpy::make_tuple(param.reg_coeff,
                           param.reg_l1_ratio,
                           param.b_learning_rate_factor,
                           param.init_var_subsample_rate,
                           param.init_var_subsample_min);
}

void SoftDecisionModel_ModelParam_bpy_pickle::
setstate(SoftDecisionModel::ModelParam& param, bpy::tuple& state) {
    if (bpy::len(state) != 5) {
        char err_msg[SIZEOF_LINE];
        snprintf(err_msg, SIZEOF_LINE, "Expected 5-item tuple in call to " \
                 "ModelParam.__setstate__; got %zd items",
                 bpy::len(state));
        PyErr_SetString(PyExc_ValueError, err_msg);
        bpy::throw_error_already_set();
    }
    param.reg_coeff               = bpy::extract<COMP_T>(state[0]);
    param.reg_l1_ratio            = bpy::extract<COMP_T>(state[1]);
    param.b_learning_rate_factor  = bpy::extract<COMP_T>(state[2]);
    param.init_var_subsample_rate = bpy::extract<COMP_T>(state[3]);
    param.init_var_subsample_min  = bpy::extract<COMP_T>(state[4]);
}

bpy::tuple SoftDecisionModel_bpy_pickle::
getinitargs(const SoftDecisionModel& model) {
    return bpy::make_tuple(model.model_param());
}

bpy::tuple SoftDecisionModel_bpy_pickle::
getstate(const SoftDecisionModel& model) {
    return bpy::make_tuple(
               model.b,
               c_array_to_numpy_1d_array(model.w_i, model.n_w_i, false),
               c_array_to_numpy_1d_array(model.w, model.dimension, false),
               c_array_to_numpy_1d_array(model.p, model.num_of_labels, false),
               c_array_to_numpy_1d_array(model.index_of_pos_sample,
                                         model.num_of_pos_sample, false),
               c_array_to_numpy_1d_array(model.index_of_neg_sample,
                                         model.num_of_neg_sample, false),
               model.num_of_samples);
}

void SoftDecisionModel_bpy_pickle::
setstate(SoftDecisionModel& model, bpy::tuple& state) {
    if (bpy::len(state) != 7) {
        char err_msg[SIZEOF_LINE];
        snprintf(err_msg, SIZEOF_LINE, "Expected 7-item tuple in call to " \
                 "SoftDecisionModel.__setstate__; got %zd items",
                 bpy::len(state));
        PyErr_SetString(PyExc_ValueError, err_msg);
        bpy::throw_error_already_set();
    }
    model.b = bpy::extract<COMP_T>(state[0]);
    numpy_1d_array_to_c_array(state[1], model.w_i, model.n_w_i);
    numpy_1d_array_to_c_array(state[2], model.w, model.dimension);
    numpy_1d_array_to_c_array(state[3], model.p, model.num_of_labels);
    numpy_1d_array_to_c_array(state[4], model.index_of_pos_sample,
                                        model.num_of_pos_sample);
    numpy_1d_array_to_c_array(state[5], model.index_of_neg_sample,
                                        model.num_of_neg_sample);
    model.num_of_samples = bpy::extract<N_DAT_T>(state[6]);
}

bpy::tuple SoftDecisionSolver_SDParam_bpy_pickle::
getstate(const SoftDecisionSolver::SDParam& param) {
    return bpy::make_tuple(param.verbosity,
                           param.show_p_each_iter,
                           param.num_of_trials,
                           param.num_of_iter_update_p_per_train,
                           param.num_of_iter_update_p_per_epoch,
                           param.num_of_iter_update_p_per_batch,
                           param.num_of_iter_confirm_converge);
}

void SoftDecisionSolver_SDParam_bpy_pickle::
setstate(SoftDecisionSolver::SDParam& param, bpy::tuple& state) {
    if (bpy::len(state) != 7) {
        char err_msg[SIZEOF_LINE];
        snprintf(err_msg, SIZEOF_LINE, "Expected 7-item tuple in call to " \
                 "SDParam.__setstate__; got %zd items", bpy::len(state));
        PyErr_SetString(PyExc_ValueError, err_msg);
        bpy::throw_error_already_set();
    }
    param.verbosity                      = bpy::extract<char>(state[0]);
    param.show_p_each_iter               = bpy::extract<bool>(state[1]);
    param.num_of_trials                  = bpy::extract<unsigned int>(state[2]);
    param.num_of_iter_update_p_per_train = bpy::extract<unsigned int>(state[3]);
    param.num_of_iter_update_p_per_epoch = bpy::extract<unsigned int>(state[4]);
    param.num_of_iter_update_p_per_batch = bpy::extract<unsigned int>(state[5]);
    param.num_of_iter_confirm_converge   = bpy::extract<unsigned int>(state[6]);
}
