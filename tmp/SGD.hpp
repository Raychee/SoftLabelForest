# ifndef _SGD_H
# define _SGD_H

# include <iostream>
# include <cstring>
# include <limits>
# include <random>
# include <chrono>

# include "my_lib.hpp"
# include "LabelStat.hpp"

/// General Gradient Descent solver model
///
/// @tparam _COMP_T    type of the value to be computed (parameters, data samples, etc)
/// @tparam _SUPV_T    type of the supervising information (classes, labels, etc)
/// @tparam _DAT_DIM_T type of the dimension of the data
/// @tparam _N_DAT_T   type of the number of the data set
template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
class GD {
public:
    class GDParam {
    public:
        GDParam(_DAT_DIM_T   _dimension            = 0,
                char         _verbosity            = 1,
                _COMP_T      _eta0                 = 0,
                unsigned int _n_iter               = 200,
                _COMP_T      _err                  = 1e-8,
                _N_DAT_T     _min_n_subsample      = 0,
                float        _eta0_try_sample_rate = 0.3,
                _COMP_T      _eta0_try_1st         = 0.1,
                _COMP_T      _eta0_try_factor      = 3,
                bool         _show_obj_each_iter   = false);
        ~GDParam() {}

        // some interfaces to write private variables in this class
        GDParam& verbosity(char _verbosity)
            { v = _verbosity; return *this; }
        GDParam& dimension(_DAT_DIM_T _dimension) {
            if (d) {
                std::cerr << "WARNING: SGD: "
                          << "Dimensionality has already been specified. "
                          << "Nothing is changed." << std::endl;
            }
            else d = _dimension;
            return * this;
        }
        GDParam& init_learning_rate(const _COMP_T _eta0)                             { eta0 = _eta0;                                 return *this; }
        GDParam& num_of_iterations(const unsigned int _n_iter)                       { n_iter = _n_iter;                             return *this; }
        GDParam& accuracy(const _COMP_T _err)                                        { err = _err;                                   return *this; }
        GDParam& min_num_of_subsamples(const _N_DAT_T _min_n_subsample)              { min_n_subsample = _min_n_subsample;           return *this; }
        GDParam& learning_rate_sample_rate(const float _eta0_try_sample_rate)        { eta0_try_sample_rate = _eta0_try_sample_rate; return *this; }
        GDParam& learning_rate_1st_try(const _COMP_T _eta0_try_1st)                  { eta0_try_1st = _eta0_try_1st;                 return *this; }
        GDParam& learning_rate_try_factor(const _COMP_T _eta0_try_factor)            { eta0_try_factor = _eta0_try_factor;           return *this; }
        GDParam& show_obj_each_iteration(const bool _compute_obj)                    { show_obj_each_iter = _compute_obj;            return *this; }
        GDParam& ostream_of_training_process(const std::ostream& _out_training_proc) { out_training_proc = &_out_training_proc;      return *this; }

        // some interfaces to read private variables in this class
        char          verbosity()                   const { return v; }
        _DAT_DIM_T    dimension()                   const { return d; }
        _COMP_T       init_learning_rate()          const { return eta0; }
        unsigned int  num_of_iterations()           const { return n_iter; }
        _COMP_T       accuracy()                    const { return err; }
        _N_DAT_T      min_num_of_subsamples()       const { return min_n_subsample; }
        float         learning_rate_sample_rate()   const { return eta0_try_sample_rate; }
        _COMP_T       learning_rate_1st_try()       const { return eta0_try_1st; }
        _COMP_T       learning_rate_try_factor()    const { return eta0_try_factor; }
        bool          show_obj_each_iteration()     const { return show_obj_each_iter; }
        std::ostream* ostream_of_training_process() const { return out_training_proc; }

        GDParam&      ostream_this(std::ostream& out);

    private:
        _DAT_DIM_T   d;             ///< dimensionality
        char         v;             ///< verbosity
        _COMP_T      eta0;          ///< the initial learning rate
        /// Stopping criteria 1: max number of iterations
        unsigned int n_iter;
        /// Stopping criteria 2: residual error upper bound
        _COMP_T      err;
        /// When GD does subsampling, the minimal number of samples should be chosen
        _N_DAT_T     min_n_subsample;
        /// the ratio of sampling from the data set for determining eta0
        float        eta0_try_sample_rate;
        /// the first guess of eta0 (if eta0 is not specified by user)
        _COMP_T      eta0_try_1st;
        /// the factor that eta0 multiplies for each try
        _COMP_T      eta0_try_factor;
        /// whether to compute the objective value after each iteration
        bool         show_obj_each_iter;
        // pointer to the ostream which is used for the output of parameters.
        // If NULL, then nothing will be output
        std::ostream* out_training_proc;
    };

    typedef LabelStat<_SUPV_T, _N_DAT_T> LABELS;

    GD();
    GD(GDParam& _gd_param);
    virtual ~GD() {}
    // The assignment operator only assigns the content that is defined in
    // "assign_to_this()", which is often the data in the derived class.
    // So be careful when using.
    // GD& operator=(GD& some);

    GD& set_param(GDParam& _gd_param) { gd_param = &_gd_param; return *this; }

    /// Feed the training data into the solver.
    ///
    /// @param[in] _X    The matrix that holds all the training data.
    /// @param[in] _Y    The array containing labels of each sample.
    /// @param[in] _N    Number of samples in the matrix.
    /// @param[in] _x    If only a subset of data is used as training data, _x is
    ///                  an index array of length _n that holds all the indexes
    ///                  of training samples. _x = NULL if all the data are for
    ///                  training.
    /// @param[in] _n    Number of training samples. Ignored if x == NULL.
    GD& training_data(_COMP_T* _X, _SUPV_T* _Y, _N_DAT_T _N,
                      _N_DAT_T* _x = NULL, _N_DAT_T _n = 0);

    /// Feed the training data into the solver.
    ///
    /// This function call is proper only when X, Y, and N have been
    /// indicated.
    GD& training_data(_N_DAT_T* _x, _N_DAT_T _n);

    /// Feed the training data into the solver.
    ///
    /// Only used when data cannot fit into memory. NOTICE: The label information
    /// is always loaded ENTIRELY into the memory, which means the indexes like x
    /// are global, not subject to the local buffer.
    /// @param[in] _Xfile   The file name of the training data.
    /// @param[in] _Yfile   The file name of the training data's labels.
    /// @param[in] _N       The number of samples that the buffer can hold.
    /// @param[in] _x       If only a subset of data is used as training data, _x is
    ///                     an index array of length _n that holds all the GLOBAL
    ///                     indexes of training samples. _x = NULL if all the data
    ///                     are for training.
    /// @param[in] _n       Number of training samples. Ignored if x == NULL.
    GD& training_data(const char* _Xfile, const char* _Yfile, _N_DAT_T _N,
                      _N_DAT_T* _x = NULL, _N_DAT_T _n = 0);

    /// Train the parameters with the explicit training data.
    GD& train(_COMP_T* _X, _SUPV_T* _Y, _N_DAT_T _N,
              _N_DAT_T* _x = NULL, _N_DAT_T _n = 0);

    /// Train the parameters with the explicit training data.
    GD& train(_N_DAT_T* _x, _N_DAT_T _n);

    /// Train the parameters with the data fed beforehand.
    virtual GD& train();

    /// Test all the N input data or _n specified by _x and output the result in
    /// the corresponding position of the array Y.
    GD& test(_COMP_T* _X, _SUPV_T* _Y, _N_DAT_T _N,
             _N_DAT_T* _x = NULL, _N_DAT_T _n = 0);

    /// Generate a permutation in the first m elements of array of length n
    GD& rand_index(_N_DAT_T* x, _N_DAT_T n, _N_DAT_T m = 0);

    /// Test one data sample.
    virtual _SUPV_T test_one(_COMP_T* Xi, _DAT_DIM_T D) const = 0;

    /// Output the SGD solver to a standard ostream.
    virtual GD& istream_this(std::istream& in)    { return *this; }
    virtual GD& ostream_this(std::ostream& out);
    virtual GD& ofstream_this(std::ofstream& out) { return ostream_this(out); }

    /// Output all the training parameters to a standard ostream.
    virtual GD& ostream_param(std::ostream& out) = 0;

protected:
    std::ifstream Xfile;    ///< the file containing training data
    std::ifstream Yfile;    ///< the file containing training data's labels

    _COMP_T*      X;        ///< data buffer
    _SUPV_T*      Y;        ///< labels
    _N_DAT_T      N;        ///< total number of samples in the buffer
    LABELS        L;        ///< All the label's statistical information about the training data in the buffer

    GDParam*      gd_param; ///< gradient descent parameters (external)
    // temporary parameters during learning
    unsigned long t;        ///< current pass (number of data samples passed)
    _COMP_T       eta;      ///< current learning rate

    virtual _COMP_T compute_learning_rate(_COMP_T eta0, unsigned long t) = 0;

    /// Determine the initial learning rate eta0 automatically according to the
    /// given data set.
    virtual GD& try_learning_rate();

    /// Train with data indicated by "_x" which refers to the indexes of samples
    /// in "_data".
    virtual GD& train_batch(_COMP_T*   X,
                            _SUPV_T*   Y,
                            _DAT_DIM_T D,
                            _N_DAT_T*  x,
                            _N_DAT_T   n,
                            _COMP_T    eta) = 0;

    /// Compute the object function (e.g. sum of empirical losses plus a
    /// regularizing term) given the current parameters and data.
    _COMP_T         compute_obj();
    virtual _COMP_T compute_obj(_COMP_T*   X,
                                _SUPV_T*   Y,
                                _DAT_DIM_T D,
                                _N_DAT_T*  x,
                                _N_DAT_T   n) = 0;

    /// Create and return a temporary duplicate of the current GD solver. Used
    /// in try_learning_rate().
    virtual GD* duplicate_this() = 0;
    /// Assign the training parameters of some GD solver to this. Used in
    /// try_learning_rate(). Tips: Remember to cast void* into derived class
    /// type of pointer in the implementation. Copy only the variables that
    /// would be modified in train_batch().
    virtual GD& assign_to_this(void* some) = 0;
};


/// General Stochastic Gradient Descent solver model
///
/// @tparam _COMP_T    type of the value to be computed (parameters, data samples, etc)
/// @tparam _SUPV_T    type of the supervising information (classes, labels, etc)
/// @tparam _DAT_DIM_T type of the dimension of the data
/// @tparam _N_DAT_T   type of the number of the data set
///
template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
class SGD : public GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T> {
public:
    /// Parameter structure for a general Stochastic Gradient Descent solver model
    class SGDParam {
    public:
        SGDParam(_N_DAT_T     _s_batch = 1,
                 unsigned int _n_epoch = 200);
        ~SGDParam() {}

        // some interfaces to write private variables in this class
        SGDParam& size_of_batches(_N_DAT_T _s_batch)    { s_batch = _s_batch; return *this; }
        SGDParam& num_of_epoches(unsigned int _n_epoch) { n_epoch = _n_epoch; return *this; }

        // some interfaces to read private variables in this class
        _N_DAT_T     size_of_batches() const { return s_batch; }
        unsigned int num_of_epoches()  const { return n_epoch; }

        SGDParam&    ostream_this(std::ostream& out);

    private:
        _N_DAT_T     s_batch;       ///< size of mini-batches. 1 is completely SGD.
        unsigned int n_epoch;       ///< number of epoches
    };//class SGDParam

    SGD(): sgd_param(NULL) {}
    SGD(typename GD::GDParam& _gd_param,
        SGDParam&             _sgd_param);
    virtual ~SGD() {};

    using GD::set_param;
    SGD& set_param(SGDParam& _sgd_param);
    SGD& set_param(typename GD::GDParam& _gd_param,
                   SGDParam&             _sgd_param);

    /// Train the parameters.
    ///
    /// The order of the data samples are automatically shuffled before each
    /// epoch. Every parameter is applied with update rule once for each input
    /// data sample.
    virtual SGD& train();

    /// Output the SGD solver to a standard ostream.
    virtual SGD& ostream_this(std::ostream& out);

protected:
    SGDParam*    sgd_param;        ///< sgd parameters (external)

    virtual SGD& try_learning_rate();
    /// Train the parameters "param" with "m" data samples in "dat" whose indexes
    /// are specified by "dat_idx".
    ///
    /// The training precedure follows exactly the order of indexes in "dat_idx".
    /// The data pass only once (1-epoch). Every parameter is applied with
    /// update rule once for each input data sample. "y" should be the WHOLE
    /// supervising data ("n", same length as "dat"), not just length of "m".
    virtual SGD& train_epoch(_DAT_DIM_T _d, _N_DAT_T* _x,
                             _N_DAT_T _s_batch,
                             _N_DAT_T _n_batch,
                             _N_DAT_T _n_remain,
                             char _v);
    virtual SGD* duplicate_this() = 0;
    virtual SGD& assign_to_this(void* some) = 0;
private:
};

/// A wrapper of function ostream_this(): operator << overload for convenience.
template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline std::ostream& operator<<(std::ostream& out,
                                GD<_COMP_T,
                                   _SUPV_T,
                                   _DAT_DIM_T,
                                   _N_DAT_T>& gd) {
    gd.ostream_this(out);
    return out;
}
template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline std::ofstream& operator<<(std::ofstream& out,
                                GD<_COMP_T,
                                   _SUPV_T,
                                   _DAT_DIM_T,
                                   _N_DAT_T>& gd) {
    gd.ofstream_this(out);
    return out;
}
template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline std::istream& operator>>(std::istream& in,
                                GD<_COMP_T,
                                   _SUPV_T,
                                   _DAT_DIM_T,
                                   _N_DAT_T>& gd) {
    gd.istream_this(in);
    return in;
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
GDParam::GDParam(_DAT_DIM_T   _dimension,
                 char         _verbosity,
                 _COMP_T      _eta0,
                 unsigned int _n_iter,
                 _COMP_T      _err,
                 _N_DAT_T     _min_n_subsample,
                 float        _eta0_try_sample_rate,
                 _COMP_T      _eta0_try_1st,
                 _COMP_T      _eta0_try_factor,
                 bool         _show_obj_each_iter):
         d(_dimension),
         v(_verbosity),
         eta0(_eta0),
         n_iter(_n_iter),
         err(_err),
         min_n_subsample(_min_n_subsample),
         eta0_try_sample_rate(_eta0_try_sample_rate),
         eta0_try_1st(_eta0_try_1st),
         eta0_try_factor(_eta0_try_factor),
         show_obj_each_iter(_show_obj_each_iter),
         out_training_proc(NULL) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline typename GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam::
ostream_this(std::ostream& out) {
    out << "Gradient Descent parameters:\n"
        << "    Verbosity = " << (int)v << "\n"
        << "    Dimension = " << d << "\n"
        << "    Max number of iterations = " << n_iter << "\n"
        << "    Stopping accuracy (error tolerance) = " << err << "\n"
        << "    Minimal number of samples when subsampling = " << min_n_subsample << "\n";
    if (eta0)
        out << "    Initial learning rate = " << eta0;
    else {
        out << "    Initial learning rate not specified. will be determined by guessing:\n"
            << "        Subsampling ratio = " << eta0_try_sample_rate * 100 << "%%\n"
            << "        First guess = " << eta0_try_1st << "\n"
            << "        Factor between guesses = " << eta0_try_factor << "\n";
    }
    out << "    Show objective value for each iteration = " << show_obj_each_iter << "\n"
        << "    Output the training process = " << (bool)out_training_proc;
    return *this;
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
GD(): X(NULL),
      Y(NULL),
      N(0),
      gd_param(NULL),
      t(0),
      eta(0) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
GD(GDParam& _gd_param)
  :X(NULL),
   Y(NULL),
   N(0),
   gd_param(&_gd_param),
   t(0) {
    eta = gd_param->init_learning_rate();
}


// template <typename _COMP_T,
//           typename _SUPV_T,
//           typename _DAT_DIM_T,
//           typename _N_DAT_T> inline
// GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
// GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
// operator=(GD& some) {
//     return assign_to_this(&some);
// }


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
training_data(_COMP_T* _X, _SUPV_T* _Y, _N_DAT_T _N,
              _N_DAT_T* _x = NULL, _N_DAT_T _n = 0) {
    X = _X;
    Y = _Y;
    N = _N;
    if (_x && _n) stat.stat(Y, _n, _x);
    else stat.stat(Y, N);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
training_data(_N_DAT_T* _x, _N_DAT_T _n) {
    stat.stat(Y, _n, _x);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
training_data(const char* _Xfile, const char* _Yfile, _N_DAT_T _N,
              _N_DAT_T* _x = NULL, _N_DAT_T _n = 0) {
    stat.stat(Y, _n, _x);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y, _N_DAT_T* _x, _N_DAT_T _s_x) {
    training_data(_data, _n, _y, _x, _s_x);
    return train();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train(_N_DAT_T* _x, _N_DAT_T _s_x) {
    training_data(_x, _s_x);
    return train();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train() {
    if (!gd_param) {
        std::cerr << "WARNING: GD: Parameters have not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    _DAT_DIM_T    dim                = gd_param->dimension();
    char          verbosity          = gd_param->verbosity();
    unsigned int  n_iter             = gd_param->num_of_iterations();
    _COMP_T       err                = gd_param->accuracy();
    bool          show_obj_each_iter = gd_param->show_obj_each_iteration();
    std::ostream* out                = gd_param->ostream_of_training_process();
    _N_DAT_T      n_sample           = stat.num_of_samples();
    _N_DAT_T*     x                  = stat.index_of_samples();
    if (!dim) {
        std::cerr << "WARNING: GD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    if (verbosity >= 1) {
        std::cout << "GD Training: \n    Data: " << n_sample << " samples, "
                  << dim << " feature dimensions.\n    Stopping Criterion: "
                  << n_iter << " iterations";
        if (err > 0) std::cout << " or accuracy higher than " << err;
        std::cout <<  ".\nGD Training: begin." << std::endl;
    }
    if (!gd_param->init_learning_rate()) try_learning_rate();
    _COMP_T eta0 = gd_param->init_learning_rate();
    if (eta0 < err) {
        if (verbosity >= 1) {
            std::cout << "GD Training: finished.\n"
                      << "    Accuracy already satisfied." << std::endl;
        }
        return *this;
    }
    _COMP_T obj0 = std::numeric_limits<_COMP_T>::max();
    _COMP_T obj1 = 0;
    if (verbosity >= 1) {
        std::cout << "Training ... ";
        if (verbosity > 1) std::cout << std::endl;
        else std::cout.flush();
    }
    if (out) {
        ostream_param(*out);
        *out << " ";
        *out << obj1 << std::endl;
    }
    unsigned int i;
    for (i = 0; i < n_iter; ++i, ++t) {
        if (verbosity >= 2) {
            std::cout << "    Iteration " << i+1 << " ... "
                      << std::flush;
        }
        eta = compute_learning_rate(eta0, t);
        train_batch(data, dim, x, n_sample, y, eta);
        obj1 = compute_obj(data, dim, x, n_sample, y);
        if (verbosity >= 2) {
            std::cout << "Done. eta = " << eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = " << obj1 << ".";
            }
            std::cout << "." << std::endl;
        }
        // DEBUG
        if (out) {
            ostream_param(*out);
            *out << " ";
            *out << obj1 << std::endl;
        }
        if (err > 0 && obj0 - obj1 < err) break;
        obj0 = obj1;
    }
    if (verbosity >= 1) {
        if (verbosity == 1) std::cout << "Done.\n";
        std::cout << "GD Training: finished. \n";
        if (i < n_iter)
            std::cout << "    Training stopped at iteration " << t + 1
                      << " with convergence.";
        else
            std::cout << "    Max number of iterations has been reached.";
        std::cout << std::endl;
    }
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
test(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y, _N_DAT_T* _x, _N_DAT_T _s_x) {
    _DAT_DIM_T dim       = gd_param->dimension();
    char       verbosity = gd_param->verbosity();
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    if (verbosity >= 1) {
        std::cout << "GD Testing: \n  Data: " << _n << " samples.\nTesting ... ";
        if (verbosity > 1) std::cout << std::endl;
        else std::cout.flush();
    }
    if (_x && _s_x) {
        for (_N_DAT_T i = 0; i < _s_x; ++i) {
            _COMP_T* dat_i = _data + dim * _x[i];
            if (verbosity >= 2)
                std::cout << "  Sample " << i << " ... " << std::flush;
            _y[i] = test_one(dat_i, dim);
            if (verbosity >= 2)
                std::cout << "Done. Predicted label: " << _y[i] << "."
                          << std::endl;
        }
    }
    else {
        _COMP_T* dat_i = _data;
        for (_N_DAT_T i = 0; i < _n; ++i, dat_i += dim) {
            if (verbosity >= 2)
                std::cout << "  Sample " << i << " ... " << std::flush;
            _y[i] = test_one(dat_i, dim);
            if (verbosity >= 2)
                std::cout << "Done. Predicted label: " << _y[i] << "."
                          << std::endl;
        }
    }
    if (verbosity == 1) std::cout << "Done.\n";
    if (verbosity >= 1) std::cout << "GD Testing: finished." << std::endl;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
rand_index(_N_DAT_T* x, _N_DAT_T n, _N_DAT_T m) {
    static std::mt19937_64 rand_gen(std::chrono::system_clock::now().time_since_epoch().count());
    if (!m || m > n) m = n;
    for (_N_DAT_T i = 0; i < m; ++i) {
        std::uniform_int_distribution<_N_DAT_T> distrib(i, n - 1);
        _N_DAT_T temp_i = distrib(rand_gen);
        _N_DAT_T temp   = x[temp_i];
        x[temp_i]       = x[i];
        x[i]            = temp;
    }
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
ostream_this(std::ostream& out) {
    stat.ostream_this(out);
    out << "\n";
    gd_param->ostream_this(out);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
try_learning_rate() {
    char       verbosity = gd_param->verbosity();
    _DAT_DIM_T dim       = gd_param->dimension();
    _COMP_T    err       = gd_param->accuracy();
    _N_DAT_T   n_sample  = stat.num_of_samples();
    _N_DAT_T*  sub_x_i   = stat.index_of_samples();
    _COMP_T    eta0_try1, eta0_try2, obj_try1, obj_try2, eta0_try_factor;
    GD*        GD_try1 = NULL, * GD_try2 = NULL, * GD_temp = NULL;
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    if (verbosity >= 1) {
        std::cout << "Initializing ... ";
        if (verbosity >= 2) {
            std::cout << "\n    No initial learning rate \"eta0\" specified. "
                      << "Deciding automatically ... ";
            if (verbosity > 2) std::cout << std::endl;
            else std::cout.flush();
        }
        std::cout.flush();
    }

    _N_DAT_T n_subsample = gd_param->learning_rate_sample_rate() * n_sample;
    if (n_subsample < gd_param->min_num_of_subsamples())
        n_subsample = gd_param->min_num_of_subsamples();
    if (n_subsample > n_sample) n_subsample = n_sample;

    rand_index(sub_x_i, n_sample, n_subsample);

    LabelStat<_SUPV_T, _N_DAT_T> stat_new(y, n_subsample, sub_x_i);

    eta0_try1 = gd_param->learning_rate_1st_try();
    if (eta0_try1 < err) {
        gd_param->init_learning_rate(eta0_try1);
        t = 1;
        return *this;
    }
    if (verbosity >= 3)
        std::cout << "        Trying eta0 = " << eta0_try1 << " ... " << std::flush;
    GD_try1       = duplicate_this();
    GD_try1->stat = stat_new;
    GD_try1->train_batch(data, dim, sub_x_i, n_subsample, y, eta0_try1);
    obj_try1 = GD_try1->compute_obj(data, dim, sub_x_i, n_subsample, y);
    if (verbosity >= 3)
        std::cout << "Done. Obj = " << obj_try1 << "." << std::endl;
    eta0_try_factor = gd_param->learning_rate_try_factor();
    eta0_try2 = eta0_try1 * eta0_try_factor;
    if (eta0_try2 < err) {
        delete GD_try1;
        gd_param->init_learning_rate(eta0_try2);
        t = 1;
        return *this;
    }
    if (verbosity >= 3)
        std::cout << "        Trying eta0 = " << eta0_try2 << "... " << std::flush;
    GD_try2       = duplicate_this();
    GD_try2->stat = stat_new;
    GD_try2->train_batch(data, dim, sub_x_i, n_subsample, y, eta0_try2);
    obj_try2 = GD_try2->compute_obj(data, dim, sub_x_i, n_subsample, y);
    if (verbosity >= 3)
        std::cout << "Done. Obj = " << obj_try2 << "." << std::endl;
    if (obj_try1 < obj_try2) {
        eta0_try_factor = 1 / eta0_try_factor;
        obj_try2        = obj_try1;
        eta0_try2       = eta0_try1;
        GD_temp         = GD_try2;
        GD_try2         = GD_try1;
        GD_try1         = GD_temp;
    }
    do {
        eta0_try1 = eta0_try2;
        obj_try1  = obj_try2;
        GD_temp   = GD_try1;
        GD_try1   = GD_try2;
        GD_try2   = GD_temp;
        eta0_try2 = eta0_try1 * eta0_try_factor;
        if (eta0_try2 < err) break;
        if (verbosity >= 3)
            std::cout << "        Trying eta0 = " << eta0_try2 << "... " << std::flush;
        GD_try2->assign_to_this(this);
        GD_try2->train_batch(data, dim, sub_x_i, n_subsample, y, eta0_try2);
        obj_try2 = GD_try2->compute_obj(data, dim, sub_x_i, n_subsample, y);
        if (verbosity >= 3)
            std::cout << "Done. Obj = " << obj_try2 << "." << std::endl;
    } while (obj_try1 >= obj_try2);
    delete GD_try2;
    if (eta0_try_factor > 1) eta0_try_factor = 1 / eta0_try_factor;
    gd_param->init_learning_rate(eta0_try1 * eta0_try_factor);
    t = 1;
    assign_to_this(GD_try1);
    delete GD_try1;
    if (verbosity == 1) std::cout << "Done." << std::endl;
    if (verbosity >= 2) {
        if (verbosity == 2) std::cout << "Done.\n";
        std::cout << "    Setting eta0 = " << eta0_try1 * eta0_try_factor << "." << std::endl;
    }
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline
_COMP_T
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
compute_obj() {
    return compute_obj(data,
                       gd_param->dimension(),
                       stat.index_of_samples(),
                       n,
                       y);
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGDParam::SGDParam(_N_DAT_T     _s_batch,
                   unsigned int _n_epoch):
          s_batch(_s_batch),
          n_epoch(_n_epoch) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T> inline typename
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::SGDParam&
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::SGDParam::
ostream_this(std::ostream& out) {
    out << "Stochastic Gradient Descent parameters:\n"
        << "    Size of each batch = " << s_batch << "\n"
        << "    Max number of epoches = " << n_epoch;
    return *this;
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGD(typename GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& _gd_param,
    SGDParam& _sgd_param)
   :GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>(_gd_param),
    sgd_param(&_sgd_param) {
}

// template <typename _COMP_T,
//           typename _SUPV_T,
//           typename _DAT_DIM_T,
//           typename _N_DAT_T>
// SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
// SGD(SGD& some)
//    :GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>(some),
//     sgd_param(some.sgd_param) {
// }

// template <typename _COMP_T,
//           typename _SUPV_T,
//           typename _DAT_DIM_T,
//           typename _N_DAT_T>
// SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
// SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
// operator=(SGD& some) {
//     GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::operator=(some);
//     if (alloc_sgd_param && sgd_param != some.sgd_param) {
//         delete sgd_param;
//         alloc_sgd_param = false;
//     }
//     sgd_param = some.sgd_param;
//     return *this;
// }

// template <typename _COMP_T,
//           typename _SUPV_T,
//           typename _DAT_DIM_T,
//           typename _N_DAT_T>
// SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
// ~SGD() {
//     if (alloc_sgd_param) delete sgd_param;
// }

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
set_param(SGDParam& _sgd_param) {
    sgd_param = &_sgd_param;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
set_param(typename GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& _gd_param,
          SGDParam& _sgd_param) {
    set_param(_gd_param);
    sgd_param = &_sgd_param;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train() {
    if (!this->gd_param || !sgd_param) {
        std::cerr << "WARNING: SGD: Parameters have not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    _DAT_DIM_T    dim                = this->gd_param->dimension();
    char          verbosity          = this->gd_param->verbosity();
    _COMP_T       err                = this->gd_param->accuracy();
    bool          show_obj_each_iter = this->gd_param->show_obj_each_iteration();
    std::ostream* out                = this->gd_param->ostream_of_training_process();
    _N_DAT_T      s_batch            = sgd_param->size_of_batches();
    unsigned int  n_epoch            = sgd_param->num_of_epoches();
    _N_DAT_T      n_sample           = this->stat.num_of_samples();
    _N_DAT_T*     rand_x             = this->stat.index_of_samples();
    _N_DAT_T      n_batch            = n_sample / s_batch;
    _N_DAT_T      n_remain           = n_sample - s_batch * n_batch;
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    if (verbosity >= 1) {
        std::cout << "SGD Training: \n    Data: " << n_sample << " samples, "
                  << dim << " feature dimensions. " << s_batch
                  << " samples per batch.\n    Stopping Criterion: "
                  << n_epoch << " epoches";
        if (err > 0) std::cout << " or accuracy higher than " << err;
        std::cout <<  ".\nSGD Training: begin." << std::endl;
    }
    if (!this->gd_param->init_learning_rate()) this->try_learning_rate();
    if (this->gd_param->init_learning_rate() < err) {
        if (verbosity >= 1) {
            std::cout << "SGD Training: finished.\n"
                      << "    Accuracy already satisfied." << std::endl;
        }
        return *this;
    }
    // _COMP_T obj0 = std::numeric_limits<_COMP_T>::max();
    // _COMP_T obj1;
    unsigned int i;
    if (verbosity >= 1) {
        std::cout << "Training ... ";
        if (verbosity > 1) std::cout << std::endl;
        else std::cout.flush();
    }
    if (out) {
        this->ostream_param(*out);
        *out << " ";
        *out << /*obj1 << */std::endl;
    }
    for (i = 0; i < n_epoch; ++i) {
        if (verbosity >= 3)
            std::cout << "    Shuffling the data set... " << std::flush;
        this->rand_index(rand_x, n_sample);
        if (verbosity >= 3) std::cout << "Done." << std::endl;
        if (verbosity >= 2) {
            std::cout << "    Epoch " << i + 1  << " ... ";
            if (verbosity >= 3) std::cout << std::endl;
            else std::cout.flush();
        }
        train_epoch(dim, rand_x, s_batch, n_batch, n_remain, verbosity);
        // obj1 = this->compute_obj(this->data, dim, rand_x, n_sample, this->y);
        if (verbosity == 2) {
            std::cout << "Done. eta = " << this->eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = "
                          << this->compute_obj(this->data,
                                               dim,
                                               rand_x,
                                               n_sample,
                                               this->y);
            }
            std::cout << "." << std::endl;
        }
        // DEBUG
        if (out) {
            this->ostream_param(*out);
            *out << " ";
            *out << /*obj1 << */std::endl;
        }
        // if (err > 0 && obj0 - obj1 < err) break;
        // obj0 = obj1;
    }

    if (verbosity >= 1) {
        if (verbosity == 1) std::cout << "Done.\n";
        std::cout << "SGD Training: finished.\n";
        if (i < n_epoch)
            std::cout << "    Training stopped at epoch " << i + 1
                      << " with convergence.";
        else
            std::cout << "    Max number of epoches has been reached.";
        std::cout << std::endl;
    }
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
ostream_this(std::ostream& out) {
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::ostream_this(out);
    out << "\n";
    sgd_param->ostream_this(out);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
try_learning_rate() {
    char       verbosity = this->gd_param->verbosity();
    _DAT_DIM_T dim       = this->gd_param->dimension();
    _COMP_T    err       = this->gd_param->accuracy();
    _N_DAT_T   s_batch   = sgd_param->size_of_batches();
    _N_DAT_T   n_sample  = this->stat.num_of_samples();
    _N_DAT_T*  sub_x_i   = this->stat.index_of_samples();
    _COMP_T    eta0_try1, eta0_try2, obj_try1, obj_try2, eta0_try_factor;
    SGD*       SGD_try1 = NULL, * SGD_try2 = NULL, * SGD_temp = NULL;
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    if (verbosity >= 1) {
        std::cout << "Initializing ... ";
        if (verbosity >= 2) {
            std::cout << "\n    No initial learning rate \"eta0\" specified. "
                      << "Deciding automatically ... ";
            if (verbosity > 2) std::cout << std::endl;
            else std::cout.flush();
        }
        std::cout.flush();
    }

    _N_DAT_T n_subsample = this->gd_param->learning_rate_sample_rate() * n_sample;
    if (n_subsample < this->gd_param->min_num_of_subsamples())
        n_subsample = this->gd_param->min_num_of_subsamples();
    if (n_subsample > n_sample) n_subsample = n_sample;
    _N_DAT_T n_batch  = n_subsample / s_batch;
    _N_DAT_T n_remain = n_subsample - s_batch * n_batch;

    this->rand_index(sub_x_i, n_sample, n_subsample);

    LabelStat<_SUPV_T, _N_DAT_T> stat_new(this->y, n_subsample, sub_x_i);

    this->t = 0;
    eta0_try1 = this->gd_param->learning_rate_1st_try();
    if (eta0_try1 < err) {
        this->gd_param->init_learning_rate(eta0_try1);
        this->t = 1;
        return *this;
    }
    if (verbosity >= 3)
        std::cout << "        Trying eta0 = " << eta0_try1 << " ... " << std::flush;
    SGD_try1       = duplicate_this();
    SGD_try1->stat = stat_new;
    this->gd_param->init_learning_rate(eta0_try1);
    SGD_try1->train_epoch(dim, sub_x_i, s_batch, n_batch, n_remain, verbosity);
    obj_try1 = SGD_try1->compute_obj(this->data, dim, sub_x_i, n_subsample, this->y);
    if (verbosity >= 3)
        std::cout << "Done. Obj = " << obj_try1 << "." << std::endl;
    eta0_try_factor = this->gd_param->learning_rate_try_factor();
    eta0_try2 = eta0_try1 * eta0_try_factor;
    if (eta0_try2 < err) {
        delete SGD_try1;
        this->gd_param->init_learning_rate(eta0_try2);
        this->t = 1;
        return *this;
    }
    if (verbosity >= 3)
        std::cout << "        Trying eta0 = " << eta0_try2 << "... " << std::flush;
    SGD_try2       = duplicate_this();
    SGD_try2->stat = stat_new;
    this->gd_param->init_learning_rate(eta0_try2);
    SGD_try2->train_epoch(dim, sub_x_i, s_batch, n_batch, n_remain, verbosity);
    obj_try2 = SGD_try2->compute_obj(this->data, dim, sub_x_i, n_subsample, this->y);
    if (verbosity >= 3)
        std::cout << "Done. Obj = " << obj_try2 << "." << std::endl;
    if (obj_try1 < obj_try2) {
        eta0_try_factor = 1 / eta0_try_factor;
        obj_try2        = obj_try1;
        eta0_try2       = eta0_try1;
        SGD_temp        = SGD_try2;
        SGD_try2        = SGD_try1;
        SGD_try1        = SGD_temp;
    }
    do {
        eta0_try1 = eta0_try2;
        obj_try1  = obj_try2;
        SGD_temp  = SGD_try1;
        SGD_try1  = SGD_try2;
        SGD_try2  = SGD_temp;
        eta0_try2 = eta0_try1 * eta0_try_factor;
        if (eta0_try2 < err) break;
        if (verbosity >= 3)
            std::cout << "        Trying eta0 = " << eta0_try2 << "... " << std::flush;
        SGD_try2->t = 0;
        SGD_try2->assign_to_this(this);
        this->gd_param->init_learning_rate(eta0_try2);
        SGD_try2->train_epoch(dim, sub_x_i, s_batch, n_batch, n_remain, verbosity);
        obj_try2 = SGD_try2->compute_obj(this->data, dim, sub_x_i, n_subsample, this->y);
        if (verbosity >= 3)
            std::cout << "Done. Obj = " << obj_try2 << "." << std::endl;
    } while (obj_try1 >= obj_try2);
    delete SGD_try2;
    if (eta0_try_factor > 1) eta0_try_factor = 1 / eta0_try_factor;
    this->gd_param->init_learning_rate(eta0_try1 * eta0_try_factor);
    this->t = 1;
    assign_to_this(SGD_try1);
    delete SGD_try1;
    if (verbosity == 1) std::cout << "Done." << std::endl;
    if (verbosity >= 2) {
        if (verbosity == 2) std::cout << "Done.\n";
        std::cout << "    Setting eta0 = " << eta0_try1 * eta0_try_factor << "." << std::endl;
    }
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train_epoch(_DAT_DIM_T _d, _N_DAT_T* _x,
            _N_DAT_T _s_batch,
            _N_DAT_T _n_batch,
            _N_DAT_T _n_remain,
            char _v) {
    _COMP_T   eta0               = this->gd_param->init_learning_rate();
    bool      show_obj_each_iter = this->gd_param->show_obj_each_iteration();
    _N_DAT_T* x_batch            = _x;
    _N_DAT_T  num_of_batches;
    if (_n_remain > 0) num_of_batches = _n_batch + 1;
    else num_of_batches = _n_batch;
    for (_N_DAT_T i = 0; i < _n_batch; ++i, ++this->t, x_batch += _s_batch) {
        if (_v >= 3) {
            std::cout << "        Batch " << i + 1 << " / " << num_of_batches
                      << " ... " << std::flush;
        }
        this->eta = this->compute_learning_rate(eta0, this->t);
        this->train_batch(this->data,
                          _d,
                          x_batch,
                          _s_batch,
                          this->y,
                          this->eta);
        if (_v >= 3) {
            std::cout << "Done. eta = " << this->eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = "
                          << this->compute_obj(this->data,
                                               _d,
                                               _x,
                                               _s_batch * _n_batch + _n_remain,
                                               this->y);
            }
            std::cout << "." << std::endl;
        }
    }
    if (_n_remain > 0) {
        if (_v >= 3) {
            std::cout << "        Batch " << num_of_batches << " / "
                      << num_of_batches << " ... " << std::flush;
        }
        this->eta = this->compute_learning_rate(eta0, this->t);
        this->train_batch(this->data,
                          _d,
                          x_batch,
                          _n_remain,
                          this->y,
                          this->eta);
        ++this->t;
        if (_v >= 3) {
            std::cout << "Done. eta = " << this->eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = "
                          << this->compute_obj(this->data,
                                               _d,
                                               _x,
                                               _s_batch * _n_batch + _n_remain,
                                               this->y);
            }
            std::cout << "." << std::endl;
        }
    }
    return *this;
}

# endif
