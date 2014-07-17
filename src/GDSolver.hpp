# ifndef GDSOLVER_HPP_
# define GDSOLVER_HPP_

# include "Q.hpp"
# include "Data.hpp"

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
class GDProblem {
  public:
    virtual _COMP_T learning_rate(_COMP_T init_learning_rate, unsigned int t);
    virtual _COMP_T objective(typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator& sample,
                              _N_DAT_T num_of_samples) = 0;
    virtual int     train_batch(typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator& sample,
                                _N_DAT_T num_of_samples, _COMP_T learning_rate) = 0;
    virtual _SUPV_T test_one(typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator& sample) = 0;

    virtual GDProblem* copy() = 0;
    virtual int        assign_from(GDProblem& some) = 0;
    virtual int        ostream_var(std::ostream& out) = 0;

    _COMP_T objective(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data);

    virtual ~GDProblem() {}
};

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
class GDSolver {
  public:
    struct GDParam {
        char          verbosity;
        bool          show_obj_each_iter;
        bool          show_learning_rate_each_iter;
        unsigned int  num_of_iter;
        _COMP_T       optimal_error;
        _COMP_T       init_learning_rate;
        _COMP_T       init_learning_rate_try_1st;
        _COMP_T       init_learning_rate_try_factor;
        _COMP_T       init_learning_rate_try_subsample_rate;
        _N_DAT_T      init_learning_rate_try_min_sample;
        std::ostream* out_stream;

        GDParam();
        GDParam(const GDParam& some);
        ~GDParam();
        const char*   out() const;
        GDParam&      out(const char* out_name);

      private:
        char          out_name[SIZEOF_PATH];
    };

    GDSolver();
    GDSolver(const GDParam& gd_param);
    virtual ~GDSolver() {}

    int            set_up(const GDParam& gd_param);
    virtual bool   is_set_up() const;
    int            reset();
    virtual int    train(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data,
                         GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem);
    int            test(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data,
                        GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem);

  protected:
    unsigned int   t;
    _COMP_T        init_learning_rate;
    const GDParam* gd_param;

    int            try_learning_rate(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data,
                                     GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem);
    virtual typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator*
                   train_iterator(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data);
    virtual int    train_iteration(typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator& sample,
                                   _N_DAT_T num_of_samples,
                                   GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem);
};

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
class SGDSolver : public GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T> {
  public:
    struct SGDParam {
        _N_DAT_T     size_of_batch;

        SGDParam();
    };

    SGDSolver();
    SGDSolver(const typename GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& gd_param,
              const SGDParam& sgd_param);
    virtual ~SGDSolver() {}

    using GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::set_up;
    int             set_up(const typename GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& gd_param,
                           const SGDParam& sgd_param);
    virtual bool    is_set_up() const;

  protected:
    const SGDParam* sgd_param;

    virtual typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::permute_iterator*
                    train_iterator(Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& data);
    virtual int     train_iteration(typename Data<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::iterator& sample,
                                    _N_DAT_T num_of_samples,
                                    GDProblem<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& problem);
};


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
struct GDSolver_GDParam_bpy_pickle : bpy::pickle_suite {
    static bpy::tuple getstate(const typename GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam&);
    static void       setstate(typename GDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam&, bpy::tuple&);
};

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
struct SGDSolver_SGDParam_bpy_pickle : bpy::pickle_suite {
    static bpy::tuple getstate(const typename SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::SGDParam&);
    static void       setstate(typename SGDSolver<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::SGDParam&, bpy::tuple&);
};

# include "GDSolver.inl"

# endif
