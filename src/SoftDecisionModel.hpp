# ifndef SOFT_DECISION_MODEL_HPP_
# define SOFT_DECISION_MODEL_HPP_

# include <boost/python.hpp>
# include "GDSolver.hpp"

namespace bpy = boost::python;

class SoftDecisionModel : public GDProblem<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> {
  public:
    struct ModelParam {
        COMP_T       reg_coeff;
        COMP_T       reg_l1_ratio;
        COMP_T       b_learning_rate_factor;
        COMP_T       init_var_subsample_rate;
        COMP_T       init_var_subsample_min;

        ModelParam();
    };

    SoftDecisionModel();
    SoftDecisionModel(const ModelParam& param);
    SoftDecisionModel(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data, ModelParam& param);
    SoftDecisionModel(const SoftDecisionModel& some);
    SoftDecisionModel(SoftDecisionModel&& some);
    virtual ~SoftDecisionModel();
    SoftDecisionModel& operator=(SoftDecisionModel& some);
    SoftDecisionModel& operator=(SoftDecisionModel&& some);

    int            init_var(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data);
    N_DAT_T        update_p(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data);

    virtual COMP_T learning_rate(COMP_T init_learning_rate, unsigned int t);
    virtual COMP_T objective(typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample,
                             N_DAT_T num_of_samples);
    virtual int    train_batch(typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample,
                               N_DAT_T num_of_samples, COMP_T learning_rate);
    virtual SUPV_T test_one(typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample);
    COMP_T         test_one(bpy::object& sample);

    using GDProblem::objective;

    virtual GDProblem* copy();
    virtual int        assign_from(GDProblem& some);
    virtual int        ostream_var(std::ostream& out);
    int                ostream_p(std::ostream& out,
                                 Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data);
    COMP_T             p_mean();
    COMP_T             pos_percentage();

    /// Boost.Python Interface
    bpy::object        ws() const;
    bpy::object        ps() const;
    COMP_T             bs() const;
    DAT_DIM_T          num_of_nonzeros() const;
    const ModelParam&  model_param() const;
    bpy::object        indices_of_pos_samples();
    bpy::object        indices_of_neg_samples();

    friend struct SoftDecisionModel_bpy_pickle;

  protected:
    COMP_T            b;
    DAT_DIM_T         n_w_i;
    DAT_DIM_T*        w_i;
    COMP_T*           w;
    COMP_T*           p;
    const ModelParam* param;

    N_DAT_T*          index_of_pos_sample;
    N_DAT_T*          index_of_neg_sample;
    N_DAT_T           num_of_pos_sample;
    N_DAT_T           num_of_neg_sample;

    DAT_DIM_T         dimension;
    SUPV_T            num_of_labels;
    N_DAT_T           num_of_samples;

    COMP_T  loss(typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample,
                 N_DAT_T num_of_samples) const;
    COMP_T  score(typename Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::iterator& sample) const;
    COMP_T  l1_norm() const;
    COMP_T  l2_norm() const;
};


class SoftDecisionSolver : public SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> {
  public:
    struct SDParam {
        char          verbosity;
        bool          show_p_each_iter;
        unsigned int  num_of_trials;
        unsigned int  num_of_iter_update_p_per_train;
        unsigned int  num_of_iter_update_p_per_epoch;
        unsigned int  num_of_iter_update_p_per_batch;
        unsigned int  num_of_iter_confirm_converge;
        std::ostream* out_stream;

        SDParam();
        SDParam(const SDParam& some);
        ~SDParam();
        const char*   out() const;
        SDParam&      out(const char* out_name);

      private:
        char          out_name[SIZEOF_PATH];
    };

    SoftDecisionSolver();
    ~SoftDecisionSolver() {}

    using SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::set_up;
    int          set_up(const typename GDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam& gd_param,
                        const typename SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& sgd_param,
                        const SDParam& sd_param);
    virtual bool is_set_up() const;
    using SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::train;
    int          train(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data,
                       SoftDecisionModel& model);

  protected:
    const SDParam* sd_param;

    int          train_alter(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data,
                             SoftDecisionModel& model);
    int          train_alter_epoch(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data,
                                   SoftDecisionModel& model);
    int          train_alter_batch(Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>& data,
                                   SoftDecisionModel& model);
};


struct SoftDecisionModel_ModelParam_bpy_pickle : bpy::pickle_suite {
    static bpy::tuple getstate(const SoftDecisionModel::ModelParam&);
    static void       setstate(SoftDecisionModel::ModelParam&, bpy::tuple&);
};

struct SoftDecisionModel_bpy_pickle : bpy::pickle_suite {
    static bpy::tuple getinitargs(const SoftDecisionModel&);
    static bpy::tuple getstate(const SoftDecisionModel&);
    static void       setstate(SoftDecisionModel&, bpy::tuple&);
};

struct SoftDecisionSolver_SDParam_bpy_pickle : bpy::pickle_suite {
    static bpy::tuple getstate(const SoftDecisionSolver::SDParam&);
    static void       setstate(SoftDecisionSolver::SDParam&, bpy::tuple&);
};

# endif
