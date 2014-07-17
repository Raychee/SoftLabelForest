# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
# define PY_ARRAY_UNIQUE_SYMBOL SOFTLABELTREE_MODULE

# include <cstring>
# include <boost/python.hpp>
# include <numpy/arrayobject.h>
# include "SoftDecisionModel.hpp"
# include "Data.hpp"
# include "Q.hpp"


using namespace boost::python;


template class Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>;
template class GDProblem<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>;
template class GDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>;
template class SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>;

template struct Data_bpy_pickle<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>;
template struct GDSolver_GDParam_bpy_pickle<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>;
template struct SGDSolver_SGDParam_bpy_pickle<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>;

typedef Data<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>      Data_;
typedef GDProblem<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> GDProblem_;
typedef GDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>  GDSolver_;
typedef SGDSolver<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> SGDSolver_;

typedef Data_bpy_pickle<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> Data_bpy_pickle_;
typedef GDSolver_GDParam_bpy_pickle<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> GDSolver_GDParam_bpy_pickle_;
typedef SGDSolver_SGDParam_bpy_pickle<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> SGDSolver_SGDParam_bpy_pickle_;


BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Data_check_overloads, Data_::check, 0, 2);


BOOST_PYTHON_MODULE(SoftDecisionModel) {
    {
        Data_& (Data_::*Data_load_obj_obj)(object&, object&) = &Data_::_load;
        Data_& (Data_::*Data_load_obj_obj_obj)(object&, object&, object&) = &Data_::_load;

        class_<Data_>("Data")
            .def(init<object&, object&>())
            .def(init<object&, object&, object&>())
            .def("load", Data_load_obj_obj, return_value_policy<reference_existing_object>())
            .def("load", Data_load_obj_obj_obj, return_value_policy<reference_existing_object>())
            .def("check", &Data_::check, Data_check_overloads())
            .add_property("num_of_samples", &Data_::num_of_samples)
            .add_property("dimension", &Data_::dimension)
            .add_property("num_of_labels", &Data_::num_of_labels)
            .add_property("entropy", &Data_::entropy)
            .add_property("data", &Data_::_data)
            .add_property("indices", &Data_::_indices)
            .add_property("labels", &Data_::_labels)
            .add_property("num_of_samples_of_each_label", &Data_::_num_of_sample_of_each_label)
            .def_pickle(Data_bpy_pickle_());
    }{
        COMP_T (SoftDecisionModel::*SoftDecisionModel_objective)(Data_&) = &GDProblem_::objective;
        COMP_T (SoftDecisionModel::*SoftDecisionModel_test_one)(object&) = &SoftDecisionModel::test_one;

        // scope scope_model =
        class_<SoftDecisionModel>("SoftDecisionModel", init<const SoftDecisionModel::ModelParam&>())
            .def(init<Data_&, SoftDecisionModel::ModelParam&>())
            .def("objective", SoftDecisionModel_objective)
            .def("test_one", SoftDecisionModel_test_one)
            .add_property("indices_of_pos_samples", &SoftDecisionModel::indices_of_pos_samples)
            .add_property("indices_of_neg_samples", &SoftDecisionModel::indices_of_neg_samples)
            .add_property("w", &SoftDecisionModel::ws)
            .add_property("b", &SoftDecisionModel::bs)
            .add_property("p", &SoftDecisionModel::ps)
            .add_property("n_nonzeros", &SoftDecisionModel::num_of_nonzeros)
            .def_pickle(SoftDecisionModel_bpy_pickle());

        class_<SoftDecisionModel::ModelParam>("ModelParam")
            .def(init<const SoftDecisionModel::ModelParam&>())
            .def_readwrite("regularizor", &SoftDecisionModel::ModelParam::reg_coeff)
            .def_readwrite("reg_l1_ratio", &SoftDecisionModel::ModelParam::reg_l1_ratio)
            .def_readwrite("bias_learning_rate_factor", &SoftDecisionModel::ModelParam::b_learning_rate_factor)
            .def_readwrite("init_var_subsample_rate", &SoftDecisionModel::ModelParam::init_var_subsample_rate)
            .def_readwrite("init_var_subsample_min", &SoftDecisionModel::ModelParam::init_var_subsample_min)
            .def_pickle(SoftDecisionModel_ModelParam_bpy_pickle());
    }{
        int (SoftDecisionSolver::*SoftDecisionSolver_set_up)(const GDSolver_::GDParam&, const SGDSolver_::SGDParam&, const SoftDecisionSolver::SDParam&) = &SoftDecisionSolver::set_up;
        int (SoftDecisionSolver::*SoftDecisionSolver_train)(Data_&, SoftDecisionModel&) = &SoftDecisionSolver::train;

        // scope scope_solver =
        class_<SoftDecisionSolver>("SoftDecisionSolver")
            .def("set_up", SoftDecisionSolver_set_up)
            .def("is_set_up", &SoftDecisionSolver::is_set_up)
            .def("train", SoftDecisionSolver_train);

        const char* (GDSolver_::GDParam::*GDSolver_GDParam_out_get)() const              = &GDSolver_::GDParam::out;
        GDSolver_::GDParam& (GDSolver_::GDParam::*GDSolver_GDParam_out_set)(const char*) = &GDSolver_::GDParam::out;

        class_<GDSolver_::GDParam>("GDParam")
            .def(init<const GDSolver_::GDParam&>())
            .def_readwrite("verbosity", &GDSolver_::GDParam::verbosity)
            .def_readwrite("show_obj_each_iter", &GDSolver_::GDParam::show_obj_each_iter)
            .def_readwrite("show_learning_rate_each_iter", &GDSolver_::GDParam::show_learning_rate_each_iter)
            .def_readwrite("num_of_iter", &GDSolver_::GDParam::num_of_iter)
            .def_readwrite("optimal_error", &GDSolver_::GDParam::optimal_error)
            .def_readwrite("init_learning_rate", &GDSolver_::GDParam::init_learning_rate)
            .def_readwrite("init_learning_rate_try_1st", &GDSolver_::GDParam::init_learning_rate_try_1st)
            .def_readwrite("init_learning_rate_try_factor", &GDSolver_::GDParam::init_learning_rate_try_factor)
            .def_readwrite("init_learning_rate_try_subsample_rate", &GDSolver_::GDParam::init_learning_rate_try_subsample_rate)
            .def_readwrite("init_learning_rate_try_min_sample", &GDSolver_::GDParam::init_learning_rate_try_min_sample)
            .add_property("out", GDSolver_GDParam_out_get, make_function(GDSolver_GDParam_out_set, return_value_policy<reference_existing_object>()))
            .def_pickle(GDSolver_GDParam_bpy_pickle_());

        class_<SGDSolver_::SGDParam>("SGDParam")
            .def(init<const SGDSolver_::SGDParam&>())
            .def_readwrite("size_of_batch", &SGDSolver_::SGDParam::size_of_batch)
            .def_pickle(SGDSolver_SGDParam_bpy_pickle_());

        const char* (SoftDecisionSolver::SDParam::*SoftDecisionSolver_SDParam_out_get)() const                       = &SoftDecisionSolver::SDParam::out;
        SoftDecisionSolver::SDParam& (SoftDecisionSolver::SDParam::*SoftDecisionSolver_SDParam_out_set)(const char*) = &SoftDecisionSolver::SDParam::out;

        class_<SoftDecisionSolver::SDParam>("SDParam")
            .def(init<const SoftDecisionSolver::SDParam&>())
            .def_readwrite("verbosity", &SoftDecisionSolver::SDParam::verbosity)
            .def_readwrite("show_p_each_iter", &SoftDecisionSolver::SDParam::show_p_each_iter)
            .def_readwrite("num_of_trials", &SoftDecisionSolver::SDParam::num_of_trials)
            .def_readwrite("num_of_iter_update_p_per_train", &SoftDecisionSolver::SDParam::num_of_iter_update_p_per_train)
            .def_readwrite("num_of_iter_update_p_per_epoch", &SoftDecisionSolver::SDParam::num_of_iter_update_p_per_epoch)
            .def_readwrite("num_of_iter_update_p_per_batch", &SoftDecisionSolver::SDParam::num_of_iter_update_p_per_batch)
            .def_readwrite("num_of_iter_confirm_converge", &SoftDecisionSolver::SDParam::num_of_iter_confirm_converge)
            .add_property("out", SoftDecisionSolver_SDParam_out_get, make_function(SoftDecisionSolver_SDParam_out_set, return_value_policy<reference_existing_object>()))
            .def_pickle(SoftDecisionSolver_SDParam_bpy_pickle());
    }

    import_array();
}

