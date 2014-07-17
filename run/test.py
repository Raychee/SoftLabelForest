import numpy as np
import scipy.io as sio
import os
import Qplot
import SoftLabelTree
import SoftDecisionModel
from time import time
from SoftDecisionModel import Data, SoftDecisionModel, SoftDecisionSolver
from SoftLabelTree import SoftLabelTree
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score, accuracy_score
from joblib import Parallel, delayed


def set_params(gd_param, sgd_param, sd_param,
               model_param, ovr_param, tree_param):
    gd_param.verbosity = '\0'
    gd_param.show_obj_each_iter = False
    gd_param.show_learning_rate_each_iter = False
    gd_param.num_of_iter = 40
    gd_param.optimal_error = 0
    gd_param.init_learning_rate = 0
    gd_param.init_learning_rate_try_1st = 0.1
    gd_param.init_learning_rate_try_factor = 10
    gd_param.init_learning_rate_try_subsample_rate = 0.3
    gd_param.init_learning_rate_try_min_sample = 1000

    sgd_param.size_of_batch = 50

    sd_param.verbosity = '\3'
    sd_param.show_p_each_iter = False
    sd_param.num_of_trials = 5
    sd_param.num_of_iter_update_p_per_train = 20
    sd_param.num_of_iter_update_p_per_epoch = 0
    sd_param.num_of_iter_update_p_per_batch = 0
    sd_param.num_of_iter_confirm_converge = 5

    model_param.regularizor = 1e-5
    model_param.reg_l1_ratio = 0.1
    model_param.bias_learning_rate_factor = 0.05
    model_param.init_var_subsample_rate = 0.5
    model_param.init_var_subsample_min = 2000

    tree_param.min_entropy = 0.5
    tree_param.max_depth = -1
    tree_param.min_num_of_samples_per_node = 20
    tree_param.ovr_in_leaves = False

    ovr_param.update({'alpha': model_param.regularizor,
                      'class_weight': 'auto',
                      'epsilon': 0.1,
                      'eta0': 0.01,
                      'fit_intercept': True,
                      'l1_ratio': model_param.reg_l1_ratio,
                      'learning_rate': 'optimal',
                      'loss': 'hinge',
                      'n_iter': gd_param.num_of_iter,
                      'n_jobs': 1,
                      'penalty': 'elasticnet',
                      'power_t': 0.5,
                      'random_state': None,
                      'rho': None,
                      'shuffle': True,
                      'verbose': 0,
                      'warm_start': False})


def print_score(X_train, X_test, score):
    print 'Test on training set:\n' \
          '\tAcc:\t{acc_train:%} ({n_acc_train}/{n_train})\n' \
          '\tmAP:\t{mAP_train:%}\n' \
          '\tAverage complexity: {avg_complexity_train}\n' \
          '\tAverage depth:      {avg_depth_train}\n' \
          'Test on test set:\n' \
          '\tAcc:\t{acc_test:%} ({n_acc_test}/{n_test})\n' \
          '\tmAP:\t{mAP_test:%}\n' \
          '\tAverage complexity: {avg_complexity_test}\n' \
          '\tAverage depth:      {avg_depth_test}\n\n' \
          'Training time:\n' \
          '\t{t_train_h:.0f}h {t_train_m:.0f}m {t_train_s}s\n\n' \
          'Test time ({n_train}+{n_test}={n_all} samples):\n' \
          '\t{t_test_h:.0f}h {t_test_m:.0f}m {t_test_s}s\n' \
          '\tavg: {t_test_avg:f}s per sample' \
          .format(n_train=X_train.shape[0], n_test=X_test.shape[0], n_all=X_train.shape[0]+X_test.shape[0],
                  t_train_h=score['time_train']//3600, t_train_m=score['time_train']%3600//60, t_train_s=score['time_train']%60,
                  t_test_h=score['time_test']//3600, t_test_m=score['time_test']%3600//60, t_test_s=score['time_test']%60,
                  t_test_avg=score['time_test']/(X_train.shape[0]+X_test.shape[0]),
                  **score)


def go(sltree, score, X_train, Y_train, X_test, Y_test):
    t_train_begin = time()
    sltree.train(X_train, Y_train)
    t_train_end = time()
    t_test_begin = time()
    Y_predict_train, AP_train, complexity_train, depths_train = sltree.test(X_train, Y_train, return_complexity=True, return_depth=True)
    Y_predict_test, AP_test, complexity_test, depths_test = sltree.test(X_test, Y_test, return_complexity=True, return_depth=True)
    t_test_end = time()
    n_acc_train = np.count_nonzero(Y_predict_train == Y_train)
    n_acc_test = np.count_nonzero(Y_predict_test == Y_test)

    score.update({'acc_train':float(n_acc_train)/Y_predict_train.shape[0],
                  'n_acc_train':n_acc_train,
                  'AP_train':AP_train,
                  'mAP_train':np.mean(AP_train),
                  'complexity_train':complexity_train,
                  'avg_complexity_train':np.mean(complexity_train),
                  'depths_train':depths_train,
                  'avg_depth_train':np.mean(depths_train),
                  'acc_test':float(n_acc_test)/Y_predict_test.shape[0],
                  'n_acc_test':n_acc_test,
                  'AP_test':AP_test,
                  'mAP_test':np.mean(AP_test),
                  'complexity_test':complexity_test,
                  'avg_complexity_test':np.mean(complexity_test),
                  'depths_test':depths_test,
                  'avg_depth_test':np.mean(depths_test),
                  'time_test':t_test_end-t_test_begin})

if __name__ == '__main__':
    data = sio.loadmat('../dat/data5.mat')
    X = data['X'].astype(np.double).T
    Y = data['Y'].astype(np.int32).ravel()
    X_train = X
    X_test = X
    Y_train = Y
    Y_test = Y

    data = Data(X, Y)
    model_param = SoftDecisionModel.Param()
    solver = SoftDecisionSolver()
    model = SoftDecisionModel(data, model_param)

    # num_of_trees = 4
    # n_jobs = 4

    # sltrees = [SoftLabelTree() for i in xrange(num_of_trees)]
    # scores = [{'acc_train':0., 'n_acc_train':0, 'AP_train':None, 'mAP_train':0.,
    #            'complexity_train':None, 'avg_complexity_train':0.,
    #            'depths_train':None, 'avg_depth_train':0.,
    #            'acc_test':0.,  'n_acc_test':0,  'AP_test':None, 'mAP_test':0.,
    #            'complexity_test':None, 'avg_complexity_test':0.,
    #            'depths_test':None,  'avg_depth_test':0.,
    #            'time_train':0., 'time_test':0.} for i in xrange(num_of_trees)]

    # Parallel(n_jobs=n_jobs)(delayed(go)(sltree, score, X_train, Y_train,
    #                                     X_test, Y_test)
    #                         for sltree, score in zip(sltrees, scores))

    # for i, score in enumerate(scores):
    #     print '************** SoftLabelTree {} **************'.format(i+1)
    #     print_score(X_train, X_test, score)
