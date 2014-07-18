import numpy as np
from sys import maxint
from collections import deque
from copy import copy
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from SoftDecisionModel import Data, SoftDecisionModel, SoftDecisionSolver
from SoftDecisionModel import ModelParam, GDParam, SGDParam, SDParam
from Q import unpack_seq, pack_seq, parse_args, average_precision


class TreeParam(object):
    def __init__(self):
        self.min_entropy = 1
        self.max_depth = 0
        self.min_num_of_samples_per_node = 1
        self.ovr_in_leaves = True


class TreeNode(object):
    def __init__(self):
        self.depth   = 0
        self.model   = None
        self.data    = None
        self.lchild  = None
        self.rchild  = None

    def is_leaf(self):
        return self.lchild is None and self.rchild is None

    def attach_lchild(self, node):
        self.lchild = node
        node.depth = self.depth + 1

    def attach_rchild(self, node):
        self.rchild = node
        node.depth = self.depth + 1

    def __copy__(self):
        node = TreeNode()
        node.depth   = self.depth
        node.model   = self.model
        node.data    = self.data
        return node


class SoftLabelTree(object):

    def __init__(self, *args, **kwargs):
        self.gd_param    = None
        self.sgd_param   = None
        self.sd_param    = None
        self.model_param = None
        self.ovr_param   = None
        self.tree_param  = None
        self.copy_from   = None
        args = unpack_seq(args)
        self.set_up(*args, **kwargs)
        if self.gd_param    is None: self.gd_param    = GDParam()
        if self.sgd_param   is None: self.sgd_param   = SGDParam()
        if self.sd_param    is None: self.sd_param    = SDParam()
        if self.model_param is None: self.model_param = ModelParam()
        if self.tree_param  is None: self.tree_param  = TreeParam()
        if self.ovr_param   is None:
            self.ovr_param = {'alpha'        : self.model_param.regularizor,
                              'class_weight' : None,
                              'epsilon'      : 0.1,
                              'eta0'         : 0.01,
                              'fit_intercept': True,
                              'l1_ratio'     : 0.15,
                              'learning_rate': 'optimal',
                              'loss'         : 'log',
                              'n_iter'       : self.gd_param.num_of_iter,
                              'n_jobs'       : 1,
                              'penalty'      : 'elasticnet',
                              'power_t'      : 0.5,
                              'random_state' : None,
                              'rho'          : None,
                              'shuffle'      : True,
                              'verbose'      : 0,
                              'warm_start'   : False}
        if self.copy_from is not None:
            self._copy(self.copy_from)
        else:
            self.root          = None
            self.min_depth     = 0
            self.max_depth     = 0
            self.avg_depth     = 0
            self.num_of_leaves = 1

    def set_up(self, *args, **kwargs):
        self.gd_param,    \
        self.sgd_param,   \
        self.sd_param,    \
        self.model_param, \
        self.ovr_param,   \
        self.tree_param,  \
        self.copy_from = parse_args(args,
                                    (GDParam,
                                     SGDParam,
                                     SDParam,
                                     ModelParam,
                                     dict,
                                     TreeParam,
                                     SoftLabelTree),
                                    (self.gd_param,
                                     self.sgd_param,
                                     self.sd_param,
                                     self.model_param,
                                     self.ovr_param,
                                     self.tree_param,
                                     self.copy_from))
        vars(self).update(kwargs)

    def params(self):
        return {'gd_param'    : self.gd_param,
                'sgd_param'   : self.sgd_param,
                'sd_param'    : self.sd_param,
                'model_param' : self.model_param,
                'ovr_param'   : self.ovr_param,
                'tree_param'  : self.tree_param}

    def is_set_up(self):
        return self.gd_param    is not None and \
               self.sgd_param   is not None and \
               self.sd_param    is not None and \
               self.model_param is not None and \
               self.ovr_param   is not None and \
               self.tree_param  is not None

    def train(self, X, Y, I=None, verbose=0):
        self.root = TreeNode()
        if I is None:
            self.root.data = Data(X, Y)
        else:
            self.root.data = Data(X, Y, I)
        return self._go_train(iter(self), verbose)

    def continue_train(self, verbose=0):
        nodes_to_go = SoftLabelTree._NodeIterator(self, start_from_root=False)
        for node in self:
            if node.is_leaf() or self._should_be_leaf(node):
                nodes_to_go.append(node)
        return self._go_train(nodes_to_go, verbose)

    def test_proba(self, *args, **kwargs):
        ''' Usage:
            Y_proba[, complexity][, depths]
                = test_proba(X, param=None,
                             return_complexity=False,
                             return_depth=False,
                             verbose=0)
        '''
        X, param, return_complexity, return_depth, verbose \
            = parse_args(args, (np.ndarray,
                                TreeParam,
                                bool,
                                bool,
                                int),
                               (None,
                                self.tree_param,
                                False,
                                False,
                                0))
        X                 = kwargs.get('X',                 X)
        param             = kwargs.get('param',             param)
        return_complexity = kwargs.get('return_complexity', return_complexity)
        return_depth      = kwargs.get('return_depth',      return_depth)
        verbose           = kwargs.get('verbose',           verbose)
        Y_proba = np.empty((X.shape[0], self.root.data.num_of_labels),
                           dtype=X.dtype)
        complexity   = None
        depths       = None
        Y_proba[...] = -np.inf
        if return_depth:      depths     = np.zeros(X.shape[0], dtype=np.int)
        if return_complexity: complexity = np.zeros(X.shape[0], dtype=np.int)
        for i in xrange(X.shape[0]):
            if verbose > 0:
                print '\rTesting sample {}/{} ...'.format(i+1, X.shape[0]),
            x = X[i]
            node = self.root
            while not (node.is_leaf() or self._should_be_leaf(node, param)):
                if return_complexity:
                    complexity[i] += node.model.n_nonzeros
                if node.model.test_one(x) > 0:
                    node = node.lchild
                else:
                    node = node.rchild
            if node.model is None or not param.ovr_in_leaves:
                num_distrib = node.data.num_of_samples_of_each_label \
                              .astype(X.dtype)
                proba = num_distrib / np.sum(num_distrib)
                Y_proba[i, node.data.labels-1] = proba
            else:
                proba = node.model.decision_function(x).ravel()
                num_model_classes = node.model.classes_.shape[0]
                if num_model_classes <= 2:
                    proba = np.r_[-proba, proba]
                    num_model_classes -= 1
                Y_proba[i, node.model.classes_-1] = proba
                if return_complexity:
                    complexity[i] += np.count_nonzero(node.model.coef_)
            if return_depth:
                    depths[i] = node.depth
        if verbose > 0:
            print '\rDone.'
        return pack_seq(Y_proba, complexity, depths)

    def test(self, *args, **kwargs):
        ''' Usage:
            Y_label[, ap][, complexity][, depths]
                = test(X, Y=None, param=None,
                       return_complexity=False, return_depth=False,
                       verbose=0)
        '''
        X, Y, param, return_complexity, return_depth, verbose \
            = parse_args(args,
                         (np.ndarray, np.ndarray, TreeParam,
                          bool,  bool,  int),
                         (None,       None,       None,
                          False, False, 0))
        X                 = kwargs.get('X',                 X)
        Y                 = kwargs.get('Y',                 Y)
        param             = kwargs.get('param',             param)
        return_complexity = kwargs.get('return_complexity', return_complexity)
        return_depth      = kwargs.get('return_depth',      return_depth)
        verbose           = kwargs.get('verbose',           verbose)
        Y_predict  = None
        ap         = None
        complexity = None
        depths     = None
        if X is not None and X.shape[-1] == self.root.data.dimension:
            result = self.test_proba(X, param, return_complexity, return_depth)
            if return_complexity and return_depth:
                Y_proba, complexity, depths = result
            elif not return_complexity and return_depth:
                Y_proba, depths = result
            elif return_complexity and not return_depth:
                Y_proba, complexity = result
            else:
                Y_proba = result
            Y_predict = np.argmax(Y_proba, axis=-1) + 1
            if Y is not None and Y.shape[0] == X.shape[0]:
                ap = average_precision(Y, Y_proba)
        return pack_seq(Y_predict, ap, complexity, depths)

    def __copy__(self):
        return SoftLabelTree(self)

    def __repr__(self):
        return 'gd_param.verbosity = {gd_repr}\n' \
               'gd_param.show_obj_each_iter = {self.gd_param.show_obj_each_iter}\n' \
               'gd_param.show_learning_rate_each_iter = {self.gd_param.show_learning_rate_each_iter}\n' \
               'gd_param.num_of_iter = {self.gd_param.num_of_iter}\n' \
               'gd_param.optimal_error = {self.gd_param.optimal_error}\n' \
               'gd_param.init_learning_rate = {self.gd_param.init_learning_rate}\n' \
               'gd_param.init_learning_rate_try_1st = {self.gd_param.init_learning_rate_try_1st}\n' \
               'gd_param.init_learning_rate_try_factor = {self.gd_param.init_learning_rate_try_factor}\n' \
               'gd_param.init_learning_rate_try_subsample_rate = {self.gd_param.init_learning_rate_try_subsample_rate}\n' \
               'gd_param.init_learning_rate_try_min_sample = {self.gd_param.init_learning_rate_try_min_sample}\n\n' \
               'sgd_param.size_of_batch = {self.sgd_param.size_of_batch}\n\n' \
               'sd_param.verbosity = {sd_repr}\n' \
               'sd_param.show_p_each_iter = {self.sd_param.show_p_each_iter}\n' \
               'sd_param.num_of_trials = {self.sd_param.num_of_trials}\n' \
               'sd_param.num_of_iter_update_p_per_train = {self.sd_param.num_of_iter_update_p_per_train}\n' \
               'sd_param.num_of_iter_update_p_per_epoch = {self.sd_param.num_of_iter_update_p_per_epoch}\n' \
               'sd_param.num_of_iter_update_p_per_batch = {self.sd_param.num_of_iter_update_p_per_batch}\n' \
               'sd_param.num_of_iter_confirm_converge = {self.sd_param.num_of_iter_confirm_converge}\n\n' \
               'model_param.regularizor = {self.model_param.regularizor}\n' \
               'model_param.reg_l1_ratio = {self.model_param.reg_l1_ratio}\n' \
               'model_param.bias_learning_rate_factor = {self.model_param.bias_learning_rate_factor}\n' \
               'model_param.init_var_subsample_rate = {self.model_param.init_var_subsample_rate}\n' \
               'model_param.init_var_subsample_min = {self.model_param.init_var_subsample_min}\n\n' \
               'tree_param.min_entropy = {self.tree_param.min_entropy}\n' \
               'tree_param.max_depth = {self.tree_param.max_depth}\n' \
               'tree_param.min_num_of_samples_per_node = {self.tree_param.min_num_of_samples_per_node}\n' \
               'tree_param.ovr_in_leaves = {self.tree_param.ovr_in_leaves}\n\n' \
               'ovr_param.update( {self.ovr_param} )\n\n' \
               'self.min_depth = {self.min_depth}\n' \
               'self.max_depth = {self.max_depth}\n' \
               'self.avg_depth = {self.avg_depth}\n' \
               'self.num_of_leaves = {self.num_of_leaves}' \
               .format(self=self, gd_repr=self.gd_param.verbosity.__repr__(),
                       sd_repr=self.sd_param.verbosity.__repr__())

    def __iter__(self):
        return SoftLabelTree._NodeIterator(self)


    def _copy(self, some):
        self.root        = None
        self.gd_param    = GDParam(some.gd_param)
        self.sgd_param   = SGDParam(some.sgd_param)
        self.sd_param    = SDParam(some.sd_param)
        self.model_param = ModelParam(some.model_param)
        self.ovr_param   = some.ovr_param.copy()
        self.tree_param  = copy(some.tree_param)
        self.min_depth = some.min_depth
        self.max_depth = some.max_depth
        self.avg_depth = some.avg_depth
        self.num_of_leaves = some.num_of_leaves
        if some.root is not None:
            self.root = copy(some.root)
            nodes_to_go = deque()
            nodes_to_go.append((some.root, self.root))
            while len(nodes_to_go) > 0:
                some_node, self_node = nodes_to_go.popleft()
                if some_node.lchild is not None:
                    self_node.attach_lchild(copy(some_node.lchild))
                    nodes_to_go.append((some_node.lchild, self_node.lchild))
                if some_node.rchild is not None:
                    self_node.attach_rchild(copy(some_node.rchild))
                    nodes_to_go.append((some_node.rchild, self_node.rchild))
        return self

    def _should_be_leaf(self, node, param=None):
        if param is None:
            param = self.tree_param
        return (param.max_depth >= 0 and
                node.depth >= param.max_depth) or \
               (node.data.num_of_samples <=
                param.min_num_of_samples_per_node) or \
               (node.data.entropy <= param.min_entropy)

    def _go_train(self, nodes_to_go, verbose=0):
        self.min_depth     = maxint
        self.num_of_leaves = 0
        sum_of_depth       = 0.
        solver = SoftDecisionSolver()
        solver.set_up(self.gd_param, self.sgd_param, self.sd_param)
        for node in nodes_to_go:
            if verbose > 0:
                print 'Node: {id}, {num_nodes} more nodes to go.' \
                      '\n\tdepth = {node.depth}, ' \
                      'num_of_labels = {data.num_of_labels}, ' \
                      'num_of_samples = {data.num_of_samples}, ' \
                      'entropy = {data.entropy}' \
                      .format(id=id(node),
                              num_nodes=len(nodes_to_go),
                              node=node, data=node.data)
            if self._should_be_leaf(node):
                if verbose > 0:
                    print '\tThis is a leaf.'
                if self.min_depth > node.depth: self.min_depth = node.depth
                if self.max_depth < node.depth: self.max_depth = node.depth
                sum_of_depth += node.depth
                self.num_of_leaves += 1
                if node.data.entropy > 0 and self.tree_param.ovr_in_leaves:
                    if verbose > 0:
                        print '\tTraining One-vs-Rest.'
                    self._train_ovr_leaf(node)
            else:
                if verbose > 0:
                    print '\tTraining SoftDecisionModel.'
                node.model = SoftDecisionModel(node.data, self.model_param)
                solver.train(node.data, node.model)
                if node.model.indices_of_pos_samples.shape[0] == 0 or \
                   node.model.indices_of_neg_samples.shape[0] == 0:
                    if verbose > 0:
                        print '\t*** Trivial solutions. This is a leaf.'
                    if self.tree_param.ovr_in_leaves:
                        if verbose > 0:
                            print '\tTraining One-vs-Rest.'
                        self._train_ovr_leaf(node)
                    else:
                        node.model = None
                else:
                    lchild = TreeNode()
                    rchild = TreeNode()
                    lchild.data = Data(node.data,
                                       node.model.indices_of_pos_samples)
                    rchild.data = Data(node.data,
                                       node.model.indices_of_neg_samples)
                    node.attach_lchild(lchild)
                    node.attach_rchild(rchild)
                    nodes_to_go.append(lchild)
                    nodes_to_go.append(rchild)
        if self.num_of_leaves == 0: self.num_of_leaves = 1
        self.avg_depth = sum_of_depth / self.num_of_leaves
        if verbose > 0:
            print 'Done.'
        return self

    def _train_ovr_leaf(self, node):
        node.model = SGDClassifier(**self.ovr_param)
        X, Y = node.data.data
        indices = node.data.indices
        node.model.fit(X[indices, :], Y[indices])
        return self

    class _NodeIterator(object):
        def __init__(self, tree, start_from_root=True):
            super(SoftLabelTree._NodeIterator, self).__init__()
            self.tree = tree
            self.nodes_queue = deque()
            if start_from_root:
                self.nodes_queue.append(tree.root)

        def next(self):
            if len(self.nodes_queue) > 0:
                node = self.nodes_queue.popleft()
                if not self.tree._should_be_leaf(node):
                    if node.lchild is not None:
                        self.nodes_queue.append(node.lchild)
                    if node.rchild is not None:
                        self.nodes_queue.append(node.rchild)
                return node
            else:
                raise StopIteration

        def append(self, node):
            self.nodes_queue.append(node)
            return self

        def __iter__(self):
            return self

        def __len__(self):
            return len(self.nodes_queue)


class ForestParam(object):
    def __init__(self):
        self.num_of_parallel_jobs = 1
        self.bootstrapping        = False


class SoftLabelForest(object):
    def __init__(self, *args, **kwargs):
        self.gd_param     = None
        self.sgd_param    = None
        self.sd_param     = None
        self.model_param  = None
        self.ovr_param    = None
        self.tree_param   = None
        self.forest_param = None
        args = unpack_seq(args)
        self.set_up(*args, **kwargs)
        if self.gd_param     is None: self.gd_param     = GDParam()
        if self.sgd_param    is None: self.sgd_param    = SGDParam()
        if self.sd_param     is None: self.sd_param     = SDParam()
        if self.model_param  is None: self.model_param  = ModelParam()
        if self.tree_param   is None: self.tree_param   = TreeParam()
        if self.forest_param is None: self.forest_param = ForestParam()
        if self.ovr_param    is None:
            self.ovr_param = {'alpha'        : self.model_param.regularizor,
                              'class_weight' : None,
                              'epsilon'      : 0.1,
                              'eta0'         : 0.01,
                              'fit_intercept': True,
                              'l1_ratio'     : 0.15,
                              'learning_rate': 'optimal',
                              'loss'         : 'log',
                              'n_iter'       : self.gd_param.num_of_iter,
                              'n_jobs'       : 1,
                              'penalty'      : 'elasticnet',
                              'power_t'      : 0.5,
                              'random_state' : None,
                              'rho'          : None,
                              'shuffle'      : True,
                              'verbose'      : 0,
                              'warm_start'   : False}
        self.forest = []
        self.avg_num_of_leaves = 0

    def set_up(self, *args, **kwargs):
        self.gd_param,    \
        self.sgd_param,   \
        self.sd_param,    \
        self.model_param, \
        self.ovr_param,   \
        self.tree_param,  \
        self.forest_param = parse_args(args,
                                       (GDParam,
                                        SGDParam,
                                        SDParam,
                                        ModelParam,
                                        dict,
                                        TreeParam,
                                        ForestParam),
                                       (self.gd_param,
                                        self.sgd_param,
                                        self.sd_param,
                                        self.model_param,
                                        self.ovr_param,
                                        self.tree_param,
                                        self.forest_param))
        vars(self).update(kwargs)

    def params(self):
        return {'gd_param'    : self.gd_param,
                'sgd_param'   : self.sgd_param,
                'sd_param'    : self.sd_param,
                'model_param' : self.model_param,
                'ovr_param'   : self.ovr_param,
                'tree_param'  : self.tree_param,
                'forest_param': self.forest_param}

    def is_set_up(self):
        return self.gd_param     is not None and \
               self.sgd_param    is not None and \
               self.sd_param     is not None and \
               self.model_param  is not None and \
               self.ovr_param    is not None and \
               self.tree_param   is not None and \
               self.forest_param is not None

    def train(self, X, Y, num_of_trees=1, verbosity=0):
        num_of_parallel_jobs = self.forest_param.num_of_parallel_jobs
        if num_of_parallel_jobs > 1 and num_of_trees > 1:
            if num_of_parallel_jobs > num_of_trees:
                num_of_parallel_jobs = num_of_trees
            self.forest.extend(
                Parallel(n_jobs=num_of_parallel_jobs, verbose=verbosity) \
                        (delayed(_parallel_train)(
                            SoftLabelTree(self.gd_param,
                                          self.sgd_param,
                                          self.sd_param,
                                          self.model_param,
                                          self.ovr_param,
                                          self.tree_param),
                            X, Y, self.forest_param.bootstrapping)
                         for i in xrange(num_of_trees)))
        else:
            for i in xrange(num_of_trees):
                if verbosity > 0:
                    print '\rTraining SoftLabelTree {}/{}.' \
                          .format(i + 1, num_of_trees),
                self.forest.append(
                    _parallel_train(
                        SoftLabelTree(self.gd_param,
                                      self.sgd_param,
                                      self.sd_param,
                                      self.model_param,
                                      self.ovr_param,
                                      self.tree_param),
                        X, Y, self.forest_param.bootstrapping))
        self.avg_num_of_leaves = self._avg_num_of_leaves()
        return self

    def test_proba(self, *args, **kwargs):
        ''' Usage:
            Y_proba[, complexity][, depths]
                = test_proba(X,
                             return_complexity=False,
                             return_depth=False,
                             verbosity=0)
        '''
        X, return_complexity, return_depth, verbosity \
            = parse_args(args, (np.ndarray, bool,  bool,  int),
                               (None,       False, False, 0))
        X                 = kwargs.get('X',                 X)
        return_complexity = kwargs.get('return_complexity', return_complexity)
        return_depth      = kwargs.get('return_depth',      return_depth)
        verbosity         = kwargs.get('verbosity',         verbosity)
        num_of_parallel_jobs = self.forest_param.num_of_parallel_jobs
        if num_of_parallel_jobs > 1 and len(self) > 1:
            if num_of_parallel_jobs > len(self):
                num_of_parallel_jobs = len(self)
            test_result = Parallel(n_jobs=num_of_parallel_jobs,
                                   verbose=verbosity) \
                                  (delayed(_parallel_test)(sltree, X,
                                                           return_complexity,
                                                           return_depth)
                                   for sltree in self)
        else:
            test_result = [None] * len(self)
            for i, sltree in enumerate(self):
                if verbosity > 0:
                    print '\rTesting SoftLabelTree {}/{}.' \
                          .format(i + 1, len(self)),
                test_result[i] = _parallel_test(sltree, X,
                                                return_complexity,
                                                return_depth)
        complexity_all_trees = None
        depths_all_trees     = None
        if return_complexity or return_depth:
            unzipped_result = zip(*test_result)
            if return_complexity and not return_depth:
                Y_proba_all_trees, complexity_all_trees = unzipped_result
            elif not return_complexity and return_depth:
                Y_proba_all_trees, depths_all_trees = unzipped_result
            else:
                Y_proba_all_trees, complexity_all_trees, depths_all_trees = \
                    unzipped_result
        else:
            Y_proba_all_trees = test_result
        Y_proba    = np.zeros(Y_proba_all_trees[0].shape)
        complexity = None
        depths     = None
        for Y_proba_each_tree in Y_proba_all_trees:
            Y_proba += Y_proba_each_tree
        if complexity_all_trees is not None:
            complexity = np.vstack(complexity_all_trees).T
        if depths_all_trees is not None:
            depths = np.vstack(depths_all_trees).T
        return pack_seq(Y_proba, complexity, depths)

    def test(self, *args, **kwargs):
        ''' Usage:
            Y_label[, ap][, complexity][, depths]
                = test(X, Y=None, param=None,
                       return_complexity=False, return_depth=False,
                       verbosity=0)
        '''
        X, Y, return_complexity, return_depth, verbosity \
            = parse_args(args,
                         (np.ndarray, np.ndarray, bool,  bool,  int),
                         (None,       None,       False, False, 0))
        X                 = kwargs.get('X',                 X)
        Y                 = kwargs.get('Y',                 Y)
        return_complexity = kwargs.get('return_complexity', return_complexity)
        return_depth      = kwargs.get('return_depth',      return_depth)
        verbosity         = kwargs.get('verbosity',         verbosity)
        Y_predict  = None
        ap         = None
        complexity = None
        depths     = None
        if X is not None:
            result = self.test_proba(X, return_complexity,
                                     return_depth, verbosity)
            if return_complexity and return_depth:
                Y_proba, complexity, depths = result
            elif not return_complexity and return_depth:
                Y_proba, depths = result
            elif return_complexity and not return_depth:
                Y_proba, complexity = result
            else:
                Y_proba = result
            Y_predict = np.argmax(Y_proba, axis=-1) + 1
            if Y is not None and Y.shape[0] == X.shape[0]:
                ap = average_precision(Y, Y_proba)
        return pack_seq(Y_predict, ap, complexity, depths)

    def __len__(self):
        return len(self.forest)

    def __getitem__(self, index):
        new_forest = SoftLabelForest(self.gd_param,
                                     self.sgd_param,
                                     self.sd_param,
                                     self.model_param,
                                     self.ovr_param,
                                     self.tree_param,
                                     self.forest_param)
        new_forest.forest = self.forest[index]
        self.avg_num_of_leaves = self._avg_num_of_leaves()
        return new_forest

    def __iter__(self):
        return iter(self.forest)

    def __copy__(self):
        return self[:]

    def __str__(self):
        return 'gd_param.verbosity                             = {gd_repr}\n' \
               'gd_param.show_obj_each_iter                    = {self.gd_param.show_obj_each_iter}\n' \
               'gd_param.show_learning_rate_each_iter          = {self.gd_param.show_learning_rate_each_iter}\n' \
               'gd_param.num_of_iter                           = {self.gd_param.num_of_iter}\n' \
               'gd_param.optimal_error                         = {self.gd_param.optimal_error}\n' \
               'gd_param.init_learning_rate                    = {self.gd_param.init_learning_rate}\n' \
               'gd_param.init_learning_rate_try_1st            = {self.gd_param.init_learning_rate_try_1st}\n' \
               'gd_param.init_learning_rate_try_factor         = {self.gd_param.init_learning_rate_try_factor}\n' \
               'gd_param.init_learning_rate_try_subsample_rate = {self.gd_param.init_learning_rate_try_subsample_rate}\n' \
               'gd_param.init_learning_rate_try_min_sample     = {self.gd_param.init_learning_rate_try_min_sample}\n\n' \
               'sgd_param.size_of_batch                        = {self.sgd_param.size_of_batch}\n\n' \
               'sd_param.verbosity                             = {sd_repr}\n' \
               'sd_param.show_p_each_iter                      = {self.sd_param.show_p_each_iter}\n' \
               'sd_param.num_of_trials                         = {self.sd_param.num_of_trials}\n' \
               'sd_param.num_of_iter_update_p_per_train        = {self.sd_param.num_of_iter_update_p_per_train}\n' \
               'sd_param.num_of_iter_update_p_per_epoch        = {self.sd_param.num_of_iter_update_p_per_epoch}\n' \
               'sd_param.num_of_iter_update_p_per_batch        = {self.sd_param.num_of_iter_update_p_per_batch}\n' \
               'sd_param.num_of_iter_confirm_converge          = {self.sd_param.num_of_iter_confirm_converge}\n\n' \
               'model_param.regularizor                        = {self.model_param.regularizor}\n' \
               'model_param.reg_l1_ratio                       = {self.model_param.reg_l1_ratio}\n' \
               'model_param.bias_learning_rate_factor          = {self.model_param.bias_learning_rate_factor}\n' \
               'model_param.init_var_subsample_rate            = {self.model_param.init_var_subsample_rate}\n' \
               'model_param.init_var_subsample_min             = {self.model_param.init_var_subsample_min}\n\n' \
               'tree_param.min_entropy                         = {self.tree_param.min_entropy}\n' \
               'tree_param.max_depth                           = {self.tree_param.max_depth}\n' \
               'tree_param.min_num_of_samples_per_node         = {self.tree_param.min_num_of_samples_per_node}\n' \
               'tree_param.ovr_in_leaves                       = {self.tree_param.ovr_in_leaves}\n\n' \
               'forest_param.num_of_parallel_jobs              = {self.forest_param.num_of_parallel_jobs}\n' \
               'forest_param.bootstrapping                     = {self.forest_param.bootstrapping}\n\n' \
               'ovr_param.update( {self.ovr_param} )\n\n' \
               'Number of trees in the forest:          {len_self}\n' \
               'Average number of leaves in the forest: {self.avg_num_of_leaves}' \
               .format(self=self,
                       gd_repr=self.gd_param.verbosity.__repr__(),
                       sd_repr=self.sd_param.verbosity.__repr__(),
                       len_self=len(self))

    def _avg_num_of_leaves(self):
        num_of_leaves = 0.
        for tree in self:
            num_of_leaves += tree.num_of_leaves
        return num_of_leaves / len(self)


def _parallel_train(tree, X, Y, bootstrapping):
    I = None
    if bootstrapping:
        I = np.random.randint(X.shape[0], size=X.shape[0])
    tree.train(X, Y, I)
    return tree


def _parallel_test(tree, *args, **kwargs):
    return tree.test_proba(*args, **kwargs)


if __name__ == '__main__':
    pass
