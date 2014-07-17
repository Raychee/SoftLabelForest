import numpy as np
from sklearn.metrics import average_precision_score


def unpack_seq(*args):
    ''' Recursively unpack sequences(tuple/list) until there are no nested
        sequences
    '''
    unpacked = []
    for it in args:
        if isinstance(it, (tuple, list)):
            unpacked.extend(unpack_seq(*it))
        else:
            unpacked.append(it)
    return tuple(unpacked)


def pack_seq(*args, **kwargs):
    ''' tuple(*args) and ignore some values (default is "None" that will be
        ignored).
    '''
    default_value = kwargs.get('ignore', None)
    packed = []
    for it in args:
        if it is not default_value and it != default_value:
            packed.append(it)
    if len(packed) == 1: packed = packed[0]
    else:                packed = tuple(packed)
    return packed


def parse_args(args_to_parse, type_args, default=None):
    ''' Parse and assign args according to different types
        Usage:
        ... = parse_args(args, type_args, default=None)
    '''
    if isinstance(default, tuple):
        parsed_args = list(default) + \
                      [default[-1]] * (len(type_args) - len(default))
    else:
        parsed_args = [default] * len(type_args)
    args_to_parse = list(args_to_parse)
    for i_parsed, type_arg in enumerate(type_args):
        for i_arg, arg in enumerate(args_to_parse):
            if isinstance(arg, type_arg):
                parsed_args[i_parsed] = arg
                del args_to_parse[i_arg]
                break
    return tuple(parsed_args)


def average_precision(Y_truth, Y_predict_score):
    assert Y_truth.shape[0] == Y_predict_score.shape[0]
    num_of_samples = Y_predict_score.shape[0]
    num_of_labels = Y_predict_score.shape[1] if Y_predict_score.ndim > 1 else 1
    if Y_truth.ndim == 1 and num_of_labels > 1:
        Y = Y_truth
        Y_truth = np.zeros(Y_predict_score.shape, dtype=np.int8)
        Y_truth[xrange(num_of_samples), Y - 1] = 1
    posinfs = np.isposinf(Y_predict_score)
    Y_predict_score[posinfs] = 0
    Y_predict_score[posinfs] = np.max(Y_predict_score) + 1
    neginfs = np.isneginf(Y_predict_score)
    Y_predict_score[neginfs] = 0
    Y_predict_score[neginfs] = np.min(Y_predict_score) - 1
    ap = np.empty(num_of_labels)
    for i_label in xrange(num_of_labels):
        ap[i_label] = average_precision_score(Y_truth[:, i_label],
                                              Y_predict_score[:, i_label])
    return ap


if __name__ == '__main__':
    pass
