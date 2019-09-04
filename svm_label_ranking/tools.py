# Authors: Yonatan-Carlos Carranza-Alarcon
# License: BSD 3-Clause

import logging
import sys
import random
import numpy as np


def generate_seeds(nb_seeds):
    return [random.randrange(pow(2, 20)) for _ in range(nb_seeds)]


def k_fold_cross_validation(data, K, randomise=False, random_seed=None):
    """
    Generates K (training, validation) pairs from the items in X.
    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    author: Sébastien Destercke
    source: https://github.com/sdestercke/classifip/classifip/evaluation/__init__.py

    :param data: the observed class value
    :type data: :class:`~classifip.dataset.arff.ArffFile`
    :param K: number of folds
    :type K: integer
    :param randomise: randomise or not data set before splitting
    :type randomise: boolean
    :param random_seed: set the seed for the randomisation to reproduce
        identical splits if needed
    :type random_seed: integer
    :returns: iterable over training/evluation pairs
    :rtype: list of :class:`~classifip.dataset.arff.ArffFile`
    """
    if randomise:
        import random
        if random_seed != None:
            random.seed(random_seed)
        random.shuffle(data.data)
    datatr = data.make_clone()
    datatst = data.make_clone()
    for k in range(K):
        datatr.data = [x for i, x in enumerate(data.data) if i % K != k]
        datatst.data = [x for i, x in enumerate(data.data) if i % K == k]
        yield datatr, datatst


def train_test_split(dataArff, test_pct=0.5, random_seed=None):
    """
       Generates partition (training, testing) pairs from the items in X
       author: Sébastien Destercke
       source: https://github.com/sdestercke/classifip/classifip/evaluation/__init__.py
    """
    training = dataArff.make_clone()
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(training.data)
    testing = training.make_clone()
    idx_end_train = int(len(dataArff.data) * (1 - test_pct))
    training.data = training.data[:idx_end_train]
    testing.data = testing.data[idx_end_train:]
    return training, testing


def create_logger(name="default", DEBUG=False):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.flush = sys.stdout.flush
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def correctness_measure(y_true, y_predict):
    """Estimate correctness measure (it is equivalent to the Spearman Footrule)

    correctness_acc = 1 - \frac{\sum_{i=1}^k |r_i - \hat{r}_i|}{0.5*k^2}

    Parameters
    ----------
    y_true : string,  e.g. "L1>L2>L3"
    y_predict : list, e.g. ["L1", "L3", "L2"]

    Returns
    -------
    measure : float,

    Example
    -------
    >>> from svm_label_raking.tools import correctness_measure
    >>> y_true = "L1>L2>L3"
    >>> y_predict = ["L1", "L3", "L2"]
    >>> correctness_measure(y_true, y_predict)
    0.5555555555555556
    """
    assert isinstance(y_predict, list), "y_predict should be a type list"
    y_true = y_true.split(">")
    if y_predict is None:
        return 0.0
    k = len(y_true)
    sum_dist = 0
    for idx, label in enumerate(y_true):
        sum_dist += abs(idx - y_predict.index(label))
    return 1 - sum_dist / (0.5 * k * k)