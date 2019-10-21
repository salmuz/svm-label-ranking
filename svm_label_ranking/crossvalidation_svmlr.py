# Copyright 2019, Yonatan-Carlos Carranza-Alarcon <salmuz@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import csv
import time
import numpy as np
from .model import SVMLR
from .tools import (
    train_test_split,
    create_logger,
    generate_seeds,
    k_fold_cross_validation,
    correctness_measure,
)
from .arff import ArffFile
import multiprocessing
from functools import partial


def __computing_training_testing_kfold(DEBUG, is_parallel, SOLVER_QP, SOLVER_LP, train_test_data):
    training, testing = train_test_data
    if is_parallel:
        pid = multiprocessing.current_process().name
    else:
        pid = 'ForkPoolWorker-0'

    # 1. Loading model with training data set to compute alpha values
    model_svmlr = SVMLR(DEBUG=DEBUG)
    model_svmlr.learn(learn_data_set=training,
                      is_shared_H_memory=not is_parallel,
                      solver=SOLVER_QP,
                      solver_lp=SOLVER_LP)

    def _pinfo(message, kwargs):
        print("[" + pid + "][" + time.strftime('%x %X %Z') + "]", "-", message % kwargs, flush=True)

    # 2. Computing the prediction and correctness accuracy
    acc_correctness = 0
    nb_tests = len(testing.data)
    for test in testing.data:
        y_prediction = model_svmlr.evaluate(data_to_predict=[test])
        y_true = test[-1]

        # to fix for many others v values (now it just code for one v optimal value)
        y_prediction = y_prediction[0]

        # Computing accuracy correctness
        accuracy = correctness_measure(y_true, y_prediction[0])
        acc_correctness = acc_correctness + accuracy / nb_tests

        # Logging information of prediction and current instance
        _desc_features = ",".join([
            '{0:.18f}'.format(feature).rstrip('0') if str(feature).upper().find("E-") > 0 else str(feature)
            for feature in test[:-1]
        ])
        _pinfo("INSTANCE-COHERENT ( %s ) (prediction, ground-truth, correctness) (%s, %s, %s)",
               (_desc_features, y_prediction, y_true, accuracy))

    print(time.strftime('%x %X %Z'), "(pid, correctness-mean) ", pid, acc_correctness, flush=True)

    return acc_correctness


def cross_validation(in_path,
                     out_path,
                     seeds=None,
                     n_times_repeat=10,
                     k_fold_cv=10,
                     nb_process=1,
                     skip_step_time=0,
                     is_H_shared_memory_disk=False,
                     DEBUG=False,
                     SOLVER_QP='quadratic',
                     SOLVER_LP='cvxopt'):
    """
    10x10 fold cross-validation procedure for experimental results
    :param in_path: the absolute path of learning data set
    :param out_path: the absolute path where putting results
    :param seeds: the list of seeds to repeat experiments
    :param n_times_repeat: Number of times to repeat the cross-validation (by default 10x10 fold-cv)
    :param k_fold_cv: Number of cross-validation (by default 10)
    :param nb_process: Number of process in parallel (by default 1 core)
    :param skip_step_time: How many repetitions want to skip (n_times_repeat)
    :param is_H_shared_memory_disk: shared memory between disk and memory for bigger matrix
                    (frank-wolfe algorithm), however it works for single core.
    :param DEBUG: if you want print the process of optimization problems
    :param SOLVER_QP: 'quadratic' if we use a quadratic solver of cvxopt  or
                   'frank-wolfe' if we use a frank-wolf algorithm
                    (by default, if size matrix is bigger and it is impossible to use cvxopt)
    :param SOLVER_LP: select the linear programing solver to use in the frank-wolfe algorithm
                      (1) cvxopt: convex optimization based in Python, cvxopt.lp(.)
                      (2) scipy: library used for scientific computing, scipy.linprog(.)
                      (3) salmuz: own solution of linear programing, svmlr_frankwolfe.__lp_with_box_constraint(.)
    :return: list of average of correctness (or accuracy) 
    """
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"
    assert skip_step_time < n_times_repeat, "It is not possible skipping most n_times_repeat"
    assert not (seeds is not None and len(seeds) != n_times_repeat), "It is not same size n_times_repeat and seeds."
    assert not (seeds is not None and len(seeds) < skip_step_time), "Skip step must be least than seeds size."

    logger = create_logger("computing_best_imprecise_mean_sampling", True)
    logger.info("Training data set (%s, %s)", in_path, out_path)
    logger.info(
        "Parameters (n_times_repeat, k_fold_cv, nbProcess, skip_step_time) (%s, %s, %s, %s)",
        n_times_repeat,
        k_fold_cv,
        nb_process,
        skip_step_time,
    )

    # Seed for get back up if process is killed
    seeds = generate_seeds(n_times_repeat) if seeds is None else seeds
    logger.debug("SEED: %s", seeds)

    # Create a CSV file for saving results
    file_csv = open(out_path, "a")
    writer = csv.writer(file_csv)

    # Reading the training data set
    data_arff = ArffFile()
    data_arff.load(in_path)
    avg_correctness = np.zeros(n_times_repeat)

    # multiprocessing pool parallel with nb_process
    # if is_H_shared_memory_disk is True, so multiprocessing is disabled
    pool, target_func_train_test = None, None
    is_multiprocessing = (not is_H_shared_memory_disk and nb_process > 1)
    if is_multiprocessing:
        pool = multiprocessing.Pool(processes=nb_process)
        target_func_train_test = partial(__computing_training_testing_kfold,
                                         DEBUG,
                                         SOLVER_QP,
                                         SOLVER_LP,
                                         is_multiprocessing)

    for time in range(skip_step_time, n_times_repeat):
        cvkfold = k_fold_cross_validation(
            data_arff,
            k_fold_cv,
            randomise=True,
            random_seed=seeds[time]
        )
        logger.debug(
            "Learning-time-step (time, seed, nb_process) (%s, %s, %s)",
            time,
            seeds[time],
            nb_process,
        )

        acc_correctness_kfold = []
        if is_multiprocessing:
            acc_correctness_kfold = pool.map(target_func_train_test, cvkfold)
        else:
            for training, testing in cvkfold:
                acc_correctness_kfold.append(__computing_training_testing_kfold(DEBUG,
                                                                                is_multiprocessing,
                                                                                SOLVER_QP,
                                                                                SOLVER_LP,
                                                                                (training, testing)))

        # save and print save partial calculations
        for kfold_time in range(k_fold_cv):
            avg_correctness[time] += acc_correctness_kfold[kfold_time]
            logger.debug(
                "Partial-k-fold_step (time, k-fold, acc_correctness, cum_avg_correctness) (%s, %s, %s, %s)",
                time,
                kfold_time,
                acc_correctness_kfold[kfold_time],
                avg_correctness[time],
            )
            writer.writerow([time, kfold_time, acc_correctness_kfold[kfold_time]])
            file_csv.flush()

        # save time-one cross-validation
        avg_correctness[time] = avg_correctness[time] / k_fold_cv
        logger.debug(
            "Partial-time-step (avg_correctness, time) (%s, %s)",
            time,
            avg_correctness[time],
        )
        writer.writerow([time, -999, avg_correctness[time]])
        file_csv.flush()

    writer.writerow([-9999, -9999, np.mean(avg_correctness)])
    file_csv.close()
    logger.debug(
        "Results Final: (acc, mean, std) (%s, %s, %s)",
        avg_correctness,
        np.mean(avg_correctness),
        np.std(avg_correctness),
    )

    return avg_correctness
