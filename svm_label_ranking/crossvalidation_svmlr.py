# Authors: Yonatan-Carlos Carranza-Alarcon
# License: BSD 3-Clause

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


def __computing_training_testing_kfold(DEBUG, train_test_data):
    training, testing = train_test_data
    pid = multiprocessing.current_process().name

    # 1. Loading model with training data set to compute alpha values
    model_svmlr = SVMLR(DEBUG=DEBUG)
    model_svmlr.learn(learn_data_set=training)

    # 2. Computing the prediction and correctness accuracy
    acc_correctness = 0
    nb_tests = len(testing.data)
    for test in testing.data:
        y_prediction = model_svmlr.evaluate(data_to_predict=[test])
        y_true = test[-1]

        # Logging information of prediction and current instance
        print(time.strftime('%x %X %Z'), "(pid, prediction, ground-truth) ",
              pid, y_prediction, y_true, flush=True)

        # to fix for many others v values (now it just code for one v optimal value)
        y_prediction = y_prediction[0]

        # Computing accuracy correctness
        accuracy = correctness_measure(y_true, y_prediction[0])
        acc_correctness = acc_correctness + accuracy / nb_tests

    return acc_correctness


def cross_validation(in_path,
                     out_path,
                     seeds=None,
                     n_times_repeat=10,
                     k_fold_cv=10,
                     nb_process=1,
                     DEBUG=False):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean_sampling", True)
    logger.info("Training data set (%s, %s)", in_path, out_path)
    logger.info(
        "Parameters (n_times_repeat, k_fold_cv, nbProcess) (%s, %s, %s)",
        n_times_repeat,
        k_fold_cv,
        nb_process,
    )

    # Seed for get back up if process is killed
    seeds = generate_seeds(n_times_repeat) if seeds is None else seeds
    logger.debug("SEED: %s", seeds)

    # Create a CSV file for saving results
    file_csv = open(out_path, "w")
    writer = csv.writer(file_csv)

    # Reading the training data set
    data_arff = ArffFile()
    data_arff.load(in_path)
    avg_correctness = np.zeros(n_times_repeat)

    # multiprocessing pool parallel with nb_process
    pool = multiprocessing.Pool(processes=nb_process)
    target_func_train_test = partial(__computing_training_testing_kfold, DEBUG)

    for time in range(0, n_times_repeat):
        cvkfold = k_fold_cross_validation(
            data_arff,
            k_fold_cv,
            randomise=True,
            random_seed=seeds[time]
        )
        # splits = list([])
        # for training, testing in cvkfold:
        #     splits.append((training, testing))
        acc_correctness_kfold = pool.map(target_func_train_test, cvkfold)

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
