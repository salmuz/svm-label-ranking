# Authors: Yonatan-Carlos Carranza-Alarcon
# License: BSD 3-Clause

import os
import csv
import numpy as np
from .model import SVMLR
from .tools import train_test_split, create_logger, generate_seeds, k_fold_cross_validation, correctness_measure
from .multitask import ManagerWorkers
from .arff import ArffFile


def __computing_training_testing_step(manager, learn_data_set, test_data_set, acc_correctness):
    # init additional parameters
    nb_tests = len(test_data_set.data)

    # Send training data model to every parallel process
    manager.addNewTraining(learn_data_set=learn_data_set)

    # Send testing data to every parallel process
    for test in test_data_set.data:
        manager.addTask({'kwargs': {'data_to_predict': [test], 'res_format_string': True}, 'y_test': test[-1]})

    # Putting poison pill all workers in order to release memory and process finished
    manager.poisonPillWorkers()
    manager.joinTraining()  # wait all process for computing results

    # Recovery all inference data of all parallel process
    shared_results = manager.getResults()
    shared_results.put('STOP')  # stop loop queue (poison pill)
    for utility in iter(shared_results.get, 'STOP'):
        y_prediction = utility['prediction']
        y_true = utility['ground_truth']

        # to fix for many others v values (now it just code for one v optimal value)
        y_prediction = y_prediction[0]

        accuracy = correctness_measure(y_true, y_prediction[0])
        acc_correctness = acc_correctness + accuracy / nb_tests

    return acc_correctness


def cross_validation(in_path,
                     out_path,
                     seeds=None,
                     n_times_repeat=10,
                     k_fold_cv=10,
                     nb_process=1):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean_sampling", True)
    logger.info('Training data set (%s, %s)', in_path, out_path)
    logger.info('Parameters (n_times_repeat, k_fold_cv, nbProcess) (%s, %s, %s)',
                n_times_repeat, k_fold_cv, nb_process)

    # Seed for get back up if process is killed
    seeds = generate_seeds(n_times_repeat) if seeds is None else seeds
    logger.debug("SEED: %s", seeds)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'w')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process)
    manager.executeAsync(class_model="svm_label_ranking.model.SVMLR")

    # Reading the training data set
    data_arff = ArffFile()
    data_arff.load(in_path)
    avg_correctness = np.zeros(n_times_repeat)

    for time in range(0, n_times_repeat):
        cvkfold = k_fold_cross_validation(data_arff, k_fold_cv, randomise=True, random_seed=seeds[time])
        k_fold = 0

        for training, testing in cvkfold:
            acc_correctness_kfold = __computing_training_testing_step(manager, training, testing, 0)
            # save and print save partial calculations
            avg_correctness[time] += acc_correctness_kfold
            logger.debug("Partial-k-fold_step (time, k-fold, acc_correctness, cum_avg_correctness) (%s, %s, %s, %s)",
                         time, k_fold, acc_correctness_kfold, avg_correctness[time])
            writer.writerow([time, k_fold, acc_correctness_kfold])
            file_csv.flush()
            k_fold += 1

        avg_correctness[time] = avg_correctness[time] / k_fold_cv
        logger.debug("Partial-time-step (avg_correctness, time) (%s, %s)", time, avg_correctness[time])
        writer.writerow([time, -999, avg_correctness[time]])
        file_csv.flush()

    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: (acc, mean, std) (%s, %s, %s)", avg_correctness,
                 np.mean(avg_correctness), np.std(avg_correctness))
    return avg_correctness
