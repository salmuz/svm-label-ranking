# Copyright 2019, Yonatan-Carlos Carranza-Alarcon <salmuz@gmail.com>
# Contributors: Zhou Xingjian and Yating Deng
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


from .tools import create_logger, timeit
import numpy as np
from .svmlr_qp import SVMLR_QP
from .svmlr_frankwolfe import SVMLR_FrankWolfe


class SVMLR(object):
    """
    SVMML implements the Multi label ranking method using the SVM for
    Label ranking problem with LDL decomposition.
    """

    def __init__(self, v_list_default=None, DEBUG=False, DEBUG_SOLVER=False):
        """
        :param v_list_default: list of values v
        """
        self.nb_var = None
        self.nb_labels = None
        self.nb_instances = None
        self.labels = None
        self.nb_preferences = None

        self.W = list()
        self.res_for_each_v = list()
        self.v_list = list([1]) if v_list_default is None else v_list_default
        # debug settings
        self._logger = create_logger("SVMLR", DEBUG)

        self.DEBUG = DEBUG
        self.DEBUG_SOLVER = DEBUG_SOLVER
        self.__solver = None

    def learn(self,
              learn_data_set,
              solver='quadratic',
              is_shared_H_memory=False,
              solver_lp='cvxopt',
              start_idx_label_disk=None):
        """
        For each hyper-parameter v in v_list, calculate the label weight of each label.
        :param learn_data_set: arff data set for model training
        :param solver: 'quadratic' if we use a quadratic solver of cvxopt  or
                       'frank-wolfe' if we use a frank-wolf algorithm
                        (by default, if size matrix is bigger and it is impossible to use cvxopt)
        :param is_shared_H_memory: shared memory between disk and memory for bigger matrix
                        (frank-wolfe algorithm), however it works for single core.
        :param solver_lp: select the linear programing solver to use in the frank-wolfe algorithm
                          (1) cvxopt: convex optimization based in Python, cvxopt.lp(.)
                          (2) scipy: library used for scientific computing, scipy.linprog(.)
                          (3) salmuz: own solution of linear programing, svmlr_frankwolfe.__lp_with_box_constraint(.)
        :param start_idx_label_disk: from which label we save in hard disk (by default half-half)
                                     (@improve to calculate the free memory before to save in disk)
        :return: list of list: for each v, a list of vector W, weights for each label
        """
        # 0. Getting the number labels and features, and others
        self.labels = learn_data_set.attribute_data["L"][:]
        self.nb_labels = len(self.labels)
        self.nb_var = len(learn_data_set.attribute_data) - 1
        training_data = learn_data_set.data
        self.nb_instances = len(training_data)

        # 1. Create list Q: in list Q, for each instance, we stock the arcs possibles of the labels.
        q = self.__stockage_Q(data=training_data)
        length_data = len(training_data)
        W = []
        for i in range(len(self.v_list)):
            W.append([])

        instances = learn_data_set.get_features_matrix()
        A = instances @ instances.T
        # case when matrix if very smaller
        self.nb_preferences = int(self.nb_labels * (self.nb_labels - 1) * 0.5)
        is_not_bigger_H_matrix = (self.nb_preferences * self.nb_instances <= int(15 * 1e+3))
        if is_not_bigger_H_matrix and solver == 'quadratic':
            self.__solver = SVMLR_QP(nb_labels=self.nb_labels,
                                     nb_instances=self.nb_instances,
                                     DEBUG=self.DEBUG,
                                     DEBUG_SOLVER=self.DEBUG_SOLVER)
        elif solver == 'frank-wolfe' or (not is_not_bigger_H_matrix and solver == 'quadratic'):
            self.__solver = SVMLR_FrankWolfe(nb_labels=self.nb_labels,
                                             nb_instances=self.nb_instances,
                                             DEBUG=self.DEBUG,
                                             DEBUG_SOLVER=self.DEBUG_SOLVER,
                                             is_shared_H_memory=is_shared_H_memory,
                                             SOLVER_LP=solver_lp,
                                             startup_idx_save_disk=start_idx_label_disk)
        else:
            raise Exception('Solver has not implemented yet ')

        # self._logger.debug("Features pair-wise inner product\n %s", A)
        # 2. For each v, we train the model and get the label weights corresponded.
        for v in self.v_list:

            # Get alpha values for the arcs in the Q list, by resolving the dual problem:
            # argmin 0.5*t(alpha)*H*alpha - t(1)*alpha with constraints.
            alpha_list = self.__solver.get_alpha(A=A, q=q, v=v)
            v_index = self.v_list.index(v)
            for i in range(1, self.nb_labels + 1):
                # Calculate the coefficient of X1, X2, ... (alpha_j)
                wl = 0
                for j in range(1, length_data + 1):
                    part_sum, part_reduce = 0, 0
                    # Check each couple in alpha_i, if there's a couple that begins with L_i
                    for couple_index in range(1, len(q[0]) + 1):
                        if "L" + str(i) == q[j - 1][couple_index - 1][0]:
                            # Search for the value corresponded in get_alpha()
                            alpha = alpha_list[(couple_index - 1) * length_data + j - 1]
                            part_sum = part_sum + alpha
                        if "L" + str(i) == q[j - 1][couple_index - 1][1]:
                            alpha = alpha_list[(couple_index - 1) * length_data + j - 1]
                            part_reduce = part_reduce + alpha
                    product = (part_sum - part_reduce) * instances[j - 1,]
                    wl = wl + product
                W[v_index].append(wl)
        self.W = W
        return W

    def evaluate(self, data_to_predict, res_format_string=True):
        """
        For each value of v, predicts the result of preference, and stock the result into a list.
        :param data_to_predict: Dataset to be predicted
        :param res_format_string: format of the prediction result
        :return: Preference prediction for every instance (list of list)
        """

        def __get_label_preference_by_weights(weights_list):
            """
            Get label preferences by weights
            :param weights_list: list
            :return: String
            """
            _pref = np.argsort(weights_list)[::-1]
            pref = ["L" + str(x + 1) for x in _pref]
            return pref

        res_predictions = []
        for w_v in self.W:
            prediction = []
            wl_list = []

            for instance in data_to_predict:

                for w_label in range(0, len(w_v)):
                    wl_list.append(np.dot(0.5 * w_v[w_label], instance[: self.nb_var]))

                if res_format_string:
                    prediction.append(__get_label_preference_by_weights(wl_list))
                else:
                    rank = np.argsort(wl_list)[::-1]
                    labels = ["L" + str(i + 1) for i in range(0, len(rank))]
                    dict1 = {}
                    for j in range(len(labels)):
                        dict1[labels[j]] = rank[j]
                    prediction.append(dict1)

                wl_list = []
            res_predictions.append(prediction)
        return res_predictions

    def plot_solver_convergence(self):
        self.__solver.plot_convergence()

    @staticmethod
    def __create_q_couples_list(labels_list):
        """
        build list Q which contains the different arcs of labels
        e.g. L1>L2>L3 ====> [[L1,L2],[L2,L3],[L1,L3]]
        :param labels_list: list
        :return: list of list
        """
        res = []
        nb_labels = len(labels_list)
        for step in range(1, nb_labels):
            for start in range(0, nb_labels):
                if start + step < nb_labels and start < nb_labels - 1:
                    res.append([labels_list[start], labels_list[start + step]])

        return res

    @timeit
    def __stockage_Q(self, data):
        """
        build list Q for all the instances
        :param data: arff data
        :return: list Q
        """
        Q = []
        length_data = len(data)
        for i in range(length_data):
            label = data[i][self.nb_var]
            labels = [i for i in label.split(">")]
            q_list_for_1_instance = SVMLR.__create_q_couples_list(labels)
            Q.append(q_list_for_1_instance)
        return Q
