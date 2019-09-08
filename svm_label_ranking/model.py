# Authors: Zhou Xingjian
#          Yating Deng
#          Yonatan-Carlos Carranza-Alarcon
# License: BSD 3-Clause


from cvxopt import solvers, matrix, spmatrix
from .tools import create_logger, timeit
import numpy as np
from ttictoc import TicToc


class SVMLR(object):
    """
    SVMML implements the Multi label ranking method using the SVM for
    Label ranking problem with LDL decomposition.
    """

    def __init__(self, v_list_default=None, DEBUG=False):
        """
        :param v_list_default: list of values v
        """
        self.nb_var = None
        self.nb_labels = None
        self.nb_instances = None
        self.labels = None
        self.W = list()
        self.res_for_each_v = list()
        self.v_list = list([1]) if v_list_default is None else v_list_default
        # debug settings
        self._logger = create_logger("SVMLR", DEBUG)
        solvers.options["show_progress"] = DEBUG
        self._t = TicToc("SVMLR")
        self._t.set_print_toc(False)
        self.DEBUG = DEBUG

    def learn(self, learn_data_set):
        """
        For each hyper-parameter v in v_list, calculate the label weight of each label.
        :param learn_data_set: arff data set for model training
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

        A = learn_data_set.get_features_matrix()
        A = A @ A.T
        # self._logger.debug("Features pair-wise inner product\n %s", A)
        # 2. For each v, we train the model and get the label weights corresponded.
        for v in self.v_list:

            # Get alpha values for the arcs in the Q list, by resolving the dual problem:
            # argmin 0.5*t(alpha)*H*alpha - t(1)*alpha with constraints.
            alpha_list = self.__get_alpha(A=A, q=q, v=v)
            v_index = self.v_list.index(v)
            for i in range(1, self.nb_labels + 1):

                # Calculate the coefficient of X1, X2, ... (alpha_j)
                wl = 0
                for j in range(1, length_data + 1):

                    part_sum = 0
                    part_reduce = 0

                    # Check each couple in alpha_i, if there's a couple that begins with L_i
                    for couple_index in range(1, len(q[0]) + 1):
                        if "L" + str(i) == q[j - 1][couple_index - 1][0]:
                            # Search for the value corresponded in get_alpha()
                            alpha = alpha_list[(couple_index - 1) * length_data + j - 1]
                            part_sum = part_sum + alpha
                        if "L" + str(i) == q[j - 1][couple_index - 1][1]:
                            alpha = alpha_list[(couple_index - 1) * length_data + j - 1]
                            part_reduce = part_reduce + alpha
                    product = np.dot(
                        part_sum - part_reduce, training_data[j - 1][: self.nb_var]
                    )
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
                    list_temp = list(np.argsort(wl_list)[::-1])
                    rank = []
                    for label in range(0, len(list_temp)):
                        point = list_temp.index(label)
                        rank.append(point)
                    labels = ["L" + str(i + 1) for i in range(0, len(rank))]
                    dict1 = {}
                    for j in range(len(labels)):
                        dict1[labels[j]] = rank[j]
                        prediction.append([dict1])

                wl_list = []
            res_predictions.append(prediction)
        return res_predictions

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

    @timeit
    def __get_alpha(self, A, q, v):
        """
        :return: list of alpha, size k(k-1)/2
        """
        # 1. Calculate matrix H
        # h = self.old_calculate_H(q, data)
        h_numpy = self.__calculate_H(q, A)
        # self._logger.debug("Full matrix\n %s", h.todense())

        # 2.Set the constraints for the dual problem
        e_i = int(0.5 * self.nb_labels * (self.nb_labels - 1))
        max_limit = float(v / e_i)
        size_H = int(0.5 * self.nb_labels * (self.nb_labels - 1) * self.nb_instances)
        res = self.__min_convex_qp(
            h_numpy,
            np.repeat(-1.0, size_H),
            np.repeat(0.0, size_H),
            np.repeat(max_limit, size_H),
            size_H,
        )

        solution = np.array([v for v in res["x"]])

        if res['status'] != 'optimal':
            self._logger.info("[Solution-not-Optimal-Not-convergence] v_default (%s)", v)

        return solution

    @timeit
    def __calculate_H(self, q, A):
        """
        :param A: numpy array
        :param q: list Q
        :return: Matrix H
        """
        row, col, data = [], [], []
        nb_preferences = int(self.nb_labels * (self.nb_labels - 1) * 0.5)

        self._logger.debug('Size H-matrix (%s, %s, %s)', nb_preferences,
                           self.nb_instances, nb_preferences * self.nb_instances)
        for r in range(1, nb_preferences + 1):
            for l in range(r, nb_preferences + 1):
                self._t.tic()
                for j in range(0, self.nb_instances):
                    _j = j if r == l else 0
                    for i in range(_j, self.nb_instances):
                        list_pq = q[i][r - 1]
                        list_ab = q[j][l - 1]
                        i_row = self.nb_instances * (r - 1) + i
                        i_col = self.nb_instances * (l - 1) + j
                        cell_data = A[i, j]

                        if list_pq[0] == list_ab[0]:
                            if i_col == i_row:
                                row.append(i_row)
                                col.append(i_col)
                                data.append(cell_data)
                            else:
                                row.extend((i_row, i_col))
                                col.extend((i_col, i_row))
                                data.extend((cell_data, cell_data))

                        elif list_pq[0] == list_ab[1]:
                            if i_col == i_row:
                                row.append(i_row)
                                col.append(i_col)
                                data.append(-1 * cell_data)
                            else:
                                row.extend((i_row, i_col))
                                col.extend((i_col, i_row))
                                data.extend((-1 * cell_data, -1 * cell_data))

                        elif list_pq[1] == list_ab[0]:
                            if i_col == i_row:
                                row.append(i_row)
                                col.append(i_col)
                                data.append(-1 * cell_data)
                            else:
                                row.extend((i_row, i_col))
                                col.extend((i_col, i_row))
                                data.extend((-1 * cell_data, -1 * cell_data))

                        elif list_pq[1] == list_ab[1]:
                            if i_col == i_row:
                                row.append(i_row)
                                col.append(i_col)
                                data.append(cell_data)
                            else:
                                row.extend((i_row, i_col))
                                col.extend((i_col, i_row))
                                data.extend((cell_data, cell_data))
                self._logger.debug('Time pair-wise preference label (%s, %s, %s)',
                                   'P' + str(r), 'P' + str(l), self._t.toc())

        size_H = int(nb_preferences * self.nb_instances)
        mat_h = spmatrix(data, row, col, size=(size_H, size_H))
        # self._logger.debug("Full matrix(mat_a)\n %s", mat_a)
        # for verification with old version
        # np.savetxt("mat_h.txt", matrix(mat_h), fmt='%0.3f')

        return mat_h

    def __min_convex_qp(self, H, q, lower, upper, d):
        ell_lower = matrix(lower, (d, 1))
        ell_upper = matrix(upper, (d, 1))
        q = matrix(q, (d, 1))
        I = matrix(0.0, (d, d))
        I[:: d + 1] = 1
        G = matrix([I, -I])
        h = matrix([ell_upper, -ell_lower])
        solvers.options["refinement"] = 2
        solvers.options["kktreg"] = 1e-9
        return solvers.qp(P=H, q=q, G=G, h=h, kktsolver="ldl", options=solvers.options)
