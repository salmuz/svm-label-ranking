# Authors: Zhou Xingjian
#          Yating Deng
#          Yonatan-Carlos Carranza-Alarcon
# License: BSD 3-Clause


from scipy import sparse, random
from cvxopt import solvers, matrix
from .tools import create_logger
import numpy as np


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
        self.labels = None
        self.W = list()
        self.res_for_each_v = list()
        self.v_list = list([1]) if v_list_default is None else v_list_default
        # debug settings
        self._logger = create_logger("SVMLR", DEBUG)
        solvers.options['show_progress'] = DEBUG

    def learn(self, learn_data_set):
        """
        For each hyper-parameter v in v_list, calculate the label weight of each label.
        :param learn_data_set: arff data set for model training
        :return: list of list: for each v, a list of vector W, weights for each label
        """
        # 0. Getting the number labels and features, and others
        self.labels = learn_data_set.attribute_data['L'][:]
        self.nb_labels = len(self.labels)
        self.nb_var = len(learn_data_set.attribute_data) - 1
        training_data = learn_data_set.data

        # 1. Create list Q: in list Q, for each instance, we stock the arcs possibles of the labels.
        q = self.__stockage_Q(data=training_data)
        length_data = len(training_data)
        W = []
        for i in range(len(self.v_list)):
            W.append([])

        # 2. For each v, we train the model and get the label weights corresponded.
        for v in self.v_list:

            # Get alpha values for the arcs in the Q list, by resolving the dual problem:
            # argmin 0.5*t(alpha)*H*alpha - t(1)*alpha with constraints.
            alpha_list = self.__get_alpha(data=training_data, v=v)
            v_index = self.v_list.index(v)
            for i in range(1, self.nb_labels + 1):

                # Calculate the coefficient of X1, X2, ... (alpha_j)
                wl = 0
                for j in range(1, length_data + 1):

                    part_sum = 0
                    part_reduce = 0

                    # Check each couple in alpha_i, if there's a couple that begins with L_i
                    for couple_index in range(1, len(q[0]) + 1):
                        if 'L' + str(i) == q[j - 1][couple_index - 1][0]:
                            # Search for the value corresponded in get_alpha()
                            alpha = alpha_list[(couple_index - 1) * length_data + j - 1]
                            part_sum = part_sum + alpha
                        if 'L' + str(i) == q[j - 1][couple_index - 1][1]:
                            alpha = alpha_list[(couple_index - 1) * length_data + j - 1]
                            part_reduce = part_reduce + alpha
                    product = np.dot(part_sum - part_reduce, training_data[j - 1][:self.nb_var])
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
            pref = ['L' + str(x + 1) for x in _pref]
            return pref

        res_predictions = []
        for w_v in self.W:
            prediction = []
            wl_list = []

            for instance in data_to_predict:

                for w_label in range(0, len(w_v)):
                    wl_list.append(np.dot(0.5 * w_v[w_label], instance[:self.nb_var]))

                if res_format_string:
                    prediction.append(__get_label_preference_by_weights(wl_list))
                else:
                    list_temp = list(np.argsort(wl_list)[::-1])
                    rank = []
                    for label in range(0, len(list_temp)):
                        point = list_temp.index(label)
                        rank.append(point)
                    labels = ['L' + str(i + 1) for i in range(0, len(rank))]
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
            labels = [i for i in label.split('>')]
            q_list_for_1_instance = SVMLR.__create_q_couples_list(labels)
            Q.append(q_list_for_1_instance)
        return Q

    def __get_alpha(self, data, v):
        """
        :return: list of alpha, size k(k-1)/2
        """
        length_data = len(data)
        # 1. Create the list Q which contains the arcs of labels
        q = self.__stockage_Q(data)

        # 2. Calculate matrix H
        h = self.__calculate_H(q, data)
        h_numpy = h.todense()
        h_numpy = h_numpy.astype(np.double)

        # 3.Set the constraints for the dual problem
        e_i = int(0.5 * self.nb_labels * (self.nb_labels - 1))
        max_limit = float(v / e_i)
        size_H = int(0.5 * self.nb_labels * (self.nb_labels - 1) * length_data)
        res = self.__min_convex_qp(h_numpy,
                                   np.repeat(-1.0, size_H),
                                   np.repeat(0.0, size_H),
                                   np.repeat(max_limit, size_H),
                                   size_H)

        solution = np.array([v for v in res['x']])

        if res['status'] != 'optimal':
            self._logger.info("[Solution-not-Optimal-Not-convergence] v_default (%s)", v)

        return solution

    def __calculate_H(self, q, data):
        """
        :param data: dataArff.data
        :param nb_label: number of labels
        :param q: list Q
        :return: Matrix H
        """
        row_a = []
        col_a = []
        data_a = []
        row_b = []
        col_b = []
        data_b = []
        row_c = []
        col_c = []
        data_c = []
        row_d = []
        col_d = []
        data_d = []

        length_data = len(data)
        for r in range(1, int(self.nb_labels * (self.nb_labels - 1) * 0.5) + 1):
            for l in range(1, int(self.nb_labels * (self.nb_labels - 1) * 0.5) + 1):
                for j in range(0, length_data):
                    for i in range(0, length_data):
                        list_pq = q[i][r - 1]
                        list_ab = q[j][l - 1]

                        if list_pq[0] == list_ab[0]:
                            row_a.append(length_data * (r - 1) + i)
                            col_a.append(length_data * (l - 1) + j)
                            x_i = np.mat(data[i][:self.nb_var])
                            x_j = np.mat(data[j][:self.nb_var])
                            data_a.append((x_i * x_j.T).item())

                        elif list_pq[0] == list_ab[1]:
                            row_b.append(length_data * (r - 1) + i)
                            col_b.append(length_data * (l - 1) + j)
                            x_i = np.mat(data[i][:self.nb_var])
                            x_j = np.mat(data[j][:self.nb_var])
                            data_b.append((x_i * x_j.T).item())

                        elif list_pq[1] == list_ab[0]:
                            row_c.append(length_data * (r - 1) + i)
                            col_c.append(length_data * (l - 1) + j)
                            x_i = np.mat(data[i][:self.nb_var])
                            x_j = np.mat(data[j][:self.nb_var])
                            data_c.append((x_i * x_j.T).item())

                        elif list_pq[1] == list_ab[1]:
                            row_d.append(length_data * (r - 1) + i)
                            col_d.append(length_data * (l - 1) + j)
                            x_i = np.mat(data[i][:self.nb_var])
                            x_j = np.mat(data[j][:self.nb_var])
                            data_d.append((x_i * x_j.T).item())

        size_H = int(0.5 * self.nb_labels * (self.nb_labels - 1) * length_data)
        mat_a = sparse.coo_matrix((data_a, (row_a, col_a)), shape=(size_H, size_H))
        mat_b = sparse.coo_matrix((data_b, (row_b, col_b)), shape=(size_H, size_H))
        mat_c = sparse.coo_matrix((data_c, (row_c, col_c)), shape=(size_H, size_H))
        mat_d = sparse.coo_matrix((data_d, (row_d, col_d)), shape=(size_H, size_H))

        mat_h = mat_a - mat_b - mat_c + mat_d

        return mat_h

    def __min_convex_qp(self, A, q, lower, upper, d):
        ell_lower = matrix(lower, (d, 1))
        ell_upper = matrix(upper, (d, 1))
        P = matrix(A)
        q = matrix(q, (d, 1))
        I = matrix(0.0, (d, d))
        I[::d + 1] = 1
        G = matrix([I, -I])
        h = matrix([ell_upper, -ell_lower])
        solvers.options['refinement'] = 2
        solvers.options['kktreg'] = 1e-9
        return solvers.qp(P=P, q=q, G=G, h=h, kktsolver='ldl', options=solvers.options)