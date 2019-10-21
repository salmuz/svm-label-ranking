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

from .tools import create_logger, timeit
from cvxopt import solvers, matrix, spmatrix, sparse
import numpy as np
from ttictoc import TicToc


class SVMLR_QP(object):

    def __init__(self, nb_labels, nb_instances, DEBUG=False, DEBUG_SOLVER=False):
        self._logger = create_logger("__SVMLR_QP", DEBUG)
        self.nb_labels = nb_labels
        self.nb_instances = nb_instances
        self._t = TicToc("__SVMLR_QP")
        self._t.set_print_toc(False)
        solvers.options["show_progress"] = DEBUG_SOLVER

    @timeit
    def get_alpha(self, A, q, v):
        """
        :return: list of alpha, size k(k-1)/2
        """
        # 1. Calculate matrix H
        # h = self.old_calculate_H(q, data)
        h_numpy = self.calculate_H(q, A)
        # np.set_printoptions(linewidth=125)
        # print(np.array(matrix(h_numpy)))
        # np.savetxt("mat_qd.txt", np.array(matrix(h_numpy)), fmt='%0.5f')

        # 2.Set the constraints for the dual problem
        e_i = int(0.5 * self.nb_labels * (self.nb_labels - 1))
        max_limit = float(v / e_i)
        size_H = int(0.5 * self.nb_labels * (self.nb_labels - 1) * self.nb_instances)
        res = self.min_convex_qp(
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
    def calculate_H(self, q, A):
        """
        :param A: numpy array
        :param q: list Q
        :return: Matrix H
        """
        row, col, data = [], [], []
        nb_preferences = int(self.nb_labels * (self.nb_labels - 1) * 0.5)

        self._logger.debug('Size H-matrix (%s, %s, %s)', nb_preferences,
                           self.nb_instances, nb_preferences * self.nb_instances)
        for r in range(0, nb_preferences):
            for l in range(r, nb_preferences):
                self._t.tic()
                for j in range(0, self.nb_instances):
                    _j = j if r == l else 0
                    for i in range(_j, self.nb_instances):
                        list_pq = q[i][r]
                        list_ab = q[j][l]
                        # creation index (row, column)
                        i_row = self.nb_instances * r + i
                        i_col = self.nb_instances * l + j
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
                                   'P' + str(r+1), 'P' + str(l+1), self._t.toc())

        size_H = int(nb_preferences * self.nb_instances)
        mat_h = spmatrix(data, row, col, size=(size_H, size_H))
        # self._logger.debug("Full matrix(mat_a)\n %s", mat_a)
        # for verification with old version
        # np.savetxt("mat_h.txt", matrix(mat_h), fmt='%0.3f')

        return mat_h

    def min_convex_qp(self, H, q, lower, upper, d):
        ell_lower = matrix(lower, (d, 1))
        ell_upper = matrix(upper, (d, 1))
        q = matrix(q, (d, 1))
        I = matrix(0.0, (d, d))
        I[:: d + 1] = 1
        G = matrix([I, -I])
        h = matrix([ell_upper, -ell_lower])
        # solvers.options["refinement"] = 2
        solvers.options["kktreg"] = 1e-9
        # https://groups.google.com/forum/#!msg/cvxopt/Umcrj8UD20g/iGY4z5YgDAAJ
        return solvers.qp(P=H, q=q, G=G, h=h, kktsolver="ldl", options=solvers.options)

    def plot_convergence(self):
        raise Exception("Not implemented yet")
