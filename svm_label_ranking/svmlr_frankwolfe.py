from .tools import create_logger, timeit, is_symmetric
from cvxopt import solvers, matrix, spmatrix, sparse
from scipy.sparse import coo_matrix
import numpy as np
from ttictoc import TicToc
import pylab as plt
import array


class SVMLR_FrankWolfe(object):

    def __init__(self, nb_labels, nb_instances, DEBUG=False, DEBUG_SOLVER=False):
        self._logger = create_logger("__SVMLR_FrankWolfe", DEBUG)
        self.nb_labels = nb_labels
        self.nb_instances = nb_instances
        self.nb_preferences = int(self.nb_labels * (self.nb_labels - 1) * 0.5)
        self.d_size = self.nb_preferences * self.nb_instances
        self._t = TicToc("__SVMLR_FrankWolfe")
        self._t.set_print_toc(False)
        solvers.options["show_progress"] = DEBUG_SOLVER
        self._trace_convergence = []
        self.DEBUG = DEBUG

    def get_alpha(self, A, q, v, max_iter=400, tol=1e-8):
        # x. Calculate the large matrix des
        H = self.calculate_H(q, A)
        # self._logger.debug("Is it semi-definite positive matrix (%s)", is_symmetric(H))
        # np.set_printoptions(linewidth=125)
        # print(H.todense())
        # print((H + H.T).todense())
        # np.savetxt("mat_fw.txt", (H + H.T).todense(), fmt='%0.5f')

        # 0. Set the constraints for the dual problem
        e_i = self.nb_preferences
        max_limit = float(v / e_i)

        # 1. Create bound constraint for linear programming
        x_bound_upper = spmatrix(1.0, range(self.d_size), range(self.d_size))
        x_bound_lower = spmatrix(-1.0, range(self.d_size), range(self.d_size))
        G = sparse([x_bound_upper, x_bound_lower])
        h = matrix(np.hstack([np.repeat(max_limit, self.d_size), -np.zeros(self.d_size)]))

        # 2. Frank-Wolf algorithm
        x_t = np.zeros(self.d_size)  # init value for algorithm frank-wolfe
        c = np.repeat(-1.0, self.d_size)
        g_t, it = 0, 0

        for it in range(max_iter):
            grad_fx = H @ x_t + H.T @ x_t + c
            res = solvers.lp(matrix(grad_fx, (self.d_size, 1)), G=G, h=h)

            if res['status'] != 'optimal':
                self._logger.info("[Solution-not-Optimal-Not-convergence] v_default (%s)", v)

            s_t = np.array([v for v in res["x"]])
            d_t = s_t - x_t
            g_t = -1 * (grad_fx.dot(d_t))
            if g_t <= tol:
                break
            Hd_t = H @ d_t + H.T @ d_t
            z_t = d_t.dot(Hd_t)
            step_size = min(-1 * (c.dot(d_t) + x_t.dot(Hd_t)) / z_t, 1.)
            x_t = x_t + step_size * d_t
            if self.DEBUG:
                self._trace_convergence.append(g_t)
                self._logger.debug("Gradient-cost-iteration (it, grad) (%s, %s)", it, g_t)

        # from scipy.optimize import linprog
        # res = linprog(grad_fx, bounds=(0, max_limit), options={"presolve": False})
        self._logger.debug("Cost-Fx-gradient and #iters (grad_fx, iters, is_optimal) (%s, %s, %s)",
                           g_t, it, it + 1 < max_iter)
        return x_t

    def calculate_H(self, q, A):
        rows, cols, data = array.array('i'), array.array('i'), array.array('d')
        self._logger.debug('Size H-matrix (nb_preference, nb_instances, d_size) (%s, %s, %s)',
                           self.nb_preferences, self.nb_instances, self.nb_preferences * self.nb_instances)

        def append(i, j, d):
            rows.append(i)
            cols.append(j)
            data.append(d)

        for r in range(0, self.nb_preferences):
            for l in range(r, self.nb_preferences):
                self._t.tic()
                for i in range(0, self.nb_instances):
                    _i = i if r == l else 0
                    for j in range(_i, self.nb_instances):
                        list_pq = q[i][r]
                        list_ab = q[j][l]
                        # creation index (row, column)
                        i_row = self.nb_instances * r + i
                        i_col = self.nb_instances * l + j
                        cell_data = A[i, j]
                        # put half value to diagonal matrix to use H + H.T
                        if i_row == i_col and r == l:
                            cell_data = 0.5 * cell_data

                        if list_pq[0] == list_ab[0]:
                            append(i_row, i_col, cell_data)

                        elif list_pq[0] == list_ab[1]:
                            append(i_row, i_col, -1 * cell_data)

                        elif list_pq[1] == list_ab[0]:
                            append(i_row, i_col, -1 * cell_data)

                        elif list_pq[1] == list_ab[1]:
                            append(i_row, i_col, cell_data)

                self._logger.debug('Time pair-wise preference label (%s, %s, %s)',
                                   'P' + str(r + 1), 'P' + str(l + 1), self._t.toc())

        rows = np.frombuffer(rows, dtype=np.int32)
        cols = np.frombuffer(cols, dtype=np.int32)
        data = np.frombuffer(data, dtype='d')
        data_coo = coo_matrix((data, (rows, cols)), shape=(self.d_size, self.d_size))
        return data_coo.tocsr()

    def plot_convergence(self):
        plt.plot(self._trace_convergence, lw=1)
        plt.yscale('log')
        plt.xlabel('Number of iterations')
        plt.ylabel('Relative Frank-Wolf gap')
        plt.title('Convergence QP')
        plt.grid()
        plt.show()
