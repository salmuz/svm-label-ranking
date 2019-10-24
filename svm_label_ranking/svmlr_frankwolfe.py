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

from .tools import create_logger, timeit, is_symmetric
from .create_H_matrix_disk_memory import sparse_matrix_H_shared_memory_and_disk, init_shared_H, dot_xt_Hr_preference, \
    dot_xt_Hr_from_disk_hard
from cvxopt import solvers, matrix, spmatrix, sparse
from scipy.sparse import coo_matrix, load_npz
from scipy.optimize import linprog
import numpy as np
from ttictoc import TicToc
import pylab as plt
import array
import time
import os
import multiprocessing
from functools import partial


class SVMLR_FrankWolfe(object):

    def __init__(self,
                 nb_labels,
                 nb_instances,
                 DEBUG=False,
                 DEBUG_SOLVER=False,
                 is_shared_H_memory=False,
                 SOLVER_LP='cvxopt',
                 startup_idx_save_disk=None,
                 process_dot_Hxt=1):
        self._logger = create_logger("__SVMLR_FrankWolfe", DEBUG)
        self.nb_labels = nb_labels
        self.nb_instances = nb_instances
        self.nb_preferences = int(self.nb_labels * (self.nb_labels - 1) * 0.5)
        self.d_size = self.nb_preferences * self.nb_instances
        self._t = TicToc("__SVMLR_FrankWolfe")
        self._t.set_print_toc(False)
        solvers.options["show_progress"] = DEBUG_SOLVER
        self._trace_convergence = []
        self.is_shared_H_memory = is_shared_H_memory
        self.DEBUG = DEBUG
        # variables to create matrix H in parallel and shared memory and disk
        self.name_matrix_H = "sparse_H_" + str(int(time.time()))
        self.in_temp_path = "/tmp/"
        self.startup_idx_save_disk = int(self.nb_preferences * 0.5 + 1) \
            if startup_idx_save_disk is None \
            else startup_idx_save_disk
        self.SOLVER_LP_DEFAULT = SOLVER_LP
        self.pool = None
        self.process_dot_Hxt = process_dot_Hxt

    def get_alpha(self, A, q, v, max_iter=100, tol=1e-8):
        # 0. calculate the large matrix des
        # it is shared memory, we use the H global variable to speed to multiplication bigger matrix
        H = self.calculate_H(q, A)
        # self._logger.debug("Is it semi-definite positive matrix (%s)", is_symmetric(H))
        if self.is_shared_H_memory:
            self.pool = multiprocessing.Pool(processes=self.process_dot_Hxt,
                                             initializer=init_shared_H,
                                             initargs=(H,))

        # 1. Set the constraints for the dual problem
        e_i = self.nb_preferences
        max_limit = float(v / e_i)

        # 2. Call wrapper linear programing solver
        lp_programing = self.__wrapper_lp_solvers(0, max_limit, solver=self.SOLVER_LP_DEFAULT)

        # 3. Frank-Wolf algorithm
        x_t = np.zeros(self.d_size)  # init value for algorithm frank-wolfe
        c = np.repeat(-1.0, self.d_size)
        g_t, it = 0, 0

        for it in range(max_iter):
            # Step 0: compute gradient of the sparse matrix
            grad_fx = self.compute_H_dot_x_grad(x_t, H, c)

            # Step 1: direction-finding sub-problem
            s_t = lp_programing(grad_fx)
            d_t = s_t - x_t
            g_t = -1 * (grad_fx.dot(d_t))
            # verify if gap is below tolerance
            if g_t <= tol:
                break
            # Step 2: set step size by line search
            Hd_t = self.compute_H_dot_x_grad(d_t, H)
            z_t = d_t.dot(Hd_t)
            step_size = min(-1 * (c.dot(d_t) + x_t.dot(Hd_t)) / z_t, 1.)
            # Step 3: update current value
            x_t = x_t + step_size * d_t
            if self.DEBUG:
                self._trace_convergence.append(g_t)
                self._logger.debug("Gradient-cost-iteration (it, grad) (%s, %s)", it, g_t)

        self._logger.debug("Cost-Fx-gradient and #iters (grad_fx, iters, is_optimal) (%s, %s, %s)",
                           g_t, it, it + 1 < max_iter)
        return x_t

    def compute_H_dot_x_grad(self, x_t, H, add_vec=None):
        # Gradient evaluate in current value x_t
        grad_fx = np.zeros(self.d_size)
        if self.is_shared_H_memory:  # multiprocessing inner product space
            if self.process_dot_Hxt > 1:
                # multiprocessing dot calculation
                fnc_target = partial(dot_xt_Hr_preference, x_t)
                res = self.pool.map(fnc_target, range(0, self.startup_idx_save_disk - 1))
                for x_memory in res:
                    grad_fx = grad_fx + x_memory

                func_target = partial(dot_xt_Hr_from_disk_hard, x_t, self.name_matrix_H, self.in_temp_path)
                res = self.pool.map(func_target, range(self.startup_idx_save_disk - 1, self.nb_preferences))
                for x_r in res:
                    grad_fx = grad_fx + x_r
            else:  # singleton processing
                for r in range(0, self.startup_idx_save_disk - 1):
                    grad_fx = grad_fx + H[r] @ x_t + H[r].T @ x_t

                for r in range(self.startup_idx_save_disk - 1, self.nb_preferences):
                    H_disk = load_npz(self.in_temp_path + self.name_matrix_H + "_" + str(r + 1) + ".npz")
                    x_disk = H_disk @ x_t + H_disk.T @ x_t
                    grad_fx = grad_fx + x_disk
        else:
            grad_fx = H @ x_t + H.T @ x_t

        if add_vec is not None:
            grad_fx = grad_fx + add_vec

        return grad_fx

    def calculate_H(self, q, A):
        if self.is_shared_H_memory:
            return sparse_matrix_H_shared_memory_and_disk(q, A,
                                                          self.nb_preferences,
                                                          self.nb_instances,
                                                          self.name_matrix_H,
                                                          self.startup_idx_save_disk,
                                                          self.in_temp_path)
        else:
            return self.all_sparse_symmetric_H(q, A)

    def all_sparse_symmetric_H(self, q, A):
        self._logger.debug('Size H-matrix (nb_preference, nb_instances, d_size) (%s, %s, %s)',
                           self.nb_preferences, self.nb_instances, self.nb_preferences * self.nb_instances)

        data_coo = None
        for r in range(0, self.nb_preferences):
            rows, cols, data = array.array('i'), array.array('i'), array.array('d')

            def append(i, j, d):
                rows.append(i)
                cols.append(j)
                data.append(d)

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
            if data_coo is not None:
                data.extend(data_coo.data)
                rows.extend(data_coo.row)
                cols.extend(data_coo.col)
            rows = np.frombuffer(rows, dtype=np.int32)
            cols = np.frombuffer(cols, dtype=np.int32)
            data = np.frombuffer(data, dtype='d')
            data_coo = coo_matrix((data, (rows, cols)), shape=(self.d_size, self.d_size))

        return data_coo.tocsr()

    def __wrapper_lp_solvers(self, lower_bound, upper_bound, solver='cvxopt'):
        self._logger.debug("Linear solver selected (%s)", solver)
        if solver == 'cvxopt':
            def __executing(grad_fx):
                res = solvers.lp(matrix(grad_fx, (self.d_size, 1)), G=G, h=h)
                if res['status'] != 'optimal':
                    self._logger.info("[Solution-not-Optimal-Not-convergence] v_default (%s)", v)
                return np.array([v for v in res["x"]])

            # 1. Create bound constraint for linear programming
            x_bound_upper = spmatrix(1.0, range(self.d_size), range(self.d_size))
            x_bound_lower = spmatrix(-1.0, range(self.d_size), range(self.d_size))
            G = sparse([x_bound_upper, x_bound_lower])
            h = matrix(np.hstack([np.repeat(upper_bound, self.d_size), -np.zeros(self.d_size)]))
            return __executing
        elif solver == 'scipy':
            def __executing(grad_fx):
                res = linprog(grad_fx, bounds=(lower_bound, upper_bound))
                if res['status'] != 0:
                    self._logger.info("[Solution-not-Optimal-Not-convergence] v_default (%s)", v)
                return np.array([v for v in res["x"]])

            return __executing
        elif solver == 'salmuz':
            def __lp_with_box_constraint(c):
                lp_solution_optimal = np.zeros(c.shape)
                idx_negative_value = np.where(c < 0)[0]
                if len(idx_negative_value) == 0:
                    return lp_solution_optimal
                lp_solution_optimal[idx_negative_value] = upper_bound
                return lp_solution_optimal

            return __lp_with_box_constraint
        else:
            raise Exception('Solver has not implemented yet')

    def plot_convergence(self):
        plt.plot(self._trace_convergence, lw=1)
        plt.yscale('log')
        plt.xlabel('Number of iterations')
        plt.ylabel('Relative Frank-Wolf gap')
        plt.title('Convergence QP')
        plt.grid()
        plt.show()

    def __del__(self):
        # remove temporal files where it save the sparse matrix
        if self.is_shared_H_memory:
            for r in range(self.startup_idx_save_disk - 1, self.nb_preferences):
                os.remove(self.in_temp_path + self.name_matrix_H + "_" + str(r + 1) + ".npz")
        # release pool process after computations
        if self.pool is not None:
            self.pool.close()
