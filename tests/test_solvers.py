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

import numpy as np
from cvxopt import solvers, matrix, spmatrix, sparse
import pylab as plt

cost_fx_obj = []
trace = []
solvers.options["show_progress"] = False


def min_convex_qp(H, q, lower, upper, d):
    ell_lower = matrix(lower, (d, 1))
    ell_upper = matrix(upper, (d, 1))
    q = matrix(q, (d, 1))
    I = matrix(0.0, (d, d))
    I[:: d + 1] = 1
    G = matrix([I, -I])
    h = matrix([ell_upper, -ell_lower])
    solvers.options["refinement"] = 2
    solvers.options["kktreg"] = 1e-9
    # https://groups.google.com/forum/#!msg/cvxopt/Umcrj8UD20g/iGY4z5YgDAAJ
    return solvers.qp(P=matrix(H, (d, d)), q=q, G=G, h=h, kktsolver="ldl", options=solvers.options)


def qd_frank_wolf(query, lower, upper, H, d):
    # 1. Create bound constraint for linear programming
    x_bound_upper = spmatrix(1.0, range(d), range(d))
    x_bound_lower = spmatrix(-1.0, range(d), range(d))
    G = sparse([x_bound_upper, x_bound_lower])
    h = matrix(np.hstack([upper, -lower]))

    # 2. Frank-Wolf algorithm
    tol = 1e-8
    x_t = np.ones(d)
    max_iter = 400

    for it in range(max_iter):
        grad_fx = x_t.T @ H + query
        res = solvers.lp(matrix(grad_fx, (d, 1)), G=G, h=h)

        if res['status'] != 'optimal':
            print("[Solution-not-Optimal-Not-convergence] v_default (%s)" % res)

        s_t = np.array([v for v in res["x"]])
        d_t = s_t - x_t
        g_t = -1 * (grad_fx.dot(d_t))
        # print("Gradient cost fx converge:", g_t, flush=True)
        if g_t <= tol:
            break
        Hd_t = d_t.T @ H
        z_t = d_t.dot(Hd_t)
        step_size = min(-1 * (query.dot(d_t) + x_t.dot(Hd_t)) / z_t, 1.)
        x_t = x_t + step_size * d_t
        trace.append(g_t)
    print("Nb. of iterations", it, g_t)
    return x_t


def __brute_force_search(query, lower, upper, H, d):
    def cost_Fx(x, query, H):
        return 0.5 * (x.T @ H @ x) + query.T @ x

    def forRecursive(lowers, uppers, level, L, optimal):
        for current in np.array([lowers[level], uppers[level]]):
            if level < L - 1:
                forRecursive(lowers, uppers, level + 1, L, np.append(optimal, current))
            else:
                sub_opt = cost_Fx(np.append(optimal, current), query, H)
                cost_fx_obj.append(sub_opt)
                print("optimal value cost : ", len(cost_fx_obj) - 1, np.append(optimal, current), sub_opt, flush=True)

    forRecursive(lower, upper, 0, d, np.array([]))


Q = np.array([[19.42941344, -12.9899322, -5.1907171, -0.25782677],
              [-12.9899322, 15.97805787, 1.87087712, -6.72150886],
              [-5.1907171, 1.87087712, 36.99333345, -16.21139038],
              [-0.25782677, -6.72150886, -16.21139038, 103.0762929]])

q = -1 * np.ones(4)  # (375) np.array([-45.3553788, 26.52058282, -99.63769322, -61.59361441])
mean_lower = np.zeros(4)  # (one iteration) np.array([4.94791667, 3.36875, 1.41666667, 0.19375])
mean_upper = np.array([5.04375, 3.46458333, 1.5125, 0.28958333])

__brute_force_search(q, mean_lower, mean_upper, Q, 4)
print(cost_fx_obj.index(min(cost_fx_obj)))
rs = min_convex_qp(Q, q, mean_lower, mean_upper, 4)
print([x for x in rs['x']])
print(qd_frank_wolf(q, mean_lower, mean_upper, Q, 4))

plt.plot(trace, lw=1)
plt.yscale('log')
plt.xlabel('Number of iterations')
plt.ylabel('Relative FW gap')
plt.title('Convergence QP')
plt.grid()
plt.show()
