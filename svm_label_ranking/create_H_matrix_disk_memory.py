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

from scipy.sparse import coo_matrix, save_npz, load_npz
import array
from ttictoc import TicToc
import numpy as np
import multiprocessing
from functools import partial
import time
from threading import Thread


# https://stackoverflow.com/a/6894023/784555
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def dot_xt_Hr_from_disk_hard(x_t, name_matrix_H, in_temp_path, r):
    file_name = in_temp_path + name_matrix_H + "_" + str(r + 1) + ".npz"
    H_disk = load_npz(file_name)
    return H_disk @ x_t + H_disk.T @ x_t


def __parallel_create_H_r_l(q, A, r, l, iis):
    nb_instances, _ = A.shape
    rows, cols, data = array.array('i'), array.array('i'), array.array('d')

    def append(i, j, d):
        rows.append(i)
        cols.append(j)
        data.append(d)

    print("[" + multiprocessing.current_process().name + ":" + time.strftime('%x %X %Z') + "] Starting worker",
          (len(iis), r, l), flush=True)
    for i in iis:
        _i = i if r == l else 0
        for j in range(_i, nb_instances):
            list_pq = q[i][r]
            list_ab = q[j][l]
            # creation index (row, column)
            i_row = nb_instances * r + i
            i_col = nb_instances * l + j
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

    print("[" + multiprocessing.current_process().name + ":" + time.strftime('%x %X %Z') + "] Finished worker",
          (len(iis), r, l), flush=True)
    return rows, cols, data


def sparse_matrix_H_shared_memory_and_disk(q, A,
                                           nb_preferences,
                                           nb_instances,
                                           name,
                                           startup_idx_save_disk,
                                           in_temp_path,
                                           nb_process=2):
    _t = TicToc("sparse_matrix_H_shared_memory_and_disk")
    _t.set_print_toc(False)
    H = dict({})

    print('Size H-matrix (nb_preference, nb_instances, d_size) (%s, %s, %s)' %
          (nb_preferences, nb_instances, nb_preferences * nb_instances), flush=True)

    def __save_data_matrix(data, rows, cols, d_size, r):
        data_coo = coo_matrix((data, (rows, cols)), shape=(d_size, d_size))
        if startup_idx_save_disk - 1 > r:
            H[r] = data_coo.tocsr()
        else:
            print("Saving pair-wise preference label (%s)" % ('P' + str(r + 1)), flush=True)
            save_npz(file=in_temp_path + name + "_" + str(r + 1) + ".npz", matrix=data_coo.tocsr())
        return True

    singleThread = None
    pool = multiprocessing.Pool(processes=nb_process)
    d_size = nb_preferences * nb_instances
    for r in range(0, nb_preferences):
        rows, cols, data = array.array('i'), array.array('i'), array.array('d')
        for l in range(r, nb_preferences):
            _t.tic()
            parallel_create_sub_matrix = partial(__parallel_create_H_r_l, q, A, r, l)
            modulo = nb_instances % nb_process
            iis = np.split(np.arange(nb_instances - modulo), nb_process)
            iis[nb_process - 1] = np.append(iis[nb_process - 1], np.arange(nb_instances - modulo, nb_instances))
            sparse_infos = pool.map(parallel_create_sub_matrix, iis)
            for rs, cs, dat in sparse_infos:
                rows.extend(rs)
                cols.extend(cs)
                data.extend(dat)
            print("Time pair-wise preference label (%s, %s, %s)" %
                  ('P' + str(r + 1), 'P' + str(l + 1), _t.toc()), flush=True)

        rows = np.frombuffer(rows, dtype=np.int32)
        cols = np.frombuffer(cols, dtype=np.int32)
        data = np.frombuffer(data, dtype='d')

        if singleThread is not None:
            singleThread.join()
        singleThread = ThreadWithReturnValue(target=__save_data_matrix, args=(data, rows, cols, d_size, r))
        singleThread.start()

    if singleThread is not None:
        singleThread.join()
    pool.close()
    return H
