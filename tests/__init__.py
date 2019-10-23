def calculate_H(self, q, A):  # 9.2 MiB 1219504 1219504
    self._logger.debug('Size H-matrix (nb_preference, nb_instances, d_size) (%s, %s, %s)',
                       self.nb_preferences, self.nb_instances, self.nb_preferences * self.nb_instances)
    data = dok_matrix((self.d_size, self.d_size))

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
                        data[i_row, i_col] += cell_data

                    elif list_pq[0] == list_ab[1]:
                        data[i_row, i_col] -= cell_data

                    elif list_pq[1] == list_ab[0]:
                        data[i_row, i_col] -= cell_data

                    elif list_pq[1] == list_ab[1]:
                        data[i_row, i_col] += cell_data

            self._logger.debug('Time pair-wise preference label (%s, %s, %s)',
                               'P' + str(r + 1), 'P' + str(l + 1), self._t.toc())
    return data.tocsr()

def calculate_H_full(self, q, A):  # 1.6 mb 1620000
    # data_col = dict({})
    self._logger.debug('Size H-matrix (nb_preference, nb_instances, d_size) (%s, %s, %s)',
                       self.nb_preferences, self.nb_instances, self.nb_preferences * self.nb_instances)

    # for i in range(self.d_size):
    #    data_col[i] = np.zeros(self.d_sizei)
    data_col = np.zeros((self.d_size, self.d_size))

    for r in range(1, self.nb_preferences + 1):
        for l in range(r, self.nb_preferences + 1):
            self._t.tic()
            for j in range(0, self.nb_instances):
                _j = j if r == l else 0
                for i in range(_j, self.nb_instances):
                    list_pq = q[i][r - 1]
                    list_ab = q[j][l - 1]
                    i_row = self.nb_instances * (r - 1) + i
                    i_col = self.nb_instances * (l - 1) + j
                    cell_data = A[i, j]
                    x_cr = data_col[i_col]  # column-row
                    x_rc = data_col[i_row]  # row-column

                    if list_pq[0] == list_ab[0]:
                        if i_col == i_row:
                            x_cr[i_row] += cell_data
                        else:
                            x_cr[i_row] += cell_data
                            x_rc[i_col] += cell_data

                    elif list_pq[0] == list_ab[1]:
                        if i_col == i_row:
                            x_cr[i_row] -= cell_data
                        else:
                            x_cr[i_row] -= cell_data
                            x_rc[i_col] -= cell_data

                    elif list_pq[1] == list_ab[0]:
                        if i_col == i_row:
                            x_cr[i_row] -= cell_data
                        else:
                            x_cr[i_row] -= cell_data
                            x_rc[i_col] -= cell_data

                    elif list_pq[1] == list_ab[1]:
                        if i_col == i_row:
                            x_cr[i_row] += cell_data
                        else:
                            x_cr[i_row] += cell_data
                            x_rc[i_col] += cell_data

            self._logger.debug('Time pair-wise preference label (%s, %s, %s)',
                               'P' + str(r), 'P' + str(l), self._t.toc())

    return data_col