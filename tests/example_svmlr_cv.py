from svm_label_ranking.crossvalidation_svmlr import cross_validation

# SEED_SCIENTIFIC = [1234]
# # Call cross-validation with single parallel process
# in_path = "iris_dense.xarff"
# out_path = "results_output.csv"
# avg_correctness = cross_validation(in_path, out_path, nb_process=1, n_times_repeat=1,
#                                    seeds=SEED_SCIENTIFIC, DEBUG=False)
# print("Avg. correctness IRIS with 1 process parallel execution:", avg_correctness, flush=True)
#
# # Call cross-validation with three parallel process
# in_path = "iris_dense.xarff"
# out_path = "results_output.csv"
# avg_correctness = cross_validation(in_path, out_path, nb_process=3, DEBUG=True)
# print("Avg. correctness IRIS with 3 process parallel execution:", avg_correctness, flush=True)

# Call cross-validation with shared memory and disk for saving H matrix
seeds = [717921, 838036, 190032, 772310, 220468, 310493, 181809, 776463, 733935, 786241]
in_path = "iris_dense.xarff"
# in_path = "/Users/salmuz/Downloads/datasets_rang/segment_dense.xarff"
out_path = "results_output.csv"
avg_correctness = cross_validation(in_path, out_path, DEBUG=True, is_H_shared_memory_disk=True, seeds=seeds,
                                   SOLVER_LP="salmuz", SOLVER_QP="frank-wolfe", start_idx_label_disk=2)
print("Avg. correctness IRIS with shared-memory-disk and frank-wolfe and own linear programming execution:",
      avg_correctness, flush=True)

# iris (before validated data)
# [0.81333333 0.75407407 0.65037037 0.6562963  0.66518519 0.71851852 0.71555556 0.73037037 0.69481481 0.70666667],
# 0.7108148148148149, 0.04643824165562087
# (after validated data, with non-negative grad)
# [0.81333333 0.74814815 0.65037037 0.72740741 0.6562963  0.69185185 0.65037037 0.74518519 0.69481481 0.72148148],
# 0.709925925925926, 0.04935712917375104

