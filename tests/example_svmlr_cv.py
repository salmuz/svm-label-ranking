from svm_label_ranking.crossvalidation_svmlr import cross_validation

SEED_SCIENTIFIC = [1234]
# Call cross-validation with single parallel process
in_path = "iris_dense.xarff"
out_path = "results_output.csv"
avg_correctness = cross_validation(in_path, out_path, nb_process=1, n_times_repeat=1,
                                   seeds=SEED_SCIENTIFIC, DEBUG=False)
print("Avg. correctness IRIS with 1 process parallel execution:", avg_correctness, flush=True)

# Call cross-validation with three parallel process
in_path = "iris_dense.xarff"
out_path = "results_output.csv"
avg_correctness = cross_validation(in_path, out_path, nb_process=3, DEBUG=True, seeds=seeds)
print("Avg. correctness IRIS with 3 process parallel execution:", avg_correctness, flush=True)

# Call cross-validation with shared memory and disk for saving H matrix
in_path = "iris_dense.xarff"
out_path = "results_output.csv"
avg_correctness = cross_validation(in_path, out_path, DEBUG=True, is_H_shared_memory_disk=True,
                                   SOLVER_LP="salmuz", SOLVER_QP="frank-wolfe", start_idx_label_disk=2)
print("Avg. correctness IRIS with shared-memory-disk and frank-wolfe and own linear programming execution:",
      avg_correctness, flush=True)

