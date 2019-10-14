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
avg_correctness = cross_validation(in_path, out_path, nb_process=2)
print("Avg. correctness IRIS with 3 process parallel execution:", avg_correctness, flush=True)
