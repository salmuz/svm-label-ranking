from svm_label_ranking.crossvalidation_svmlr import cross_validation

# Call cross-validation with single parallel process
in_path = "iris_dense.xarff"
out_path = "results_output.csv"
avg_correctness = cross_validation(in_path, out_path)
print("Avg. correctness IRIS with 1 process parallel execution:", avg_correctness, flush=True)

# Call cross-validation with three parallel process
in_path = "iris_dense.xarff"
out_path = "results_output.csv"
avg_correctness = cross_validation(in_path, out_path, nb_process=2)
print("Avg. correctness IRIS with 3 process parallel execution:", avg_correctness, flush=True)
