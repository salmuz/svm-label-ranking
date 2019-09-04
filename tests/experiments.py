from svm_label_ranking.crossvalidation_svmlr import cross_validation
import sys

# inputs values arguments
in_path = sys.argv[1]
out_path = sys.argv[2]
nb_process = int(sys.argv[3]) if len(sys.argv) > 3 else 1

# execute cross-validation svm label-ranking model
cross_validation(in_path, out_path, nb_process=nb_process)
