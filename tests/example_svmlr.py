from svm_label_ranking import arff
from svm_label_ranking.model import SVMLR

# Example script of executing the svm multi label model.
# By default, we set the possible value of hyper-parameter v=[0.1, 0.4, 0.7, 0.9, 8, 24, 32, 128]. It could be
# assigned to other values by modifying the __init__ of the class.

# We start by creating an instance of the base classifier we want to use
print("Example of SVM label ranking - Data set IRIS \n")
v = list([0.1, 0.4, 0.7, 0.9, 8, 24, 32, 128])

model = SVMLR(DEBUG=True)
data_arff = arff.ArffFile()
data_arff.load("iris_dense.xarff")

# Learning
model.learn(data_arff)
print("Process learning with quadratic-programming algorithm finished")

# Prediction
predictions = model.evaluate([data_arff.data[120]])

print("Prediction with quadratic-programing cvxopt is \n")
print(predictions, data_arff.data[120][-1:])
print(model.evaluate([data_arff.data[0]]), data_arff.data[0][-1:])
print("\n")

# Learning
model.learn(data_arff, solver='frank-wolfe')
print("Process learning with frank-wolfe algorithm finished")

predictions = model.evaluate([data_arff.data[120]])
print("Prediction with frank-wolfe is \n")
print(predictions, data_arff.data[120][-1:])
print(model.evaluate([data_arff.data[0]]), data_arff.data[0][-1:])
print("\n")
model.plot_solver_convergence()



