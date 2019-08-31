from svm_label_raking import arff
from svm_label_raking.model import SVMLR

# Example script of executing the svm multi label model.
# By default, we set the possible value of hyper-parameter v=[0.1, 0.4, 0.7, 0.9, 8, 24, 32, 128]. It could be
# assigned to other values by modifying the __init__ of the class.

# We start by creating an instance of the base classifier we want to use
print("Example of SVM label ranking - Data set IRIS \n")
v = list([0.1, 0.4, 0.7, 0.9, 8, 24, 32, 128])

model = SVMLR(v_list_default=v)
data_arff = arff.ArffFile()
data_arff.load("iris_dense.xarff")

# Learning
model.learn(data_arff)
print("Process learning finished")

# Prediction
predictions = model.predict([data_arff.data[1], data_arff.data[2]])

print("Prediction is \n")
print(predictions)
print("\n")

