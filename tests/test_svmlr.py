# Sample Test passing with nose and pytest
from svm_label_ranking import arff
from svm_label_ranking.model import SVMLR


def test_load_data():
    data_arff = arff.ArffFile()
    data_arff.load("testing_data_2x3.xarff")

    assert data_arff.data[0][2] == "L1>L2>L3", "Should be L1>L2>L3"


def test_learning_model():
    data_arff = arff.ArffFile()
    data_arff.load("testing_data_3x4.xarff")

    svmlr = SVMLR(DEBUG=True)
    svmlr.learn(data_arff)

