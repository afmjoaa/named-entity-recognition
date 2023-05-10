from pre_process import PreProcess
from word_to_vector import WordToVector
from one_hot_encoding import OneHotEncoder
import numpy as np
from training import DnnTraining
from inference import DnnInference
from keras.models import load_model


def pipeline():
    train_data, train_label, val_data, val_label, test_data, test_label = PreProcess.loadAndSPlitDataSet(
        "../data/en_train.conll", "../data/en_dev.conll")

    training = DnnTraining(input_dim=300, output_dim=3)
    training.startTraining(train_data, train_label, val_data, val_label, epochs=50)
    training.saveTrainedModel()


def predict(test_data, test_label):
    model = load_model('dnn_model.h5')
    inference = DnnInference(test_data, test_label)
    inference.inferenceAndPrint(model, 20)


if __name__ == "__main__":
    model = load_model('dnn_model.h5')

