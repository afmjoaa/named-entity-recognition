from pre_process import PreProcess
from word_to_vector import WordToVector
from one_hot_encoding import OneHotEncoder
import numpy as np
from training import DnnTraining
from inference import DnnInference
from keras.models import load_model

def pipeline():
    # Get word array and label array
    wordArr, labelArr = PreProcess.getTrainingTuple(dataFile='dummy.conll', onlyBioTagging=True)
    print(wordArr, labelArr)

    # Get wordToVector from [wordArr] and oneHotEncoding from [labelArr]
    wordToVecArr = WordToVector.getPretrainedWordToVecList(wordArr)
    oneHotEncodingArr = OneHotEncoder.getOneHotEncodingOfOutput(labelArr)

    # Convert python array to num py array
    np_wordToVecArr = np.array(wordToVecArr)
    np_oneHotEncodingArr = np.array(oneHotEncodingArr)
    training = DnnTraining(input_dim=300, output_dim=3)
    training.startTraining(np_wordToVecArr, np_oneHotEncodingArr, np_wordToVecArr, np_oneHotEncodingArr, epochs=20)
    training.saveTrainedModel()


def predict():
    wordArr, labelArr = PreProcess.getTrainingTuple(dataFile='dummy.conll', onlyBioTagging=True)
    model = load_model('dnn_model.h5')
    inference = DnnInference(wordArr, labelArr)
    inference.inferenceAndPrint(model, 5)


if __name__ == "__main__":
    predict()
