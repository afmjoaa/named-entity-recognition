from pre_process import PreProcess
from word_to_vector import WordToVector
from one_hot_encoding import OneHotEncoder
import numpy as np
from training import DnnTraining
from inference import DnnInference
from keras.models import load_model

def pipeline():
    # Get word array and label array
    wordArr, labelArr = PreProcess.getTrainingTuple(dataFile='train.conll', onlyBioTagging=True)
    # print(wordArr, labelArr)

    # Get wordToVector from [wordArr] and oneHotEncoding from [labelArr]
    wordToVecArr = WordToVector.getPretrainedWordToVecList(wordArr)
    oneHotEncodingArr = OneHotEncoder.getOneHotEncodingOfOutput(labelArr)
    # Convert python array to num py array
    np_wordToVecArr = np.array(wordToVecArr)
    np_oneHotEncodingArr = np.array(oneHotEncodingArr)


    # validation part
    val_wordArr, val_labelArr = PreProcess.getTrainingTuple(dataFile='val.conll', onlyBioTagging=True)

    # Get wordToVector from [wordArr] and oneHotEncoding from [labelArr]
    val_wordToVecArr = WordToVector.getPretrainedWordToVecList(val_wordArr)
    val_oneHotEncodingArr = OneHotEncoder.getOneHotEncodingOfOutput(val_labelArr)
    # Convert python array to num py array
    val_np_wordToVecArr = np.array(val_wordToVecArr)
    val_np_oneHotEncodingArr = np.array(val_oneHotEncodingArr)

    training = DnnTraining(input_dim=300, output_dim=3)
    training.startTraining(np_wordToVecArr, np_oneHotEncodingArr, val_np_wordToVecArr, val_np_oneHotEncodingArr, epochs=50)
    training.saveTrainedModel()


def predict():
    wordArr, labelArr = PreProcess.getTrainingTuple(dataFile='test.conll', onlyBioTagging=True)
    model = load_model('dnn_model.h5')
    inference = DnnInference(wordArr, labelArr)
    inference.inferenceAndPrint(model, 20)


if __name__ == "__main__":
    predict()
