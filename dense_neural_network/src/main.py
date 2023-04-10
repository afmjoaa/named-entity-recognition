from pre_process import PreProcess
from word_to_vector import WordToVector
from one_hot_encoding import OneHotEncoder
from training import DnnTraining

def pipeline():
    # Get word array and label array
    wordArr, labelArr = PreProcess.getTrainingTuple(dataFile='dummy.conll', onlyBioTagging=True)
    print(wordArr, labelArr)

    # Get wordToVector from [wordArr] and oneHotEncoding from [labelArr]
    wordToVecArr = WordToVector.getPretrainedWordToVecList(wordArr)
    oneHotEncodingArr = OneHotEncoder.getOneHotEncodingOfOutput(labelArr)
    training = DnnTraining()
    training.startTraining(wordToVecArr, oneHotEncodingArr,wordToVecArr, oneHotEncodingArr)
    training.saveTrainedModel()
    # print(wordToVecArr)
    # print(oneHotEncodingArr)


if __name__ == "__main__":
    pipeline()
