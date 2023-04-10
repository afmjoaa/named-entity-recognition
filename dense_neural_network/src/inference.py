from word_to_vector import WordToVector
import numpy as np
from one_hot_encoding import OneHotEncoder

class DnnInference:
    def __init__(self, testWordArray, testLabelArray):
        self.testWordArray = testWordArray
        self.testLabelArray = testLabelArray

    def inferenceAndPrint(self, model, limit=100):
        for i in range(min(limit, len(self.testWordArray))):
            currentWord = self.testWordArray[i]
            wordToVec = WordToVector.getPretrainedWordToVec(currentWord)
            np_wordToVec = np.array(wordToVec)
            input_data = np_wordToVec.reshape((1, np_wordToVec.shape[0]))
            predictions = model.predict(input_data)
            print(currentWord, self.testLabelArray[i], OneHotEncoder.getDecodedLabel(predictions[0]), predictions[0],)

    @staticmethod
    def inferenceFromTrainedWordToVec(wordToVecList, model, limit: int):
        for i in range(limit):
            predictions = model.predict(wordToVecList[i])
            print(predictions)

